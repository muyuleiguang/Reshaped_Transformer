# -*- coding: utf-8 -*-
"""
多任务 Transformer 模型（轴承故障分类 + 振动趋势预测）
=========================================================
本文件包含三大改进：
1. **多尺度卷积嵌入**（Multi‑Scale ConvEmbedding）
2. **T‑PE 位置先验偏置**（Temporal Positional Bias）
3. **门控稀疏注意力**（Gated Sparse Attention）  ← 🆕 本次新增
   - 局部窗口 + 全局锚点掩码
   - 点积阈值门控 `τ`，显式筛除弱相关

所有代码已用中文注释，修改段落以 `# === 新增 ===` / `# === 修改 ===` 标记。
"""
from typing import Tuple, List, Optional, Sequence
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# === 保留: 多尺度卷积嵌入模块 ===
# ---------------------------------------------------------------------------
class MultiScaleConvEmbedding(nn.Module):
    """多尺度卷积嵌入"""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        kernel_sizes: Tuple[int, ...] | List[int] = (3, 5, 9),
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, conv_channels, k, stride=stride, padding=k // 2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(inplace=True),
            )
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)                       # (B, C, L)
        outs = [path(x) for path in self.paths]
        x = torch.cat(outs, dim=1)                   # (B, C*, L)
        return x.permute(0, 2, 1)                    # (B, L, C*)

# ---------------------------------------------------------------------------
# === 保留: T‑PE 位置先验偏置 ===
# ---------------------------------------------------------------------------

def build_tpe_bias(
    seq_len: int,
    *,
    sigma: Optional[float] = None,
    period: int = 401,
    bias_strength: float = 0.3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """构造 T‑PE 偏置矩阵 B = B_gauss + B_per。"""
    device = device or torch.device("cpu")
    idx = torch.arange(seq_len, device=device)
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    sigma = sigma or (period / 2)
    gauss = torch.exp(-(d.float() ** 2) / (2 * sigma ** 2))
    periodic = bias_strength * (d % period == 0).float()
    return gauss + periodic                                # (L, L)

# ---------------------------------------------------------------------------
# === 保留: 正弦位置编码 ===
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-(torch.log(torch.tensor(10000.0)) / d_model)))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1), :]

# ---------------------------------------------------------------------------
# === 新增: 门控稀疏注意力模块 ===
# ---------------------------------------------------------------------------
class GatedSparseAttention(nn.Module):
    """门控稀疏多头注意力

    - 仅计算局部窗口 `w` 内或全局锚点之间的注意力。
    - 对允许位置再用阈值 `τ` 进行门控，去除弱相关。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),  # 默认把首 token 设为全局（类 [CLS]）
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        # QKV 线性投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 稀疏/门控参数
        self.window_size = window_size
        self.tau = tau
        self.register_buffer("global_mask", self._build_global_mask(global_indices), persistent=False)

    # -----------------------------
    # 辅助: 构造全局锚点布尔向量
    # -----------------------------
    def _build_global_mask(self, indices: Sequence[int]) -> torch.Tensor:
        mask = torch.zeros(1, dtype=torch.bool)  # placeholder, 真正大小在 forward 构造
        if len(indices):
            mask = torch.tensor(indices, dtype=torch.long)
        return mask

    # -----------------------------
    # 前向传播
    # -----------------------------
    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) ; pos_bias: (L, L)
        B, L, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        device = x.device

        # (B, L, D) → (B, H, L, Dh)
        def _proj(linear, t):
            return linear(t).view(B, L, H, Dh).transpose(1, 2)  # (B, H, L, Dh)

        q = _proj(self.q_proj, x)
        k = _proj(self.k_proj, x)
        v = _proj(self.v_proj, x)

        # 缩放点积
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # ---------- 构造允许关注掩码 (局部 + 全局) ----------
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()          # (L, L)
        local_mask = dist <= self.window_size                       # (L, L)

        if self.global_mask.numel() == 1:                           # 未显式设置
            g_idx = torch.tensor([0], device=device)                # 默认 CLS=0
        else:
            g_idx = self.global_mask.to(device)
        i_global = torch.zeros(L, dtype=torch.bool, device=device)
        j_global = torch.zeros(L, dtype=torch.bool, device=device)
        i_global[g_idx] = True
        j_global[g_idx] = True
        allowed = local_mask | i_global.unsqueeze(1) | j_global.unsqueeze(0)  # (L, L)
        allowed = allowed.unsqueeze(0).unsqueeze(0)                            # (1,1,L,L)

        # ---------- 阈值门控 ----------
        gate = (scores > self.tau).to(scores.dtype)               # (B,H,L,L)

        # 对不允许的位置置 -inf，允许但 gate=0 也置 -inf
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg_inf)
        scores = scores.masked_fill((allowed & (gate == 0)), neg_inf)

        # ---------- 加位置偏置，Softmax ----------
        scores = scores + pos_bias.to(scores.dtype).unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # ---------- 加权求和 ----------
        out = torch.matmul(attn, v)                                 # (B,H,L,Dh)
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)     # (B,L,D)
        return self.out_proj(out)                                   # (B,L,D)

# ---------------------------------------------------------------------------
# === 修改: Transformer 编码器块，使用门控稀疏注意力 ===
# ---------------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """单层 Transformer 编码器（Gated Sparse Attention + FFN）"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = GatedSparseAttention(
            embed_dim,
            num_heads,
            window_size=window_size,
            tau=tau,
            global_indices=global_indices,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor):
        # ---- 门控稀疏注意力 ----
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.attn(x, pos_bias))

        # ---- 前馈网络 ----
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x

# ---------------------------------------------------------------------------
# === 修改: Transformer Backbone 传递稀疏参数 ===
# ---------------------------------------------------------------------------
class TransformerBackbone(nn.Module):
    """由若干 `TransformerEncoderBlock` 组成的编码器栈"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        *,
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim,
                num_heads,
                ffn_dim,
                window_size=window_size,
                tau=tau,
                global_indices=global_indices,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor):
        for layer in self.layers:
            x = layer(x, pos_bias)
        return x

# ---------------------------------------------------------------------------
# === 修改: 多任务模型，新增稀疏注意力超参 ===
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    """多任务 Transformer 模型（分类 + 趋势预测）"""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        conv_channels: int = 128,
        kernel_sizes: Tuple[int, ...] | List[int] = (3, 5, 9),
        embed_dim: Optional[int] = None,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        num_classes: int = 10,
        num_regression: int = 1024,
        dropout: float = 0.1,
        # --- T‑PE 参数 ---
        period: int = 401,
        sigma: Optional[float] = None,
        bias_strength: float = 0.3,
        # --- 稀疏注意力参数 ---
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        self.period = period
        self.sigma = sigma
        self.bias_strength = bias_strength

        # 嵌入层
        self.embedding = MultiScaleConvEmbedding(in_channels, conv_channels, kernel_sizes)
        conv_out_dim = conv_channels * len(kernel_sizes)
        self.embed_dim = embed_dim or conv_out_dim
        self.proj = nn.Identity() if conv_out_dim == self.embed_dim else nn.Linear(conv_out_dim, self.embed_dim)

        # 位置编码 & Backbone
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        self.backbone = TransformerBackbone(
            self.embed_dim,
            num_heads,
            num_layers,
            ffn_dim,
            window_size=window_size,
            tau=tau,
            global_indices=global_indices,
            dropout=dropout,
        )

        # 任务头
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.regressor = nn.Linear(self.embed_dim, num_regression)

    # -----------------------------
    # 前向传播
    # -----------------------------
    def forward(self, x: torch.Tensor):
        B, L, _ = x.shape
        device = x.device

        # 构造 T‑PE 偏置矩阵
        pos_bias = build_tpe_bias(
            L,
            sigma=self.sigma,
            period=self.period,
            bias_strength=self.bias_strength,
            device=device,
        )  # (L, L)

        # 嵌入 + 投影
        x = self.proj(self.embedding(x))
        x = self.positional_encoding(x)

        # 带稀疏注意力的 Transformer 主干
        x = self.backbone(x, pos_bias)

        pooled = x[:, 0, :]
        return self.classifier(pooled), self.regressor(pooled)

# ---------------------------------------------------------------------------
# 快速单元测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L = 2, 256
    dummy = torch.randn(B, L, 1)
    model = MultiTaskModel()
    cls, reg = model(dummy)
    print(cls.shape, reg.shape)
# Output: torch.Size([2, 10]) torch.Size([2, 1024])
