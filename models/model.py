# -*- coding: utf-8 -*-
"""
多任务 Transformer 模型（轴承故障分类 + 振动趋势预测）
=========================================================
本文件包含三大改进：
1. **多尺度卷积嵌入**（Multi‑Scale ConvEmbedding）
2. **T‑PE 位置先验偏置**（Temporal Positional Bias，现已支持可学习的 sigma 和 bias_strength）
3. **门控稀疏注意力**（Gated Sparse Attention）  ← 本次新增
   - 局部窗口 + 全局锚点掩码
   - 点积阈值门控 `τ`，显式筛除弱相关

所有代码均配有中文注释，新增/修改部分用 `# === 新增 ===` 与 `# === 修改 ===` 标记。
"""
from typing import Tuple, List, Optional, Sequence
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# === 保留: 多尺度卷积嵌入模块 ===
# ---------------------------------------------------------------------------
class MultiScaleConvEmbedding(nn.Module):
    """多尺度卷积嵌入：并行多个一维卷积，捕获多种感受野下的局部特征。"""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        kernel_sizes: Tuple[int, ...] | List[int] = (3, 5, 9),
        stride: int = 1,
    ) -> None:
        super().__init__()
        # paths 列表中每个模块对应一个卷积核大小 k
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, conv_channels, k, stride=stride, padding=k // 2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(inplace=True),
            )
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (batch, seq_len, in_channels)
        # 转换为 (batch, in_channels, seq_len) 以适配 Conv1d
        x = x.permute(0, 2, 1)  # (B, C_in, L)
        # 并行计算各卷积分支输出，形状均为 (B, conv_channels, L)
        outs = [path(x) for path in self.paths]
        # 在通道维拼接：(B, conv_channels * M, L)
        x = torch.cat(outs, dim=1)
        # 恢复到 (batch, seq_len, conv_channels * M)
        return x.permute(0, 2, 1)

# ---------------------------------------------------------------------------
# === 保留: T‑PE 位置先验偏置 ===
# ---------------------------------------------------------------------------

def build_tpe_bias(
    seq_len: int,
    *,
    sigma: torch.Tensor | float = None,
    period: int = 401,
    bias_strength: torch.Tensor | float = 0.3,
    device: torch.device | None = None,
) -> torch.Tensor:
    """构造 T‑PE 偏置矩阵：B = B_gauss + B_per。

    参数说明：
    - seq_len: 序列长度 L
    - sigma: 高斯衰减标准差，可为标量或可学习张量；若为 None，则默认 period/2
    - period: 周期 T，用于周期性强化
    - bias_strength: 周期性偏置幅度 b，可为张量或浮点数
    - device: 张量所在设备

    返回：
    - B: 偏置矩阵，形状 (L, L)
    """
    device = device or torch.device("cpu")
    # 生成位置索引 [0,1,...,L-1]
    idx = torch.arange(seq_len, device=device)
    # 计算绝对距离矩阵 d_ij = |i - j|
    d = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (L, L)

    # 如果 sigma 是标量，则转为张量
    if isinstance(sigma, float) or isinstance(sigma, int) or sigma is None:
        sigma_val = float(sigma) if sigma is not None else float(period) / 2.0
        sigma_tensor = torch.tensor(sigma_val, device=device)
    else:
        sigma_tensor = sigma.to(device)
    # 高斯衰减 B_gauss = exp(-d^2 / (2σ^2))
    gauss = torch.exp(-(d.float() ** 2) / (2 * sigma_tensor ** 2))  # (L, L)

    # 如果 bias_strength 为标量则转张量
    if isinstance(bias_strength, float) or isinstance(bias_strength, int):
        b_val = float(bias_strength)
        b_tensor = torch.tensor(b_val, device=device)
    else:
        b_tensor = bias_strength.to(device)
    # 周期性强化：当 d % T == 0 时额外加偏置 b
    periodic = b_tensor * (d % period == 0).float()  # (L, L)

    return gauss + periodic  # (L, L)

# ---------------------------------------------------------------------------
# === 保留: 正弦位置编码 ===
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """经典正弦/余弦位置编码，可与 T-PE 叠加使用。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 构造位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(torch.log(torch.tensor(10000.0)) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为缓冲区，不参与梯度更新，但会随模型保存/加载
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        # 直接将前 L 长度的编码叠加到 x 上
        return x + self.pe[:, : x.size(1), :]

# ---------------------------------------------------------------------------
# === 新增: 门控稀疏注意力模块 ===
# ---------------------------------------------------------------------------
class GatedSparseAttention(nn.Module):
    """门控稀疏多头注意力

    1. 局部滑动窗口+全局锚点：每个位置仅与自身 +/- window_size 范围内，以及全局锚点产生注意力交互。
    2. 阈值门控 (threshold gating)：对注意力分数小于 tau 的条目直接屏蔽。
    3. 最后再加入 T-PE 偏置，并做 Softmax。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),  # 默认第 0 个位置为全局锚点
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # QKV 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # 稀疏注意力相关参数
        self.window_size = window_size
        self.tau = tau
        # global_mask 存储全局索引列表，用于 forward 时构造全局掩码
        self.register_buffer("global_mask", self._build_global_mask(global_indices), persistent=False)

    def _build_global_mask(self, indices: Sequence[int]) -> torch.Tensor:
        """辅助函数：根据传入的索引列表构造全局锚点列表（一维）。"""
        if len(indices) == 0:
            return torch.zeros(1, dtype=torch.long)
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        """
        前向计算：
        - x: (B, L, D)
        - pos_bias: T-PE 偏置，形状 (L, L)
        返回：
        - 输出 (B, L, D)
        """
        B, L, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        device = x.device

        # 线性投影并拆分多头：得到 (B, H, L, Dh)
        def project_to_heads(linear, t: torch.Tensor) -> torch.Tensor:
            # 先 (B, L, D) -> (B, L, D)，再 reshape (B, L, H, Dh) -> transpose (B, H, L, Dh)
            return linear(t).view(B, L, H, Dh).transpose(1, 2)

        q = project_to_heads(self.q_proj, x)
        k = project_to_heads(self.k_proj, x)
        v = project_to_heads(self.v_proj, x)

        # 计算缩放点积注意力分数 (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # QK^T / sqrt(d)

        # ---------------- 构造允许关注掩码：局部 + 全局 ----------------
        idx = torch.arange(L, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (L, L)
        local_mask = dist <= self.window_size                 # (L, L)

        # 取出全局索引
        if self.global_mask.numel() == 1 and self.global_mask[0] == 0:
            # 若 global_mask 里仅含 0，则默认位置 0 为锚点
            g_idx = torch.tensor([0], device=device)
        else:
            g_idx = self.global_mask.to(device)
        # i_global 和 j_global 标记全局位置布尔向量
        i_global = torch.zeros(L, dtype=torch.bool, device=device)
        j_global = torch.zeros(L, dtype=torch.bool, device=device)
        i_global[g_idx] = True
        j_global[g_idx] = True
        # allowed = local_mask 或 i_global 或 j_global
        allowed = local_mask | i_global.unsqueeze(1) | j_global.unsqueeze(0)  # (L, L)
        # 扩展到 (B, H, L, L) 方便与 scores 逐元素对比
        allowed = allowed.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        # ---------------- 阈值门控：scores > tau 才保留 ----------------
        gate = (scores > self.tau).to(scores.dtype)  # (B, H, L, L)
        # 定义负无穷
        neg_inf = torch.finfo(scores.dtype).min
        # 对不在 allowed 的位置置 -inf
        scores = scores.masked_fill(~allowed, neg_inf)
        # 对 allowed 且分数 < tau 的位置也置 -inf
        scores = scores.masked_fill((allowed & (gate == 0)), neg_inf)

        # --------------- 加入 T-PE 偏置后 Softmax ---------------
        # pos_bias: (L, L)，broadcast 到 (B, H, L, L)
        scores = scores + pos_bias.to(scores.dtype).unsqueeze(0).unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)  # (B, H, L, L)
        attn = self.dropout(attn)

        # --------------- 注意力加权求和 ---------------
        out = torch.matmul(attn, v)                   # (B, H, L, Dh)
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)  # (B, L, D)
        return self.out_proj(out)  # (B, L, D)

# ---------------------------------------------------------------------------
# === 修改: Transformer 编码器块，使用门控稀疏注意力 ===
# ---------------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """单层 Transformer 编码器（Gated Sparse Attention + 前馈网络）"""

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
        # 使用门控稀疏注意力替代原生 MultiheadAttention
        self.attn = GatedSparseAttention(
            embed_dim,
            num_heads,
            window_size=window_size,
            tau=tau,
            global_indices=global_indices,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        # 前馈网络：两层全连接
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        # ---- 门控稀疏注意力子层 ----
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.dropout(self.attn(x_norm, pos_bias))

        # ---- 前馈网络子层 ----
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(self.ffn(x_norm))
        return x

# ---------------------------------------------------------------------------
# === 修改: Transformer Backbone，传递稀疏注意力相关超参 ===
# ---------------------------------------------------------------------------
class TransformerBackbone(nn.Module):
    """由若干 `TransformerEncoderBlock` 层级联组成的编码器栈"""

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

    def forward(self, x: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_bias)
        return x

# ---------------------------------------------------------------------------
# === 修改: 多任务模型，新增可学习的 sigma 和 bias_strength ===
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
        # --- T‑PE 参数（sigma 和 bias_strength 转为可学习） ---
        period: int = 401,
        sigma: float = None,
        bias_strength: float = 0.3,
        # --- 稀疏注意力参数 ---
        window_size: int = 5,
        tau: float = 0.0,
        global_indices: Sequence[int] = (0,),
    ) -> None:
        super().__init__()
        # period 仍然作为超参数，不做可学习
        self.period = period
        # === 新增：将 sigma 设为可学习参数 ===
        sigma_init = float(sigma) if sigma is not None else float(period) / 2.0
        self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))
        # === 新增：将 bias_strength 设为可学习参数 ===
        self.bias_strength = nn.Parameter(torch.tensor(float(bias_strength), dtype=torch.float32))

        # 嵌入层：多尺度卷积
        self.embedding = MultiScaleConvEmbedding(in_channels, conv_channels, kernel_sizes)
        conv_out_dim = conv_channels * len(kernel_sizes)
        # 如果未指定 embed_dim，则直接使用 conv_out_dim
        self.embed_dim = embed_dim or conv_out_dim
        # 投影层：当 conv_out_dim != embed_dim 时，将通道数映射到 embed_dim
        self.proj = nn.Identity() if conv_out_dim == self.embed_dim else nn.Linear(conv_out_dim, self.embed_dim)

        # 位置编码（正弦编码）
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        # Transformer 主干：使用门控稀疏注意力
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

        # 任务头：分类和回归
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.regressor = nn.Linear(self.embed_dim, num_regression)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播流程：
        1. 生成 T-PE 偏置矩阵（使用可学习的 sigma 与 bias_strength）
        2. 多尺度卷积嵌入 + 可选投影
        3. 正弦位置编码
        4. 带稀疏注意力的 Transformer 主干
        5. 取序列首位特征用于分类和回归任务头

        输入：
            x: (batch, seq_len, in_channels)
        返回：
            class_out: 分类 logits，形状 (batch, num_classes)
            reg_out:  回归输出，形状 (batch, num_regression)
        """
        B, L, _ = x.shape
        device = x.device

        # === 新增：构造 T-PE 偏置矩阵 ===
        pos_bias = build_tpe_bias(
            L,
            sigma=self.sigma,               # 可学习张量
            period=self.period,              # 固定超参数
            bias_strength=self.bias_strength,  # 可学习张量
            device=device,
        )  # (L, L)

        # --- 多尺度卷积嵌入 + 投影 ---
        x = self.embedding(x)  # (B, L, conv_out_dim)
        x = self.proj(x)       # (B, L, embed_dim)

        # --- 正弦位置编码 ---
        x = self.positional_encoding(x)  # (B, L, embed_dim)

        # --- 带稀疏注意力的 Transformer 主干 ---
        x = self.backbone(x, pos_bias)  # (B, L, embed_dim)

        # --- 池化：取序列首位作为整体特征 ---
        pooled = x[:, 0, :]  # (B, embed_dim)
        # --- 任务头 ---
        class_out = self.classifier(pooled)  # (B, num_classes)
        reg_out = self.regressor(pooled)     # (B, num_regression)
        return class_out, reg_out

# ---------------------------------------------------------------------------
# 快速单元测试：用于检查模型维度是否匹配
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, L = 2, 256
    dummy = torch.randn(B, L, 1)
    model = MultiTaskModel()
    cls, reg = model(dummy)
    print("分类输出形状:", cls.shape)  # 应为 (2, num_classes)
    print("回归输出形状:", reg.shape)  # 应为 (2, num_regression)
