# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

# 卷积嵌入：将原始输入经卷积层提取特征
class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # x 假定为 (batch, seq_len, features)
        # 转换为 (batch, features, seq_len) 以适配 Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 恢复维度顺序为 (batch, seq_len', conv_channels)
        x = x.permute(0, 2, 1)
        return x

# 位置编码：为序列添加位置编码信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # 位置编码：偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x 假定形状为 (batch, seq_len, d_model)
        seq_len = x.size(1)
        # 将位置编码加到输入 x 上
        x = x + self.pe[:, :seq_len, :]
        return x

# Transformer 编码器块：包含多头自注意力和前馈网络
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x 形状 (batch, seq_len, embed_dim)
        # 先归一化，然后应用自注意力
        residual = x
        x = self.norm1(x)
        # PyTorch 的 MultiheadAttention 期望 (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.mha(x, x, x)
        attn_output = attn_output.transpose(0, 1)
        x = residual + self.dropout(attn_output)
        # 再进行前馈网络和残差连接
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x

# Transformer 主干网络：堆叠多个 TransformerEncoderBlock
class TransformerBackbone(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, dropout) 
            for _ in range(num_layers)
        ])
    def forward(self, x):
        # x 形状 (batch, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x

# 多任务模型：包含 Transformer 主干网络及多个任务头
class MultiTaskModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        num_classes: int = 10,       # 十类别
        num_regression: int = 1024
    ):
        super().__init__()
        # 卷积嵌入和位置编码
        self.embedding = ConvEmbedding(in_channels, conv_channels)
        self.positional_encoding = PositionalEncoding(embed_dim)
        # Transformer 主干网络
        self.backbone = TransformerBackbone(embed_dim, num_heads, num_layers, ffn_dim)
        # 任务头：分类和回归
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.regressor  = nn.Linear(embed_dim, num_regression)

    def forward(self, x):
        # 输入 x 假定形状为 (batch, seq_len, in_channels)
        # 先通过卷积嵌入提取特征
        x = self.embedding(x)
        # 为了与 Transformer 期望的维度一致，将 conv 输出转为 embed_dim
        # 假设 conv_channels == embed_dim，否则需要额外线性变换
        # 添加位置编码
        x = self.positional_encoding(x)
        # 通过 Transformer 主干网络
        x = self.backbone(x)
        # 取序列第一个位置或进行池化作为整体特征
        # 这里直接使用序列首位置作为分类/回归输入
        pooled = x[:, 0, :]  # (batch, embed_dim)
        # 生成分类和回归输出
        class_out = self.classifier(pooled)
        reg_out = self.regressor(pooled)
        return class_out, reg_out
