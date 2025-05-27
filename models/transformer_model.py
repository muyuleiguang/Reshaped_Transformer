import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv_embedding import ConvEmbedding
from models.attention import GatedSparseAttention

class TransformerEncoderLayerSparse(nn.Module):
    """
    Transformer Encoder Layer that uses GatedSparseAttention instead of nn.MultiheadAttention.
    """
    def __init__(self, embed_dim=128, num_heads=8, dim_feedforward=256, local_window_size=5,
                 gating_threshold=0.0, gauss_sigma=2.0, periodic_strength=1.0, period=100.0, dropout=0.1):
        super(TransformerEncoderLayerSparse, self).__init__()
        self.self_attn = GatedSparseAttention(embed_dim, num_heads, local_window_size,
                                              gating_threshold, gauss_sigma, periodic_strength, period, dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        src: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Self-attention block
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # Feed-forward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerWithSparseAttention(nn.Module):
    """
    Transformer model that applies multi-scale ConvEmbedding and multiple TransformerEncoder layers with sparse attention.
    """
    def __init__(self, in_channels=1, embed_dim=128, kernel_sizes=(3,5,9),
                 num_layers=4, num_heads=8, dim_feedforward=256, local_window_size=5, gating_threshold=0.0,
                 gauss_sigma=2.0, periodic_strength=1.0, period=100.0, dropout=0.1):
        super(TransformerWithSparseAttention, self).__init__()
        self.conv_embed = ConvEmbedding(in_channels, embed_dim, kernel_sizes)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerSparse(embed_dim, num_heads, dim_feedforward,
                                          local_window_size, gating_threshold,
                                          gauss_sigma, periodic_strength, period, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, in_channels, seq_len]
        returns: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Convolutional embedding
        x = self.conv_embed(x)  # [batch, seq_len, embed_dim]
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)
        return x
