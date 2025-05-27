import torch
import torch.nn as nn
from models.transformer_model import TransformerWithSparseAttention

class MultiTaskModel(nn.Module):
    """
    Multi-task model with a shared Transformer encoder and separate classification and regression heads.
    """
    def __init__(self, in_channels=1, embed_dim=128, kernel_sizes=(3,5,9),
                 num_layers=4, num_heads=8, dim_feedforward=256, local_window_size=5,
                 gating_threshold=0.0, gauss_sigma=2.0, periodic_strength=1.0, period=100.0, dropout=0.1):
        super(MultiTaskModel, self).__init__()
        # Shared Transformer encoder
        self.transformer = TransformerWithSparseAttention(
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_sizes=kernel_sizes,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            local_window_size=local_window_size,
            gating_threshold=gating_threshold,
            gauss_sigma=gauss_sigma,
            periodic_strength=periodic_strength,
            period=period,
            dropout=dropout
        )
        # Classification head (4 classes)
        self.classifier = nn.Linear(embed_dim, 4)
        # Regression head (predict next 1024 points)
        self.regressor = nn.Linear(embed_dim, 1024)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, in_channels, seq_len]
        returns: (cls_logits, reg_output)
          cls_logits: [batch_size, 4], raw classification scores
          reg_output: [batch_size, 1024], regression predictions
        """
        # Pass through transformer
        features = self.transformer(x)  # [batch, seq_len, embed_dim]
        # Pooling: take mean over time dimension
        pooled = features.mean(dim=1)  # [batch, embed_dim]
        # Classification head
        cls_logits = self.classifier(pooled)
        # Regression head
        reg_output = self.regressor(pooled)
        return cls_logits, reg_output
