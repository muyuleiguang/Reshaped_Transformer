import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSparseAttention(nn.Module):
    """
    Gated Sparse Attention module with Longformer-style local+global patterns, gating, and temporal positional bias.
    """
    def __init__(self, embed_dim, num_heads=8, local_window_size=5, gating_threshold=0.0, 
                 gauss_sigma=2.0, periodic_strength=1.0, period=401.0, dropout=0.1):
        super(GatedSparseAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.local_window = local_window_size
        self.gating_threshold = gating_threshold
        self.gauss_sigma = gauss_sigma
        self.periodic_strength = periodic_strength
        self.period = period

        # Linear projections for Q, K, V and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, embed_dim]
        returns: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.size()
        # Project inputs to multi-head Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, embed_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Reshape and transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).permute(0,2,1,3)
        # Compute raw attention scores [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Create local attention mask
        idxs = torch.arange(seq_len, device=x.device)
        dist = idxs[:, None] - idxs[None, :]
        local_mask = (dist.abs() <= (self.local_window // 2)).float()
        # Global attention: token 0 attends to all and all attend token 0
        global_mask = torch.zeros_like(local_mask)
        global_mask[0, :] = 1.0
        global_mask[:, 0] = 1.0
        attn_mask = local_mask | global_mask  # allowed positions

        # Mask out disallowed positions
        scores = scores.masked_fill(attn_mask[None, None, :, :] == 0, float('-1e9'))

        # Add temporal positional bias (Gaussian + periodic)
        with torch.no_grad():
            diff = dist.float()
            gauss_bias = torch.exp(-0.5 * (diff / self.gauss_sigma)**2)
            periodic_bias = self.periodic_strength * torch.cos(2 * torch.pi * diff / self.period)
            pos_bias = gauss_bias + periodic_bias
        scores = scores + pos_bias.unsqueeze(0).unsqueeze(0)

        # Apply gating: mask out low scores
        scores = scores.masked_fill(scores < self.gating_threshold, float('-1e9'))

        # Attention probabilities
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        # Weighted sum of values
        context = torch.matmul(attn_probs, V)  # [batch, num_heads, seq_len, head_dim]
        # Merge heads
        context = context.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, self.embed_dim)
        # Final linear projection
        output = self.out_proj(context)
        return output
