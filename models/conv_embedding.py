import torch
import torch.nn as nn

class ConvEmbedding(nn.Module):
    """
    CoMer style multi-scale convolution embedding.
    Applies parallel 1D convolutions with different kernel sizes and concatenates outputs.
    """
    def __init__(self, in_channels=1, embed_dim=128, kernel_sizes=(3,5,9)):
        super(ConvEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.kernel_sizes = kernel_sizes
        num_kernels = len(kernel_sizes)
        # Determine output channels for each conv so that total equals embed_dim
        base_channels = embed_dim // num_kernels
        remainder = embed_dim - base_channels * num_kernels
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Distribute remainder channels
            out_channels = base_channels + (1 if i < remainder else 0)
            padding = (k - 1) // 2  # keep length same after conv
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=padding)
            self.convs.append(conv)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, in_channels, seq_len]
        returns: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Apply each convolution
        conv_outputs = [conv(x) for conv in self.convs]  # list of [batch, out_channels_i, seq_len]
        # Concatenate along channel dimension
        out = torch.cat(conv_outputs, dim=1)  # [batch, embed_dim, seq_len]
        # Transpose to [batch, seq_len, embed_dim] for transformer input
        out = out.permute(0, 2, 1).contiguous()
        return out
