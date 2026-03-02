import torch
import torch.nn as nn

class Linear2DPositionalEncoding(nn.Module):
    """
    Trainable linear positional encoding for 2D (channel, time) grids.

    Given channel index c and time index t, this module computes
        pe = W @ [c_norm, t_norm] + b
    where W has shape (embed_dim, 2) and b has shape (embed_dim).

    The indices are normalized to [0, 1] so the learned slopes can generalize
    across different numbers of channels/time steps.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # weight: (embed_dim, 2), bias: (embed_dim)
        self.linear = nn.Linear(2, embed_dim, bias=True)

    def forward(self, num_channels: int, num_time_patches: int) -> torch.Tensor:
        # Create coordinate grid (C*T, 2)
        c = torch.arange(num_channels, dtype=torch.float32)
        t = torch.arange(num_time_patches, dtype=torch.float32)
        # Normalize to [0, 1]
        c_norm = c / max(num_channels - 1, 1)
        t_norm = t / max(num_time_patches - 1, 1)
        # Broadcast to grid
        cc = c_norm.unsqueeze(1).repeat(1, num_time_patches).reshape(-1)
        tt = t_norm.unsqueeze(0).repeat(num_channels, 1).reshape(-1)
        coords = torch.stack([cc, tt], dim=1)
        # Compute linear positional embeddings
        return self.linear(coords)

class Learnable2DPositionalEncoding(nn.Module):
    """
    Trainable 2D positional encoding using separate embeddings for channels and
    time patches. For each position (channel, time), the embedding is the sum
    (or concatenation) of the channel and time embeddings.
    """
    def __init__(self, embed_dim: int, combine: str = "sum"):
        super().__init__()
        assert combine in ("sum", "concat"), "combine must be 'sum' or 'concat'"
        self.combine = combine
        if combine == "sum":
            self.channel_embed = None  # to be created on first call
            self.time_embed = None
            self.embed_dim = embed_dim
        else:
            # If concatenating, each embedding has half dimension
            assert embed_dim % 2 == 0, "embed_dim must be even when concatenating"
            self.embed_dim = embed_dim

    def _init_embeddings(self, num_channels: int, num_time_patches: int):
        if self.combine == "sum":
            self.channel_embed = nn.Embedding(num_channels, self.embed_dim)
            self.time_embed = nn.Embedding(num_time_patches, self.embed_dim)
        else:
            half_dim = self.embed_dim // 2
            self.channel_embed = nn.Embedding(num_channels, half_dim)
            self.time_embed = nn.Embedding(num_time_patches, half_dim)

    def forward(self, num_channels: int, num_time_patches: int) -> torch.Tensor:
        # Lazily initialize embeddings when dimensions are known
        if self.channel_embed is None or self.time_embed is None:
            self._init_embeddings(num_channels, num_time_patches)
        # Indices for channels and time
        c_idx = torch.arange(num_channels).unsqueeze(1).repeat(1, num_time_patches).reshape(-1)
        t_idx = torch.arange(num_time_patches).unsqueeze(0).repeat(num_channels, 1).reshape(-1)
        if self.combine == "sum":
            return self.channel_embed(c_idx) + self.time_embed(t_idx)
        else:
            return torch.cat([self.channel_embed(c_idx), self.time_embed(t_idx)], dim=1)

