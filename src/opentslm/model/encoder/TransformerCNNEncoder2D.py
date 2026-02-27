# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn

from opentslm.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    MAE-style 1D sin-cos embedding.
    pos: (M,) float tensor of positions
    returns: (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    device, dtype = pos.device, pos.dtype

    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (base ** omega)  # (D/2,)

    out = pos.unsqueeze(1) * omega.unsqueeze(0)  # (M, D/2)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_ts(
    embed_dim: int,
    num_channels: int,
    num_time_patches: int,
    base: float = 10000.0,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Fixed 2D sin-cos PE for a (channel x time_patch) grid.

    Token order assumed: channel-major then time-major, i.e.
      idx = c * num_time_patches + t
    which matches Conv2d output flatten(2) when shape is (B, D, C, T).

    Returns: (C*T, embed_dim)
    """
    # MAE-style 2D sincos uses half dims for one axis and half for the other, then concatenates.
    # That requires embed_dim divisible by 4 so each half is even.
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos concat PE."

    device = device or "cpu"

    # positions laid out to match flatten order: [c0t0, c0t1, ..., c1t0, ...]
    t = torch.arange(num_time_patches, device=device, dtype=dtype).unsqueeze(0).repeat(num_channels, 1).reshape(-1)
    c = torch.arange(num_channels, device=device, dtype=dtype).unsqueeze(1).repeat(1, num_time_patches).reshape(-1)

    # time first, channel second (matches common MAE code path where "w goes first")
    emb_t = _get_1d_sincos_pos_embed(embed_dim // 2, t, base=base)  # (C*T, D/2)
    emb_c = _get_1d_sincos_pos_embed(embed_dim // 2, c, base=base)  # (C*T, D/2)
    return torch.cat([emb_t, emb_c], dim=1)  # (C*T, D)



class TransformerCNNEncoder2D(TimeSeriesEncoderBase):
    """
    ViT-like patch embedding + fixed 2D sin-cos PE for multivariate time series.

    Input:
      x: (B, C, L) or (B, L) which will be treated as C=1

    Output:
      (B, C*T, D) by default, or (B, C, T, D) if return_4d=True
    """

    def __init__(
        self,
        args=None,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
        num_heads: int = 8,
        num_layers: int = 6,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 1024,
        pe_base: float = 10000.0,
        return_4d: bool = False,
    ):
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.pe_base = pe_base
        self.return_4d = return_4d

        # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.args = args


    def forward(self, x: torch.Tensor, chan_id: torch.Tensor) -> torch.Tensor:
        C, L = x.shape


        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        T = L // self.patch_size  # number of time patches


        # ---- patch embedding, weight-shared per channel
        # (C, L) -> (C, 1, L) -> Conv1d -> (C, D, T) -> (C, T, D)
        x_bc = x.reshape(C, 1, L)
        tok = self.patch_embed(x_bc)  # (C, D, T)
        D = tok.size(1)
        tok = tok.transpose(1, 2).reshape(C, T, D)  # (C, T, D)

        t_pos = torch.arange(T, device=tok.device, dtype=torch.float32)  # (T,)
        c_pos = chan_id.to(device=tok.device, dtype=torch.float32)       # (C,)


        if self.args.use_only_time_pe:
            pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)  # (T, D)
            pe = pe_t.unsqueeze(0).expand(C, -1, -1)  # (C, T, D)
        elif self.args.use_only_channel_pe:
            pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)  # (C, D)
            pe = pe_c.unsqueeze(1).expand(-1, T, -1)  # (C, T, D)
        else:
            if D % 4 == 0:
                pe_t = _get_1d_sincos_pos_embed(D // 2, t_pos, base=self.pe_base).to(tok.dtype)  # (T, D/2)
                pe_c = _get_1d_sincos_pos_embed(D // 2, c_pos, base=self.pe_base).to(tok.dtype)  # (C, D/2)
                pe = torch.cat(
                    [
                        pe_t.unsqueeze(0).expand(C, -1, -1),      # (C, T, D/2)
                        pe_c.unsqueeze(1).expand(-1, T, -1),          # (C, T, D/2)
                    ],
                    dim=-1,
                )  # (C, T, D)
            else:
                pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)        # (T, D)
                pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)        # (C, D)
                pe = pe_t.unsqueeze(0) + pe_c.unsqueeze(1)                                       # (C, T, D)


        tok = tok + pe  # shape-safe: (C, T, D)

        # norm, dropout, encoder
        tok = self.input_norm(tok)
        tok = self.input_dropout(tok)
        tok = self.encoder(tok)

        return tok
