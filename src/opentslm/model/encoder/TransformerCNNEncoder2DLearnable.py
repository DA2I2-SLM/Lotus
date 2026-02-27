# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
import torch
import torch.nn as nn

from opentslm.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase
from linear_positional_encoding import Learnable2DPositionalEncoding



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

        # ViT-style patch embedding is a Conv2d with kernel=stride=patch_size.
        # Here we patch along time only: (1, patch_size).
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=(1, patch_size),
            stride=(1, patch_size),
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
        self.positional_encoding = Learnable2DPositionalEncoding(embed_dim=128, combine="sum")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)

        if x.dim() != 3:
            raise ValueError(f"Expected x of shape (B, C, L) or (B, L), got {tuple(x.shape)}")

        B, C, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        T = L // self.patch_size  # number of time patches

        # Treat as a single-channel "image": (B, 1, H=C, W=L)
        x2d = x.unsqueeze(1)

        # Patch embed: (B, 1, C, L) -> (B, D, C, T)
        tok = self.patch_embed(x2d)

        # Flatten to tokens: (B, D, C, T) -> (B, C*T, D)
        tok = tok.flatten(2).transpose(1, 2)

        # Fixed 2D PE for (channel, time_patch)
        pos = self.positional_encoding(num_channels=C, num_time_patches=T)
        tok = tok + pos.unsqueeze(0)

        tok = self.input_norm(tok)
        tok = self.input_dropout(tok)
        tok = self.encoder(tok)

        # NOTE: We dont want 4d. 
        # if self.return_4d:
        #     # (B, C*T, D) -> (B, C, T, D)
        #     tok = tok.reshape(B, C, T, tok.size(-1))

        return tok
