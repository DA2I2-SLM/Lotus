# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from opentslm.model_config import TRANSFORMER_INPUT_DIM, PATCH_SIZE
from opentslm.model.decoder.TimeSeriesDecoderBase import TimeSeriesDecoderBase


class TransformerCNNDecoder2D(TimeSeriesDecoderBase):
    """
    MLP next-patch decoder.

    Takes one hidden vector per patch position (output of decoder_projector,
    already in encoder space) and independently maps each to predicted next
    patch values. No cross-position attention — all temporal context is
    provided by the causal LLM and causal encoder upstream.

    Input:
      x: (N, input_dim)        — flat batch of hidden vectors
         (C, T, input_dim)     — sequence of hidden vectors (training)
      chan_id: ignored (kept for API compatibility)

    Output:
      (N, patch_size)           or
      (C, T * patch_size)
    """

    def __init__(
        self,
        input_dim: int = 768,
        decoder_dim: int = 512,
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.1,
        # kept for API compatibility — not used
        num_heads: int = 8,
        num_layers: int = 12,
        ff_dim: int = 2048,
        pe_base: float = 10000.0,
        use_conv_decoder: bool = True,
    ):
        super().__init__(input_dim, patch_size, dropout)
        self.input_dim = input_dim
        self.decoder_dim = decoder_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, patch_size),
        )

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Decoder: {total:,} params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor, chan_id=None) -> torch.Tensor:
        """
        x: (N, input_dim)     → (N, patch_size)
           (C, T, input_dim)  → (C, T * patch_size)
        chan_id: ignored.
        """
        shape = x.shape
        if x.dim() == 3:
            C, T, D = shape
            x = x.reshape(C * T, D)
        x = self.norm(x)
        x = self.mlp(x)  # [..., patch_size]
        if len(shape) == 3:
            return x.reshape(C, T * self.patch_size)
        return x