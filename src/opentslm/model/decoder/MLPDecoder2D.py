# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Lightweight MLP-based patch decoder.

Drop-in replacement for ``TransformerCNNDecoder2D`` with the same forward
signature: ``forward(x) -> (N, T, patch_size)`` for ``x`` of shape
``(N, T, input_dim)``, and ``(N, patch_size)`` for ``(N, input_dim)``.

The contextual transformer stack is replaced with a position-wise 2-layer MLP.
No cross-position attention — each position's patch prediction depends only
on its own input hidden state.
"""

import torch
import torch.nn as nn

from opentslm.model_config import PATCH_SIZE
from opentslm.model.decoder.TimeSeriesDecoderBase import TimeSeriesDecoderBase


class MLPDecoder2D(TimeSeriesDecoderBase):
    """Position-wise 2-layer MLP decoder.

    Input:
        (N, T, input_dim)  →  (N, T, patch_size)    full sequence
        (N, input_dim)     →  (N, patch_size)        single-step convenience
    """

    def __init__(
        self,
        input_dim: int = 768,
        decoder_dim: int = 512,
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.1,
        ff_dim: int = 2048,
        # legacy / unused kwargs kept for API compatibility with the transformer
        # decoder so callers can swap them freely.
        pe_base: float = 10000.0,
        use_conv_decoder: bool = True,
        num_heads: int = 8,
        num_layers: int = 2,
        **kwargs,
    ):
        super().__init__(input_dim, patch_size, dropout)
        self.input_dim = input_dim
        self.decoder_dim = decoder_dim

        # Project input to decoder width (Identity when already equal).
        self.in_proj = (
            nn.Linear(input_dim, decoder_dim)
            if input_dim != decoder_dim
            else nn.Identity()
        )

        # 2-layer position-wise MLP with residual.
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, decoder_dim),
        )
        self.out_norm = nn.LayerNorm(decoder_dim)

        # Patch prediction head: decoder_dim → patch_size.
        self.head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, patch_size),
        )

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 MLPDecoder2D: {total:,} params ({trainable:,} trainable)")

    def _decode_hidden(self, x: torch.Tensor):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                # (N, 1, input_dim)
        out = self.in_proj(x)                  # (N, T, decoder_dim)
        out = out + self.mlp(out)              # position-wise residual MLP
        out = self.out_norm(out)
        return out, squeeze

    def forward(self, x: torch.Tensor, chan_id=None) -> torch.Tensor:
        out, squeeze = self._decode_hidden(x)
        out = self.head(out)                   # (N, T, patch_size)
        if squeeze:
            return out[:, 0, :]
        return out

    def forecast_teacher_forced(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "Teacher-forced decoder rollout has been removed. "
            "Use the encoder -> projector -> decoder next-patch workflow instead."
        )

    def forecast_autoregressive(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "Decoder-internal autoregressive rollout has been removed. "
            "Use an outer loop that re-encodes predicted patches through the model workflow."
        )
