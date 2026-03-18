# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from opentslm.model_config import PATCH_SIZE
from opentslm.model.decoder.TimeSeriesDecoderBase import TimeSeriesDecoderBase


class TransformerCNNDecoder2D(TimeSeriesDecoderBase):
    """
    Causal self-attention decoder over the growing LLM hidden-state sequence.

    The input is the projected LLM last hidden states for every TS patch
    processed so far — both history (observed) and forecast (generated).
    Since the LLM is frozen, these hidden states are fully determined by
    the encoder + projector outputs; gradients flow back through the frozen
    LLM to train the encoder and projector.

    Training (teacher forcing):
        The full N-patch sequence is passed at once under a causal mask.
        Position t predicts the next patch (t+1) in parallel.

    Inference (autoregressive):
        step 0 : input = [hist_0, …, hist_{N-1}]                     → patch 0
        step 1 : input = [hist_0, …, hist_{N-1}, gen_0]              → patch 1
        step t : input = [hist_0, …, hist_{N-1}, gen_0, …, gen_{t-1}]→ patch t

    Input shapes accepted:
        (N, T, input_dim)  →  (N, T, patch_size)    full sequence
        (N, input_dim)     →  (N, patch_size)        single-step convenience
    """

    def __init__(
        self,
        input_dim: int = 768,       # enc_dim — output of decoder_projector
        decoder_dim: int = 512,     # internal width of the causal transformer
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 2048,
        # legacy / unused kwargs kept for API compatibility
        pe_base: float = 10000.0,
        use_conv_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, patch_size, dropout)
        self.input_dim = input_dim
        self.decoder_dim = decoder_dim

        # Project from enc_dim → decoder_dim (Identity if equal)
        self.in_proj = (
            nn.Linear(input_dim, decoder_dim)
            if input_dim != decoder_dim
            else nn.Identity()
        )

        # GPT-style causal self-attention stack (no cross-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,      # pre-LN for stable deep training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(decoder_dim),
        )

        # Patch prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, patch_size),
        )

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Decoder: {total:,} params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor, chan_id=None) -> torch.Tensor:
        """
        x : (N, T, input_dim)  — sequence of projected LLM hidden states
            (N, input_dim)     — single hidden vector (inference convenience)

        chan_id: unused, kept for API compatibility.

        Returns:
            (N, T, patch_size)  when x is 3-D
            (N, patch_size)     when x is 2-D
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)              # (N, 1, input_dim)

        N, T, _ = x.shape
        out = self.in_proj(x)               # (N, T, decoder_dim)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=out.dtype
        )
        out = self.transformer(out, mask=causal_mask, is_causal=True)   # (N, T, decoder_dim)
        out = self.head(out)                # (N, T, patch_size)

        if squeeze:
            return out[:, 0, :]             # (N, patch_size)
        return out                          # (N, T, patch_size)
