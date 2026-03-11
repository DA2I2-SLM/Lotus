# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from opentslm.model_config import TRANSFORMER_INPUT_DIM, PATCH_SIZE
from opentslm.model.decoder.TimeSeriesDecoderBase import TimeSeriesDecoderBase


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """MAE-style 1D sin-cos embedding."""
    assert embed_dim % 2 == 0
    device, dtype = pos.device, pos.dtype
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (base ** omega)
    out = pos.unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class TransformerCNNDecoder2D(TimeSeriesDecoderBase):
    """
    Transformer decoder for time-series reconstruction.

    Mirrors the encoder architecture to provide sufficient capacity for
    inverting the encoder's nonlinear representations. Uses a multi-layer
    transformer followed by ConvTranspose1d to project back to patch space.

    Architecture:
        input (C, T, input_dim)
        → LayerNorm + Dropout
        → Transformer (12 layers, pre-LN, GELU)
        → Output projection (if dims differ)
        → ConvTranspose1d → (C, L)

    Scaled for large-scale pretraining (~80-100M params):
      - 12 transformer layers (asymmetric: ~half the encoder depth)
      - d_model = 768, ff_dim = 3072
      - 12 attention heads

    The decoder is intentionally smaller than the encoder following the
    MAE principle: a lightweight decoder encourages the encoder to learn
    richer representations rather than deferring work to the decoder.

    Input:
      x: (C, T, D) — latent representation
      chan_id: channel indices for positional encoding

    Output:
      (C, L) — reconstructed time series where L = T * patch_size
    """

    def __init__(
        self,
        input_dim: int = 768,
        decoder_dim: int = 512,
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 12,
        ff_dim: int = 2048,
        pe_base: float = 10000.0,
        use_conv_decoder: bool = True,
    ):
        super().__init__(input_dim, patch_size, dropout)
        self.input_dim = input_dim
        self.decoder_dim = decoder_dim
        self.use_conv_decoder = use_conv_decoder
        self.pe_base = pe_base

        # ── Input projection (encoder dim → decoder dim) ─────────────────────
        # Allows the decoder to operate at a different (narrower) width
        if input_dim != decoder_dim:
            self.input_proj = nn.Linear(input_dim, decoder_dim)
        else:
            self.input_proj = nn.Identity()

        self.input_norm = nn.LayerNorm(decoder_dim)
        self.input_dropout = nn.Dropout(dropout)

        # ── Transformer decoder layers ───────────────────────────────────────
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN for stable deep training
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(decoder_dim),  # final norm
        )

        # ── Output: project back to time-series patches ──────────────────────
        self.output_norm = nn.LayerNorm(decoder_dim)

        if use_conv_decoder:
            # Need to go from decoder_dim → patch values via ConvTranspose1d
            # ConvTranspose1d expects (B, C_in, L_in) → (B, C_out, L_out)
            self.patch_unembed = nn.ConvTranspose1d(
                in_channels=decoder_dim,
                out_channels=1,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            )
        else:
            self.patch_unembed = nn.Linear(decoder_dim, patch_size)

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Decoder: {total:,} params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor, chan_id) -> torch.Tensor:
        """
        Decode latent representation to time series.

        Args:
            x: (C, T, D) — latent representation from encoder or LLM
            chan_id: channel indices (list or tensor) for positional encoding

        Returns:
            (C, L) — reconstructed time series where L = T * patch_size
        """
        C, T, D = x.shape

        # ── Input projection ─────────────────────────────────────────────────
        x = self.input_proj(x)  # (C, T, decoder_dim)
        dec_dim = x.size(-1)

        # ── Positional encoding ──────────────────────────────────────────────
        # Add position info so the transformer knows patch ordering.
        # Use same 2D sin-cos scheme as encoder for consistency.
        t_pos = torch.arange(T, device=x.device, dtype=torch.float32)

        if isinstance(chan_id, list):
            c_pos = torch.tensor(chan_id, device=x.device, dtype=torch.float32)
        else:
            c_pos = chan_id.to(device=x.device, dtype=torch.float32)

        if dec_dim % 4 == 0:
            pe_t = _get_1d_sincos_pos_embed(dec_dim // 2, t_pos, base=self.pe_base).to(x.dtype)
            pe_c = _get_1d_sincos_pos_embed(dec_dim // 2, c_pos, base=self.pe_base).to(x.dtype)
            pe = torch.cat(
                [
                    pe_t.unsqueeze(0).expand(C, -1, -1),   # (C, T, D/2)
                    pe_c.unsqueeze(1).expand(-1, T, -1),   # (C, T, D/2)
                ],
                dim=-1,
            )  # (C, T, D)
        else:
            pe_t = _get_1d_sincos_pos_embed(dec_dim, t_pos, base=self.pe_base).to(x.dtype)
            pe_c = _get_1d_sincos_pos_embed(dec_dim, c_pos, base=self.pe_base).to(x.dtype)
            pe = pe_t.unsqueeze(0) + pe_c.unsqueeze(1)

        x = x + pe

        # ── Transformer (bidirectional) ──────────────────────────────────────
        # Full attention across all T patch positions.  Inference matches by
        # passing the full growing history+generated sequence to the decoder
        # (see generate()), so both training and inference use bidirectional
        # context over the available sequence.
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = self.transformer(x)                    # (C, T, decoder_dim)

        # ── Reconstruct patches ──────────────────────────────────────────────
        x = self.output_norm(x)

        if self.use_conv_decoder:
            x = x.transpose(1, 2)            # (C, decoder_dim, T)
            x = self.patch_unembed(x)        # (C, 1, L)
            x = x.squeeze(1)                 # (C, L)
        else:
            x = self.patch_unembed(x)        # (C, T, patch_size)
            x = x.reshape(C, T * self.patch_size)  # (C, L)

        return x