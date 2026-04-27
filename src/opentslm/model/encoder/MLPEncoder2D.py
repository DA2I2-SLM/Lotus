# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Lightweight MLP-based 2D patch encoder.

Drop-in replacement for ``TransformerCNNEncoder2D`` with the same
``forward(x, chan_id, attention_mask=None) -> (C, T, output_dim)`` signature.
Patch embedding and 2D sin-cos positional encoding are identical to the
transformer variant, but the contextual stack is replaced with a position-wise
2-layer MLP (no attention, no cross-patch mixing).
"""

from typing import List
import torch
import torch.nn as nn

from opentslm.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


def _get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """MAE-style 1D sin-cos embedding. pos: (M,) float → (M, embed_dim)."""
    assert embed_dim % 2 == 0
    device, dtype = pos.device, pos.dtype
    omega = torch.arange(embed_dim // 2, device=device, dtype=dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (base ** omega)
    out = pos.unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class MLPEncoder2D(TimeSeriesEncoderBase):
    """Position-wise MLP encoder with 2D (time, channel) sin-cos PE.

    Input : x of shape (C, L), chan_id of shape (C,), optional attention_mask (C, T).
    Output: (C, T, output_dim), same shape contract as TransformerCNNEncoder2D.

    The contextual stack is a 2-layer MLP applied independently to each patch
    embedding. Useful as a lightweight baseline.
    """

    def __init__(
        self,
        args=None,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 3072,
        pe_base: float = 10000.0,
    ):
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.pe_base = pe_base
        self.transformer_input_dim = transformer_input_dim
        self.args = args

        # Patch embedding: (C, 1, L) → (C, D, T) via strided conv over time.
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # Position-wise 2-layer MLP (same depth as requested).
        self.mlp = nn.Sequential(
            nn.Linear(transformer_input_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ff_dim, transformer_input_dim),
        )
        self.out_norm = nn.LayerNorm(transformer_input_dim)

        if transformer_input_dim != output_dim:
            self.output_proj = nn.Linear(transformer_input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 MLPEncoder2D: {total:,} params ({trainable:,} trainable)")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _embed_and_pe(self, tok: torch.Tensor, chan_id: torch.Tensor, t_offset: int = 0) -> torch.Tensor:
        """Add 2D sin-cos PE to a (C, T, D) patch-embed tensor."""
        C, T, D = tok.shape
        t_pos = torch.arange(t_offset, t_offset + T, device=tok.device, dtype=torch.float32)
        c_pos = chan_id.to(device=tok.device, dtype=torch.float32)

        use_only_time_pe = bool(getattr(self.args, "use_only_time_pe", False))
        use_only_channel_pe = bool(getattr(self.args, "use_only_channel_pe", False))

        if use_only_time_pe:
            pe = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
            pe = pe.unsqueeze(0).expand(C, -1, -1)
        elif use_only_channel_pe:
            pe = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
            pe = pe.unsqueeze(1).expand(-1, T, -1)
        else:
            if D % 4 == 0:
                pe_t = _get_1d_sincos_pos_embed(D // 2, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D // 2, c_pos, base=self.pe_base).to(tok.dtype)
                pe = torch.cat(
                    [pe_t.unsqueeze(0).expand(C, -1, -1),
                     pe_c.unsqueeze(1).expand(-1, T, -1)],
                    dim=-1,
                )
            else:
                pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
                pe = pe_t.unsqueeze(0).expand(C, -1, -1) + pe_c.unsqueeze(1).expand(-1, T, -1)

        return tok + pe

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """x: (C, L), chan_id: (C,), attention_mask: (C, T) or None. Returns (C, T, output_dim)."""
        C, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        tok = self.patch_embed(x.reshape(C, 1, L)).transpose(1, 2)   # (C, T, D)
        tok = self._embed_and_pe(tok, chan_id, t_offset=0)
        tok = self.input_norm(tok)
        tok = self.input_dropout(tok)

        # Position-wise MLP with residual.
        tok = tok + self.mlp(tok)
        tok = self.out_norm(tok)

        # Mask-out padding patches (zero their features) so downstream averaging
        # or projection doesn't leak padding content.
        if attention_mask is not None:
            m = attention_mask.to(dtype=tok.dtype, device=tok.device).unsqueeze(-1)
            tok = tok * m

        return self.output_proj(tok)   # (C, T, output_dim)

    # ------------------------------------------------------------------
    # Compatibility stubs for callers that expect the transformer's
    # incremental API. MLP is position-wise so "incremental" is just a
    # single patch forward; we reuse `forward` for simplicity.
    # ------------------------------------------------------------------

    def tokenize_patches(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
        t_offset: int = 0,
    ) -> torch.Tensor:
        if x.size(1) % self.patch_size != 0:
            raise ValueError(f"L={x.size(1)} not divisible by patch_size={self.patch_size}")
        C = x.size(0)
        tok = self.patch_embed(x.reshape(C, 1, -1)).transpose(1, 2)
        tok = self._embed_and_pe(tok, chan_id, t_offset=t_offset)
        return self.input_norm(tok)
