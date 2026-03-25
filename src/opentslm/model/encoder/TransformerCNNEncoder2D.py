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


class TransformerCNNEncoder2D(TimeSeriesEncoderBase):
    """
    Bidirectional patch encoder — each patch attends to all other patches
    in the observed history (no causal mask).

    Runs ONCE on the full history window and produces a fixed memory M
    that the causal forecast decoder cross-attends to at every generation
    step. Bidirectional attention maximises the information available to
    M without leaking future: the encoder only ever sees observed patches.

    - 24 bidirectional self-attention layers (TransformerEncoderLayer, no mask)
    - d_model = 768, ff_dim = 3072
    - 12 attention heads

    Input:
      x: (C, L)  — C channels (total across batch), L timesteps
                    (padded to a multiple of patch_size)

    Output:
      (C, T, D)  — T = L / patch_size patch representations
    """

    def __init__(
        self,
        args=None,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        transformer_input_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 24,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 3072,
        pe_base: float = 10000.0,
        return_4d: bool = False,
    ):
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self.pe_base = pe_base
        self.return_4d = return_4d
        self.transformer_input_dim = transformer_input_dim

        # ── Patch embedding: (B, 1, L) → (B, D, T) ──────────────────────────
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # ── Frequency patch embedding (optional) ─────────────────────────────
        # rfft on a patch of length P gives P//2+1 complex values → 2*(P//2+1) reals
        self._use_freq_embed = getattr(args, "use_freq_embed", False)
        if self._use_freq_embed:
            freq_input_dim = (patch_size // 2 + 1) * 2
            self.freq_embed = nn.Linear(freq_input_dim, transformer_input_dim, bias=False)
            self.concat_proj = nn.Linear(2 * transformer_input_dim, transformer_input_dim, bias=False)

        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        # ── Transformer encoder (24 layers) ──────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN for stable deep training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(transformer_input_dim),  # final norm
        )

        if transformer_input_dim != output_dim:
            self.output_proj = nn.Linear(transformer_input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        self.args = args

        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Encoder: {total:,} params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor, chan_id: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        NOTE: chan_id is needed to reconstruct the 2D positional encodings. During processing, the time series are batched into shape (C,L), where C is the TOTAL number of channels across the batched examples
        For example, if the batch contains 2 examples, each with 3 channels, then C = 6. chan_id is then [0, 1, 2, 0, 1, 2], which results in similar channel positional encoding for similar channel. L is the length 
        (padded to the same length across all series in the batch). Now, we will use chan_id to identify which of the 6 channels correspond to the same original channel across the 2 examples, so that we can assign 
        them the same channel positional encoding.
        """
        C, L = x.shape
        D = self.transformer_input_dim

        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        T = L // self.patch_size

        x_bc = x.reshape(C, 1, L)
        tok = self.patch_embed(x_bc)           # (C, D, T)
        tok = tok.transpose(1, 2)              # (C, T, D)

        # ── Frequency embedding branch ────────────────────────────────────────
        if self._use_freq_embed:
            # reshape into patches, apply rfft, stack real+imag
            x_patches = x.reshape(C, T, self.patch_size)          # (C, T, P)
            x_fft = torch.fft.rfft(x_patches, dim=-1)             # (C, T, P//2+1) complex
            x_fft = torch.cat([x_fft.real, x_fft.imag], dim=-1)   # (C, T, (P//2+1)*2)
            freq_tok = self.freq_embed(x_fft.to(tok.dtype))        # (C, T, D)
            tok = self.concat_proj(torch.cat([tok, freq_tok], dim=-1))  # (C, T, D)

        t_pos = torch.arange(T, device=tok.device, dtype=torch.float32)
        c_pos = chan_id.to(device=tok.device, dtype=torch.float32)

        if self.args.use_only_time_pe:
            pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
            pe = pe_t.unsqueeze(0).expand(C, -1, -1)
        elif self.args.use_only_channel_pe:
            pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
            pe = pe_c.unsqueeze(1).expand(-1, T, -1)
        else:
            if D % 4 == 0:
                pe_t = _get_1d_sincos_pos_embed(D // 2, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D // 2, c_pos, base=self.pe_base).to(tok.dtype)
                pe = torch.cat(
                    [
                        pe_t.unsqueeze(0).expand(C, -1, -1),
                        pe_c.unsqueeze(1).expand(-1, T, -1),
                    ],
                    dim=-1,
                )
            else:
                pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
                pe = pe_t.unsqueeze(0) + pe_c.unsqueeze(1)

    
        tok = tok + pe

        tok = self.input_norm(tok)
        tok = self.input_dropout(tok)
        # attention_mask: (C, T) with 1=valid, 0=padding (HuggingFace convention)
        # nn.TransformerEncoder expects src_key_padding_mask: True=ignore

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # tok = self.encoder(tok, src_key_padding_mask=src_key_padding_mask, is_causal=True)  # (C, T, D)

        T = tok.size(1)  # sequence length since batch_first=True
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tok.device, dtype=tok.dtype
        )

        tok = self.encoder(
            tok,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )

        # ── Output projection ────────────────────────────────────────────────
        tok = self.output_proj(tok)          # (C, T, output_dim)

        return tok