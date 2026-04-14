# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from typing import List, Tuple
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


def _infer_sample_slices(chan_id: torch.Tensor) -> List[slice]:
    """Recover per-sample channel groups from flattened chan_id values."""
    if chan_id.dim() != 1:
        raise ValueError(f"Expected chan_id to be 1-D, got {tuple(chan_id.shape)}")

    chan_vals = chan_id.detach().cpu().tolist()
    if not chan_vals:
        return []

    starts = [0]
    for idx in range(1, len(chan_vals)):
        if chan_vals[idx] <= chan_vals[idx - 1]:
            starts.append(idx)

    starts.append(len(chan_vals))
    return [slice(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


class TransformerCNNEncoder2D(TimeSeriesEncoderBase):
    """
    Causal patch encoder — each patch attends only to itself and all
    preceding patches (causal self-attention mask).

    Runs ONCE on the full history window and produces a fixed memory M
    that the causal forecast decoder cross-attends to at every generation
    step. Causal attention ensures patch representation i encodes only
    patches 0..i, preventing the LLM from reading future patch information
    out of earlier positions and making the next-patch prediction task
    non-trivial.

    - 24 causal self-attention layers (TransformerEncoderLayer, causal mask)
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

    # ------------------------------------------------------------------
    # Incremental (KV-cache) helpers
    # ------------------------------------------------------------------

    def _compute_tok(
        self,
        x: torch.Tensor,          # [C, T_new * patch_size]
        chan_id: torch.Tensor,     # [C]
        t_offset: int = 0,        # time-position index of the first new patch
    ) -> torch.Tensor:
        """Patch-embed x and add 2D sin-cos positional encoding.

        Returns [C, T_new, transformer_input_dim] after input_norm + dropout.
        Mirrors the embedding block in forward() but supports an arbitrary
        starting time position (t_offset) so that new patches receive the
        correct absolute time PE.
        """
        C = x.size(0)
        T_new = x.size(1) // self.patch_size
        D = self.transformer_input_dim

        tok = self.patch_embed(x.reshape(C, 1, -1)).transpose(1, 2)  # [C, T_new, D]

        if self._use_freq_embed:
            x_patches = x.reshape(C, T_new, self.patch_size)
            x_fft = torch.fft.rfft(x_patches, dim=-1)
            x_fft = torch.cat([x_fft.real, x_fft.imag], dim=-1)
            freq_tok = self.freq_embed(x_fft.to(tok.dtype))
            tok = self.concat_proj(torch.cat([tok, freq_tok], dim=-1))

        t_pos = torch.arange(t_offset, t_offset + T_new, device=tok.device, dtype=torch.float32)
        c_pos = chan_id.to(device=tok.device, dtype=torch.float32)

        if self.args.use_only_time_pe:
            pe = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
            pe = pe.unsqueeze(0).expand(C, -1, -1)
        elif self.args.use_only_channel_pe:
            pe = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
            pe = pe.unsqueeze(1).expand(-1, T_new, -1)
        else:
            if D % 4 == 0:
                pe_t = _get_1d_sincos_pos_embed(D // 2, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D // 2, c_pos, base=self.pe_base).to(tok.dtype)
                pe = torch.cat(
                    [pe_t.unsqueeze(0).expand(C, -1, -1),
                     pe_c.unsqueeze(1).expand(-1, T_new, -1)],
                    dim=-1,
                )
            else:
                pe_t = _get_1d_sincos_pos_embed(D, t_pos, base=self.pe_base).to(tok.dtype)
                pe_c = _get_1d_sincos_pos_embed(D, c_pos, base=self.pe_base).to(tok.dtype)
                pe = pe_t.unsqueeze(0).expand(C, -1, -1) + pe_c.unsqueeze(1).expand(-1, T_new, -1)

        tok = tok + pe
        tok = self.input_norm(tok)
        tok = self.input_dropout(tok)
        return tok  # [C, T_new, D]

    def tokenize_patches(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
        t_offset: int = 0,
    ) -> torch.Tensor:
        """
        Return patch tokens before the encoder stack.

        This exposes the encoder's patch embedding and positional encoding
        without running the contextual encoder stack.
        """
        if x.size(1) % self.patch_size != 0:
            raise ValueError(f"L={x.size(1)} not divisible by patch_size={self.patch_size}")
        return self._compute_tok(x, chan_id, t_offset=t_offset)

    def build_kv_cache(
        self,
        x: torch.Tensor,      # [C, L]  L divisible by patch_size
        chan_id: torch.Tensor, # [C]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Full encoder forward that also returns per-layer input caches.

        The cache stores the un-normed input to each transformer layer for
        every history patch.  forward_incremental() uses these as K/V so
        that a new patch can attend to all history without re-encoding it.

        Returns:
            output    : [C, T, output_dim]  (same as forward())
            kv_caches : list of num_layers tensors, each [C, T, transformer_dim]
        """
        if x.size(1) % self.patch_size != 0:
            raise ValueError(f"L={x.size(1)} not divisible by patch_size={self.patch_size}")

        tok = self._compute_tok(x, chan_id, t_offset=0)  # [C, T, D]
        T = tok.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tok.device, dtype=tok.dtype
        )

        kv_caches: List[torch.Tensor] = []
        h = tok
        for layer in self.encoder.layers:
            kv_caches.append(h)                             # store pre-layer activation
            h = layer(h, src_mask=causal_mask, is_causal=True)

        if self.encoder.norm is not None:
            h = self.encoder.norm(h)
        output = self.output_proj(h)
        return output, kv_caches

    def forward_incremental(
        self,
        x_new: torch.Tensor,           # [C, patch_size]  one new patch per channel
        chan_id: torch.Tensor,          # [C]
        t_offset: int,                  # absolute time index of the new patch
        kv_caches: List[torch.Tensor],  # per-layer [C, T_past, D]; updated in-place
    ) -> torch.Tensor:
        """Encode one new patch using cached K/V activations from all past patches.

        Because the encoder is causal, position t can only attend to 0..t, so
        past positions' outputs never change.  Only the new position's output
        must be computed, attending to the cached history via the saved
        per-layer pre-norm activations.

        kv_caches is updated in-place: each entry grows by one column so
        successive calls build up the growing sequence correctly.

        Returns: [C, 1, output_dim]
        """
        # 1. Embed the new patch with correct absolute time PE
        h_new = self._compute_tok(x_new, chan_id, t_offset=t_offset)  # [C, 1, D]

        # 2. Propagate through each transformer layer with cached K/V
        for l, layer in enumerate(self.encoder.layers):
            h_past = kv_caches[l]               # [C, T_past, D]  — pre-layer inputs

            # norm_first=True: K/V are the normed layer inputs
            kv_past  = layer.norm1(h_past)       # [C, T_past, D]
            q_kv_new = layer.norm1(h_new)        # [C, 1, D]

            # K and V: all past + new token (new token is the last so no causal
            # masking needed — it can attend to every preceding position)
            k = torch.cat([kv_past, q_kv_new], dim=1)   # [C, T_past+1, D]
            v = k

            attn_out = layer.self_attn(q_kv_new, k, v, need_weights=False)[0]  # [C, 1, D]

            # Store the pre-layer input for the new position before overwriting h_new
            kv_caches[l] = torch.cat([h_past, h_new], dim=1)   # grow cache by 1

            # Apply residual + feed-forward (norm_first path)
            h_new = h_new + layer.dropout1(attn_out)
            h_new = h_new + layer._ff_block(layer.norm2(h_new))

        # 3. Final norm + projection
        if self.encoder.norm is not None:
            h_new = self.encoder.norm(h_new)
        return self.output_proj(h_new)   # [C, 1, output_dim]

    def _forward_same_channel_count(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        C, L = x.shape
        D = self.transformer_input_dim
        n_ch = int(chan_id.max().item()) + 1
        if C % n_ch != 0:
            raise ValueError(
                f"Expected total channel count {C} to be divisible by n_ch={n_ch}. "
                f"chan_id={chan_id.tolist()}"
            )
        B = C // n_ch
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

        # ── Merge channels into the sequence so all channels at the same time
        #    step can attend to each other (cross-channel attention).
        #    Layout: time-major interleaving — [t0c0, t0c1, ..., t0c_{n_ch-1}, t1c0, ...]
        #    (C, T, D) = (B*n_ch, T, D) → (B, n_ch, T, D) → (B, T, n_ch, D) → (B, T*n_ch, D)
        tok = tok.reshape(B, n_ch, T, D)
        tok = tok.permute(0, 2, 1, 3).reshape(B, T * n_ch, D)

        # ── Block-causal mask (T*n_ch × T*n_ch) ─────────────────────────────
        # Token i has time index i // n_ch.  Token j is visible to i iff
        # j's time <= i's time (all channels at the same step see each other).
        seq_len = T * n_ch
        t_idx = torch.arange(seq_len, device=tok.device)
        can_attend = (t_idx // n_ch).unsqueeze(1) >= (t_idx // n_ch).unsqueeze(0)  # (seq_len, seq_len)
        causal_mask = torch.zeros(seq_len, seq_len, dtype=tok.dtype, device=tok.device)
        causal_mask.masked_fill_(~can_attend, float('-inf'))

        # ── Padding mask: (C, T) → time-major (B, T*n_ch) ───────────────────
        # Additive convention: 0.0 = attend, -inf = ignore.
        src_key_padding_mask = None
        if attention_mask is not None:
            bool_pad = ~attention_mask.bool()                          # (C, T) = (B*n_ch, T)
            bool_pad = bool_pad.reshape(B, n_ch, T).permute(0, 2, 1)  # (B, T, n_ch)
            bool_pad = bool_pad.reshape(B, T * n_ch)                   # (B, T*n_ch)
            src_key_padding_mask = torch.zeros(
                bool_pad.shape, dtype=tok.dtype, device=tok.device
            ).masked_fill_(bool_pad, float('-inf'))

        tok = self.encoder(
            tok,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=False,  # causality is enforced via explicit causal_mask above
        )

        # ── Undo time-major reshape back to (C, T, D) ────────────────────────
        # (B, T*n_ch, D) → (B, T, n_ch, D) → (B, n_ch, T, D) → (B*n_ch, T, D)
        tok = tok.reshape(B, T, n_ch, D)
        tok = tok.permute(0, 2, 1, 3).reshape(C, T, D)

        # ── Output projection ────────────────────────────────────────────────
        tok = self.output_proj(tok)          # (C, T, output_dim)

        return tok

    def forward(self, x: torch.Tensor, chan_id: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        NOTE: chan_id is needed to reconstruct the 2D positional encodings. During
        processing, time series are flattened across the batch into shape (C, L),
        where C is the total number of channels. chan_id restarts at 0 for each
        sample, e.g. [0, 1, 2, 0, 1, 2] for two 3-channel examples.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape (C, L), got {tuple(x.shape)}")
        if chan_id.dim() != 1 or chan_id.numel() != x.size(0):
            raise ValueError(
                f"Expected chan_id of shape ({x.size(0)},), got {tuple(chan_id.shape)}"
            )
        if x.size(1) % self.patch_size != 0:
            raise ValueError(f"L={x.size(1)} not divisible by patch_size={self.patch_size}")

        T = x.size(1) // self.patch_size
        if attention_mask is not None and attention_mask.shape != (x.size(0), T):
            raise ValueError(
                f"Expected attention_mask of shape {(x.size(0), T)}, got {tuple(attention_mask.shape)}"
            )

        if x.size(0) == 0:
            return x.new_empty((0, T, self.output_dim))
        # Fast path for the common fixed-width layouts: pure univariate batches
        # (all zeros) or uniform multivariate batches where each sample has the
        # same channel count.
        sample_slices = _infer_sample_slices(chan_id)
        if not sample_slices:
            return x.new_empty((0, T, self.output_dim))

        group_sizes = [slc.stop - slc.start for slc in sample_slices]
        if torch.all(chan_id == 0) or len(set(group_sizes)) == 1:
            return self._forward_same_channel_count(x, chan_id, attention_mask)

        outputs: List[torch.Tensor] = []
        for slc in sample_slices:
            x_group = x[slc]
            mask_group = attention_mask[slc] if attention_mask is not None else None
            local_chan_id = torch.arange(
                x_group.size(0), device=chan_id.device, dtype=chan_id.dtype
            )
            outputs.append(
                self._forward_same_channel_count(
                    x_group,
                    local_chan_id,
                    mask_group,
                )
            )

        return torch.cat(outputs, dim=0)
