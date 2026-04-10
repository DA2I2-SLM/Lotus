# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opentslm.model_config import ENCODER_OUTPUT_DIM, PATCH_SIZE, TRANSFORMER_INPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


def _infer_sample_slices(chan_id: torch.Tensor) -> List[slice]:
    """
    Recover per-sample channel groups from the flattened chan_id convention used
    by OpenTSLM, e.g. [0, 1, 2, 0, 1] -> [slice(0, 3), slice(3, 5)].
    """
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


class _RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, base: float = 10000.0):
        super().__init__()
        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be a positive even integer, got {rotary_dim}")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.cos_cached is not None
            and self.sin_cached is not None
            and self._cached_len >= seq_len
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.cos_cached = freqs.cos().to(dtype=dtype)
        self.sin_cached = freqs.sin().to(dtype=dtype)
        self._cached_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor) -> torch.Tensor:
        max_seq = int(seq_id.max().item()) + 1
        self._build_cache(max_seq, device=x.device, dtype=x.dtype)

        cos = self.cos_cached[seq_id].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[seq_id].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)


class _TimeAttention(nn.Module):
    """
    Timer-XL-style TimeAttention:
    - flattened multivariate tokens
    - causal temporal masking expanded across variables
    - temporal RoPE inside attention
    - learnable same-variable / different-variable attention bias
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_fraction: float = 0.5,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        rotary_dim = int(self.head_dim * rope_fraction)
        rotary_dim -= rotary_dim % 2
        if rotary_dim <= 0:
            raise ValueError(
                f"rope_fraction={rope_fraction} produced rotary_dim={rotary_dim}; "
                "choose a larger fraction or a wider head dimension."
            )
        self.rotary_dim = rotary_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = _RotaryEmbedding(rotary_dim=rotary_dim, base=rope_base)
        self.var_bias = nn.Embedding(2, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self._layout_cache: Dict[Tuple[int, int, str, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _layout_cache_key(self, n_vars: int, n_tokens: int, device: torch.device) -> Tuple[int, int, str, int]:
        return (
            int(n_vars),
            int(n_tokens),
            device.type,
            device.index if device.index is not None else -1,
        )

    def _get_layout_masks(
        self,
        n_vars: int,
        n_tokens: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self._layout_cache_key(n_vars, n_tokens, device)
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        temporal_mask = torch.triu(
            torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=device),
            diagonal=1,
        )
        var_id = torch.arange(n_vars, device=device).repeat_interleave(n_tokens)
        same_var = var_id[:, None].eq(var_id[None, :])
        causal_mask = torch.kron(
            torch.ones(n_vars, n_vars, dtype=torch.bool, device=device),
            temporal_mask,
        )
        self._layout_cache[key] = (same_var, causal_mask)
        return same_var, causal_mask

    def forward(
        self,
        x: torch.Tensor,
        n_vars: int,
        n_tokens: int,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        mask_fill_value = -1e4 if x.dtype in (torch.float16, torch.bfloat16) else -1e9

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        seq_id = torch.arange(L, device=x.device)
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
        q = torch.cat([self.rope(q_rot, seq_id), q_pass], dim=-1)
        k = torch.cat([self.rope(k_rot, seq_id), k_pass], dim=-1)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        if key_padding_mask is not None:
            valid_keys = key_padding_mask.to(device=x.device, dtype=torch.bool)
            key_bias = torch.zeros((B, 1, 1, L), dtype=q.dtype, device=x.device)
            key_bias = key_bias.masked_fill(~valid_keys[:, None, None, :], mask_fill_value)
        else:
            key_bias = None

        if n_vars == 1:
            if key_bias is None:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                _, causal_mask = self._get_layout_masks(1, n_tokens, x.device)
                attn_bias = torch.zeros((1, 1, L, L), dtype=q.dtype, device=x.device)
                attn_bias = attn_bias.masked_fill(causal_mask.view(1, 1, L, L), mask_fill_value)
                attn_bias = attn_bias + key_bias
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                    is_causal=False,
                )
        else:
            same_var, causal_mask = self._get_layout_masks(n_vars, n_tokens, x.device)
            same_weight = self.var_bias.weight[1].to(dtype=q.dtype).view(1, self.num_heads, 1, 1)
            diff_weight = self.var_bias.weight[0].to(dtype=q.dtype).view(1, self.num_heads, 1, 1)
            attn_bias = torch.where(
                same_var.view(1, 1, L, L),
                same_weight,
                diff_weight,
            )
            attn_bias = attn_bias.masked_fill(causal_mask.view(1, 1, L, L), mask_fill_value)
            if key_bias is not None:
                attn_bias = attn_bias + key_bias
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=False,
            )

        out = out.permute(0, 2, 1, 3).reshape(B, L, self.d_model)
        return self.out_proj(out)


class _TimerXLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.attention = _TimeAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.conv1 = nn.Conv1d(d_model, ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(ff_dim, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        x: torch.Tensor,
        n_vars: int,
        n_tokens: int,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.attention(
                x,
                n_vars=n_vars,
                n_tokens=n_tokens,
                key_padding_mask=key_padding_mask,
            )
        )

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class TransformerTimerXLEncoder2D(TimeSeriesEncoderBase):
    """
    Timer-XL-style patch encoder adapted to the current OpenTSLM encoder API.

    Public interface compatibility:
    - forward(x, chan_id, attention_mask=None) -> [C, T, D]
    - build_kv_cache / forward_incremental are kept for the current generation
      path, using a safe recompute-on-growth compatibility implementation
    """

    def __init__(
        self,
        args=None,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.1,
        transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
        num_heads: int = 12,
        num_layers: int = 24,
        patch_size: int = PATCH_SIZE,
        ff_dim: int = 3072,
        pe_base: float = 10000.0,
        return_4d: bool = False,
    ):
        super().__init__(output_dim, dropout)
        self.args = args
        self.patch_size = patch_size
        self.pe_base = pe_base
        self.return_4d = return_4d
        self.transformer_input_dim = transformer_input_dim

        self.patch_embed = nn.Linear(patch_size, transformer_input_dim)
        self.blocks = nn.ModuleList(
            [
                _TimerXLBlock(
                    d_model=transformer_input_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(transformer_input_dim)

        if transformer_input_dim != output_dim:
            self.output_proj = nn.Linear(transformer_input_dim, output_dim)
        else:
            self.output_proj = nn.Identity()

        if getattr(args, "use_only_time_pe", False) or getattr(args, "use_only_channel_pe", False) or getattr(args, "use_freq_embed", False):
            print(
                "⚠️  TransformerTimerXLEncoder2D ignores legacy PE/frequency flags; "
                "it always uses Timer-XL-style RoPE + variable bias."
            )

        self._print_param_count()

    def _print_param_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🔢 Timer-XL Encoder: {total:,} params ({trainable:,} trainable)")

    def _encode_univariate_batch(
        self,
        x_batch: torch.Tensor,
        mask_batch: torch.Tensor = None,
    ) -> torch.Tensor:
        B, L = x_batch.shape
        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        T = L // self.patch_size
        patches = x_batch.reshape(B, T, self.patch_size)
        h = self.patch_embed(patches)

        flat_mask = None
        if mask_batch is not None:
            flat_mask = mask_batch.to(device=x_batch.device, dtype=torch.bool)

        for block in self.blocks:
            h = block(h, n_vars=1, n_tokens=T, key_padding_mask=flat_mask)

        h = self.final_norm(h)
        h = self.output_proj(h)

        if mask_batch is not None:
            h = h.masked_fill(~mask_batch.unsqueeze(-1).bool(), 0.0)

        return h

    def _encode_group(
        self,
        x_group: torch.Tensor,
        mask_group: torch.Tensor = None,
    ) -> torch.Tensor:
        V, L = x_group.shape
        if L % self.patch_size != 0:
            raise ValueError(f"L={L} not divisible by patch_size={self.patch_size}")

        T = L // self.patch_size
        patches = x_group.reshape(V, T, self.patch_size)
        tokens = self.patch_embed(patches)

        flat_tokens = tokens.reshape(1, V * T, self.transformer_input_dim)
        flat_mask = None
        if mask_group is not None:
            flat_mask = mask_group.reshape(1, V * T).to(device=x_group.device, dtype=torch.bool)

        h = flat_tokens
        for block in self.blocks:
            h = block(h, n_vars=V, n_tokens=T, key_padding_mask=flat_mask)

        h = self.final_norm(h)
        h = self.output_proj(h)
        h = h.reshape(V, T, -1)

        if mask_group is not None:
            h = h.masked_fill(~mask_group.unsqueeze(-1).bool(), 0.0)

        return h

    def build_kv_cache(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        output = self.forward(x, chan_id)
        cache = [x.detach().clone(), chan_id.detach().clone()]
        return output, cache

    def forward_incremental(
        self,
        x_new: torch.Tensor,
        chan_id: torch.Tensor,
        t_offset: int,
        kv_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        del t_offset

        if len(kv_caches) != 2:
            raise ValueError(
                "Expected compatibility cache [raw_history, chan_id], "
                f"got {len(kv_caches)} entries."
            )

        raw_history, cached_chan_id = kv_caches
        if cached_chan_id.shape != chan_id.shape or not torch.equal(cached_chan_id, chan_id):
            raise ValueError("chan_id changed between build_kv_cache() and forward_incremental().")

        full_history = torch.cat([raw_history, x_new], dim=1)
        full_output = self.forward(full_history, chan_id)
        kv_caches[0] = full_history.detach()
        return full_output[:, -1:, :]

    def forward(
        self,
        x: torch.Tensor,
        chan_id: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
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

        if torch.all(chan_id == 0):
            return self._encode_univariate_batch(x, attention_mask)

        sample_slices = _infer_sample_slices(chan_id)
        outputs: List[torch.Tensor] = []
        singleton_rows: List[int] = []
        singleton_positions: List[int] = []

        for out_idx, slc in enumerate(sample_slices):
            if (slc.stop - slc.start) == 1:
                singleton_rows.append(slc.start)
                singleton_positions.append(out_idx)
                outputs.append(None)
                continue

            x_group = x[slc]
            mask_group = attention_mask[slc] if attention_mask is not None else None
            outputs.append(self._encode_group(x_group, mask_group=mask_group))

        if singleton_rows:
            mask_batch = attention_mask[singleton_rows] if attention_mask is not None else None
            uni_out = self._encode_univariate_batch(x[singleton_rows], mask_batch)
            for pos, out in zip(singleton_positions, uni_out.split(1, dim=0)):
                outputs[pos] = out

        return torch.cat(outputs, dim=0)
