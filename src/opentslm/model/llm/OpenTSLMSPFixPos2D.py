# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from typing import List, Dict, Any, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder2D import TransformerCNNEncoder2D
from ..decoder.TransformerCNNDecoder2D import TransformerCNNDecoder2D
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.model_config import PATCH_SIZE




class OpenTSLMSPFixPos2D(TimeSeriesLLM):
    def __init__(
        self,
        args,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.args = args

        # 1) tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        print("Length of tokenizer vocab:", len(self.tokenizer))

        # adding special tokens
        special_tokens = {"additional_special_tokens": ["<ts>", "</ts>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Store the token ids for use during forward pass
        self.ts_start_token_id = self.tokenizer.convert_tokens_to_ids("<ts>")
        self.ts_end_token_id = self.tokenizer.convert_tokens_to_ids("</ts>")

        print("Length after adding special tokens:", len(self.tokenizer))
        print("Tokenizer special tokens:", self.tokenizer.special_tokens_map)

        # 3) encoder + projector (now internal)
        self.encoder = TransformerCNNEncoder2D(args=args).to(device)
        self.decoder = TransformerCNNDecoder2D().to(device)

        # Keep patch-size source of truth consistent across model components.
        self.patch_size = int(self.encoder.patch_size)
        if int(self.decoder.patch_size) != self.patch_size:
            raise ValueError(
                "Encoder/decoder patch size mismatch: "
                f"encoder={self.encoder.patch_size}, decoder={self.decoder.patch_size}"
            )

        # Debug step counter (prints a sample once every N training steps)
        self._debug_step = 0
        self.counter = 0

        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = None

        # Freeze the LLM backbone
        for p in self.llm.parameters():
            p.requires_grad = False

        self.llm.gradient_checkpointing_enable()

        enc_dim = self.encoder.output_dim
        llm_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        ).to(device)

        self.decoder_projector = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, enc_dim),
        ).to(device)

    def _chunked_cross_entropy(self, logits, labels, chunk_size=128):
        """Memory-efficient cross-entropy that avoids materializing full float32 logits."""
        # Work directly on the original tensors — no .contiguous() copy
        # Shift: logits[..., :-1, :] predicts labels[..., 1:]
        seq_len = logits.size(1) - 1
        total_loss = 0.0
        num_tokens = 0

        for j in range(0, seq_len, chunk_size):
            end = min(j + chunk_size, seq_len)
            chunk_logits = logits[:, j:end, :].float()
            chunk_labels = labels[:, j+1:end+1]

            loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            total_loss += loss
            num_tokens += (chunk_labels != -100).sum()

            del chunk_logits, chunk_labels, loss

        return total_loss / num_tokens.clamp(min=1)    
    


    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Enable LoRA fine-tuning for the LLM component.

        Args:
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of module names to apply LoRA to. If None, uses defaults.
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine-tuning. Please install with: pip install peft"
            )

        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled. Call disable_lora() first if you want to reconfigure LoRA."
            )

        # Store reference to original model before applying LoRA
        self.original_llm = self.llm

        # Default target modules for common architectures
        if target_modules is None:
            target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        try:
            # Apply LoRA to the model
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True

            # Print LoRA info
            lora_params = sum(
                p.numel()
                for name, p in self.llm.named_parameters()
                if p.requires_grad and "lora_" in name
            )
            trainable_params = sum(
                p.numel() for p in self.llm.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA enabled:")
            print(f"   LoRA parameters: {lora_params:,}")
            print(f"   Total trainable parameters: {trainable_params:,}")
            print(f"   Total parameters: {total_params:,}")
            print(f"   LoRA %: {100 * lora_params / total_params:.2f}%")
            print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

        except Exception as e:
            print(f"❌ Failed to enable LoRA: {e}")
            print(
                "   This might be due to incompatible target modules for your model architecture."
            )
            print(
                "   Try specifying different target_modules or check your model's layer names."
            )
            raise

    def get_lora_parameters(self):
        """Get LoRA parameters for the optimizer."""
        if not self.lora_enabled:
            return []

        lora_params = []
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_params.append(param)
        return lora_params

    def disable_lora(self):
        """Disable LoRA and revert to original frozen LLM."""
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is not enabled. Cannot disable LoRA when it's not active."
            )

        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None

        self.lora_enabled = False
        print("✅ LoRA disabled, reverted to frozen LLM")


    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, Any]],
        return_targets: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TL;DR:
            This function is probably the most crucial part of OpenTSLM-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
        """
        device = self.device
        H = self.llm.config.hidden_size
        B = len(batch)

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) Tokenize & embed all texts
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        )

        input_ids = tok.input_ids.to(device, non_blocking=True)          # [N_all, P_max]
        attn_mask = tok.attention_mask.to(device, non_blocking=True)     # [N_all, P_max]
        text_embeds = self.llm.get_input_embeddings()(input_ids)         # [N_all, P_max, H]

        # 3) Batch time-series encode & project
        ts_list: List[torch.Tensor] = []
        ts_chan_id: List[int] = []
        # Track the original (pre-collate-padding) number of real time steps per series.
        # The collate_fn pads each series to a patch_size multiple BEFORE we see it here,
        # so ts.size(0) would over-count real steps and mark padding patches as valid.
        # We detect real length by finding the last non-zero value.
        ts_real_lens: List[int] = []

        for b, sample in enumerate(batch):
            for c, ts in enumerate(sample["time_series"]):
                # Normalize to [T, 1].  Channel-independent datasets emit one
                # [T] tensor per channel; accept [T,1] too.
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)          # [T] -> [T, 1]
                if ts.size(-1) != 1:
                    raise ValueError(
                        f"Expected univariate series [T] or [T,1], got {tuple(ts.shape)}. "
                        "For multivariate data use channel-independent prompts (one per channel)."
                    )

                # ── Detect real length (before collate-fn zero-padding) ──────
                # The collate_fn right-pads with 0.0 to align to patch_size.
                # Find the last non-zero position to recover the true length.
                flat = ts.squeeze(-1)          # [T]
                nonzero_idx = flat.nonzero(as_tuple=False)
                if nonzero_idx.numel() > 0:
                    real_len = int(nonzero_idx[-1].item()) + 1
                else:
                    real_len = flat.size(0)    # all-zero series — keep full length
                ts_real_lens.append(real_len)

                # ── Per-series z-normalization before encoding ────────────────
                # Ensures the encoder always receives mean-0 / std-1 input
                # regardless of upstream normalization differences.
                # For already-normalized series this is a near no-op (mean≈0, std≈1).
                m = flat.mean()
                s = flat.std().clamp(min=1e-8)
                ts = ((flat - m) / s).unsqueeze(-1)   # re-normalized [T, 1]

                ts_list.append(ts)
                ts_chan_id.append(c)

        # Store per-series normalization stats so callers can de-normalize outputs.
        # Access via self._ts_norm_stats after any call to pad_and_apply_batch.
        self._ts_norm_stats: List[Tuple[float, float]] = []
        for sample in batch:
            for ts in sample["time_series"]:
                flat = ts if ts.dim() == 1 else ts.squeeze(-1)
                m = flat.mean().item()
                s = float(flat.std().clamp(min=1e-8).item())
                self._ts_norm_stats.append((m, s))

        ts_patch_targets_shifted = None
        ts_patch_valid = None

        if ts_list:
            ts_lens = [ts.size(0) for ts in ts_list]  # padded lengths (for pad_sequence)
            ts_padded = pad_sequence(ts_list, batch_first=True).to(device, non_blocking=True)

            # pad time dim to multiple of patch_size
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)

            if return_targets:
                ts_patch_targets = ts_padded.squeeze(-1).reshape(ts_padded.size(0), -1, self.patch_size)
                N_ts_total, N_patches, P = ts_patch_targets.shape

                ts_patch_targets_shifted = torch.zeros_like(ts_patch_targets)
                ts_patch_targets_shifted[:, :-1, :] = ts_patch_targets[:, 1:, :]

                # Mark patch i as valid for next-patch prediction only when patch i+1
                # is a REAL (non-padding) patch.
                # Use ts_real_lens (detected from last non-zero) not ts_lens (padded)
                # so that collate-fn zero-padding is excluded.
                ts_patch_valid = torch.zeros(N_ts_total, N_patches, device=device, dtype=torch.bool)
                for s, real_len in enumerate(ts_real_lens):
                    n_real = (real_len + self.patch_size - 1) // self.patch_size  # ceil(real_len / P)
                    if n_real > 1:
                        ts_patch_valid[s, :n_real - 1] = True

            ts_enc = self.encoder(
                ts_padded.squeeze(-1),
                chan_id=torch.tensor(ts_chan_id, device=device),
            )
            ts_proj = self.projector(ts_enc).to(text_embeds.dtype)
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)
            self._ts_norm_stats = []

        # 4) Re-assemble per sample
        all_seq_embeds: List[torch.Tensor] = []
        all_seq_masks: List[torch.Tensor] = []
        all_seq_ids: List[torch.Tensor] = [] if return_targets else None
        all_ts_flags: List[torch.Tensor] = [] if return_targets else None

        bos_id = self.tokenizer.bos_token_id

        # Helper: slice only active (non-padding) tokens from a tokenized row.
        # Makes left-padding vs right-padding irrelevant and ensures no 0s inside the assembled mask.
        def _extract_active(
            row_embeds: torch.Tensor,   # [P_max, H]
            row_mask: torch.Tensor,     # [P_max]
            row_ids: torch.Tensor,      # [P_max]
            strip_bos: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            active = row_mask.bool()
            seg_ids = row_ids[active]       # [L_seg]
            seg_emb = row_embeds[active]    # [L_seg, H]

            if strip_bos and seg_ids.numel() > 0 and seg_ids[0].item() == bos_id:
                seg_ids = seg_ids[1:]
                seg_emb = seg_emb[1:]

            seg_len = seg_ids.numel()
            seg_mask = torch.ones(seg_len, device=device, dtype=row_mask.dtype)
            return seg_emb, seg_mask, seg_ids

        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]   # [1+N_ts+1, P_max, H]
            sample_masks  = attn_mask[start:end]     # [1+N_ts+1, P_max]
            sample_ids    = input_ids[start:end]     # [1+N_ts+1, P_max]

            seq_embeds, seq_masks = [], []
            seq_ids, seq_flags = ([], []) if return_targets else (None, None)

            # pre_prompt (keep BOS if present)
            pre_emb, pre_m, pre_ids = _extract_active(
                sample_embeds[0], sample_masks[0], sample_ids[0], strip_bos=False
            )
            seq_embeds.append(pre_emb)
            seq_masks.append(pre_m)
            if return_targets:
                seq_ids.append(pre_ids)
                seq_flags.append(torch.zeros(pre_ids.numel(), device=device, dtype=torch.long))

            # each (text_i, ts_i)
            for i in range(n_ts):
                idx = 1 + i

                txt_emb, txt_m, txt_ids = _extract_active(
                    sample_embeds[idx], sample_masks[idx], sample_ids[idx], strip_bos=True
                )

                seq_embeds.append(txt_emb)
                seq_masks.append(txt_m)

                proj = ts_proj[ts_offset + i]
                n_patches = proj.size(0)

                seq_embeds.append(proj)
                seq_masks.append(torch.ones(n_patches, device=device, dtype=torch.long))

                if return_targets:
                    seq_ids.append(txt_ids)
                    seq_flags.append(torch.zeros(txt_ids.numel(), device=device, dtype=torch.long))

                    seq_ids.append(torch.full((n_patches,), -100, device=device, dtype=torch.long))
                    seq_flags.append(torch.ones(n_patches, device=device, dtype=torch.long))

                # inject </ts> closing token after each TS block ──
                close_emb = self.llm.get_input_embeddings()(
                    torch.tensor([self.ts_end_token_id], device=device)
                )  # [1, H]
                seq_embeds.append(close_emb)
                seq_masks.append(torch.ones(1, device=device, dtype=torch.long))

                if return_targets:
                    seq_ids.append(torch.tensor([self.ts_end_token_id], device=device))
                    seq_flags.append(torch.zeros(1, device=device, dtype=torch.long))

            ts_offset += n_ts

            post_embeds, post_masks, post_ids = _extract_active(
                sample_embeds[-1], sample_masks[-1], sample_ids[-1], strip_bos=True
            )

            seq_embeds.append(post_embeds)
            seq_masks.append(post_masks)
            if return_targets:
                seq_ids.append(post_ids)
                seq_flags.append(torch.zeros(post_ids.numel(), device=device, dtype=torch.long))

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))
            if return_targets:
                all_seq_ids.append(torch.cat(seq_ids, dim=0))
                all_ts_flags.append(torch.cat(seq_flags, dim=0))

        # 5) Batch-pad the final sequences
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)   # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)   # [B, L_max]

        if return_targets:
            token_ids = pad_sequence(all_seq_ids, batch_first=True, padding_value=-100)
            ts_mask   = pad_sequence(all_ts_flags, batch_first=True, padding_value=0).bool()
            return inputs_embeds, attention_mask, token_ids, ts_mask, ts_patch_targets_shifted, ts_patch_valid, ts_chan_id

        return inputs_embeds, attention_mask, None, None, None, None, ts_chan_id


    def generate(
        self,
        batch: List[Dict[str, Any]],
        max_new_tokens: int = 200,
        n_forecast_patches: Optional[int] = None,
        **generate_kwargs,
    ) -> Tuple[List[str], List[List[torch.Tensor]]]:
        """
        Custom autoregressive loop that supports TS patch generation.

        Encoder runs on input <ts>..</ts> segments in the prompt (unchanged).
        When the LLM emits <ts> during generation and n_forecast_patches is set:
          1. The hidden state at the <ts> position is decoded → first TS patch.
          2. Each predicted patch is re-encoded (encoder + projector) and fed
             back as the next LLM input → next hidden state → next patch.
          3. After n_forecast_patches patches, </ts> is injected and text
             generation continues.
        Decoder is only active inside TS generation blocks.

        NOTE: TS generation is triggered only when ALL batch items simultaneously
        emit <ts>.  For mixed text/TS batches, use batch_size=1.

        Args:
            batch:              list of sample dicts (same format as compute_loss).
            max_new_tokens:     maximum number of text tokens to generate.
            n_forecast_patches: patches to generate per <ts> block.
                                If None, <ts> is treated as a plain text token
                                (original behaviour, no decoder involved).

        Returns:
            text_outputs:     List[str]
            ts_patch_outputs: List[List[Tensor[N_patches, patch_size]]]
                              outer list = batch; inner list = one tensor per
                              <ts> block emitted during generation.
        """
        device = self.device
        B = len(batch)

        # ── 1. Build prompt embeddings (encoder handles input TS segments) ──────
        # return_targets=True to obtain ts_mask, which marks TS patch positions.
        # We use it below to extract all history hidden states for the decoder buffer.
        inputs_embeds, attn_mask, _, ts_mask, _, _, _ = self.pad_and_apply_batch(
            batch, return_targets=True
        )  # [B, L_prompt, H], [B, L_prompt], ts_mask: [B, L_prompt] bool

        # ── 2. Full-prompt forward pass ─────────────────────────────────────────
        with torch.no_grad():
            prompt_out = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        past_kv = prompt_out.past_key_values
        next_logits = prompt_out.logits[:, -1, :]   # [B, V]

        # ── Pre-build decoder history buffer ─────────────────────────────────
        # Collect ALL history patch hidden states and project them into
        # encoder space.  At each generation step the decoder receives the
        # full growing sequence [history | generated so far] and reads the
        # output at the last position — identical to the training setup where
        # the decoder processes T=N_patches positions simultaneously.
        prompt_hidden = prompt_out.hidden_states[-1]   # [B, L_prompt, H]
        hist_buf_list = []
        for b in range(B):
            patch_positions = ts_mask[b].nonzero(as_tuple=True)[0]
            if len(patch_positions) > 0:
                h_hist = prompt_hidden[b, patch_positions, :]        # [N_hist, H]
            else:
                h_hist = prompt_hidden[b, -1:, :]                    # fallback: last token
            h_enc = self.decoder_projector(h_hist.float())           # [N_hist, enc_dim]
            h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
            hist_buf_list.append(h_enc)
        # Pad to uniform length across batch items
        N_hist_max = max(h.size(0) for h in hist_buf_list)
        enc_dim    = hist_buf_list[0].size(-1)
        hist_dec_buf = prompt_hidden.new_zeros(B, N_hist_max, enc_dim,
                                               dtype=hist_buf_list[0].dtype)
        for b, h in enumerate(hist_buf_list):
            hist_dec_buf[b, :h.size(0)] = h          # [B, N_hist_max, enc_dim]

        eos_id = self.tokenizer.eos_token_id
        generated_ids:     List[List[int]]          = [[] for _ in range(B)]
        ts_patch_outputs:  List[List[torch.Tensor]] = [[] for _ in range(B)]

        # ── helpers ─────────────────────────────────────────────────────────────
        def _step(embeds: torch.Tensor):
            """Feed [B, 1, H] embeds with KV-cache; update past_kv and attn_mask."""
            nonlocal past_kv, attn_mask
            attn_mask = torch.cat(
                [attn_mask, torch.ones(B, 1, device=device, dtype=attn_mask.dtype)],
                dim=1,
            )
            with torch.no_grad():
                out = self.llm(
                    inputs_embeds=embeds,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            past_kv = out.past_key_values
            return out

        def _hidden_to_patch(dec_buf: torch.Tensor) -> torch.Tensor:
            """[B, T, enc_dim] → predicted TS patch [B, patch_size] at last position.
            dec_buf is already projected (decoder_projector + layer_norm applied)."""
            B_local, T_local, _ = dec_buf.shape
            patch_out = self.decoder(dec_buf, [0] * B_local)  # [B, T*patch_size]
            return patch_out.reshape(B_local, T_local, self.patch_size)[:, -1, :]

        def _patch_to_embed(patch: torch.Tensor) -> torch.Tensor:
            """[B, patch_size] → LLM embed [B, 1, H] via encoder+projector.
            Treats the patch as a single-window series of length patch_size."""
            enc = self.encoder(
                patch,
                chan_id=torch.zeros(B, device=device, dtype=torch.long),
            )                                                   # [B, 1, enc_dim]
            emb = self.projector(enc).to(inputs_embeds.dtype)  # [B, 1, H]
            return emb[:, :1, :]

        # ── 3. Autoregressive generation loop ───────────────────────────────────
        for _ in range(max_new_tokens):
            next_tokens = next_logits.argmax(dim=-1)            # [B]

            # ── TS generation block (all items must emit <ts> together) ──────────
            if n_forecast_patches is not None and (next_tokens == self.ts_start_token_id).all():
                for b in range(B):
                    generated_ids[b].append(self.ts_start_token_id)

                # Feed <ts> to update KV cache (hidden state not used as seed)
                _step(self.llm.get_input_embeddings()(
                    torch.full((B, 1), self.ts_start_token_id, device=device)
                ))

                # Decoder buffer starts as the full projected history sequence.
                # At each step the decoder attends over [history | generated so far]
                # and reads the output at the last (most recent) position.
                current_dec_buf = hist_dec_buf.clone()           # [B, N_hist, enc_dim]

                batch_patches: List[List[torch.Tensor]] = [[] for _ in range(B)]
                for patch_idx in range(n_forecast_patches):
                    patch_pred = _hidden_to_patch(current_dec_buf)   # [B, patch_size]
                    for b in range(B):
                        batch_patches[b].append(patch_pred[b].detach().cpu())

                    if patch_idx < n_forecast_patches - 1:
                        # Re-encode predicted patch → new LLM hidden state
                        patch_emb = _patch_to_embed(patch_pred.detach())  # [B, 1, H]
                        out = _step(patch_emb)
                        new_hidden = out.hidden_states[-1][:, -1, :]      # [B, H]
                        # Project into enc_dim and append to decoder buffer
                        new_enc = self.decoder_projector(new_hidden.float())
                        new_enc = F.layer_norm(new_enc, [new_enc.size(-1)])
                        current_dec_buf = torch.cat(
                            [current_dec_buf, new_enc.unsqueeze(1)], dim=1
                        )  # [B, N_hist + patch_idx + 1, enc_dim]

                for b in range(B):
                    ts_patch_outputs[b].append(
                        torch.stack(batch_patches[b], dim=0)    # [N_patches, patch_size]
                    )

                # Inject </ts> and continue text generation
                for b in range(B):
                    generated_ids[b].append(self.ts_end_token_id)
                out = _step(
                    self.llm.get_input_embeddings()(
                        torch.full((B, 1), self.ts_end_token_id, device=device)
                    )
                )
                next_logits = out.logits[:, -1, :]
                continue

            # ── Normal text step ─────────────────────────────────────────────────
            for b in range(B):
                generated_ids[b].append(next_tokens[b].item())

            if eos_id is not None and all(t.item() == eos_id for t in next_tokens):
                break

            out = _step(self.llm.get_input_embeddings()(next_tokens.unsqueeze(1)))
            next_logits = out.logits[:, -1, :]

        # ── 4. Decode text (strip TS special tokens) ─────────────────────────────
        ts_token_set = {self.ts_start_token_id, self.ts_end_token_id}
        text_outputs = []
        for b in range(B):
            ids = [i for i in generated_ids[b] if i >= 0 and i not in ts_token_set]
            text_outputs.append(self.tokenizer.decode(ids, skip_special_tokens=True))

        return text_outputs, ts_patch_outputs


    # ------------------------------------------------------------------
    # Training debug printer
    # ------------------------------------------------------------------

    def _debug_batch(self, batch: List[Dict[str, Any]], step: int) -> None:
        """Print one representative sample per task type found in the batch.
        Only runs on rank 0 in distributed settings; always flushes stdout."""
        import sys
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        seen = {}
        for row in batch:
            t = row.get("type", "unknown")
            if t not in seen:
                seen[t] = row

        lines = []
        sep = "=" * 70
        lines.append(f"\n{sep}")
        lines.append(f"[DEBUG] Training step {step}  |  batch size={len(batch)}")
        lines.append(f"  Task mix: { {t: sum(1 for r in batch if r.get('type')==t) for t in seen} }")

        for task_type, row in seen.items():
            lines.append(f"\n--- Task: {task_type.upper()} ---")

            # After _format_sample the dict has: pre_prompt, post_prompt,
            # time_series (list of series), time_series_text, answer, type
            pre    = row.get("pre_prompt", "")
            post   = row.get("post_prompt", "")
            ts     = row.get("time_series", [])   # list of series (each a list/tensor)
            answer = row.get("answer", "")

            # ---- Input ----
            lines.append(f"  pre_prompt : {str(pre)[:150]}")
            lines.append(f"  post_prompt: {str(post)[:150]}")
            ts_len = len(ts) if isinstance(ts, (list, tuple)) else ts.shape[0]
            if ts_len > 0:
                first_ts = ts[0]
                if hasattr(first_ts, 'tolist'):
                    first_ts = first_ts.tolist()
                # flatten one level if multivariate (list-of-lists)
                flat = first_ts[0] if (first_ts and isinstance(first_ts[0], list)) else first_ts
                lines.append(f"  time_series: len={len(flat)}, "
                              f"first 8={[round(float(v), 4) for v in flat[:8]]}")
            else:
                lines.append(f"  time_series: (none)")

            # ---- Target ----
            lines.append(f"  answer     : {str(answer)[:150]}")

        lines.append(sep)
        print("\n".join(lines), flush=True)
        sys.stdout.flush()

    @torch.no_grad()
    def compute_loss_per_task(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return {task_type: loss} for each task type present in the batch.
        Also computes TSQA letter-accuracy (checks if first non-space char of
        the greedy-decoded answer matches the gold answer letter)."""
        results = {}
        task_types = list({r.get("type", "unknown") for r in batch})

        for task in task_types:
            sub = [r for r in batch if r.get("type") == task]
            if not sub:
                continue
            try:
                loss = self.compute_loss(sub).item()
            except Exception:
                loss = float("nan")
            results[f"loss/{task}"] = loss

        # TSQA letter accuracy: check first alphabetic char of greedy top-1
        tsqa_rows = [r for r in batch if r.get("type") == "tsqa"]
        if tsqa_rows:
            correct = 0
            for row in tsqa_rows:
                gold = row.get("answer", "").strip().lstrip("(").upper()[:1]
                if not gold:
                    continue
                # run a short greedy generation (just need first token after prompt)
                try:
                    text_out, _ = self.generate([row], max_new_tokens=5)
                    pred = text_out[0].strip().lstrip("(").upper()[:1] if text_out else ""
                    if pred == gold:
                        correct += 1
                except Exception:
                    pass
            results["accuracy/tsqa"] = correct / len(tsqa_rows) if tsqa_rows else 0.0

        return results

    @torch.no_grad()
    def generate_phase1_samples(self, batch: List[Dict[str, Any]], max_new_tokens: int = 80) -> None:
        """Print one greedy-decoded sample per task type (rank-0 only)."""
        import sys
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        seen = {}
        for row in batch:
            t = row.get("type", "unknown")
            if t not in seen:
                seen[t] = row

        sep = "=" * 70
        lines = [f"\n{sep}", "[GENERATION SAMPLES]"]
        for task_type, row in seen.items():
            lines.append(f"\n--- {task_type.upper()} ---")
            lines.append(f"  PRE : {row.get('pre_prompt', '')[:120]}")
            lines.append(f"  POST: {row.get('post_prompt', '')[:120]}")
            lines.append(f"  GOLD: {row.get('answer', '')[:150]}")
            try:
                text_out, _ = self.generate([row], max_new_tokens=max_new_tokens)
                lines.append(f"  PRED: {text_out[0][:150] if text_out else '(empty)'}")
            except Exception as e:
                lines.append(f"  PRED: ERROR — {e}")
        lines.append(sep)
        print("\n".join(lines), flush=True)
        sys.stdout.flush()

    def compute_loss(self, batch: List[Dict[str, Any]], log_every: int = 1000) -> torch.Tensor:
        self._debug_step += 1
        if self._debug_step % log_every == 1:
            self._debug_batch(batch, self._debug_step)

        answers = [b["answer"] for b in batch]

        inputs_embeds, attention_mask, _, _, _, _, _ = self.pad_and_apply_batch(batch, return_targets=False)
        B = inputs_embeds.size(0)

        # tokenize answers (get per-sample lengths)
        ans_tok = self.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,          # still ok; we will slice per sample
            truncation=True,
            max_length=4096,           # just a large number to avoid truncation; we will slice per sample
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)          # [B, A_max]
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)   # [B, A_max]

        # embed all answers once (batched), then slice per sample (efficient + avoids A_max append)
        ans_emb_all = self.llm.get_input_embeddings()(ans_ids)                 # [B, A_max, H]

        bos_id = self.tokenizer.bos_token_id

        new_embeds_list = []
        new_masks_list = []
        new_labels_list = []

        for i in range(B):
            valid_len_i = int(attention_mask[i].sum().item())  # actual prompt length (excludes padding)
            a_len = int(ans_mask[i].sum().item())              # true answer length for sample i

            # Strip BOS that tokenizers prepend to the answer (Llama, Gemma, etc.).
            # BOS is never a prediction target — it is injected, not generated.
            ans_start = 1 if (bos_id is not None and a_len > 0 and ans_ids[i, 0].item() == bos_id) else 0
            a_len_eff = a_len - ans_start  # effective answer length after BOS strip

            # concat (prompt without padding) + (answer without BOS)
            emb_i = torch.cat([inputs_embeds[i, :valid_len_i], ans_emb_all[i, ans_start:a_len, :]], dim=0)  # [valid_len_i+a_len_eff, H]
            mask_i = torch.cat([attention_mask[i, :valid_len_i], torch.ones(a_len_eff, device=self.device, dtype=attention_mask.dtype)], dim=0)

            # labels: ignore prompt part, supervise only answer tokens (no BOS)
            lab_i = torch.full((valid_len_i + a_len_eff,), -100, device=self.device, dtype=torch.long)
            lab_i[valid_len_i:] = ans_ids[i, ans_start:a_len]

            new_embeds_list.append(emb_i)
            new_masks_list.append(mask_i)
            new_labels_list.append(lab_i)

        # pad to batch max AFTER slicing (much smaller than using A_max for everyone)
        inputs_embeds = pad_sequence(new_embeds_list, batch_first=True)   # [B, L_new, H]
        attention_mask = pad_sequence(new_masks_list, batch_first=True)   # [B, L_new]
        labels = pad_sequence(new_labels_list, batch_first=True, padding_value=-100)  # [B, L_new]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,       # avoid materializing full float32 logits internally
            return_dict=True,
        )
        return self._chunked_cross_entropy(outputs.logits, labels)
        




    def compute_joint_phase1_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint Phase-1 loss: CE on text tasks + MSE next-patch on Chronos.

        Chronos answers are exact floats (<ts>...</ts>), making CE on them
        near-random noise. CE is therefore restricted to non-Chronos samples
        (e.g. TSQA, M4 captioning). Chronos contributes only via MSE.

        Args:
            batch: mixed batch of {"type": "chronos"|"tsqa"|"m4", ...} dicts
            alpha: weight for MSE decoder loss relative to CE loss

        Returns:
            (total_loss, loss_ce, loss_mse)
            loss_mse is 0 when no Chronos samples are in the batch.
            loss_ce  is 0 when no text-task samples are in the batch.
        """
        ce_batch      = [s for s in batch if s.get("type") != "chronos"]
        chronos_batch = [s for s in batch if s.get("type") == "chronos"]


        # 1. CE on all text tasks (TSQA, M4, ...)
        if ce_batch:
            loss_ce = self.compute_loss(ce_batch)
        else:
            loss_ce = next(self.parameters()).new_zeros(())

        # 2. MSE next-patch on Chronos only
        if not chronos_batch:
            return loss_ce, loss_ce, loss_ce.new_zeros(())

        _, _, loss_mse = self.compute_pretrain_loss(chronos_batch)

        total = loss_ce + alpha * loss_mse
        return total, loss_ce, loss_mse



    def compute_pretrain_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: Optional[float] = None,
        debug: bool = False,
        debug_every: int = 10,
        gen_every: int = 0,              # set to e.g. 200 to enable occasional greedy generation
        max_new_tokens: int = 50,
        plot_ts: bool = False,           # enable TS plot
        plot_every: int = 1000,           # plot frequency
        plot_dir: str = "./ts_debug",    # where to save PNGs
        plot_max_series: int = 1,        # how many series to plot per trigger
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pretraining objective:
        1) Next-token prediction (NTP) on text tokens (cross-entropy, ignore_index=-100)
        2) Next-patch prediction on time series patches via decoder (MSE)

        Returns:
        total_loss, ntp_loss, recon_loss
        """
        # ----------------------------- helpers -----------------------------
        def _is_rank0() -> bool:
            # Avoid multi-rank file write spam under DDP
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    return torch.distributed.get_rank() == 0
            except Exception:
                pass
            return True

        def _stitch_patches(patches_2d: torch.Tensor, stride: Optional[int] = None) -> torch.Tensor:
            """
            patches_2d: [n_patches, P]
            If stride is None or stride == P, simply concatenates patches.
            If stride < P, uses overlap-add with averaging.
            Returns: [T] stitched 1D series
            """
            assert patches_2d.dim() == 2, f"expected [n_patches, P], got {patches_2d.shape}"
            n_patches, P = patches_2d.shape
            if n_patches == 0:
                return patches_2d.new_zeros((0,))

            if stride is None:
                stride = int(getattr(self, "patch_stride", P))

            stride = int(stride)
            if stride >= P:
                return patches_2d.reshape(-1)

            # overlap-add
            T = (n_patches - 1) * stride + P
            out = patches_2d.new_zeros((T,))
            cnt = patches_2d.new_zeros((T,))
            for i in range(n_patches):
                s = i * stride
                out[s:s + P] += patches_2d[i]
                cnt[s:s + P] += 1
            out = out / cnt.clamp(min=1)
            return out

        # ----------------------------- main -----------------------------
        self.counter += 1

        if alpha is None:
            alpha = getattr(self, "recon_alpha", 1.0)

        (
            inputs_embeds,              # [B, L, H]
            attention_mask,             # [B, L]
            token_ids,                  # [B, L]   -100 at TS positions
            ts_mask,                    # [B, L]   True at TS patch positions
            ts_patch_targets_shifted,   # [N_ts_total, N_patches, P]
            ts_patch_valid,             # [N_ts_total, N_patches] bool
            ts_chan_id,                 # List[int] length N_ts_total
        ) = self.pad_and_apply_batch(batch, return_targets=True)

        B, L, H = inputs_embeds.size()
        device = inputs_embeds.device

        need_recon = (ts_patch_targets_shifted is not None) and (ts_patch_targets_shifted.numel() > 0)

        # forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=need_recon,
            return_dict=True,
        )

        # Loss 1: Next-token prediction on text tokens
        logits = outputs.logits                          # [B, L, V]
        shift_logits = logits[:, :-1, :].contiguous()    # [B, L-1, V]
        shift_labels = token_ids[:, 1:].contiguous()     # [B, L-1]

        ntp_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Loss 2: Next-patch recon on TS patches
        recon_loss = torch.tensor(0.0, device=device)

        recon_std = None
        N_ts_total = N_patches = P = 0
        valid_count = 0
        valid_ratio = 0.0
        ts_recon = None  # [N_ts_total, N_patches, P] — kept for optional plotting

        if need_recon:
            N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
            last_hidden = outputs.hidden_states[-1]
            ts_hidden = last_hidden[ts_mask]

            expected_flat = N_ts_total * N_patches
            actual_flat = ts_hidden.size(0)
            assert actual_flat == expected_flat, (
                f"ts_mask extracted {actual_flat} hidden vectors but expected "
                f"N_ts_total({N_ts_total}) × N_patches({N_patches}) = {expected_flat}."
            )

            # Project LLM hidden → encoder dim, then NORMALIZE to stabilize decoder input
            ts_hidden = self.decoder_projector(ts_hidden.float())
            ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])  # ← FIX 1: tame the scale
            ts_hidden_3d = ts_hidden.reshape(N_ts_total, N_patches, -1)
            ts_recon = self.decoder(ts_hidden_3d, ts_chan_id)
            ts_recon = ts_recon.reshape(N_ts_total, N_patches, P)

            # Just compute MSE directly on valid patches.
            targets_flat = ts_patch_targets_shifted.reshape(-1, P)
            recon_flat = ts_recon.reshape(-1, P).to(targets_flat.dtype)
            valid_flat = ts_patch_valid.reshape(-1)

            valid_count = int(valid_flat.sum().item())
            valid_ratio = float(valid_count / max(1, valid_flat.numel()))

            if valid_count > 0:
                recon_loss = F.mse_loss(recon_flat[valid_flat], targets_flat[valid_flat])

                with torch.no_grad():
                    recon_std = float(recon_flat[valid_flat].std().item())
                    if recon_std < 1e-3:
                        print(
                            f"⚠️  [collapse guard] decoder output std={recon_std:.2e}. "
                            f"Decoder may be ignoring LLM hidden states."
                        )
            else:
                recon_loss = torch.tensor(0.0, device=device)


        total_loss = ntp_loss + alpha * recon_loss
        # ----------------------------- debug prints -----------------------------
        do_print = debug and (self.counter % max(1, debug_every) == 0)
        if do_print:
            V = shift_logits.size(-1)
            attn_sum = attention_mask.sum(dim=1)

            ts_pos_count = int((token_ids == -100).sum().item())
            text_pos_count = int(((token_ids != -100) & attention_mask.bool()).sum().item())

            print("\n" + "=" * 90)
            print(f"[STEP {self.counter}]")
            print("\n[INPUT]")
            print(f"  inputs_embeds: {tuple(inputs_embeds.shape)}")
            print(
                f"  attention_mask: {tuple(attention_mask.shape)} | "
                f"attn_sum min/mean/max = "
                f"{int(attn_sum.min().item())}/"
                f"{float(attn_sum.float().mean().item()):.1f}/"
                f"{int(attn_sum.max().item())}"
            )
            print(
                f"  token_ids: {tuple(token_ids.shape)} | "
                f"text_positions={text_pos_count} | ts_positions(-100)={ts_pos_count}"
            )

            print("\n[OUTPUT / LOSSES]")
            print(f"  logits: {tuple(logits.shape)} (vocab={V})")
            print(f"  alpha: {alpha}")
            print(f"  ntp_loss:   {float(ntp_loss.item()):.6f}")
            print(f"  recon_loss: {float(recon_loss.item()):.6f}")
            print(f"  total_loss: {float(total_loss.item()):.6f}")

            # Per-token CE breakdown (text vs boundary after TS)
            with torch.no_grad():
                B2, Lm1, V2 = shift_logits.shape

                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, V2),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(B2, Lm1)

                shift_text_mask = (shift_labels != -100) & attention_mask[:, 1:].bool()

                # Boundary: first text token after a TS position, plus next 4 tokens
                shift_ts_boundary = torch.zeros_like(shift_text_mask)
                for b in range(B2):
                    ts_pos = (token_ids[b] == -100)
                    for i in range(L - 1):
                        if ts_pos[i] and (not ts_pos[i + 1]) and bool(attention_mask[b, i + 1].item()):
                            for k in range(0, 5):
                                j = i + k
                                if j < Lm1 and shift_labels[b, j].item() != -100 and bool(attention_mask[b, j + 1].item()):
                                    shift_ts_boundary[b, j] = True
                            break

                text_loss_vals = per_token_loss[shift_text_mask]
                boundary_loss_vals = per_token_loss[shift_ts_boundary]

                print("\n[PER-REGION NTP LOSS]")
                if text_loss_vals.numel() > 0:
                    print(
                        f"  all text tokens: mean={float(text_loss_vals.mean().item()):.4f} | "
                        f"count={text_loss_vals.numel()}"
                    )
                if boundary_loss_vals.numel() > 0:
                    nb = (~shift_ts_boundary) & shift_text_mask
                    nb_mean = float(per_token_loss[nb].mean().item()) if nb.any() else float("nan")
                    print(
                        f"  post-TS boundary (first 5): mean={float(boundary_loss_vals.mean().item()):.4f} | "
                        f"count={boundary_loss_vals.numel()}"
                    )
                    print(f"  non-boundary text:         mean={nb_mean:.4f}")

                print("\n[BOUNDARY TOKEN PREDICTIONS] (top-5 at first few boundary tokens)")
                for b in range(min(B2, 2)):
                    idxs = shift_ts_boundary[b].nonzero(as_tuple=True)[0]
                    if idxs.numel() == 0:
                        print(f"  sample {b}: no boundary tokens found")
                        continue
                    for idx in idxs[:5]:
                        idx = int(idx.item())
                        true_id = int(shift_labels[b, idx].item())
                        top5 = shift_logits[b, idx].topk(5).indices.detach().cpu().tolist()
                        try:
                            true_tok = self.tokenizer.decode([true_id]) if true_id != -100 else "<IGN>"
                            pred_toks = [self.tokenizer.decode([t]) for t in top5]
                        except Exception:
                            true_tok = str(true_id)
                            pred_toks = [str(t) for t in top5]
                        print(
                            f"  sample {b}, pos {idx}: true={repr(true_tok)} | "
                            f"top5={[repr(p) for p in pred_toks]} | "
                            f"loss={float(per_token_loss[b, idx].item()):.4f}"
                        )

            print("\n[RECON DIAGNOSTICS]")
            if not need_recon:
                print("  (no TS targets in this batch)")
            else:
                print(f"  ts_patch_targets_shifted: [N_ts_total={N_ts_total}, N_patches={N_patches}, P={P}]")
                print(f"  valid next-patch mask: count={valid_count} | ratio={valid_ratio:.3f}")
                if recon_std is not None:
                    print(f"  recon_norm std over valid positions: {recon_std:.3e}")

        # ----------------------------- TS decode + plot -----------------------------
        do_plot = (
            plot_ts
            and need_recon
            and (ts_recon is not None)
            and (self.counter % max(1, plot_every) == 0)
            and _is_rank0()
        )

        if do_plot:
            os.makedirs(plot_dir, exist_ok=True)
            with torch.no_grad():
                n_plot = min(int(plot_max_series), int(N_ts_total))
                for s in range(n_plot):
                    gt_p = ts_patch_targets_shifted[s]  # [N_patches, P]
                    pr_p = ts_recon[s]                  # [N_patches, P]
                    v_p = ts_patch_valid[s]             # [N_patches]

                    # Keep only valid patches (exclude padding and last patch)
                    idx = v_p.nonzero(as_tuple=True)[0]
                    gt_p = gt_p[idx]
                    pr_p = pr_p[idx]

                    # Stitch into 1D series
                    gt_1d = _stitch_patches(gt_p).detach().float().cpu().numpy()
                    pr_1d = _stitch_patches(pr_p).detach().float().cpu().numpy()

                    # Make plot
                    fig = plt.figure()
                    plt.plot(gt_1d, label="ground_truth_next_patch")
                    plt.plot(pr_1d, label="pred_next_patch", alpha=0.35)
                    plt.title(f"TS next-patch recon (series={s}) step={self.counter}")
                    plt.legend()

                    out_path = os.path.join(plot_dir, f"ts_recon_step{self.counter:07d}_series{s}.png")
                    plt.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)

                    print(f"[TS PLOT] saved: {out_path}")

        # ----------------------------- optional generation -----------------------------
        if gen_every > 0 and self.counter % gen_every == 0:
            print(f"\n[GENERATION] (greedy, max_new_tokens={max_new_tokens})")
            with torch.no_grad():
                for b in range(min(B, 2)):
                    valid_len = int(attention_mask[b].sum().item())
                    prompt_embeds = inputs_embeds[b:b + 1, :valid_len, :]
                    prompt_mask = attention_mask[b:b + 1, :valid_len]

                    prompt_token_ids = token_ids[b, :valid_len].tolist()
                    prompt_text_ids = [t for t in prompt_token_ids if t != -100]
                    try:
                        prompt_text = self.tokenizer.decode(prompt_text_ids, skip_special_tokens=False)
                        if len(prompt_text) > 240:
                            prompt_text = prompt_text[:120] + " ... " + prompt_text[-120:]
                    except Exception:
                        prompt_text = "(could not decode)"
                    print(f"\n  sample {b} input text: {repr(prompt_text)}")

                    generated_ids = []
                    cur_embeds = prompt_embeds
                    cur_mask = prompt_mask

                    for _ in range(max_new_tokens):
                        out = self.llm(inputs_embeds=cur_embeds, attention_mask=cur_mask, return_dict=True)
                        next_id = out.logits[:, -1, :].argmax(dim=-1)  # [1]
                        nid = int(next_id.item())
                        generated_ids.append(nid)

                        if hasattr(self.tokenizer, "eos_token_id") and nid == self.tokenizer.eos_token_id:
                            break

                        next_embed = self.llm.get_input_embeddings()(next_id).unsqueeze(1)  # [1, 1, H]
                        cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                        cur_mask = torch.cat(
                            [cur_mask, torch.ones(1, 1, device=cur_mask.device, dtype=cur_mask.dtype)],
                            dim=1,
                        )

                    try:
                        gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    except Exception:
                        gen_text = f"(raw ids: {generated_ids})"

                    print(f"  sample {b} generated: {repr(gen_text)}")
                    print(f"  sample {b} gen ids  : {generated_ids[:20]}{'...' if len(generated_ids) > 20 else ''}")

        return total_loss, ntp_loss, recon_loss


    def compute_phase2_loss(
        self,
        batch: List[Dict[str, Any]],
        lambda_mse: float = 0.1,
        pred_len: int = 96,
        patch_size: int = PATCH_SIZE,
    ) -> torch.Tensor:
        """Phase-2 loss: CE on all text tokens + MSE on forecasting future values.

        For forecasting samples the answer contains ``<ts> v0 v1 … </ts>``.
        - ``<ts>`` and ``</ts>`` tokens: CE loss (normal next-token prediction).
        - Float tokens between them: masked out from CE (-100), used for MSE.

        The MSE gradient path reuses the existing decoder pipeline:
          LLM hidden state at ``<ts>`` position
            → decoder_projector  (llm_dim → enc_dim)
            → layer_norm
            → expand to (1, n_future_patches, enc_dim)
            → decoder             (produces (1, PRED_LEN))
            → MSE vs ts_future

        Adaptive balancing: scale = (CE / MSE).clamp(0.01, 100), detached, so
        MSE is rescaled to the same order of magnitude as CE.
        """
        n_future_patches = pred_len // patch_size  # 24

        answers = [b["answer"] for b in batch]

        inputs_embeds, attention_mask, _, _, _, _, _ = self.pad_and_apply_batch(batch, return_targets=False)
        B, L, H = inputs_embeds.size()

        ans_tok = self.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        ans_ids  = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb_all = self.llm.get_input_embeddings()(ans_ids)

        bos_id = self.tokenizer.bos_token_id

        new_embeds_list: List[torch.Tensor] = []
        new_masks_list:  List[torch.Tensor] = []
        new_labels_list: List[torch.Tensor] = []

        # Maps sample index to absolute position of <ts> in the assembled sequence
        ts_hidden_positions: Dict[int, int] = {}

        for i in range(B):
            valid_len_i = int(attention_mask[i].sum().item())
            a_len       = int(ans_mask[i].sum().item())

            # Strip BOS prepended by the tokenizer (never a prediction target)
            ans_start = (
                1 if (bos_id is not None and a_len > 0 and ans_ids[i, 0].item() == bos_id)
                else 0
            )
            a_len_eff = a_len - ans_start

            emb_i = torch.cat(
                [inputs_embeds[i, :valid_len_i], ans_emb_all[i, ans_start:a_len]],
                dim=0,
            )
            mask_i = torch.cat(
                [
                    attention_mask[i, :valid_len_i],
                    torch.ones(a_len_eff, device=self.device, dtype=attention_mask.dtype),
                ],
                dim=0,
            )
            lab_i = torch.full(
                (valid_len_i + a_len_eff,), -100, device=self.device, dtype=torch.long
            )
            lab_i[valid_len_i:] = ans_ids[i, ans_start:a_len]

            # For forecasting samples: mask float tokens between <ts> and </ts> from CE
            if batch[i].get("is_forecasting", False) and batch[i].get("ts_future") is not None:
                eff_ids   = ans_ids[i, ans_start:a_len]
                open_pos  = (eff_ids == self.ts_start_token_id).nonzero(as_tuple=True)[0]
                close_pos = (eff_ids == self.ts_end_token_id).nonzero(as_tuple=True)[0]

                if len(open_pos) > 0 and len(close_pos) > 0:
                    p_open  = open_pos[0].item()
                    p_close = close_pos[0].item()

                    # Mask the float tokens strictly between <ts> and </ts>
                    if p_close > p_open + 1:
                        lab_i[valid_len_i + p_open + 1 : valid_len_i + p_close] = -100

                    # Record absolute position of <ts> for the MSE path
                    ts_hidden_positions[i] = valid_len_i + p_open

            new_embeds_list.append(emb_i)
            new_masks_list.append(mask_i)
            new_labels_list.append(lab_i)

        inputs_embeds  = pad_sequence(new_embeds_list, batch_first=True)
        attention_mask = pad_sequence(new_masks_list,  batch_first=True)
        labels         = pad_sequence(new_labels_list, batch_first=True, padding_value=-100)

        need_hidden = bool(ts_hidden_positions)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=need_hidden,
            return_dict=True,
        )

        # Extract what we need and free the rest immediately
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1] if need_hidden else None
        del outputs
        torch.cuda.empty_cache()

        loss_ce = self._chunked_cross_entropy(logits, labels)
        del logits
        torch.cuda.empty_cache()

        zero_mse = loss_ce.new_zeros(1).squeeze()

        if not ts_hidden_positions:
            return loss_ce, loss_ce, zero_mse

        # ---- MSE path (forecasting samples only) ----
        mse_per_sample = []
        for i, ts_pos in ts_hidden_positions.items():
            # Detach from LLM graph: MSE should only train decoder_projector,
            # not flow gradients back into LoRA / the language model.
            h_ts = last_hidden[i, ts_pos, :].float().detach()

            # Project to encoder dimension
            h_enc = self.decoder_projector(h_ts)
            h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])

            # Expand to (1, n_future_patches, enc_dim) so the decoder can
            # produce one patch embedding per future step
            h_dec = h_enc.unsqueeze(0).unsqueeze(0).expand(1, n_future_patches, -1).contiguous()

            # Decoder: (1, n_future_patches, enc_dim) → (1, PRED_LEN)
            pred_ts = self.decoder(h_dec, chan_id=[0]).squeeze(0)  # (PRED_LEN,)

            gt = batch[i]["ts_future"].to(self.device, dtype=pred_ts.dtype)
            mse_per_sample.append(F.mse_loss(pred_ts, gt))

        loss_mse = torch.stack(mse_per_sample).mean()

        # Adaptive balancing: rescale MSE to the same order as CE
        with torch.no_grad():
            scale = (loss_ce / (loss_mse + 1e-8)).clamp(0.01, 100.0)

        total_loss = loss_ce + lambda_mse * scale * loss_mse
        return total_loss, loss_ce, loss_mse



    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def store_to_file(self, path: str):
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "decoder_projector_state": self.decoder_projector.state_dict(),
        }

        # Add LoRA state to checkpoint
        self.save_lora_state_to_checkpoint(checkpoint)

        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        self.decoder.load_state_dict(ckpt["decoder_state"])
        self.decoder_projector.load_state_dict(ckpt["decoder_projector_state"])

        # Load LoRA state if present (allow missing for backward compatibility)
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)

        print(f"📥 Loaded model from epoch {ckpt.get('epoch', '?')}")

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """
        Load LoRA adapters from a checkpoint.

        Args:
            checkpoint: Checkpoint dictionary containing potential LoRA state
            allow_missing: If True, don't raise exception when checkpoint has no LoRA but model expects it

        Raises:
            RuntimeError: When there's a mismatch between checkpoint and current LoRA state
        """
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)

        if checkpoint_has_lora and "lora_state" in checkpoint:
            # Checkpoint has LoRA adapters
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )

            # Load LoRA adapters
            try:
                lora_state = checkpoint["lora_state"]
                loaded_count = 0
                missing_keys = []

                # Track which LoRA parameters we expect to find
                expected_lora_params = {
                    name
                    for name, param in self.llm.named_parameters()
                    if param.requires_grad and "lora_" in name
                }

                for name, param in self.llm.named_parameters():
                    if name in lora_state and param.requires_grad and "lora_" in name:
                        param.data.copy_(lora_state[name])
                        loaded_count += 1
                    elif param.requires_grad and "lora_" in name:
                        missing_keys.append(name)

                if missing_keys and not allow_missing:
                    raise RuntimeError(
                        f"Could not find LoRA parameters in checkpoint: {missing_keys[:5]}..."
                    )

                print(f"📥 Loaded LoRA adapters: {loaded_count} parameters")
                return loaded_count

            except Exception as e:
                if "Could not find LoRA parameters" in str(e):
                    raise  # Re-raise our custom exception
                raise RuntimeError(f"Failed to load LoRA adapters: {e}")

        elif checkpoint_has_lora:
            raise RuntimeError(
                "Checkpoint indicates LoRA was enabled but no LoRA state found"
            )

        # Handle case where checkpoint has no LoRA but model expects it
        if not checkpoint_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError(
                    "Loading checkpoint from before LoRA was enabled, but LoRA is currently enabled. "
                    "LoRA adapters will be randomly initialized. Set allow_missing=True to allow this."
                )
            else:
                print("⚠️  Loading checkpoint from before LoRA was enabled.")
                print("   LoRA adapters will be randomly initialized.")

        return 0

    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """
        Save LoRA adapters to a checkpoint dictionary.

        Args:
            checkpoint: Checkpoint dictionary to add LoRA state to

        Returns:
            int: Number of LoRA parameters saved
        """
        checkpoint["lora_enabled"] = self.lora_enabled

        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            try:
                # Save LoRA adapter weights
                lora_state = {}
                for name, param in self.llm.named_parameters():
                    if param.requires_grad and "lora_" in name:
                        lora_state[name] = param.data.clone()

                if lora_state:
                    checkpoint["lora_state"] = lora_state
                    checkpoint["lora_config"] = self.llm.peft_config
                    print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
                    return len(lora_state)
            except Exception as e:
                raise RuntimeError(f"Failed to save LoRA adapters: {e}")

        return 0

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 30000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """

        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, normalize=normalize
        )
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]
