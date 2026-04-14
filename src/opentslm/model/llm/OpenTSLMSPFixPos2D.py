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
from opentslm.model_config import PATCH_SIZE, PRED_LEN




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
            attn_implementation="sdpa",
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

        # 3) encoder + decoder + projectors
        encoder_patch_size = int(getattr(args, "patch_size", PATCH_SIZE))
        self.encoder = TransformerCNNEncoder2D(
            args=args,
            patch_size=encoder_patch_size,
        ).to(device)

        # Keep patch-size source of truth consistent across model components.
        self.patch_size = int(self.encoder.patch_size)

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

        # Decoder: causal self-attention over growing LLM hidden-state sequence.
        # input_dim must match decoder_projector output (enc_dim).
        # patch_size taken from encoder so a single --patch_size flag controls all.
        self.decoder = TransformerCNNDecoder2D(
            input_dim=enc_dim,
            patch_size=self.patch_size,
        ).to(device)

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

        # Separate projector for the enc_mse training path.
        # Reads directly from encoder output (enc_dim → enc_dim), bypassing the
        # LLM projector entirely.  Trained only by MSE loss → no CE pressure to
        # collapse patch representations into a single "TS token" direction.
        self.mse_projector = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, enc_dim),
        ).to(device)

        # Temporary storage for raw encoder output, set by pad_and_apply_batch
        # when return_targets=True. Used by the numeric forecasting losses.
        self._last_ts_enc: Optional[torch.Tensor] = None
        self._last_ts_raw: Optional[torch.Tensor] = None
        self._last_ts_chan_id: Optional[torch.Tensor] = None
        self._last_ts_patch_mask: Optional[torch.Tensor] = None
        self._last_ts_patch_target_valid: Optional[torch.Tensor] = None
        self._last_pretrain_next_patch_mse: Optional[torch.Tensor] = None
        self._last_pretrain_final_patch_mse: Optional[torch.Tensor] = None

    def _chunked_cross_entropy(self, logits, labels, chunk_size=128):
        """Memory-efficient cross-entropy that avoids materializing full float32 logits."""
        # Work directly on the original tensors — no .contiguous() copy
        # Shift: logits[..., :-1, :] predicts labels[..., 1:]
        seq_len = logits.size(1) - 1
        total_loss = 0.0
        num_tokens = 0

        for j in range(0, seq_len, chunk_size):
            end = min(j + chunk_size, seq_len)
            chunk_logits = logits[:, j:end, :]
            chunk_labels = labels[:, j+1:end+1]

            loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)).float(),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction='sum'
            )
            total_loss += loss
            num_tokens += (chunk_labels != -100).sum()

            del chunk_logits, chunk_labels, loss

        return total_loss / num_tokens.clamp(min=1)

    def _chunked_cross_entropy_with_row_stats(self, logits, labels, chunk_size=128):
        """Chunked CE that also returns per-row loss sums/token counts."""
        seq_len = logits.size(1) - 1
        batch_size = logits.size(0)

        row_loss_sums = logits.new_zeros((batch_size,), dtype=torch.float32)
        row_token_counts = labels.new_zeros((batch_size,), dtype=torch.long)

        for j in range(0, seq_len, chunk_size):
            end = min(j + chunk_size, seq_len)
            chunk_logits = logits[:, j:end, :]
            chunk_labels = labels[:, j + 1 : end + 1]

            per_token_loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)).float(),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(batch_size, -1)
            valid = chunk_labels != -100

            row_loss_sums += (per_token_loss * valid).sum(dim=1)
            row_token_counts += valid.sum(dim=1)

            del chunk_logits, chunk_labels, per_token_loss, valid

        total_tokens = row_token_counts.sum()
        total_loss = row_loss_sums.sum() / total_tokens.clamp(min=1)
        return total_loss, row_loss_sums, row_token_counts

    def _llm_and_chunked_ce(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        return_row_stats: bool = False,
        chunk_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard HuggingFace LLM forward + chunked CE.

        Uses self.llm(...) directly so all model-specific logit transforms
        (softcapping, scaling, etc.) are applied exactly as in the original
        forward.  _chunked_cross_entropy then processes outputs.logits in
        chunks to avoid upcasting the full [B, L, vocab_size] tensor to float32
        all at once.

        Returns:
            ce_loss:     scalar, token-mean CE (label=-100 ignored)
            last_hidden: [B, L, H] last hidden state for MSE / TS patch prediction
        """
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        if return_row_stats:
            ce_loss, row_loss_sums, row_token_counts = (
                self._chunked_cross_entropy_with_row_stats(outputs.logits, labels, chunk_size=chunk_size)
            )
        else:
            ce_loss = self._chunked_cross_entropy(outputs.logits, labels, chunk_size=chunk_size)
        last_hidden = outputs.hidden_states[-1]
        del outputs
        if return_row_stats:
            return ce_loss, last_hidden, row_loss_sums, row_token_counts
        return ce_loss, last_hidden

    def _llm_last_hidden_only(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return final hidden states without running the LM head.

        This is used by pure-MSE stages, which need the LLM backbone features
        but do not use token logits at all.
        """
        backbone = self.llm.get_base_model() if hasattr(self.llm, "get_base_model") else self.llm
        transformer = backbone.model if hasattr(backbone, "model") else backbone

        outputs = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        del outputs
        return last_hidden


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
        detach_encoder: bool = False,
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
        ts_real_lens: List[int] = []
        ts_num_patches: List[int] = []
        ts_norm_stats: List[Tuple[float, float]] = []

        for b, sample in enumerate(batch):
            ts_lengths = sample.get("time_series_lengths")
            ts_normed = bool(sample.get("ts_already_normalized", False))
            sample_norm_stats = sample.get("time_series_norm_stats")

            for c, ts in enumerate(sample["time_series"]):
                ts = torch.as_tensor(ts, dtype=torch.float32)
                # Normalize to [T, 1].  Channel-independent datasets emit one
                # [T] tensor per channel; accept [T,1] too.
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)          # [T] -> [T, 1]
                if ts.size(-1) != 1:
                    raise ValueError(
                        f"Expected univariate series [T] or [T,1], got {tuple(ts.shape)}. "
                        "For multivariate data use channel-independent prompts (one per channel)."
                    )

                flat = ts.squeeze(-1)          # [T]
                if ts_lengths is not None and len(ts_lengths) > c:
                    length_val = ts_lengths[c]
                    real_len = int(length_val.item() if torch.is_tensor(length_val) else length_val)
                    real_len = max(0, min(real_len, flat.size(0)))
                else:
                    real_len = flat.size(0)
                flat_real = flat[:real_len]
                ts_real_lens.append(real_len)
                ts_num_patches.append((real_len + self.patch_size - 1) // self.patch_size)

                # Normalize only the real, unpadded prefix. Datasets that already
                # normalize upstream can opt out via ts_already_normalized=True.
                if ts_normed:
                    ts_proc = flat_real
                else:
                    m = flat_real.mean()
                    s = flat_real.std().clamp(min=1e-8)
                    ts_proc = (flat_real - m) / s

                if sample_norm_stats is not None and len(sample_norm_stats) > c:
                    raw_m, raw_s = sample_norm_stats[c]
                    ts_norm_stats.append((float(raw_m), float(raw_s)))
                elif ts_normed and len(sample["time_series"]) == 1 and "norm_mean" in sample and "norm_std" in sample:
                    ts_norm_stats.append((float(sample["norm_mean"]), float(sample["norm_std"])))
                else:
                    # Fall back to stats in the current encoder-input space.
                    ts_norm_stats.append((
                        float(flat_real.mean().item()),
                        float(flat_real.std().clamp(min=1e-8).item()),
                    ))

                ts_list.append(ts_proc.unsqueeze(-1))
                ts_chan_id.append(c)

        # Best-effort per-series stats for downstream plotting/de-normalization.
        # When datasets expose raw normalization stats we preserve them; otherwise
        # these are stats of the encoder-input space.
        self._ts_norm_stats = ts_norm_stats

        # if any(s.get("type") == "chronos" for s in batch):
        #     print("=" * 60)
        #     print("[DEBUG] first chronos batch")
        #     for i, s in enumerate(batch):
        #         if s.get("type") != "chronos":
        #             continue
        #         ts_list_raw = s.get("time_series", [])
        #         ts_text_list = s.get("time_series_text", [])
        #         print(f"  sample[{i}]  n_ts={len(ts_list_raw)}  n_texts={len(ts_text_list)}")
        #         for j, ts in enumerate(ts_list_raw):
        #             arr = ts.numpy() if hasattr(ts, 'numpy') else ts
        #             import numpy as _np
        #             arr = _np.asarray(arr)
        #             print(f"    ts[{j}]: shape={arr.shape}  dtype={arr.dtype}")
        #         print(f"    time_series_text={ts_text_list}")
        #         print(f"    keys={[k for k in s.keys() if k not in ('time_series', 'time_series_text')]}")
        #         break  # just the first chronos sample is enough
        #     print(f"[DEBUG] ts_chan_id (full batch): {ts_chan_id}")
        #     print("=" * 60)
        #     quit()

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
                ts_padded = torch.cat([ts_padded, pad], dim=1) # (n_ts, T_max, 1)

            max_patches = ts_padded.size(1) // self.patch_size
            n_real_t = torch.tensor(ts_num_patches, device=device)  # [N_ts]
            ts_patch_mask = (
                torch.arange(max_patches, device=device).unsqueeze(0) < n_real_t.unsqueeze(1)
            )

            if return_targets:
                ts_patch_targets = ts_padded.squeeze(-1).reshape(ts_padded.size(0), -1, self.patch_size)
                N_ts_total, N_patches, P = ts_patch_targets.shape

                ts_patch_targets_shifted = torch.zeros_like(ts_patch_targets)
                ts_patch_targets_shifted[:, :-1, :] = ts_patch_targets[:, 1:, :]

                # Mark patch i as valid when patch i+1 exists as a real patch
                # (even if partial).  Per-timestep masking in the loss handles
                # any trailing zeros inside the last real patch.
                ts_patch_valid = (
                    torch.arange(N_patches, device=device).unsqueeze(0)
                    < (n_real_t - 1).clamp(min=0).unsqueeze(1)
                )

                # Per-timestep validity for shifted targets: True where the
                # actual timestep (j+1)*P + p is within real_len for that series.
                # Shape: [N_ts, N_patches, P]
                patch_idx = torch.arange(N_patches, device=device)   # [N_patches]
                step_idx  = torch.arange(P, device=device)            # [P]
                global_ts = (patch_idx + 1).unsqueeze(1) * P + step_idx.unsqueeze(0)  # [N_patches, P]
                real_lens_t = torch.tensor(ts_real_lens, device=device)               # [N_ts]
                ts_patch_target_valid = global_ts.unsqueeze(0) < real_lens_t.view(-1, 1, 1)

            ts_raw = ts_padded.squeeze(-1)

            # # ── DEBUG PADDING ─────────────────────────────────────────────────
            # print("=" * 70)
            # print(f"[PAD DEBUG] patch_size={self.patch_size}  N_ts={len(ts_list)}  "
            #       f"T_max(before align)={T_max}  T_max(after align)={ts_padded.size(1)}  "
            #       f"max_patches={max_patches}")
            # print(f"  {'series':>6}  {'real_len':>8}  {'n_real_patches':>14}  "
            #       f"{'partial_last?':>13}  {'valid_j_range':>13}  {'real_steps_in_last_patch':>24}")
            # print("  " + "-" * 72)
            # for i in range(len(ts_list)):
            #     real_t  = ts_real_lens[i]
            #     n_ceil  = ts_num_patches[i]
            #     partial = (real_t % self.patch_size) != 0
            #     # valid j: j < n_real_t - 1
            #     j_range = f"0..{n_ceil - 2}" if n_ceil >= 3 else ("0" if n_ceil == 2 else "none")
            #     real_in_last = real_t % self.patch_size if partial else self.patch_size
            #     print(f"  {i:>6}  {real_t:>8}  {n_ceil:>14}  "
            #           f"{'YES' if partial else 'no':>13}  {j_range:>13}  "
            #           f"{real_in_last:>24}")
            # if return_targets:
            #     print(f"\n[PAD DEBUG] ts_patch_target_valid (True=real timestep in target patch):")
            #     print(ts_patch_target_valid.cpu())
            # print("=" * 70)
            # quit()
            # ── END DEBUG ──────────────────────────────────────────────────────
            ts_chan_tensor = torch.tensor(ts_chan_id, device=device)

            ts_enc = self.encoder(
                ts_raw,
                chan_id=ts_chan_tensor,
                attention_mask=ts_patch_mask,
            )
            # Save raw encoder output for the enc_mse loss path.  Only set when
            # return_targets=True (i.e. for Chronos in joint training), so CE
            # calls (return_targets=False) never overwrite this.
            if return_targets:
                self._last_ts_enc = ts_enc   # [N_ts, T_patches, enc_dim]
                self._last_ts_raw = ts_raw
                self._last_ts_chan_id = ts_chan_tensor
                self._last_ts_patch_mask = ts_patch_mask
                self._last_ts_patch_target_valid = ts_patch_target_valid  # [N_ts, N_patches, P]
            # detach_encoder=True: stop CE gradients from flowing back through
            # the encoder (used in joint_phase1_enc_mse so only MSE trains encoder).
            enc_for_proj = ts_enc.detach() if detach_encoder else ts_enc
            ts_proj = self.projector(enc_for_proj).to(text_embeds.dtype)
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)
            self._ts_norm_stats = []
            if return_targets:
                self._last_ts_enc = None
                self._last_ts_raw = None
                self._last_ts_chan_id = None
                self._last_ts_patch_mask = None
                self._last_ts_patch_target_valid = None

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
        for sample_idx, ((start, end), n_ts) in enumerate(zip(text_ptrs, ts_counts)):
            sample = batch[sample_idx]
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
                    sample_embeds[idx], sample_masks[idx], sample_ids[idx], strip_bos=True,
                )

                seq_embeds.append(txt_emb)
                seq_masks.append(txt_m)

                if return_targets:
                    seq_ids.append(txt_ids)
                    seq_flags.append(torch.zeros(txt_ids.numel(), device=device, dtype=torch.long))

                ts_text_list = sample.get("time_series_text", [])
                ts_text_i = ts_text_list[i] if i < len(ts_text_list) else ""
                if "<ts>" not in str(ts_text_i):
                    open_emb = self.llm.get_input_embeddings()(
                        torch.tensor([self.ts_start_token_id], device=device)
                    )  # [1, H]
                    seq_embeds.append(open_emb)
                    seq_masks.append(torch.ones(1, device=device, dtype=torch.long))

                    if return_targets:
                        seq_ids.append(torch.tensor([self.ts_start_token_id], device=device))
                        seq_flags.append(torch.zeros(1, device=device, dtype=torch.long))

                n_real = ts_num_patches[ts_offset + i]
                proj = ts_proj[ts_offset + i, :n_real]
                n_patches = n_real

                seq_embeds.append(proj)
                seq_masks.append(torch.ones(n_patches, device=device, dtype=torch.long))

                if return_targets:
                    seq_ids.append(torch.full((n_patches,), -100, device=device, dtype=torch.long))
                    seq_flags.append(torch.ones(n_patches, device=device, dtype=torch.long))

                # inject </ts> closing token after each TS block
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
                sample_embeds[-1], sample_masks[-1], sample_ids[-1], strip_bos=True,
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

    def _project_foundational_hidden(
        self,
        ts_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Project encoder states through the same adapters used by the joint path."""
        ts_llm = self.projector(ts_enc.float())
        ts_llm = F.layer_norm(ts_llm, [ts_llm.size(-1)])
        ts_hidden = self.decoder_projector(ts_llm)
        return F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

    def _build_foundational_history_hidden_from_raw(
        self,
        ts_raw: torch.Tensor,
        chan_id: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode raw series into foundational history states for forecasting."""
        if ts_raw.size(1) % self.patch_size != 0:
            raise ValueError(
                f"L={ts_raw.size(1)} not divisible by patch_size={self.patch_size}"
            )

        if attention_mask is None:
            n_patches = ts_raw.size(1) // self.patch_size
            attention_mask = torch.ones(
                ts_raw.size(0),
                n_patches,
                device=ts_raw.device,
                dtype=torch.bool,
            )

        ts_enc = self.encoder(
            ts_raw,
            chan_id=chan_id,
            attention_mask=attention_mask,
        )
        return self._project_foundational_hidden(ts_enc)

    def build_foundational_history_hidden(self) -> torch.Tensor:
        """
        Build foundational history states for forecasting-only Phase 1.

        Reuses the full contextual encoder stack, then maps the resulting
        patch states into the numeric decoder space via the dedicated
        forecasting projector.
        """
        if self._last_ts_enc is not None:
            return self._project_foundational_hidden(self._last_ts_enc)

        if self._last_ts_raw is None or self._last_ts_chan_id is None:
            raise RuntimeError(
                "Foundational history tokens are unavailable. "
                "Call pad_and_apply_batch(..., return_targets=True) first."
            )

        return self._build_foundational_history_hidden_from_raw(
            self._last_ts_raw,
            self._last_ts_chan_id,
            attention_mask=self._last_ts_patch_mask,
        )


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
          3. After each patch, the LLM logits are checked: if argmax equals
             </ts>, the patch loop stops (model self-detects the boundary).
             n_forecast_patches acts as a safety cap if </ts> is never predicted.
        Decoder is only active inside TS generation blocks.

        NOTE: TS generation is triggered only when ALL batch items simultaneously
        emit <ts>.  For mixed text/TS batches, use batch_size=1.

        Args:
            batch:              list of sample dicts (same format as compute_loss).
            max_new_tokens:     maximum number of text tokens to generate.
            n_forecast_patches: maximum patches to generate per <ts> block (safety
                                cap).  Once the model emits <ts>, forecasting is
                                delegated to the numeric decoder for exactly this
                                many patches, then </ts> is injected to resume text
                                generation.  If None, <ts> is treated as a plain
                                text token (original behaviour, no decoder involved).

        Returns:
            text_outputs:     List[str]
            ts_patch_outputs: List[List[Tensor[N_patches, patch_size]]]
                              outer list = batch; inner list = one tensor per
                              <ts> block emitted during generation.
        """
        if n_forecast_patches is not None and n_forecast_patches < 1:
            raise ValueError(
                f"n_forecast_patches must be >= 1 when provided, got {n_forecast_patches}."
            )

        # Gradient checkpointing is incompatible with use_cache=True (HuggingFace
        # silently sets use_cache=False when GC is on, so past_key_values is never
        # built and every autoregressive step re-runs the full sequence).
        # Disable GC for generation so KV-cache actually works, then restore.
        gc_was_enabled = self.llm.is_gradient_checkpointing
        if gc_was_enabled:
            self.llm.gradient_checkpointing_disable()

        try:
            return self._generate_impl(batch, max_new_tokens, n_forecast_patches)
        finally:
            if gc_was_enabled:
                self.llm.gradient_checkpointing_enable()

    def _generate_impl(
        self,
        batch: List[Dict[str, Any]],
        max_new_tokens: int,
        n_forecast_patches: Optional[int],
    ) -> Tuple[List[str], List[List[torch.Tensor]]]:
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
        valid_prompt_lens = attn_mask.sum(dim=1).long().clamp(min=1)
        batch_idx = torch.arange(B, device=device)
        next_logits = prompt_out.logits[batch_idx, valid_prompt_lens - 1, :]   # [B, V]

        prompt_hidden = prompt_out.hidden_states[-1]   # [B, L_prompt, H]
        hist_buf_list = None

        if n_forecast_patches is not None:
            def _num_ts_segments(sample: Dict[str, Any]) -> int:
                ts_segs = sample.get("time_series", [])
                if isinstance(ts_segs, torch.Tensor):
                    return 1 if ts_segs.dim() <= 1 else int(ts_segs.size(0))
                if isinstance(ts_segs, (list, tuple)):
                    return len(ts_segs)
                return 0

            invalid = [
                (idx, _num_ts_segments(sample))
                for idx, sample in enumerate(batch)
                if _num_ts_segments(sample) != 1
            ]
            if invalid:
                summary = ", ".join(
                    f"sample {idx}: {count} segments" for idx, count in invalid[:4]
                )
                if len(invalid) > 4:
                    summary += f", ... ({len(invalid)} total)"
                raise NotImplementedError(
                    "generate() with n_forecast_patches currently supports exactly "
                    f"one time-series segment per sample; got {summary}."
                )

            # ── Pre-build decoder history buffer ─────────────────────────────
            # Collect all history patch hidden states and project them into
            # encoder space. The decoder then reads [history | generated so far].
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

        eos_id = self.tokenizer.eos_token_id
        generated_ids:     List[List[int]]          = [[] for _ in range(B)]
        ts_patch_outputs:  List[List[torch.Tensor]] = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
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

        # ── 3. Autoregressive generation loop ───────────────────────────────────
        for _ in range(max_new_tokens):
            next_tokens = next_logits.argmax(dim=-1)            # [B]
            next_tokens = next_tokens.masked_fill(finished, pad_id)
            active_mask = ~finished

            # ── TS generation block (all items must emit <ts> together) ──────────
            if (
                n_forecast_patches is not None
                and active_mask.any()
                and (next_tokens[active_mask] == self.ts_start_token_id).all()
            ):
                entered_ts_mask = active_mask.clone()
                for b in range(B):
                    if entered_ts_mask[b]:
                        generated_ids[b].append(self.ts_start_token_id)

                # Feed <ts> to update KV cache (hidden state not used as seed)
                ts_start_ids = torch.full((B, 1), pad_id, device=device, dtype=torch.long)
                ts_start_ids[entered_ts_mask] = self.ts_start_token_id
                _step(self.llm.get_input_embeddings()(ts_start_ids))

                raise NotImplementedError(
                    "Numeric patch generation through decoder-internal autoregression "
                    "has been removed. Use the direct encoder -> projector -> decoder "
                    "workflow with explicit re-encoding of predicted patches."
                )

                # The controller knows the requested horizon, so it injects
                # </ts> after the numeric decoder emits exactly that many patches.
                for b in range(B):
                    if entered_ts_mask[b]:
                        generated_ids[b].append(self.ts_end_token_id)
                ts_end_ids = torch.full((B, 1), pad_id, device=device, dtype=torch.long)
                ts_end_ids[entered_ts_mask] = self.ts_end_token_id
                out = _step(self.llm.get_input_embeddings()(ts_end_ids))
                next_logits = out.logits[:, -1, :]
                continue

            # ── Normal text step ─────────────────────────────────────────────────
            for b in range(B):
                if active_mask[b]:
                    generated_ids[b].append(next_tokens[b].item())

            if eos_id is not None:
                finished = finished | (active_mask & (next_tokens == eos_id))
                if finished.all():
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
            lines.append(f"  answer     : {str(answer)[:400]}")

        lines.append(sep)
        print("\n".join(lines), flush=True)
        sys.stdout.flush()

    @torch.no_grad()
    def compute_loss_per_task(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return {task_type: loss} for each task type present in the batch.
        Chronos uses MSE (decoder next-patch loss); all other tasks use CE.
        Also computes TSQA letter-accuracy."""
        results = {}
        task_types = list({r.get("type", "unknown") for r in batch})

        for task in task_types:
            sub = [r for r in batch if r.get("type") == task]
            if not sub:
                continue
            try:
                if task == "chronos":
                    # CE on float tokens is meaningless — use decoder MSE instead
                    _, _, loss = self.compute_pretrain_loss(sub)
                else:
                    loss = self.compute_loss(sub)
                results[f"loss/{task}"] = loss.item()
            except Exception:
                results[f"loss/{task}"] = float("nan")

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

        ce_loss, _ = self._llm_and_chunked_ce(inputs_embeds, attention_mask, labels)
        return ce_loss
        




    def compute_joint_phase1_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
        dynamic_weighting: bool = False,
        detach_llm_for_mse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint Phase-1 loss: CE on text tasks + MSE next-patch on Chronos.

        Single LLM forward pass over all samples:
        - CE samples (tsqa/m4): [prompt | answer] with labels at answer positions.
        - Chronos: [prompt only] with all-(-100) labels → zero CE gradient.
        MSE gradient flows back through encoder+projector (detached at LLM hidden states), while CE gradient is isolated to
        CE samples via the all-(-100) labels on Chronos rows.

        Args:
            detach_llm_for_mse: When True (used for LoRA stages), the MSE decoder
                path reads from chr_embeds (projector output) rather than
                last_hidden (LLM output).  Forecasting gradients then flow
                encoder → projector → decoder_projector → decoder without
                touching LLM/LoRA parameters.  When False (default), the
                existing behaviour is preserved: last_hidden is used and
                MSE gradients reach the LLM.

        Returns:
            (total_loss, loss_ce, loss_tsqa, loss_m4, loss_mse)
        """
        device = self.device
        zero   = next(self.parameters()).new_zeros(())

        tsqa_batch    = [s for s in batch if s.get("type") == "tsqa"]
        m4_batch      = [s for s in batch if s.get("type") in ("m4", "m4_captioning")]
        ce_batch      = tsqa_batch + m4_batch
        chronos_batch = [s for s in batch if s.get("type") == "chronos"]

        B_ce  = len(ce_batch)
        B_chr = len(chronos_batch)

        # ── Step 1: Embed prompts (encoder + projector) ───────────────────
        if ce_batch:
            (
                ce_embeds,   # [B_ce, L_ce, H]
                ce_attn,     # [B_ce, L_ce]
                _, _, _, _, _,
            ) = self.pad_and_apply_batch(ce_batch, return_targets=False)

        if chronos_batch:
            (
                chr_embeds,           # [B_chr, L_chr, H]
                chr_attn,             # [B_chr, L_chr]
                _,
                chr_ts_mask,          # [B_chr, L_chr]  True at INPUT TS patch positions
                chr_targets_shifted,  # [N_ts_chr, N_patches, P]
                chr_ts_valid,         # [N_ts_chr, N_patches] bool
                _,
            ) = self.pad_and_apply_batch(chronos_batch, return_targets=True)

        # ── Step 2: Tokenize & embed CE answers ───────────────────────────
        if ce_batch:
            ans_tok = self.tokenizer(
                [s["answer"] for s in ce_batch],
                return_tensors="pt", padding=True, truncation=True, max_length=4096,
            )
            ans_ids  = ans_tok.input_ids.to(device, non_blocking=True)   # [B_ce, A_max]
            ans_mask = ans_tok.attention_mask.to(device, non_blocking=True)
            ans_emb  = self.llm.get_input_embeddings()(ans_ids)          # [B_ce, A_max, H]
            bos_id   = self.tokenizer.bos_token_id

        # ── Step 3: Build per-sample combined sequences ───────────────────
        all_embeds_list: List[torch.Tensor] = []
        all_masks_list:  List[torch.Tensor] = []
        all_labels_list: List[torch.Tensor] = []

        # CE: [prompt | answer], labels at answer positions
        if ce_batch:
            for i in range(B_ce):
                valid_i   = int(ce_attn[i].sum().item())
                a_len     = int(ans_mask[i].sum().item())
                ans_start = (1 if (bos_id is not None and a_len > 0
                                   and ans_ids[i, 0].item() == bos_id) else 0)
                a_eff = a_len - ans_start

                emb_i  = torch.cat([ce_embeds[i, :valid_i],
                                     ans_emb[i, ans_start:a_len]], dim=0)
                mask_i = torch.cat([ce_attn[i, :valid_i],
                                     torch.ones(a_eff, device=device,
                                                dtype=ce_attn.dtype)], dim=0)
                lab_i  = torch.full((valid_i + a_eff,), -100,
                                     device=device, dtype=torch.long)
                lab_i[valid_i:] = ans_ids[i, ans_start:a_len]

                all_embeds_list.append(emb_i)
                all_masks_list.append(mask_i)
                all_labels_list.append(lab_i)

        # Chronos: [prompt only], all-(-100) labels → zero CE gradient
        if chronos_batch:
            for j in range(B_chr):
                valid_j = int(chr_attn[j].sum().item())
                all_embeds_list.append(chr_embeds[j, :valid_j])
                all_masks_list.append(chr_attn[j, :valid_j])
                all_labels_list.append(
                    torch.full((valid_j,), -100, device=device, dtype=torch.long)
                )

        # ── Step 4: Single LLM forward ────────────────────────────────────
        inputs_embeds_comb  = pad_sequence(all_embeds_list, batch_first=True)
        attention_mask_comb = pad_sequence(all_masks_list,  batch_first=True)
        labels              = pad_sequence(all_labels_list, batch_first=True,
                                            padding_value=-100)
        L_max = inputs_embeds_comb.size(1)

        chronos_present = bool(chronos_batch) and chr_targets_shifted.numel() > 0

        # ── Step 5: CE loss ───────────────────────────────────────────────
        need_task_breakdown = bool(tsqa_batch and m4_batch)
        if need_task_breakdown:
            loss_ce_raw, last_hidden, row_loss_sums, row_token_counts = (
                self._llm_and_chunked_ce(
                    inputs_embeds_comb,
                    attention_mask_comb,
                    labels,
                    return_row_stats=True,
                )
            )
        else:
            loss_ce_raw, last_hidden = self._llm_and_chunked_ce(
                inputs_embeds_comb, attention_mask_comb, labels
            )
        loss_ce = loss_ce_raw if ce_batch else zero

        # ── Step 6: Per-task logging from the existing CE forward ──────────
        if need_task_breakdown:
            n_tsqa = len(tsqa_batch)
            tsqa_tokens = row_token_counts[:n_tsqa].sum()
            m4_tokens = row_token_counts[n_tsqa : n_tsqa + len(m4_batch)].sum()
            loss_tsqa = (
                row_loss_sums[:n_tsqa].sum() / tsqa_tokens.clamp(min=1)
                if tsqa_tokens.item() > 0
                else zero
            )
            loss_m4 = (
                row_loss_sums[n_tsqa : n_tsqa + len(m4_batch)].sum()
                / m4_tokens.clamp(min=1)
                if m4_tokens.item() > 0
                else zero
            )
        elif tsqa_batch:
            loss_tsqa = loss_ce.detach()
            loss_m4   = zero.detach()
        elif m4_batch:
            loss_tsqa = zero.detach()
            loss_m4   = loss_ce.detach()
        else:
            loss_tsqa = zero.detach()
            loss_m4   = zero.detach()

        # ── Step 7: MSE for Chronos (mirrors compute_pretrain_loss) ───────
        if not chronos_present:
            return loss_ce, loss_ce, loss_tsqa, loss_m4, zero

        # last_hidden already set above

        # Extend chr_ts_mask from L_chr to L_max (pad with False; those are
        # padding positions in the combined sequence and contain no TS tokens).
        L_chr = chr_ts_mask.size(1)
        if L_chr < L_max:
            chr_ts_mask_ext = torch.cat([
                chr_ts_mask,
                chr_ts_mask.new_zeros(B_chr, L_max - L_chr, dtype=torch.bool),
            ], dim=1)  # [B_chr, L_max]
        else:
            chr_ts_mask_ext = chr_ts_mask[:, :L_max]

        # Flat extraction of TS hidden states for the MSE decoder path.
        # detach_llm_for_mse=True (LoRA stage): use chr_embeds (projector output)
        # so forecasting gradients flow encoder→projector→decoder_projector→decoder
        # without touching LLM/LoRA parameters.
        # detach_llm_for_mse=False (default): use last_hidden (LLM output) as before.
        if detach_llm_for_mse:
            # chr_embeds is [B_chr, L_chr, H]; chr_ts_mask is [B_chr, L_chr].
            ts_hidden = chr_embeds[chr_ts_mask]  # [sum(patch_counts), H]
        else:
            # last_hidden[B_ce:] contains the Chronos rows; chr_ts_mask_ext selects
            # exactly the INPUT TS patch positions (same logic as compute_pretrain_loss).
            ts_hidden = last_hidden[B_ce:][chr_ts_mask_ext]  # [sum(patch_counts), H]

        N_ts_chr, N_patches, P = chr_targets_shifted.shape
        patch_counts = (chr_ts_valid.sum(dim=1) + 1).tolist()
        assert ts_hidden.size(0) == int(sum(patch_counts)), (
            f"ts_mask extracted {ts_hidden.size(0)} hidden vectors but expected "
            f"{int(sum(patch_counts))}."
        )

        # Project + normalize, then batched decoder call
        ts_hidden    = self.decoder_projector(ts_hidden.float())
        ts_hidden    = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])
        ts_hidden_3d = pad_sequence(
            list(torch.split(ts_hidden, patch_counts)), batch_first=True
        )  # [N_ts_chr, N_patches, enc_dim]

        ts_pred = self.decoder(ts_hidden_3d)  # [N_ts_chr, N_patches, P]

        targets_flat = chr_targets_shifted.reshape(-1, P)
        pred_flat    = ts_pred.reshape(-1, P).to(targets_flat.dtype)
        valid_flat   = chr_ts_valid.reshape(-1)
        valid_count  = int(valid_flat.sum().item())

        loss_mse = (
            F.mse_loss(pred_flat[valid_flat], targets_flat[valid_flat])
            if valid_count > 0 else zero
        )

        if dynamic_weighting and ce_batch:
            mse_weight = (loss_ce.detach() / (loss_mse.detach() + 1e-8)).clamp(0.01, 100.0)
        else:
            mse_weight = alpha

        total = loss_ce + mse_weight * loss_mse
        return total, loss_ce, loss_tsqa, loss_m4, loss_mse

    def compute_joint_phase1_enc_mse_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
        dynamic_weighting: bool = False,
        detach_llm_for_mse: bool = False,  # unused; kept for API parity
        detach_enc_for_ce: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint Phase-1 loss with an independent MSE path that reads directly
        from encoder output, bypassing the projector and frozen LLM entirely.

        Gradient paths:
          CE  path: encoder → projector → LLM (frozen) → CE loss
                    (identical to existing joint_phase1_extend behaviour)
          MSE path: encoder → mse_projector → decoder → MSE loss
                    (encoder output has cosine-sim ≈ 0.80 — temporally diverse;
                     mse_projector is never touched by CE loss, so it cannot
                     collapse all patches into a single "TS token" direction)

        The two paths share only the encoder.  mse_projector and
        decoder_projector are separate parameters with separate optimiser groups.

        detach_enc_for_ce (default False):
            When True, encoder output is detached before the projector so CE
            loss trains projector but NOT encoder.  The encoder is then trained
            exclusively by the MSE path.  Used by joint_phase1_enc_mse_detach.

        Returns:
            (total_loss, loss_ce, loss_tsqa, loss_m4, loss_mse)
        """
        zero = next(self.parameters()).new_zeros(())

        tsqa_batch    = [s for s in batch if s.get("type") == "tsqa"]
        m4_batch      = [s for s in batch if s.get("type") in ("m4", "m4_captioning")]
        ce_batch      = tsqa_batch + m4_batch
        chronos_batch = [s for s in batch if s.get("type") == "chronos"]

        # ── CE path ──────────────────────────────────────────────────────────
        # detach_enc_for_ce=True (joint_phase1_enc_mse_detach only):
        #   encoder output is detached before projector → CE trains projector
        #   only; encoder gradient comes exclusively from the MSE path.
        if ce_batch:
            (
                ce_embeds, ce_attn, _, _, _, _, _,
            ) = self.pad_and_apply_batch(ce_batch, return_targets=False,
                                         detach_encoder=detach_enc_for_ce)

        if chronos_batch:
            (
                chr_embeds, chr_attn, _,
                chr_ts_mask,
                chr_targets_shifted,   # [N_ts, N_patches, P]
                chr_ts_valid,          # [N_ts, N_patches] bool
                _,
            ) = self.pad_and_apply_batch(chronos_batch, return_targets=True)
            # _last_ts_enc was set during pad_and_apply_batch above
            chr_enc_out = self._last_ts_enc   # [N_ts, T_patches, enc_dim]

        # Build combined LLM input (CE samples only; Chronos contributes no CE loss)
        from torch.nn.utils.rnn import pad_sequence as _pad
        device = self.device
        all_embeds, all_masks, all_labels = [], [], []

        if ce_batch:
            ans_tok = self.tokenizer(
                [s["answer"] for s in ce_batch],
                return_tensors="pt", padding=True, truncation=True, max_length=4096,
            )
            ans_ids  = ans_tok.input_ids.to(device, non_blocking=True)
            ans_mask = ans_tok.attention_mask.to(device, non_blocking=True)
            ans_emb  = self.llm.get_input_embeddings()(ans_ids)
            bos_id   = self.tokenizer.bos_token_id
            B_ce     = len(ce_batch)

            for i in range(B_ce):
                valid_i   = int(ce_attn[i].sum().item())
                a_len     = int(ans_mask[i].sum().item())
                ans_start = (1 if (bos_id is not None and a_len > 0
                                   and ans_ids[i, 0].item() == bos_id) else 0)
                a_eff     = a_len - ans_start
                emb_i  = torch.cat([ce_embeds[i, :valid_i],
                                    ans_emb[i, ans_start:a_len]], dim=0)
                mask_i = torch.cat([ce_attn[i, :valid_i],
                                    torch.ones(a_eff, device=device,
                                               dtype=ce_attn.dtype)], dim=0)
                lab_i  = torch.full((valid_i + a_eff,), -100,
                                    device=device, dtype=torch.long)
                lab_i[valid_i:] = ans_ids[i, ans_start:a_len]
                all_embeds.append(emb_i)
                all_masks.append(mask_i)
                all_labels.append(lab_i)

        if chronos_batch:
            for j in range(len(chronos_batch)):
                valid_j = int(chr_attn[j].sum().item())
                all_embeds.append(chr_embeds[j, :valid_j])
                all_masks.append(chr_attn[j, :valid_j])
                all_labels.append(
                    torch.full((valid_j,), -100, device=device, dtype=torch.long)
                )

        if all_embeds:
            loss_ce_raw, _ = self._llm_and_chunked_ce(
                _pad(all_embeds, batch_first=True),
                _pad(all_masks,  batch_first=True),
                _pad(all_labels, batch_first=True, padding_value=-100),
            )
        else:
            loss_ce_raw = zero

        loss_ce = loss_ce_raw if ce_batch else zero

        if tsqa_batch and not m4_batch:
            loss_tsqa, loss_m4 = loss_ce.detach(), zero.detach()
        elif m4_batch and not tsqa_batch:
            loss_tsqa, loss_m4 = zero.detach(), loss_ce.detach()
        else:
            loss_tsqa, loss_m4 = zero.detach(), zero.detach()

        # ── MSE path: encoder output → mse_projector → decoder ───────────────
        if not chronos_batch or chr_enc_out is None:
            return loss_ce, loss_ce, loss_tsqa, loss_m4, zero

        N_ts, N_patches, P = chr_targets_shifted.shape
        patch_counts = (chr_ts_valid.sum(dim=1) + 1).tolist()

        # Flatten the encoder output to [sum(patch_counts), enc_dim]
        # chr_ts_valid marks patches 0..N_patches-2 as valid (patch_counts[i] = n_valid[i]+1)
        # We need all T_patches positions per series (patch_counts[i] of them).
        enc_flat_list = []
        for i, pc in enumerate(patch_counts):
            enc_flat_list.append(chr_enc_out[i, :int(pc), :])
        enc_flat = torch.cat(enc_flat_list, dim=0).float()  # [sum(patch_counts), enc_dim]

        ts_hidden = self.mse_projector(enc_flat)
        ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])
        ts_hidden_3d = _pad(
            list(torch.split(ts_hidden, [int(pc) for pc in patch_counts])),
            batch_first=True,
        )  # [N_ts, N_patches, enc_dim]

        ts_pred = self.decoder(ts_hidden_3d)   # [N_ts, N_patches, P]

        targets_flat = chr_targets_shifted.reshape(-1, P)
        pred_flat    = ts_pred.reshape(-1, P).to(targets_flat.dtype)
        valid_flat   = chr_ts_valid.reshape(-1)
        valid_count  = int(valid_flat.sum().item())

        loss_mse = (
            F.mse_loss(pred_flat[valid_flat], targets_flat[valid_flat])
            if valid_count > 0 else zero
        )

        if dynamic_weighting and ce_batch:
            mse_weight = (loss_ce.detach() / (loss_mse.detach() + 1e-8)).clamp(0.01, 100.0)
        else:
            mse_weight = alpha

        total = loss_ce + mse_weight * loss_mse
        return total, loss_ce, loss_tsqa, loss_m4, loss_mse


    def compute_joint_phase2_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
        dynamic_weighting: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint Phase-2 loss: CE through LLM + MSE through decoder. Single forward pass.

        Gradient paths (consistent with Phase 1 pretraining):

        CE path  — all samples, one LLM forward pass:
          - Chronos: CE on text tokens in answer (<ts>, </ts>, surrounding text).
            Float values between <ts>…</ts> are masked (-100) — CE never
            supervises noisy float tokens directly.
          - CoT (har/sleep/ecg): CE on full answer tokens.
          Gradient: LoRA / LLM weights only.

        MSE path — Chronos samples only, reuses the *same* LLM forward pass:
          INPUT TS patches → encoder (frozen) → projector (frozen) → LLM input
          → LLM hidden states at INPUT TS patch positions
          → decoder_projector → decoder → predicted future patches
          → MSE vs ts_future
          Gradient: decoder_projector + decoder + LoRA weights.
          Frozen modules (encoder / projector / base LLM) stay unchanged
          because their parameters have requires_grad=False.

        Args:
            batch: mixed list of {"type": "chronos"|"har"|"sleep"|"ecg", ...}
            alpha: fixed MSE weight when dynamic_weighting=False
            dynamic_weighting: if True, rescale MSE each step so gradients
                match CE in magnitude

        Returns:
            (total_loss, loss_ce, loss_mse)
        """
        device = self.device
        zero   = next(self.parameters()).new_zeros(())

        # ── Step 1: Prompt embeddings + ts_mask ──────────────────────────
        # return_targets=True gives us ts_mask (INPUT TS patch positions).
        # ts_patch_targets_shifted is not used — we target ts_future instead.
        (
            inputs_embeds_prompt,   # [B, L_prompt, H]
            attention_mask_prompt,  # [B, L_prompt]
            _token_ids,
            ts_mask,                # [B, L_prompt]  True at INPUT TS patch positions
            _ts_patch_targets,
            _ts_patch_valid,
            _ts_chan_id,
        ) = self.pad_and_apply_batch(batch, return_targets=True)

        B, L_prompt, H = inputs_embeds_prompt.size()

        # ── Step 2: Tokenize answers ──────────────────────────────────────
        answers     = [s["answer"] for s in batch]
        ans_tok     = self.tokenizer(
            answers, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        )
        ans_ids     = ans_tok.input_ids.to(device, non_blocking=True)
        ans_mask    = ans_tok.attention_mask.to(device, non_blocking=True)
        ans_emb_all = self.llm.get_input_embeddings()(ans_ids)

        bos_id = self.tokenizer.bos_token_id

        # ── Step 3: Build combined (prompt + answer) per sample ───────────
        new_embeds_list, new_masks_list, new_labels_list = [], [], []
        valid_prompt_lens: List[int] = []

        for i in range(B):
            valid_i   = int(attention_mask_prompt[i].sum().item())
            a_len     = int(ans_mask[i].sum().item())
            ans_start = (
                1 if (bos_id is not None and a_len > 0
                      and ans_ids[i, 0].item() == bos_id)
                else 0
            )
            a_eff = a_len - ans_start

            emb_i  = torch.cat([
                inputs_embeds_prompt[i, :valid_i],
                ans_emb_all[i, ans_start:a_len],
            ], dim=0)
            mask_i = torch.cat([
                attention_mask_prompt[i, :valid_i],
                torch.ones(a_eff, device=device, dtype=attention_mask_prompt.dtype),
            ], dim=0)
            lab_i  = torch.full((valid_i + a_eff,), -100, device=device, dtype=torch.long)
            lab_i[valid_i:] = ans_ids[i, ans_start:a_len]

            # Chronos: mask float tokens between <ts> and </ts> in the answer.
            # CE is supervised only on boundary tokens and surrounding text.
            if batch[i].get("type") == "chronos":
                eff_ids   = ans_ids[i, ans_start:a_len]
                open_pos  = (eff_ids == self.ts_start_token_id).nonzero(as_tuple=True)[0]
                close_pos = (eff_ids == self.ts_end_token_id).nonzero(as_tuple=True)[0]
                if len(open_pos) > 0 and len(close_pos) > 0:
                    p_open, p_close = open_pos[0].item(), close_pos[0].item()
                    if p_close > p_open + 1:
                        lab_i[valid_i + p_open + 1: valid_i + p_close] = -100

            new_embeds_list.append(emb_i)
            new_masks_list.append(mask_i)
            new_labels_list.append(lab_i)
            valid_prompt_lens.append(valid_i)

        inputs_embeds_comb  = pad_sequence(new_embeds_list, batch_first=True)
        attention_mask_comb = pad_sequence(new_masks_list,  batch_first=True)
        labels              = pad_sequence(new_labels_list, batch_first=True, padding_value=-100)

        # ── Step 4: Single LLM forward pass (CE + hidden states for MSE) ─
        chronos_present = any(
            s.get("type") == "chronos" and s.get("ts_future") is not None
            for s in batch
        )
        # ── Step 5: CE loss ───────────────────────────────────────────────
        # chunk_size=32: Gemma-3's 262K vocab makes each float32 chunk
        # [B*chunk, vocab] expensive; 32 keeps peak allocation ~4× smaller.
        #
        # Use return_row_stats=True so we can compute per-sample CE and then
        # average equally across chronos vs chatts groups. Without this,
        # ChatTS samples (hundreds of answer tokens) completely drown out
        # Chronos samples (only 3 tokens: <ts>, </ts>, EOS), making the
        # CE gradient through Chronos negligible.
        _, last_hidden, row_loss_sums, row_token_counts = self._llm_and_chunked_ce(
            inputs_embeds_comb, attention_mask_comb, labels,
            chunk_size=32, return_row_stats=True,
        )

        # Per-sample mean CE — equalises samples regardless of answer length.
        per_sample_ce = row_loss_sums / row_token_counts.float().clamp(min=1)

        # Average separately per type, then combine 50/50 so neither group
        # dominates when the batch has unequal counts of each.
        is_chronos = torch.tensor(
            [b.get("type") == "chronos" for b in batch], device=device
        )
        has_chronos = is_chronos.any()
        has_chatts  = (~is_chronos).any()
        if has_chronos and has_chatts:
            loss_ce = (
                0.5 * per_sample_ce[is_chronos].mean()
                + 0.5 * per_sample_ce[~is_chronos].mean()
            )
        else:
            loss_ce = per_sample_ce.mean()

        if not chronos_present:
            return loss_ce, loss_ce, zero

        # last_hidden already set above  # [B, L_comb, H]

        # ── Step 6: MSE — same path as Phase 1 pretraining ───────────────
        # Extract LLM hidden states at INPUT TS patch positions (from prompt
        # portion of the combined sequence). Keep the computation graph intact
        # so forecast MSE can update LoRA in phase-2 downstream.
        #
        # Optimisation: collect h_ts tensors for all Chronos samples, run
        # decoder_projector in a single batched call (one GEMM instead of N),
        # then pass each sample's slice to the decoder individually (variable
        # sequence lengths prevent full batching of the causal decoder).

        # Gather per-channel TS patch embeddings and first future patch targets.
        # pad_and_apply_batch() inserts one TS block per channel, so we must
        # decode one next patch per channel rather than collapsing all channel
        # histories from the same sample into a single target.
        chronos_targets: List[torch.Tensor] = []
        h_ts_list: List[torch.Tensor] = []
        patch_counts: List[int] = []

        rank = int(os.environ.get("RANK", "0"))
        debug = (rank == 0)

        # ── Batch composition ─────────────────────────────────────────────
        if debug:
            n_chronos = sum(1 for s in batch if s.get("type") == "chronos")
            n_chatts  = len(batch) - n_chronos
            datasets  = [s.get("_dataset", "?") for s in batch if s.get("type") == "chronos"]
            print(
                f"\n[p2d MSE] batch={len(batch)} ({n_chronos} chronos, {n_chatts} chatts) "
                f"datasets={datasets}",
                flush=True,
            )

        per_sample_debug: List[dict] = []

        for i, sample in enumerate(batch):
            if sample.get("type") != "chronos" or sample.get("ts_future") is None:
                continue
            valid_i   = valid_prompt_lens[i]
            ts_mask_i = ts_mask[i, :valid_i].bool()
            n_p_total = int(ts_mask_i.sum().item())
            if n_p_total == 0:
                if debug:
                    print(f"  [sample {i}] WARNING ts_mask=0 patch positions "
                          f"(prompt_len={valid_i}) — skipped", flush=True)
                continue
            n_channels = int(sample.get("n_channels", len(sample.get("time_series", [])) or 1))
            if n_channels <= 0:
                continue
            if n_p_total % n_channels != 0:
                raise RuntimeError(
                    "joint_phase2_downstream expected TS patch positions to split evenly "
                    f"across channels, but sample {i} has n_p_total={n_p_total} and "
                    f"n_channels={n_channels}."
                )

            h_ts = last_hidden[i, :valid_i, :][ts_mask_i].float()  # [C * N_p, H]
            n_p_per_channel = n_p_total // n_channels
            h_ts = h_ts.reshape(n_channels, n_p_per_channel, H)

            gt = sample["ts_future"].to(device=device, dtype=h_ts.dtype)
            if gt.dim() != 2 or gt.size(0) != n_channels:
                raise RuntimeError(
                    "joint_phase2_downstream expected ts_future to have shape "
                    f"[n_channels, pred_len], but sample {i} has {tuple(gt.shape)} "
                    f"for n_channels={n_channels}."
                )
            gt_next = gt[:, :self.patch_size]  # [C, patch_size]

            if debug:
                h_norms = h_ts.norm(dim=-1)  # [C, N_p]
                per_sample_debug.append({
                    "i": i, "ds": sample.get("_dataset", "?"),
                    "hist": sample.get("_history_len", "?"),
                    "pred": sample.get("_pred_len", "?"),
                    "C": n_channels, "Np": n_p_per_channel, "L": valid_i,
                    "h_mean": h_ts.mean().item(), "h_std": h_ts.std().item(),
                    "hn_mean": h_norms.mean().item(),
                    "gt_mean": gt_next.mean().item(), "gt_std": gt_next.std().item(),
                    "gt_max": gt_next.abs().max().item(),
                })

            for c in range(n_channels):
                h_ts_list.append(h_ts[c])
                patch_counts.append(n_p_per_channel)
                chronos_targets.append(gt_next[c])

        if debug and per_sample_debug:
            print(f"  {'i':>3} {'dataset':>18} {'hist':>5} {'pred':>5} "
                  f"{'C':>3} {'Np':>3} {'L':>5} "
                  f"{'h_mean':>7} {'h_std':>7} {'|h|':>7} "
                  f"{'gt_mean':>7} {'gt_std':>7} {'gt_max':>7}", flush=True)
            for d in per_sample_debug[:8]:
                print(f"  {d['i']:>3} {str(d['ds']):>18} {str(d['hist']):>5} {str(d['pred']):>5} "
                      f"{d['C']:>3} {d['Np']:>3} {d['L']:>5} "
                      f"{d['h_mean']:>7.3f} {d['h_std']:>7.3f} {d['hn_mean']:>7.3f} "
                      f"{d['gt_mean']:>7.3f} {d['gt_std']:>7.3f} {d['gt_max']:>7.3f}",
                      flush=True)
            if len(per_sample_debug) > 8:
                print(f"  ... ({len(per_sample_debug)} total)", flush=True)

        if not chronos_targets:
            if debug:
                print("[p2d MSE] no chronos targets — CE-only loss", flush=True)
            return loss_ce, loss_ce, zero

        # Single batched projection across all Chronos samples.
        h_cat     = torch.cat(h_ts_list, dim=0)               # [sum(N_p), H]
        h_enc_cat = self.decoder_projector(h_cat)              # [sum(N_p), enc_dim]
        h_enc_cat = F.layer_norm(h_enc_cat, [h_enc_cat.size(-1)])

        if debug:
            print(f"  projector out (post LN): mean={h_enc_cat.mean().item():.3f} "
                  f"std={h_enc_cat.std().item():.3f}  (expect ~N(0,1))", flush=True)

        # Batched decoder call: right-pad each sample to N_p_max and run once.
        N_c     = len(patch_counts)
        N_p_max = max(patch_counts)
        h_enc_batch = pad_sequence(
            list(torch.split(h_enc_cat, patch_counts, dim=0)), batch_first=True
        )  # [N_c, N_p_max, enc_dim]  (right-padded with zeros)

        pred_batch = self.decoder(h_enc_batch)                  # [N_c, N_p_max, patch_size]

        # Index the last real position for each sample and compute MSE in one shot.
        gt_stacked = torch.stack(chronos_targets).float()                  # [N_c, patch_size]
        last_idx   = torch.tensor([n_p - 1 for n_p in patch_counts], device=device)
        pred_last  = pred_batch[torch.arange(N_c, device=device), last_idx].float() # [N_c, patch_size]
        loss_mse   = F.mse_loss(pred_last, gt_stacked)

        if not torch.isfinite(loss_mse):
            loss_mse = zero

        if debug:
            per_ch_mse = ((pred_last - gt_stacked) ** 2).mean(dim=1)  # [N_c]
            worst3 = per_ch_mse.topk(min(3, N_c)).indices
            print(
                f"  decoder: pred mean={pred_last.mean().item():.3f} "
                f"std={pred_last.std().item():.3f} abs_max={pred_last.abs().max().item():.3f}\n"
                f"  target:  pred mean={gt_stacked.mean().item():.3f} "
                f"std={gt_stacked.std().item():.3f} abs_max={gt_stacked.abs().max().item():.3f}\n"
                f"  N_c={N_c} N_p_max={N_p_max} loss_mse={loss_mse.item():.6f}",
                flush=True,
            )
            print(f"  worst 3 channels by MSE:", flush=True)
            for idx in worst3:
                p = pred_last[idx].detach()
                t = gt_stacked[idx].detach()
                print(f"    ch[{idx.item()}] mse={per_ch_mse[idx].item():.4f} | "
                      f"pred[:5]={[round(x,3) for x in p[:5].tolist()]} | "
                      f"target[:5]={[round(x,3) for x in t[:5].tolist()]}",
                      flush=True)

        print(
            f"[rank{rank}] phase2 loss_mse={loss_mse.item():.6f} "
            f"loss_ce={loss_ce.item():.6f} "
            f"pred_shape={tuple(pred_last.shape)} target_shape={tuple(gt_stacked.shape)} "
            f"pred_abs_max={pred_last.abs().max().item():.4e} "
            f"target_abs_max={gt_stacked.abs().max().item():.4e}",
            flush=True,
        )
        if dynamic_weighting:
            w = (loss_ce.detach() / (loss_mse.detach() + 1e-8)).clamp(0.01, 100.0)
        else:
            w = alpha

        total = loss_ce + w * loss_mse
        return total, loss_ce, loss_mse


    def compute_pretrain_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: Optional[float] = None,
        mse_only: bool = False,
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

        For forecasting-only foundational pretraining (``mse_only=True``),
        the numeric path mirrors standard autoregressive next-token training
        while still exercising the same encoder-side entry/exit adapters that
        the joint LLM workflow relies on:

            [patch_0, patch_1, ..., patch_n, true_next_patch]
              -> encoder -> projector -> decoder_projector -> decoder
              -> one-step shifted MSE over valid positions

        So position t predicts position t+1 across the whole patch sequence, and
        the last observed history patch predicts the first unseen future patch.

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

        has_ts = (ts_patch_targets_shifted is not None) and (ts_patch_targets_shifted.numel() > 0)

        # Loss 1: NTP on text tokens + get hidden states for patch prediction.
        # MSE-only stages skip the LM head entirely and reuse only the backbone
        # hidden states for decoder supervision.
        if mse_only:
            ntp_loss = inputs_embeds.new_zeros(())
            last_hidden = None
        else:
            # _llm_and_chunked_ce avoids materializing the full [B, L, vocab_size]
            # tensor in float32 all at once.
            ntp_loss, last_hidden = self._llm_and_chunked_ce(
                inputs_embeds, attention_mask, token_ids
            )

        # Loss 2: next-patch prediction via the causal decoder.
        #
        # This is the numeric analogue of next-token training:
        # position t attends to [0..t] and predicts the patch at t+1.
        #
        # Gradient path:
        #   mse_only=False: encoder -> projector -> LLM backbone -> decoder_projector -> decoder -> shifted MSE
        #   mse_only=True:  [history, true next patch] -> encoder -> projector -> decoder_projector -> decoder -> shifted MSE
        safe_zero = inputs_embeds.sum() * 0.0
        mse_loss = safe_zero
        next_patch_mse = safe_zero.detach()
        final_patch_mse = safe_zero.detach()

        pred_std = None
        N_ts_total = N_patches = P = 0
        valid_count = 0
        valid_ratio = 0.0
        ts_pred = None  # (N_ts_total, N_patches, P) — kept for optional plotting

        if has_ts:
            N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
            patch_counts = (ts_patch_valid.sum(dim=1) + 1).tolist()
            if mse_only:
                if self._last_ts_raw is None or self._last_ts_chan_id is None:
                    raise RuntimeError(
                        "Raw encoder inputs are unavailable for mse_only pretraining. "
                        "pad_and_apply_batch(return_targets=True) should populate "
                        "the cached foundational forecasting tensors before "
                        "compute_pretrain_loss()."
                    )

                full_series_list = []
                full_patch_counts = []
                full_chan_ids = []
                future_targets = []
                final_pred_indices = []

                chronos_ts_idx = 0
                for sample in batch:
                    if sample.get("type") != "chronos":
                        continue
                    fut = sample.get("ts_future")
                    if fut is None:
                        chronos_ts_idx += 1
                        continue

                    n_hist = int(patch_counts[chronos_ts_idx])
                    hist_steps = n_hist * self.patch_size
                    hist_raw = self._last_ts_raw[chronos_ts_idx, :hist_steps]
                    gt_fut = torch.as_tensor(
                        fut,
                        device=device,
                        dtype=hist_raw.dtype,
                    )[:, :self.patch_size]
                    full_series_list.append(torch.cat([hist_raw, gt_fut], dim=0))
                    full_patch_counts.append(n_hist + 1)
                    full_chan_ids.append(self._last_ts_chan_id[chronos_ts_idx])
                    future_targets.append(gt_fut)
                    final_pred_indices.append(n_hist - 1)
                    chronos_ts_idx += 1

                if full_series_list:
                    full_raw = pad_sequence(full_series_list, batch_first=True).to(device)
                    max_full_patches = full_raw.size(1) // self.patch_size
                    full_patch_mask = (
                        torch.arange(max_full_patches, device=device).unsqueeze(0)
                        < torch.tensor(full_patch_counts, device=device).unsqueeze(1)
                    )
                    full_hidden_3d = self._build_foundational_history_hidden_from_raw(
                        full_raw,
                        chan_id=torch.stack(full_chan_ids).to(device=device, dtype=torch.long),
                        attention_mask=full_patch_mask,
                    )
                    ts_pred = self.decoder(full_hidden_3d)

                    full_targets = full_raw.reshape(len(full_series_list), -1, self.patch_size)
                    N_ts_total = full_targets.size(0)
                    N_patches = full_targets.size(1)
                    P = full_targets.size(2)
                    shifted_targets = torch.zeros_like(full_targets)
                    shifted_targets[:, :-1, :] = full_targets[:, 1:, :]
                    valid_next = (
                        torch.arange(full_targets.size(1), device=device).unsqueeze(0)
                        < (torch.tensor(full_patch_counts, device=device) - 1).unsqueeze(1)
                    )

                    targets_flat = shifted_targets.reshape(-1, P).float()
                    pred_flat = ts_pred.reshape(-1, P).float()
                    valid_flat = valid_next.reshape(-1)
                    valid_count = int(valid_flat.sum().item())
                    valid_ratio = float(valid_count / max(1, valid_flat.numel()))

                    if valid_count > 0:
                        valid_pred = pred_flat[valid_flat]
                        valid_target = targets_flat[valid_flat]
                        next_patch_mse = F.mse_loss(valid_pred, valid_target)
                        if torch.isfinite(next_patch_mse):
                            mse_loss = next_patch_mse
                        else:
                            if _is_rank0():
                                print(
                                    "[phase1_chronos_mse] Non-finite next_patch_mse detected; "
                                    f"skipping batch. pred_abs_max={valid_pred.abs().max().item():.4e} "
                                    f"target_abs_max={valid_target.abs().max().item():.4e}"
                                )
                            next_patch_mse = safe_zero.detach()

                    if final_pred_indices:
                        final_pred = ts_pred[
                            torch.arange(len(final_pred_indices), device=device),
                            torch.tensor(final_pred_indices, device=device),
                        ].float()
                        final_target = torch.stack(future_targets).to(device=device).float()
                        final_patch_mse = F.mse_loss(final_pred, final_target)
                        if not torch.isfinite(final_patch_mse):
                            if _is_rank0():
                                print(
                                    "[phase1_chronos_mse] Non-finite final_patch_mse detected. "
                                    f"pred_abs_max={final_pred.abs().max().item():.4e} "
                                    f"target_abs_max={final_target.abs().max().item():.4e}"
                                )
                            final_patch_mse = safe_zero.detach()

                    with torch.no_grad():
                        if valid_count > 0:
                            pred_std = float(pred_flat[valid_flat].std().item())
                else:
                    ts_pred = None
                    valid_count = 0
                    valid_ratio = 0.0
                    mse_loss = inputs_embeds.sum() * 0.0
            else:
                ts_hidden = last_hidden[ts_mask]

                expected_flat = int(sum(patch_counts))
                actual_flat = ts_hidden.size(0)
                assert actual_flat == expected_flat, (
                    f"ts_mask extracted {actual_flat} hidden vectors but expected "
                    f"{expected_flat} real patch vectors."
                )

                # Project LLM hidden → decoder input dim, normalise for stability
                ts_hidden = self.decoder_projector(ts_hidden.float())
                ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

                # Vectorised scatter: split the flat tensor by per-series counts and
                # right-pad to N_patches with zeros — identical to the original loop.
                # max(patch_counts) == N_patches always (verified by the T_max padding
                # in pad_and_apply_batch), so pad_sequence produces exactly
                # [N_ts_total, N_patches, enc_dim].
                ts_hidden_3d = pad_sequence(
                    list(torch.split(ts_hidden, patch_counts)), batch_first=True
                )  # [N_ts_total, N_patches, enc_dim]

            if not mse_only:
                # Causal decoder: self-attends over the patch sequence and
                # predicts the next patch at every valid position in parallel.
                ts_pred = self.decoder(ts_hidden_3d)    # (N_ts_total, N_patches, P)
                targets_flat = ts_patch_targets_shifted.reshape(-1, P)
                pred_flat = ts_pred.reshape(-1, P).to(targets_flat.dtype)
                valid_flat = ts_patch_valid.reshape(-1)

                valid_count = int(valid_flat.sum().item())
                valid_ratio = float(valid_count / max(1, valid_flat.numel()))

                if valid_count > 0:
                    valid_pred = pred_flat[valid_flat].float()
                    valid_target = targets_flat[valid_flat].float()
                    # Mask out zero-padded timesteps within partial last-real patches.
                    # ts_patch_target_valid[i,j,p] is True only when global timestep
                    # (j+1)*P+p < real_len[i], so trailing zeros never contribute.
                    # Drop zero-padded timesteps from partial last-real patches before
                    # computing loss — index directly so F.mse_loss sees only real values.
                    ts_mask_flat = self._last_ts_patch_target_valid.reshape(-1, P)[valid_flat].reshape(-1)
                    next_patch_mse = F.mse_loss(valid_pred.reshape(-1)[ts_mask_flat],
                                                valid_target.reshape(-1)[ts_mask_flat])
                    if torch.isfinite(next_patch_mse):
                        mse_loss = next_patch_mse
                    else:
                        if _is_rank0():
                            print(
                                "[forecasting] Non-finite next_patch_mse detected; skipping batch. "
                                f"pred_abs_max={valid_pred.abs().max().item():.4e} "
                                f"target_abs_max={valid_target.abs().max().item():.4e}"
                            )
                        next_patch_mse = safe_zero.detach()
                    with torch.no_grad():
                        pred_std = float(valid_pred.std().item())
                else:
                    mse_loss = safe_zero

            if pred_std is not None and pred_std < 1e-3:
                print(
                    f"⚠️  [collapse guard] decoder output std={pred_std:.2e}. "
                    f"Decoder may be ignoring decoder inputs."
                )

            if (
                not mse_only
                and valid_count > 0
                and self.counter % 5000 == 0
                and _is_rank0()
            ):
                patch_mse_sum = torch.zeros(N_patches, device=device)
                patch_mse_count = torch.zeros(N_patches, device=device)
                for s in range(N_ts_total):
                    valid_pos = ts_patch_valid[s]
                    n_valid = int(valid_pos.sum().item())
                    if n_valid == 0:
                        continue
                    s_pred_s = ts_pred[s, valid_pos].float()
                    s_target_s = ts_patch_targets_shifted[s, valid_pos].float()
                    tv = self._last_ts_patch_target_valid[s, valid_pos]  # [n_valid, P] bool
                    valid_idx = torch.nonzero(valid_pos, as_tuple=False).squeeze(-1)
                    sq_err = (s_pred_s - s_target_s).pow(2)
                    tv_f = tv.float()
                    per_patch_count = tv_f.sum(dim=1)
                    keep = per_patch_count > 0
                    if not keep.any():
                        continue
                    s_mse = (sq_err * tv_f).sum(dim=1)[keep] / per_patch_count[keep]
                    patch_mse_sum[valid_idx[keep]] += s_mse
                    patch_mse_count[valid_idx[keep]] += 1

                valid_positions = int((patch_mse_count > 0).sum().item())
                avg_patch_mse = patch_mse_sum / patch_mse_count.clamp(min=1)

                def _batch_patch_str(indices):
                    parts = []
                    for i in indices:
                        if 0 <= i < N_patches and patch_mse_count[i] > 0:
                            parts.append(
                                f"  patch[{i:3d}]: mse={avg_patch_mse[i].item():.4f}"
                                f" (n={int(patch_mse_count[i].item())})"
                            )
                    return "\n".join(parts)

                mid = valid_positions // 2
                first_idx = list(range(min(5, valid_positions)))
                mid_idx = list(range(max(0, mid - 2), min(valid_positions, mid + 3)))
                last_idx = list(range(max(0, valid_positions - 5), valid_positions))

                print(
                    f"\n[step {self.counter}] per-patch MSE averaged over {N_ts_total} series "
                    f"({valid_positions} valid patch positions, P={P}):\n"
                    f"  -- first 5 --\n{_batch_patch_str(first_idx)}\n"
                    f"  -- middle 5 --\n{_batch_patch_str(mid_idx)}\n"
                    f"  -- last 5 --\n{_batch_patch_str(last_idx)}\n"
                    f"  next-patch mse={mse_loss.item():.4f}  pred_std={pred_std:.4f}"
                )

            if self.counter % 5000 == 0 and _is_rank0():
                print(
                    f"[step {self.counter}] foundational next_patch_mse={next_patch_mse.item():.4f} "
                    f"final_patch_mse={final_patch_mse.item():.4f}"
                )

        recon_loss = mse_loss  # alias kept so the rest of the function is unchanged
        self._last_pretrain_next_patch_mse = next_patch_mse.detach()
        self._last_pretrain_final_patch_mse = final_patch_mse.detach()


        total_loss = ntp_loss + alpha * recon_loss
        # ----------------------------- debug prints -----------------------------
        do_print = debug and (self.counter % max(1, debug_every) == 0)
        if do_print:
            # Materialise logits only for the debug path (not kept for training).
            with torch.no_grad():
                _debug_logits = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
            shift_logits = _debug_logits[:, :-1, :].contiguous()    # [B, L-1, V]
            shift_labels = token_ids[:, 1:].contiguous()     # [B, L-1]
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
            print(f"  logits: {tuple(_debug_logits.shape)} (vocab={V})")
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

            print("\n[NEXT-PATCH PREDICTION DIAGNOSTICS]")
            if not has_ts:
                print("  (no TS targets in this batch)")
            else:
                print(f"  targets: [N_ts_total={N_ts_total}, N_patches={N_patches}, P={P}]")
                print(f"  valid positions: count={valid_count} | ratio={valid_ratio:.3f}")
                if pred_std is not None:
                    print(f"  decoder output std over valid positions: {pred_std:.3e}")

        # ----------------------------- TS plot -----------------------------
        do_plot = (
            plot_ts
            and has_ts
            and (ts_pred is not None)
            and (self.counter % max(1, plot_every) == 0)
            and _is_rank0()
        )

        if do_plot:
            os.makedirs(plot_dir, exist_ok=True)
            with torch.no_grad():
                n_plot = min(int(plot_max_series), int(N_ts_total))
                for s in range(n_plot):
                    gt_p = ts_patch_targets_shifted[s]  # [N_patches, P]
                    pr_p = ts_pred[s]                   # [N_patches, P]
                    v_p = ts_patch_valid[s]             # [N_patches]

                    idx = v_p.nonzero(as_tuple=True)[0]
                    gt_p = gt_p[idx]
                    pr_p = pr_p[idx]

                    gt_1d = _stitch_patches(gt_p).detach().float().cpu().numpy()
                    pr_1d = _stitch_patches(pr_p).detach().float().cpu().numpy()

                    fig = plt.figure()
                    plt.plot(gt_1d, label="ground_truth_next_patch")
                    plt.plot(pr_1d, label="pred_next_patch", alpha=0.35)
                    plt.title(f"TS next-patch prediction (series={s}) step={self.counter}")
                    plt.legend()

                    out_path = os.path.join(plot_dir, f"ts_pred_step{self.counter:07d}_series{s}.png")
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


    def compute_chronos_seq2seq_loss(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Seq2seq teacher-forced Chronos loss has been removed. "
            "Use the direct encoder/projector/decoder next-patch workflow."
        )

        """
        Autoregressive seq2seq MSE loss for Chronos forecasting (Phase-1).

        Each sample has the structure:
            [text: "Given T_in steps:"] [input_patch_1..N]
            [text: "Predict next T_out steps:"] [target_patch_1..M]

        The LLM processes the full combined sequence with causal attention, so
        the hidden state at each history patch position carries context from
        the text and all preceding observed patches. Forecasting then switches
        to the numeric decoder's own patch-feedback path:

          history hidden states
            → decoder_projector
            → causal decoder
            → teacher-forced target patch feedback
            → predicted forecast patches

        Hidden states are NOT detached, so forecast MSE gradients still flow
        back through the LLM to the encoder and projector on the history side.
        Future teacher forcing uses raw target patches directly, which aligns
        training with autoregressive decoding at inference time.

        Transfusion-inspired training:
        - the LLM sees ``[text | history_patches | text | target_patches]``
          in one causal pass
        - optimization is driven by next-patch MSE over the forecast horizon
        - text CE on the scaffolding prompts is computed only as an auxiliary
          monitoring metric and does not contribute gradients

        Returns (total_loss, aux_text_ce_loss, next_patch_mse_loss).
        """
        zero = next(self.parameters()).new_zeros(())

        (
            inputs_embeds,
            attention_mask,
            token_ids,          # real token IDs for text, -100 at TS positions
            ts_mask,
            _ts_patch_targets_shifted,
            _ts_patch_valid,
            _,
        ) = self.pad_and_apply_batch(batch, return_targets=True)

        if ts_mask is None or not ts_mask.any():
            return zero, zero, zero

        # Shared forward: obtain the final hidden states for seq2seq MSE while
        # also measuring auxiliary scaffold-text CE. Hidden states are NOT
        # detached, so MSE gradients flow back through the LLM to the encoder
        # and projector.
        ntp_loss, last_hidden = self._llm_and_chunked_ce(inputs_embeds, attention_mask, token_ids)
        ntp_loss = ntp_loss.detach()

        pred_chunks: List[torch.Tensor] = []
        target_chunks: List[torch.Tensor] = []

        for i, sample in enumerate(batch):
            ts_lengths = sample.get("time_series_lengths")
            ts_values = sample.get("time_series")

            if ts_lengths is None or ts_values is None:
                continue
            if len(ts_lengths) != 2:
                raise ValueError(
                    "phase1_chronos_seq2seq expects exactly 2 time-series blocks "
                    f"(history, target), got {len(ts_lengths)} for sample {i}."
                )

            in_steps = int(ts_lengths[0])
            tgt_steps = int(ts_lengths[1])
            if in_steps <= 0 or tgt_steps <= 0:
                continue
            if in_steps % self.patch_size != 0 or tgt_steps % self.patch_size != 0:
                raise ValueError(
                    "Seq2seq sample lengths must be divisible by patch_size: "
                    f"sample {i} has in_steps={in_steps}, tgt_steps={tgt_steps}, "
                    f"patch_size={self.patch_size}."
                )

            n_in = in_steps // self.patch_size
            n_out = tgt_steps // self.patch_size

            sample_hidden = last_hidden[i][ts_mask[i]]
            expected_patches = n_in + n_out
            if sample_hidden.size(0) != expected_patches:
                raise ValueError(
                    "Seq2seq hidden-state extraction mismatch: "
                    f"sample {i} expected {expected_patches} TS patch states "
                    f"but found {sample_hidden.size(0)}."
                )

            history_hidden = sample_hidden[:n_in]
            history_hidden = self.decoder_projector(history_hidden.float())
            history_hidden = F.layer_norm(history_hidden, [history_hidden.size(-1)])

            target_series = ts_values[1, :tgt_steps].to(
                device=history_hidden.device, dtype=history_hidden.dtype
            )
            target_patches = target_series.reshape(n_out, self.patch_size)
            pred_target = self.decoder.forecast_teacher_forced(
                history_hidden.unsqueeze(0),
                target_patches.unsqueeze(0),
            ).squeeze(0)
            if pred_target.size(0) != n_out:
                raise ValueError(
                    "Seq2seq forecast supervision window mismatch: "
                    f"sample {i} expected {n_out} predictions, got {pred_target.size(0)}."
                )

            pred_chunks.append(pred_target)
            target_chunks.append(target_patches)

        if not pred_chunks:
            return ntp_loss, ntp_loss, zero

        pred_flat = torch.cat(pred_chunks, dim=0)
        target_flat = torch.cat(target_chunks, dim=0)
        mse_loss = F.mse_loss(pred_flat, target_flat)

        # Optimize only the forecast MSE. alpha remains as a global scale knob
        # for compatibility with the other phase-1 reconstruction objectives.
        alpha = getattr(self, "recon_alpha", 1.0)
        total_loss = alpha * mse_loss
        return total_loss, ntp_loss, mse_loss

    def compute_chronos_seq2seq_foundational_loss(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Seq2seq teacher-forced foundational loss has been removed. "
            "Use the direct encoder -> projector -> decoder_projector -> decoder "
            "next-patch workflow."
        )

    def compute_chronos_seq2seq_enc_mse_loss(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Seq2seq teacher-forced encoder/MSE loss has been removed. "
            "Use the direct encoder -> projector -> decoder_projector -> decoder "
            "next-patch workflow."
        )

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
        if pred_len % patch_size != 0:
            raise ValueError(
                f"pred_len ({pred_len}) must be divisible by patch_size ({patch_size})."
            )
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
        loss_ce, last_hidden_state = self._llm_and_chunked_ce(inputs_embeds, attention_mask, labels)
        last_hidden = last_hidden_state if need_hidden else None

        zero_mse = loss_ce.new_zeros(1).squeeze()

        if not ts_hidden_positions:
            return loss_ce, loss_ce, zero_mse

        # ---- MSE path (forecasting samples only) — fully batched ----
        items = list(ts_hidden_positions.items())                    # [(sample_idx, ts_pos)]

        # Stack all hidden vectors into one batch, project once, then decode
        # the full forecast horizon in patch space.
        h_ts_batch = torch.stack(
            [last_hidden[i, pos, :] for i, pos in items]
        ).float()                                                    # [N, H]
        h_enc_seed = self.decoder_projector(h_ts_batch)             # [N, enc_dim]
        h_enc_seed = F.layer_norm(h_enc_seed, [h_enc_seed.size(-1)])
        h_enc_seq  = h_enc_seed.unsqueeze(1).expand(-1, n_future_patches, -1)
        pred_patches = self.decoder(h_enc_seq)                      # [N, n_future_patches, patch_size]
        pred_ph2 = pred_patches.reshape(len(items), n_future_patches * patch_size)

        gt_ph2_list = []
        for i, _ in items:
            future = batch[i]["ts_future"].to(self.device, dtype=pred_ph2.dtype)
            if future.numel() < pred_len:
                raise ValueError(
                    f"ts_future for sample {i} is shorter than pred_len: "
                    f"{future.numel()} < {pred_len}"
                )
            gt_ph2_list.append(future[:pred_len])
        gt_ph2 = torch.stack(gt_ph2_list)                           # [N, pred_len]
        loss_mse = F.mse_loss(pred_ph2, gt_ph2)

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
            "mse_projector_state": self.mse_projector.state_dict(),
        }

        # Add LoRA state to checkpoint
        self.save_lora_state_to_checkpoint(checkpoint)

        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        decoder_state = ckpt["decoder_state"]
        decoder_load = self.decoder.load_state_dict(decoder_state, strict=False)
        allowed_legacy_keys = {
            "feedback_proj.0.weight",
            "feedback_proj.0.bias",
            "feedback_proj.2.weight",
            "feedback_proj.2.bias",
        }
        missing_keys = set(decoder_load.missing_keys)
        unexpected_keys = set(decoder_load.unexpected_keys)
        disallowed_missing = missing_keys - allowed_legacy_keys
        disallowed_unexpected = unexpected_keys - allowed_legacy_keys
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "Decoder checkpoint is incompatible with the current model. "
                f"Missing keys: {sorted(missing_keys)} | "
                f"Unexpected keys: {sorted(unexpected_keys)}"
            )
        if missing_keys or unexpected_keys:
            print(
                "⚠️  Loaded a legacy decoder checkpoint with deprecated "
                "teacher-forcing feedback weights. These keys are ignored."
            )
        self.decoder_projector.load_state_dict(ckpt["decoder_projector_state"])
        if "mse_projector_state" in ckpt:
            self.mse_projector.load_state_dict(ckpt["mse_projector_state"])

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
        text_outputs, _ = self.generate(batch, max_new_tokens=max_new_tokens)
        return text_outputs[0]
