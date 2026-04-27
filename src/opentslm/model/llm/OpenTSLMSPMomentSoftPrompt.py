# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch.optim import AdamW

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# MOMENT
try:
    from momentfm import MOMENTPipeline
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False

from .TimeSeriesLLM import TimeSeriesLLM
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


class PatchwiseDecoderHead(nn.Module):
    """
    Patch-wise decoder head (literature-style):
    one small head per patch token: h -> patch_len values.
    """
    def __init__(self, d_model: int, patch_len: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, patch_len)
        # Stable init (similar spirit to many TS heads)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_tokens, d_model] or [B, N_tokens, d_model]
        return self.proj(self.dropout(x))


class OpenTSLMSPMomentSoftPrompt(TimeSeriesLLM):
    """
    Soft-prompt TS->LLM bridge using a frozen pretrained MOMENT encoder.

    Pretraining objective:
      1) NTP on text tokens (cross-entropy)
      2) Next-patch prediction on TS tokens (MSE), patch-wise head

    Trains:
      - encoder_projector (MOMENT d_model -> LLM hidden)
      - decoder_projector (LLM hidden -> MOMENT d_model)
      - decoder_head      (MOMENT d_model -> patch_len)
      - optionally LoRA adapters on LLM

    Frozen:
      - MOMENT encoder
      - LLM backbone (unless LoRA)
    """

    def __init__(
        self,
        args,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        moment_id: str = "AutonLab/MOMENT-1-base",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.args = args
        self.device = device

        if not MOMENT_AVAILABLE:
            raise RuntimeError(
                "momentfm is required for this training path. Install with: pip install momentfm"
            )

        # -------------------------
        # Tokenizer + LLM (frozen)
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add TS boundary tokens (optional, but keep consistent with your codebase)
        special_tokens = {"additional_special_tokens": ["<ts>", "</ts>"]}
        self.tokenizer.add_special_tokens(special_tokens)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.ts_start_token_id = self.tokenizer.convert_tokens_to_ids("<ts>")
        self.ts_end_token_id = self.tokenizer.convert_tokens_to_ids("</ts>")

        # Freeze LLM backbone by default
        for p in self.llm.parameters():
            p.requires_grad = False

        # -------------------------
        # Pretrained MOMENT (frozen)
        # -------------------------
        # IMPORTANT:
        # We want per-patch embeddings, so we use task_name="embedding" and reduction="none".
        self.moment = MOMENTPipeline.from_pretrained(
            moment_id,
            model_kwargs={"task_name": "embedding"},
        ).to(device)
        self.moment.init()
        self.moment.eval()
        for p in self.moment.parameters():
            p.requires_grad = False

        # Read MOMENT config
        self.moment_d_model = int(self.moment.config.d_model)
        self.moment_patch_len = int(self.moment.patch_len)
        # Many MOMENT checkpoints assume a fixed seq_len in config.
        # We will pad/truncate TS segments to this length for stable behavior.
        self.moment_seq_len = int(getattr(self.moment.config, "seq_len", 512))

        # -------------------------
        # Trainable bridges + head
        # -------------------------
        llm_dim = int(self.llm.config.hidden_size)

        # MOMENT -> LLM
        self.encoder_projector = nn.Linear(self.moment_d_model, llm_dim).to(device)

        # LLM -> MOMENT
        self.decoder_projector = nn.Linear(llm_dim, self.moment_d_model).to(device)

        # Patch-wise numeric head (per patch token)
        self.decoder_head = PatchwiseDecoderHead(
            d_model=self.moment_d_model,
            patch_len=self.moment_patch_len,
            dropout=float(getattr(args, "ts_head_dropout", 0.0)),
        ).to(device)

        # LoRA state
        self.lora_enabled = False
        self.original_llm = None

        # Debug counter
        self.counter = 0

        print("[OpenTSLMSPMomentSoftPrompt]")
        print(f"  LLM: {llm_id} (hidden={llm_dim}) frozen=True (unless LoRA)")
        print(f"  MOMENT: {moment_id} (d_model={self.moment_d_model}, patch_len={self.moment_patch_len}, seq_len={self.moment_seq_len}) frozen=True")
        print("  Trainable: encoder_projector, decoder_projector, decoder_head")

    # -------------------------
    # LoRA (optional)
    # -------------------------
    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is required for LoRA. Install with: pip install peft")
        if self.lora_enabled:
            raise RuntimeError("LoRA already enabled")

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        self.original_llm = self.llm
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.lora_enabled = True

        # Freeze non-LoRA params, train LoRA
        for name, p in self.llm.named_parameters():
            p.requires_grad = ("lora_" in name)

        print("✅ LoRA enabled (LLM backbone remains frozen; LoRA trainable).")

    def get_lora_parameters(self):
        if not self.lora_enabled:
            return []
        return [p for n, p in self.llm.named_parameters() if p.requires_grad and "lora_" in n]

    # -------------------------
    # Stage config
    # -------------------------
    def set_stage_moment_softprompt_pretrain(self):
        # Frozen: MOMENT, LLM (except LoRA)
        for p in self.moment.parameters():
            p.requires_grad = False
        for name, p in self.llm.named_parameters():
            if self.lora_enabled and ("lora_" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Train: bridges + head
        for p in self.encoder_projector.parameters():
            p.requires_grad = True
        for p in self.decoder_projector.parameters():
            p.requires_grad = True
        for p in self.decoder_head.parameters():
            p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print("Stage: MOMENT-softprompt pretrain")
        print("  ❄️  MOMENT encoder frozen")
        print("  ❄️  LLM frozen (LoRA trainable if enabled)")
        print("  🔥 encoder_projector trainable")
        print("  🔥 decoder_projector trainable")
        print("  🔥 decoder_head trainable (patch-wise)")
        print(f"  trainable params: {trainable:,}")
        print("=" * 70)

    def get_optimizer_moment_softprompt_pretrain(
        self,
        lr_bridge: float = 2e-4,
        lr_head: float = 2e-4,
        lr_lora: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        param_groups = [
            {"params": list(self.encoder_projector.parameters()), "lr": lr_bridge, "weight_decay": weight_decay},
            {"params": list(self.decoder_projector.parameters()), "lr": lr_bridge, "weight_decay": weight_decay},
            {"params": list(self.decoder_head.parameters()),       "lr": lr_head,   "weight_decay": weight_decay},
        ]
        if self.lora_enabled:
            lora_params = self.get_lora_parameters()
            if lora_params:
                param_groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": 0.0})

        return AdamW(param_groups)

    # -------------------------
    # Internal: TS shape handling + MOMENT embed + targets
    # -------------------------
    @staticmethod
    def _to_T_C(ts: torch.Tensor) -> torch.Tensor:
        # Accept [T], [T,C], or [C,T] (heuristic)
        if ts.dim() == 1:
            return ts.unsqueeze(-1)  # [T,1]
        if ts.dim() != 2:
            raise ValueError(f"Expected ts dim 1 or 2, got {tuple(ts.shape)}")

        T0, T1 = ts.shape
        # Heuristic: if first dim looks like channels and second looks like time, transpose
        if T0 <= 32 and T1 >= 64:
            # likely [C,T]
            return ts.transpose(0, 1).contiguous()  # [T,C]
        return ts  # [T,C]

    def _pad_trunc_to_moment(
        self, ts_TC: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          x_enc:      [C, moment_seq_len]
          input_mask: [moment_seq_len]  (1 for real, 0 for pad)
        """
        ts_TC = ts_TC.to(self.device)
        T, C = ts_TC.shape
        L = self.moment_seq_len

        # truncate
        if T >= L:
            x = ts_TC[:L, :]  # [L,C]
            mask = torch.ones(L, device=self.device, dtype=torch.long)
        else:
            pad = torch.zeros((L - T, C), device=self.device, dtype=ts_TC.dtype)
            x = torch.cat([ts_TC, pad], dim=0)  # [L,C]
            mask = torch.zeros(L, device=self.device, dtype=torch.long)
            mask[:T] = 1

        # MOMENT expects [B,C,T], so transpose to [C,L]
        x_enc = x.transpose(0, 1).contiguous()  # [C,L]
        return x_enc, mask  # [C,L], [L]

    def _moment_embed_and_targets_grouped(
        self, ts_list: List[torch.Tensor]
    ):
        """
        For each TS segment:
          - get per-patch embeddings from frozen MOMENT (reduction="none")
          - flatten into patch-major token sequence: [n_patches*C, d_model]
          - build next-patch targets aligned to the prefix tokens: [(n_patches-1)*C, patch_len]
          - build valid mask: [(n_patches-1)*C]
        Returns lists aligned with ts_list order.
        """
        if len(ts_list) == 0:
            return [], [], [], []

        # Group by channel count so MOMENT sees constant n_channels inside a batch
        groups: Dict[int, List[int]] = {}
        packed_x: Dict[int, List[torch.Tensor]] = {}
        packed_m: Dict[int, List[torch.Tensor]] = {}

        for i, ts in enumerate(ts_list):
            ts_TC = self._to_T_C(ts)
            x_enc_CL, mask_L = self._pad_trunc_to_moment(ts_TC)  # [C,L], [L]
            C = x_enc_CL.shape[0]
            groups.setdefault(C, []).append(i)
            packed_x.setdefault(C, []).append(x_enc_CL)
            packed_m.setdefault(C, []).append(mask_L)

        embeds_out = [None] * len(ts_list)
        targets_out = [None] * len(ts_list)
        valid_out = [None] * len(ts_list)
        meta_out = [None] * len(ts_list)

        for C, idxs in groups.items():
            x_enc = torch.stack(packed_x[C], dim=0)  # [B,C,L]
            input_mask = torch.stack(packed_m[C], dim=0)  # [B,L]

            # Frozen MOMENT, so no_grad is safe and saves memory
            with torch.no_grad():
                # Per-patch embeddings: [B,C,n_patches,d_model]
                moment_out = self.moment(x_enc=x_enc, input_mask=input_mask, reduction="none")
                enc = moment_out.embeddings

                # Normalize inputs the same way MOMENT does (for targets)
                x_norm = self.moment.normalizer(x=x_enc, mask=input_mask, mode="norm")
                x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

                # Patch view: [B,C,n_patches,patch_len]
                patches = self.moment.tokenizer(x=x_norm)
                B, C2, n_patches, patch_len = patches.shape
                assert C2 == C
                assert patch_len == self.moment_patch_len
                # Flatten embeddings into patch-major tokens: [B, n_patches*C, d_model]
                enc_flat = enc.permute(0, 2, 1, 3).reshape(B, n_patches * C, self.moment_d_model)

                # Next-patch targets: target token at (p,c) is patch (p+1,c)
                # targets_next: [B,C,n_patches-1,patch_len]
                targets_next = patches[:, :, 1:, :]
                # patch-major flatten: [B, (n_patches-1)*C, patch_len]
                targets_flat = targets_next.permute(0, 2, 1, 3).reshape(B, (n_patches - 1) * C, patch_len)

                # Valid mask: only if both current patch and next patch contain real data
                # input_mask is [B,L]; L is moment_seq_len and should be multiple of patch_len for standard MOMENT ckpts
                if (input_mask.shape[1] % patch_len) != 0:
                    # conservative fallback: mark all as valid; still works, just noisier
                    valid_flat = torch.ones((B, (n_patches - 1) * C), device=self.device, dtype=torch.bool)
                else:
                    patch_mask = input_mask.view(B, n_patches, patch_len).any(dim=-1)  # [B,n_patches]
                    pred_valid_patch = patch_mask[:, :-1] & patch_mask[:, 1:]           # [B,n_patches-1]
                    valid_flat = pred_valid_patch.unsqueeze(-1).expand(B, n_patches - 1, C).reshape(B, (n_patches - 1) * C)

            # Store back per original sample index
            for b, orig_i in enumerate(idxs):
                embeds_out[orig_i] = enc_flat[b]                         # [n_patches*C, d_model]
                targets_out[orig_i] = targets_flat[b].to(torch.float32)  # [pred_tokens, patch_len]
                valid_out[orig_i] = valid_flat[b]                        # [pred_tokens]
                meta_out[orig_i] = {
                    "n_channels": C,
                    "n_patches": int(enc.shape[2]),
                    "patch_len": patch_len,
                    "n_tokens_total": int(enc_flat.shape[1]),
                    "n_tokens_pred": int(targets_flat.shape[1]),
                }

        return embeds_out, targets_out, valid_out, meta_out

    # -------------------------
    # Batch assembly (text + TS soft prompt tokens)
    # -------------------------
    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
        return_targets: bool = False,
    ):
        device = self.device
        H = int(self.llm.config.hidden_size)
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
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)          # [N_all, P_max]
        attn_mask = tok.attention_mask.to(device, non_blocking=True)     # [N_all, P_max]
        text_embeds = self.llm.get_input_embeddings()(input_ids)         # [N_all, P_max, H]

        # 3) Gather all TS segments across the batch (one segment per time_series_text entry)
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                ts_list.append(ts)

        # 4) MOMENT embeddings + targets (aligned with ts_list order)
        if ts_list:
            moment_embeds_list, target_list, valid_list, meta_list = self._moment_embed_and_targets_grouped(ts_list)

            # Project MOMENT d_model -> LLM hidden, cast to text dtype
            ts_proj_list = []
            for enc_flat in moment_embeds_list:
                # enc_flat: [n_tokens_total, d_model]
                enc_flat = enc_flat.to(device=device, dtype=torch.float32)
                proj = self.encoder_projector(enc_flat).to(dtype=text_embeds.dtype)  # [n_tokens_total, H]
                ts_proj_list.append(proj)
        else:
            moment_embeds_list, target_list, valid_list, meta_list = [], [], [], []
            ts_proj_list = []

        # 5) Re-assemble per sample
        all_seq_embeds: List[torch.Tensor] = []
        all_seq_masks: List[torch.Tensor] = []
        all_seq_ids: List[torch.Tensor] = [] if return_targets else None
        all_ts_flags: List[torch.Tensor] = [] if return_targets else None

        # Return-time metadata so compute_pretrain_loss can split TS tokens correctly
        ts_segments = [] if return_targets else None

        bos_id = self.tokenizer.bos_token_id

        def _extract_active(row_embeds, row_mask, row_ids, strip_bos: bool):
            active = row_mask.bool()
            seg_ids = row_ids[active]
            seg_emb = row_embeds[active]
            if strip_bos and seg_ids.numel() > 0 and seg_ids[0].item() == bos_id:
                seg_ids = seg_ids[1:]
                seg_emb = seg_emb[1:]
            seg_len = seg_ids.numel()
            seg_mask = torch.ones(seg_len, device=device, dtype=row_mask.dtype)
            return seg_emb, seg_mask, seg_ids

        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]   # [1+N_ts+1, P_max, H]
            sample_masks  = attn_mask[start:end]
            sample_ids    = input_ids[start:end]

            seq_embeds, seq_masks = [], []
            seq_ids, seq_flags = ([], []) if return_targets else (None, None)

            # pre_prompt (keep BOS)
            pre_emb, pre_m, pre_ids = _extract_active(sample_embeds[0], sample_masks[0], sample_ids[0], strip_bos=False)
            seq_embeds.append(pre_emb)
            seq_masks.append(pre_m)
            if return_targets:
                seq_ids.append(pre_ids)
                seq_flags.append(torch.zeros(pre_ids.numel(), device=device, dtype=torch.long))

            # each (text_i, ts_i)
            for i in range(n_ts):
                idx = 1 + i

                txt_emb, txt_m, txt_ids = _extract_active(sample_embeds[idx], sample_masks[idx], sample_ids[idx], strip_bos=True)
                seq_embeds.append(txt_emb)
                seq_masks.append(txt_m)

                proj = ts_proj_list[ts_offset + i]  # [n_tokens_total, H]
                n_tokens_total = proj.size(0)
                seq_embeds.append(proj)
                seq_masks.append(torch.ones(n_tokens_total, device=device, dtype=torch.long))

                if return_targets:
                    seq_ids.append(txt_ids)
                    seq_flags.append(torch.zeros(txt_ids.numel(), device=device, dtype=torch.long))

                    seq_ids.append(torch.full((n_tokens_total,), -100, device=device, dtype=torch.long))
                    seq_flags.append(torch.ones(n_tokens_total, device=device, dtype=torch.long))

                    meta = meta_list[ts_offset + i]
                    ts_segments.append({
                        "n_tokens_total": meta["n_tokens_total"],
                        "n_tokens_pred": meta["n_tokens_pred"],
                        "targets": target_list[ts_offset + i],  # [n_tokens_pred, patch_len] float32
                        "valid":   valid_list[ts_offset + i],   # [n_tokens_pred] bool
                    })

            ts_offset += n_ts

            # post_prompt
            post_emb, post_m, post_ids = _extract_active(sample_embeds[-1], sample_masks[-1], sample_ids[-1], strip_bos=True)
            seq_embeds.append(post_emb)
            seq_masks.append(post_m)
            if return_targets:
                seq_ids.append(post_ids)
                seq_flags.append(torch.zeros(post_ids.numel(), device=device, dtype=torch.long))

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))
            if return_targets:
                all_seq_ids.append(torch.cat(seq_ids, dim=0))
                all_ts_flags.append(torch.cat(seq_flags, dim=0))

        # 6) Batch-pad
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)   # [B, L_max, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)   # [B, L_max]

        if return_targets:
            token_ids = pad_sequence(all_seq_ids, batch_first=True, padding_value=-100)
            ts_mask   = pad_sequence(all_ts_flags, batch_first=True, padding_value=0).bool()
            return inputs_embeds, attention_mask, token_ids, ts_mask, ts_segments

        return inputs_embeds, attention_mask, None, None, None

    # -------------------------
    # Pretraining loss
    # -------------------------
    def compute_pretrain_loss(
        self,
        batch: List[Dict[str, any]],
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        total = ntp_loss + alpha * recon_loss

        recon_loss is patch-wise next-patch MSE:
          token at patch p predicts numeric values of patch p+1, per channel.
        """
        if alpha is None:
            alpha = float(getattr(self.args, "recon_alpha", 1.0))

        (
            inputs_embeds,
            attention_mask,
            token_ids,
            ts_mask,
            ts_segments,
        ) = self.pad_and_apply_batch(batch, return_targets=True)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 1) NTP loss on text tokens only (TS tokens are -100)
        logits       = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()

        ntp_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 2) Next-patch loss on TS tokens
        if (ts_segments is None) or (len(ts_segments) == 0):
            recon_loss = torch.tensor(0.0, device=self.device)
            return ntp_loss + alpha * recon_loss, ntp_loss, recon_loss

        last_hidden = outputs.hidden_states[-1]  # [B, L, H]
        ts_hidden_all = last_hidden[ts_mask]     # [sum_ts_tokens, H]

        # Split by segments (in the exact insertion order used in pad_and_apply_batch)
        offset = 0
        sse = torch.tensor(0.0, device=self.device)
        denom = torch.tensor(0.0, device=self.device)

        for seg in ts_segments:
            n_total = int(seg["n_tokens_total"])
            n_pred  = int(seg["n_tokens_pred"])

            h_seg = ts_hidden_all[offset: offset + n_total]
            offset += n_total

            if n_pred <= 0:
                continue

            h_pred = h_seg[:n_pred]  # prefix corresponds to patches 0..n_patches-2 (patch-major flatten)

            z = self.decoder_projector(h_pred.to(torch.float32))  # [n_pred, d_model]
            pred = self.decoder_head(z)                            # [n_pred, patch_len]

            targets = seg["targets"].to(self.device)               # [n_pred, patch_len] float32
            valid   = seg["valid"].to(self.device)                 # [n_pred] bool

            if valid.any():
                diff = (pred - targets)[valid]                     # [n_valid, patch_len]
                sse = sse + (diff * diff).sum()
                denom = denom + diff.numel()

        recon_loss = sse / torch.clamp(denom, min=1.0)

        total = ntp_loss + alpha * recon_loss
        return total, ntp_loss, recon_loss

    # -------------------------
    # Generation + downstream loss (unchanged pattern)
    # -------------------------
    def generate(self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs) -> List[str]:
        inputs_embeds, attention_mask, _, _, _ = self.pad_and_apply_batch(batch, return_targets=False)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token

    def eval_prompt(self, prompt: FullPrompt, max_new_tokens: int = 30000, normalize: bool = False) -> str:
        batch = [prompt.to_dict()]
        self.eval()
        batch = extend_time_series_to_match_patch_size_and_aggregate(batch, normalize=normalize)
        output = self.generate(batch, max_new_tokens=max_new_tokens)
        return output[0]