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

from opentslm.model_config import ENCODER_OUTPUT_DIM
from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder2D import TransformerCNNEncoder2D
from ..decoder.TransformerCNNDecoder2D import TransformerCNNDecoder2D
from ..projector.MLPProjector import MLPProjector, DecoderProjector
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

from torch.optim import AdamW
from opentslm.model_config import (
    LR_ENCODER, LR_PROJECTOR, LR_DECODER, LR_DECODER_PROJECTOR, WEIGHT_DECAY
)


def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick: z = μ + σ * ε,   ε ~ N(0, I)

    Using log_var = log(σ²) for numerical stability.

    Args:
        mu:      [..., enc_dim]
        log_var: [..., enc_dim]
    Returns:
        z:       [..., enc_dim]  differentiable sample from q(z|x)
    """
    std = torch.exp(0.5 * log_var)   # σ = exp(log_σ)
    eps = torch.randn_like(std)       # ε ~ N(0, I)
    return mu + std * eps


def _kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL( N(μ, σ²) || N(0, I) ) — closed form.

    = -0.5 * mean(1 + log_σ² - μ² - exp(log_σ²))

    Averaged over all elements for stable scaling across batch sizes.

    Args:
        mu:      [..., enc_dim]
        log_var: [..., enc_dim]
    Returns:
        kl:  scalar
    """
    return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())



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

        # self.projector = MLPProjector(
        #     ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        # ).to(device)

        # self.decoder_projector = DecoderProjector(
        #     self.llm.config.hidden_size, ENCODER_OUTPUT_DIM, device=device
        # ).to(device)


        self.patch_size = 4

        # MASKING token for masked autoencoding (Stage 1a)
        self.mask_token = nn.Parameter(torch.zeros(self.patch_size))
        nn.init.normal_(self.mask_token, std=0.02)
        
        
        # LoRA-related attributes
        self.lora_enabled = False
        self.original_llm = (
            None  # Keep reference to original model for backward compatibility
        )

        # Freeze the LLM backbone for SP model (internally)
        for p in self.llm.parameters():
            p.requires_grad = False


        # for debugging
        self.counter = 0
        
        # setting up for VAE training Encoder Decoder
        enc_dim = self.encoder.output_dim
        llm_dim = self.llm.config.hidden_size
        self.mu_head      = nn.Linear(enc_dim, enc_dim).to(device)
        self.log_var_head = nn.Linear(enc_dim, enc_dim).to(device)
        self.projector = nn.Linear(enc_dim, llm_dim).to(device)
        self.decoder_projector = nn.Linear(llm_dim, enc_dim).to(device)

# =========================================================================
    # set_stage_1a — deterministic masked autoencoder (no VAE, no KL)
    # =========================================================================

    def set_stage_1a(self):
        """
        Configure model for Stage 1a — masked autoencoder pretraining.

        No VAE, no KL. Pure deterministic:
            masked_patches → encoder → enc_repr → decoder → reconstructed_patches
            loss = MSE on masked positions only

        Trains  : encoder, decoder
        Frozen  : projector, decoder_projector, LLM (not called at all)
        """
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

        # not used in stage 1a
        for p in self.projector.parameters():
            p.requires_grad = False
        for p in self.decoder_projector.parameters():
            p.requires_grad = False
        for p in self.llm.parameters():
            p.requires_grad = False

        # mu/log_var heads: not used
        if hasattr(self, "mu_head"):
            for p in self.mu_head.parameters():
                p.requires_grad = False
        if hasattr(self, "log_var_head"):
            for p in self.log_var_head.parameters():
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 60)
        print("Stage 1a: Masked Autoencoder (deterministic, no LLM, no VAE)")
        print("  Pipeline: masked_patches → encoder → decoder → recon_patches")
        print("  Loss: MSE on masked positions only (prevents trivial copy)")
        print(f"  🔥 encoder    trainable")
        print(f"  🔥 decoder    trainable")
        print(f"  ❄️  projector  frozen  (not used)")
        print(f"  ❄️  LLM        frozen  (not called)")
        print(f"  trainable params: {trainable:,}")
        print("=" * 60)


    # =========================================================================
    # compute_pretrain_loss_1a — masked autoencoder ELBO (no VAE)
    # =========================================================================

    def compute_pretrain_loss_1a(
        self,
        batch,
        mask_ratio: float = 0.75,
    ):
        """
        Stage 1a — Masked Autoencoder reconstruction loss.

        Pipeline:
            1. Split series into patches
            2. Randomly mask `mask_ratio` fraction of patches
               (replace with learnable mask_token — encoder cannot see target values)
            3. Encoder processes the masked input
            4. Decoder reconstructs all patches
            5. MSE loss computed ONLY on masked patches

        Masking prevents the trivial solution: without it, the Transformer
        encoder attends to the target patch directly and the decoder just
        copies it through, giving loss → 0 with no learning.

        Args:
            batch:      list of sample dicts
            mask_ratio: fraction of patches to hide (0.75 matches MOMENT/MAE)

        Returns:
            (recon_loss, recon_loss, zero_tensor)
            — returns 3 values to keep the training loop signature consistent
              with the VAE variant (total, recon, kl) where kl=0 here
        """
        device = self.device

        # ── 1. Collect series ─────────────────────────────────────────────────
        ts_list    = []
        ts_chan_id = []
        for sample in batch:
            for c, ts in enumerate(sample["time_series"]):
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                if ts.dim() == 2 and ts.size(1) != 1:
                    ts = ts.transpose(0, 1)
                ts_list.append(ts.squeeze(-1))   # [T]
                ts_chan_id.append(c)

        if not ts_list:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        # ── 2. Pad all series to the same length ──────────────────────────────
        ts_padded = pad_sequence(
            [t.to(device) for t in ts_list], batch_first=True
        )   # [N, T_max]

        # ── 3. Per-series instance normalization ──────────────────────────────
        # Critical: without this, high-amplitude series dominate MSE gradient.
        ts_means = ts_padded.mean(dim=1, keepdim=True)
        ts_stds  = ts_padded.std(dim=1, keepdim=True).clamp(min=1e-8)
        ts_norm  = (ts_padded - ts_means) / ts_stds   # [N, T_max]  mean=0, std=1

        # ── 4. Pad to multiple of patch_size ──────────────────────────────────
        rem = ts_norm.size(1) % self.patch_size
        if rem:
            ts_norm = F.pad(ts_norm, (0, self.patch_size - rem))

        N, T_padded = ts_norm.shape
        P           = self.patch_size
        N_patches   = T_padded // P

        # ── 5. Reshape into patches ───────────────────────────────────────────
        ts_patches = ts_norm.reshape(N, N_patches, P)   # [N, N_patches, P]
        # ts_patches is the reconstruction TARGET (original, unmasked)

        # ── 6. Build random mask ──────────────────────────────────────────────
        # mask[i, j] = True  →  patch j of series i is MASKED (hidden from encoder)
        rand = torch.rand(N, N_patches, device=device)
        mask = rand < mask_ratio   # [N, N_patches]

        # Safety: guarantee at least one visible patch per series
        # (avoids degenerate case where encoder sees nothing)
        all_masked = mask.all(dim=1)
        if all_masked.any():
            random_idx      = torch.randint(0, N_patches, (all_masked.sum().item(),), device=device)
            affected        = all_masked.nonzero(as_tuple=True)[0]
            mask[affected, random_idx] = False

        # ── 7. Replace masked patches with learnable mask token ───────────────
        # Using a learned token (rather than zeros) so the encoder cannot
        # infer masking from the value and learns a true masked-reconstruction objective.
        # mask_token is defined in __init__: nn.Parameter(torch.zeros(patch_size))
        mask_token_3d = self.mask_token.to(device).view(1, 1, P).expand(N, N_patches, P)
        mask_3d       = mask.unsqueeze(-1).expand_as(ts_patches)
        ts_input      = torch.where(mask_3d, mask_token_3d, ts_patches)   # [N, N_patches, P]

        # flatten back to [N, T_padded] for encoder
        ts_input_flat = ts_input.reshape(N, T_padded)   # [N, T_padded]

        # ── 8. Encoder ────────────────────────────────────────────────────────
        enc_repr = self.encoder(
            ts_input_flat,
            chan_id=torch.tensor(ts_chan_id, device=device),
        )   # [N, N_patches, enc_dim]

        # ── 9. Decoder ────────────────────────────────────────────────────────
        recon_flat = self.decoder(enc_repr, ts_chan_id)   # [N, T_padded]  (flat output)
        recon      = recon_flat.reshape(N, N_patches, P).to(ts_patches.dtype)
        # recon: [N, N_patches, P]

        # ── 10. MSE only on MASKED patches ───────────────────────────────────
        # Supervising only masked positions forces the encoder to learn
        # generalised temporal representations — it cannot copy visible values.
        recon_masked   = recon[mask]      # [n_masked, P]
        targets_masked = ts_patches[mask] # [n_masked, P]

        if recon_masked.numel() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero, zero

        recon_loss = F.mse_loss(recon_masked, targets_masked)

        # ── 11. Collapse guard ────────────────────────────────────────────────
        # If decoder outputs constant ~0 for everything, std → 0.
        # This is the failure mode we saw previously.
        with torch.no_grad():
            recon_std = recon_masked.std().item()
            mask_frac = mask.float().mean().item()
            if recon_std < 0.05:
                print(
                    f"⚠️  [collapse] recon std={recon_std:.4f} — "
                    f"decoder outputting near-constant values. "
                    f"Check encoder gradients and LR."
                )

        zero = torch.tensor(0.0, device=device)
        return recon_loss, recon_loss, zero   # (total, recon, kl=0)



    # =============================================================================
    # _get_optimizer_1a — VAE version
    # =============================================================================

    def _get_optimizer_1a(self):
        """
        Stage 1a VAE optimizer.
        Includes all trainable components: inference network, bridge, generative network.
        LLM is excluded (frozen).
        """

        model = self._get_model()
        param_groups = [
            # inference network
            {"params": list(model.encoder.parameters()),
            "lr": LR_ENCODER,           "weight_decay": WEIGHT_DECAY},
            {"params": list(model.mu_head.parameters()),
            "lr": LR_ENCODER,           "weight_decay": WEIGHT_DECAY},
            {"params": list(model.log_var_head.parameters()),
            "lr": LR_ENCODER,           "weight_decay": WEIGHT_DECAY},
            # bridge
            {"params": list(model.projector.parameters()),
            "lr": LR_PROJECTOR,         "weight_decay": WEIGHT_DECAY},
            # generative network
            {"params": list(model.decoder_projector.parameters()),
            "lr": LR_DECODER_PROJECTOR, "weight_decay": WEIGHT_DECAY},
            {"params": list(model.decoder.parameters()),
            "lr": LR_DECODER,           "weight_decay": WEIGHT_DECAY},
        ]

        if self.rank == 0:
            trainable = sum(p.numel() for g in param_groups for p in g["params"])
            print(f"📊 Stage 1a VAE optimizer  |  trainable params={trainable:,}")
            print(f"   encoder / mu / log_var LR : {LR_ENCODER:.2e}")
            print(f"   projector LR             : {LR_PROJECTOR:.2e}")
            print(f"   decoder_projector LR       : {LR_DECODER_PROJECTOR:.2e}")
            print(f"   decoder LR                 : {LR_DECODER:.2e}")

        return AdamW(param_groups)

    def set_stage_1b(self):
        """
        Configure model for Stage 1b — projector alignment.

        Trains  : projector (+ LoRA if already enabled)
        Frozen  : encoder, decoder, decoder_projector, LLM backbone

        Rationale: encoder and decoder have converged in Stage 1a so their
        representations are stable.  The projector now learns to translate
        those fixed encoder outputs into the frozen LLM's embedding space,
        guided purely by next-token-prediction loss on text tokens.
        """
        # encoder: freeze (converged in 1a)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # decoder: freeze (not used in 1b)
        for p in self.decoder.parameters():
            p.requires_grad = False

        # mu/log_var heads: freeze (not used in 1b)
        for p in self.mu_head.parameters():
            p.requires_grad = False 
        for p in self.log_var_head.parameters():
            p.requires_grad = False

        # bridge: train
        for p in self.projector.parameters():
            p.requires_grad = True

        # decoder_projector: freeze (not used in 1b)
        for p in self.decoder_projector.parameters():
            p.requires_grad = False


        # LLM backbone: frozen; LoRA adapters remain trainable if present
        for name, p in self.llm.named_parameters():
            if "lora_" in name:
                p.requires_grad = True   # preserve LoRA if already enabled
            else:
                p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 55)
        print("Stage 1b: Projector alignment (frozen encoder/decoder)")
        print(f"  ❄️  encoder          frozen")
        print(f"  ❄️  decoder          frozen")
        print(f"  ❄️  decoder_projector frozen")
        print(f"  🔥 projector        trainable")
        print(f"  ❄️  LLM              frozen  (LoRA trainable if enabled)")
        print(f"  trainable params: {trainable:,}")
        print("=" * 55)



    def compute_pretrain_loss_1b(
        self,
        batch: List[Dict[str, any]],
        debug: bool = False,
    ) -> torch.Tensor:
        self.counter += 1

        (
            inputs_embeds,
            attention_mask,
            token_ids,
            _ts_mask,
            _ts_targets,
            _ts_valid,
            _ts_chan_id,
        ) = self.pad_and_apply_batch(batch, return_targets=True)

        # ── LLM forward ──
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        logits       = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = token_ids[:, 1:].contiguous()

        ntp_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )


        # Debugging print
        if self.counter % 10 == 0:

            B, Lm1, V = shift_logits.shape
            L = Lm1 + 1  # original sequence length
            # ── 5. Output analysis ──
            print(f"\n[OUTPUT]")
            print(f"  logits shape: {logits.shape}  (vocab size = {V})")
            print(f"  ntp_loss: {ntp_loss.item():.6f}")

            # ── 6. Per-region loss breakdown ──
            with torch.no_grad():
                per_token_loss = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='none',
                ).view(B, Lm1)

                shift_ts_mask   = (token_ids[:, 1:] == -100)
                shift_text_mask = (token_ids[:, 1:] != -100) & (attention_mask[:, 1:].bool())

                # Loss on text tokens right AFTER TS patches (boundary tokens)
                # These are the ones that should be most affected by projector quality
                shift_ts_boundary = torch.zeros_like(shift_text_mask)
                for b in range(B):
                    ts_pos = (token_ids[b] == -100)
                    for i in range(L - 1):
                        if ts_pos[i] and not ts_pos[i + 1] and attention_mask[b, i + 1]:
                            # token at i+1 is first text token after TS block
                            # in shifted labels this is index i
                            shift_ts_boundary[b, i] = True
                            # also mark next few tokens
                            for k in range(1, min(6, Lm1 - i)):
                                if i + k < Lm1 and shift_labels[b, i + k] != -100:
                                    shift_ts_boundary[b, i + k] = True

                text_loss_vals = per_token_loss[shift_text_mask]
                boundary_loss_vals = per_token_loss[shift_ts_boundary]

                print(f"\n[PER-REGION LOSS]")
                if text_loss_vals.numel() > 0:
                    print(f"  All text tokens — mean: {text_loss_vals.mean():.4f}, "
                        f"count: {text_loss_vals.numel()}")
                if boundary_loss_vals.numel() > 0:
                    print(f"  Post-TS boundary (first 5 text tokens after </ts>) — "
                        f"mean: {boundary_loss_vals.mean():.4f}, count: {boundary_loss_vals.numel()}")
                    print(f"  Non-boundary text — mean: "
                        f"{per_token_loss[shift_text_mask & ~shift_ts_boundary].mean():.4f}")

                # ── 7. Top predictions vs labels at boundary ──
                print(f"\n[BOUNDARY TOKEN PREDICTIONS] (first text tokens after TS patches)")
                for b in range(min(B, 2)):
                    boundary_idx = shift_ts_boundary[b].nonzero(as_tuple=True)[0]
                    if len(boundary_idx) == 0:
                        print(f"  Sample {b}: no boundary tokens found")
                        continue
                    for idx in boundary_idx[:5]:  # first 5 boundary tokens
                        pred_top5 = shift_logits[b, idx].topk(5)
                        true_id = shift_labels[b, idx].item()
                        try:
                            true_tok = self.tokenizer.decode([true_id])
                            pred_toks = [self.tokenizer.decode([t.item()]) for t in pred_top5.indices]
                        except:
                            true_tok = str(true_id)
                            pred_toks = [str(t.item()) for t in pred_top5.indices]
                        print(f"  Sample {b}, pos {idx.item()}: "
                            f"true={repr(true_tok)} | top5={[repr(p) for p in pred_toks]} | "
                            f"loss={per_token_loss[b, idx]:.4f}")

                # ── 8. Greedy generation to see what the LLM actually outputs ──
                        print(f"\n[GENERATION] (greedy decode, max 50 new tokens)")
                        with torch.no_grad():
                            for b in range(min(B, 2)):
                                # Find the last non-pad position
                                valid_len = attention_mask[b].sum().item()
                                prompt_embeds = inputs_embeds[b:b+1, :valid_len, :]  # [1, valid_len, H]
                                prompt_mask   = attention_mask[b:b+1, :valid_len]    # [1, valid_len]

                                # Decode the text portion of the input as "prompt"
                                prompt_token_ids = token_ids[b, :valid_len].tolist()
                                prompt_text_ids = [t for t in prompt_token_ids if t != -100]
                                try:
                                    prompt_text = self.tokenizer.decode(prompt_text_ids)
                                    if len(prompt_text) > 200:
                                        prompt_text = prompt_text[:100] + " ... " + prompt_text[-100:]
                                except:
                                    prompt_text = "(could not decode)"
                                print(f"\n  Sample {b} input text: {repr(prompt_text)}")

                                # Greedy auto-regressive generation
                                generated_ids = []
                                cur_embeds = prompt_embeds
                                cur_mask   = prompt_mask

                                for step in range(50):
                                    out = self.llm(
                                        inputs_embeds=cur_embeds,
                                        attention_mask=cur_mask,
                                        return_dict=True,
                                    )
                                    next_logit = out.logits[:, -1, :]          # [1, V]
                                    next_id    = next_logit.argmax(dim=-1)      # [1]
                                    generated_ids.append(next_id.item())

                                    # Check for EOS
                                    if hasattr(self.tokenizer, 'eos_token_id') and next_id.item() == self.tokenizer.eos_token_id:
                                        break

                                    # Get embedding for next token and append
                                    next_embed = self.llm.get_input_embeddings()(next_id).unsqueeze(1)  # [1, 1, H]
                                    cur_embeds = torch.cat([cur_embeds, next_embed], dim=1)
                                    cur_mask   = torch.cat([cur_mask, torch.ones(1, 1, device=cur_mask.device, dtype=cur_mask.dtype)], dim=1)

                                try:
                                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                                except:
                                    generated_text = f"(raw ids: {generated_ids})"

                                print(f"  Sample {b} generated : {repr(generated_text)}")

                                # Also show top-5 token probabilities for first 10 generated tokens
                                # (already generated, re-derive from the loop isn't clean, so just show the text)
                                print(f"  Sample {b} gen ids   : {generated_ids[:20]}{'...' if len(generated_ids) > 20 else ''}")

            print("=" * 80)
        elif self.counter % 99 == 0:
            print("Finished debuggging")
            quit()
        return ntp_loss

    def _sincos_1d(self, positions: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
        # positions: [P]
        device = positions.device
        positions = positions.float().unsqueeze(1)  # [P,1]
        half = dim // 2
        freqs = torch.arange(half, device=device, dtype=torch.float32)
        freqs = 1.0 / (base ** (freqs / max(1, half)))   # [half]
        ang = positions * freqs.unsqueeze(0)             # [P,half]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [P,2*half]
        if dim % 2 == 1:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), 1)], dim=1)
        return emb


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
        batch: List[Dict[str, any]],
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

        for b, sample in enumerate(batch):
            for c, ts in enumerate(sample["time_series"]):
                # ensure [T] -> [T,1]
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)

                # NOTE: fix ordering bug: transpose BEFORE appending, otherwise ts_list gets the wrong layout
                if ts.dim() == 2 and ts.size(1) != 1:
                    # transpose from [D,T] to [T,D]
                    ts = ts.transpose(0, 1)

                if ts.dim() != 2 or ts.size(1) != 1:
                    raise ValueError(f"Expected time series shape [T,1], got {tuple(ts.shape)}")

                ts_list.append(ts)
                ts_chan_id.append(c)

        ts_patch_targets_shifted = None
        ts_patch_valid = None

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(device, non_blocking=True)

            # pad time dim to multiple of patch_size
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)

            # NOTE: store raw patch targets for reconstruction
            if return_targets:
                ts_patch_targets = ts_padded.squeeze(-1).reshape(ts_padded.size(0), -1, self.patch_size)
                N_ts_total, N_patches, P = ts_patch_targets.shape

                ts_patch_targets_shifted = torch.zeros_like(ts_patch_targets)
                ts_patch_targets_shifted[:, :-1, :] = ts_patch_targets[:, 1:, :]

                ts_patch_valid = torch.ones(N_ts_total, N_patches, device=device, dtype=torch.bool)
                ts_patch_valid[:, -1] = False

            ts_enc = self.encoder(
                ts_padded.squeeze(-1),
                chan_id=torch.tensor(ts_chan_id, device=device),
            )
            ts_proj = self.projector(ts_enc).to(text_embeds.dtype)
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) Re-assemble per sample
        all_seq_embeds: List[torch.Tensor] = []
        all_seq_masks: List[torch.Tensor] = []
        all_seq_ids: List[torch.Tensor] = [] if return_targets else None
        all_ts_flags: List[torch.Tensor] = [] if return_targets else None

        bos_id = self.tokenizer.bos_token_id

        # NOTE: new helper to slice ONLY active tokens using attention_mask.
        # This makes left-padding vs right-padding irrelevant and guarantees there are no 0s inside the final assembled mask.
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
            seg_mask = torch.ones(seg_len, device=device, dtype=row_mask.dtype)  # NOTE: always ones, so final masks are prefix-ones
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

                # NOTE: strip BOS for non-first text segments in a padding-safe way (check the first ACTIVE token)
                txt_emb, txt_m, txt_ids = _extract_active(
                    sample_embeds[idx], sample_masks[idx], sample_ids[idx], strip_bos=True
                )

                seq_embeds.append(txt_emb)
                seq_masks.append(txt_m)

                proj = ts_proj[ts_offset + i]  # [N_patches, H]
                n_patches = proj.size(0)

                seq_embeds.append(proj)
                seq_masks.append(torch.ones(n_patches, device=device, dtype=torch.long))

                if return_targets:
                    seq_ids.append(txt_ids)
                    seq_flags.append(torch.zeros(txt_ids.numel(), device=device, dtype=torch.long))

                    seq_ids.append(torch.full((n_patches,), -100, device=device, dtype=torch.long))
                    seq_flags.append(torch.ones(n_patches, device=device, dtype=torch.long))

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
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        inputs_embeds, attention_mask, _, _, _, _, ts_chan_id = self.pad_and_apply_batch(batch, return_targets=self.args.is_pretraining)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        answers = [b["answer"] for b in batch]

        inputs_embeds, attention_mask, token_ids, ts_mask, ts_patch_targets_shifted, ts_patch_valid, ts_chan_id = self.pad_and_apply_batch(batch, return_targets=self.args.is_pretraining)  # [B, L, H], [B, L]
        B, L, H = inputs_embeds.size()

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

        new_embeds_list = []
        new_masks_list = []
        new_labels_list = []

        for i in range(B):
            a_len = int(ans_mask[i].sum().item())  # true answer length for sample i

            # concat (prompt+ts) + (answer up to a_len)
            emb_i = torch.cat([inputs_embeds[i], ans_emb_all[i, :a_len, :]], dim=0)  # [L+a_len, H]
            mask_i = torch.cat([attention_mask[i], torch.ones(a_len, device=self.device, dtype=attention_mask.dtype)], dim=0)

            # labels: ignore prompt part, supervise only answer tokens
            lab_i = torch.full((L + a_len,), -100, device=self.device, dtype=torch.long)
            lab_i[L:] = ans_ids[i, :a_len]

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
            labels=labels,
            return_dict=True,
        )
        return outputs.loss
        



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
        if not hasattr(self, "counter"):
            self.counter = 0
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
        t_std_min = t_std_med = t_std_max = None
        N_ts_total = N_patches = P = 0
        valid_count = 0
        valid_ratio = 0.0

        # Keep these for plotting/debug
        ts_recon = None  # [N_ts_total, N_patches, P]

        if need_recon:
            N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
            last_hidden = outputs.hidden_states[-1]      # [B, L, H]

            # Extract hidden states at TS patch positions
            ts_hidden = last_hidden[ts_mask]             # [N_ts_total * N_patches, H]

            # Safe reshape contract check
            expected_flat = N_ts_total * N_patches
            actual_flat = ts_hidden.size(0)
            assert actual_flat == expected_flat, (
                f"ts_mask extracted {actual_flat} hidden vectors but expected "
                f"N_ts_total({N_ts_total}) × N_patches({N_patches}) = {expected_flat}. "
                f"Check that pad_and_apply_batch pads all series to the same T_max."
            )

            # Project + decode
            ts_hidden = self.decoder_projector(ts_hidden.float())      # [N_ts_total*N_patches, enc_dim]
            ts_hidden_3d = ts_hidden.reshape(N_ts_total, N_patches, -1)
            ts_recon = self.decoder(ts_hidden_3d, ts_chan_id)
            ts_recon = ts_recon.reshape(N_ts_total, N_patches, P)      # [N_ts_total, N_patches, P]

            # Per-series instance normalization on targets (apply same to recon for loss)
            targets_2d = ts_patch_targets_shifted.reshape(N_ts_total, -1)
            t_mean = targets_2d.mean(dim=1, keepdim=True)
            t_std = targets_2d.std(dim=1, keepdim=True).clamp(min=1e-6)

            with torch.no_grad():
                t_std_cpu = t_std.squeeze(1).detach()
                t_std_min = float(t_std_cpu.min().item())
                t_std_med = float(t_std_cpu.median().item())
                t_std_max = float(t_std_cpu.max().item())

            t_mean_3d = t_mean.unsqueeze(-1).expand_as(ts_patch_targets_shifted)
            t_std_3d = t_std.unsqueeze(-1).expand_as(ts_patch_targets_shifted)

            targets_norm = (ts_patch_targets_shifted - t_mean_3d) / t_std_3d
            recon_norm = (ts_recon - t_mean_3d) / t_std_3d

            # Flatten and apply validity mask (exclude last patch)
            targets_flat = targets_norm.reshape(-1, P)
            recon_flat = recon_norm.reshape(-1, P).to(targets_flat.dtype)
            valid_flat = ts_patch_valid.reshape(-1)

            valid_count = int(valid_flat.sum().item())
            valid_ratio = float(valid_count / max(1, valid_flat.numel()))

            if valid_count > 0:
                recon_loss = F.mse_loss(recon_flat[valid_flat], targets_flat[valid_flat])

                # Collapse guard
                with torch.no_grad():
                    recon_std = float(recon_flat[valid_flat].std().item())
                    if recon_std < 1e-3:
                        print(
                            f"⚠️  [collapse guard] decoder output std={recon_std:.2e}. "
                            f"Decoder may be ignoring LLM hidden states. "
                            f"Consider reducing alpha={alpha} or checking decoder_projector gradients."
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
                if t_std_min is not None:
                    print(
                        f"  per-series target std (after clamp): "
                        f"min/med/max = {t_std_min:.3e}/{t_std_med:.3e}/{t_std_max:.3e}"
                    )
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

        
        os.makedirs(plot_dir, exist_ok=True)

        if self.counter % 100 == 0:
            with torch.no_grad():
                n_plot = min(int(plot_max_series), int(N_ts_total))
                for s in range(n_plot):
                    gt_p = ts_patch_targets_shifted[s]  # [N_patches, P]
                    pr_p = ts_recon[s]                  # [N_patches, P]
                    v_p = ts_patch_valid[s]             # [N_patches]

                    # Keep only valid patches (exclude last patch, and any other invalids)
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
        if self.counter % 100 == 0:
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
