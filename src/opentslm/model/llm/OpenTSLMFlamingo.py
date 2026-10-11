# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from opentslm.model.encoder.TransformerCNNEncoder2D import TransformerCNNEncoder2D
from opentslm.model.decoder.TransformerCNNDecoder2D import TransformerCNNDecoder2D
from opentslm.model.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from typing import Any, List, Dict, Tuple
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

class _EncoderFlamingoWrapper(nn.Module):
    """Thin wrapper around TransformerCNNEncoder2D that satisfies the Flamingo
    vision_encoder interface (forward(x) with x: [N, L], no chan_id argument).
    Weights are shared with ``OpenTSLMFlamingo.ts_encoder``.
    """

    def __init__(self, encoder: TransformerCNNEncoder2D):
        super().__init__()
        self.encoder = encoder  # shared object — same parameters as self.ts_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L)  — data is already z-normalised by the dataset
        chan_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.encoder(x, chan_id)  # (N, N_patches, enc_dim)


# Monkey-patch FlamingoLayer to add attention_type property for compatibility with newer transformers
from open_flamingo.src.flamingo_lm import FlamingoLayer


def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


# Add the attention_type property to FlamingoLayer
FlamingoLayer.attention_type = property(_attention_type_property) # type: ignore


class OpenTSLMFlamingo(TimeSeriesLLM):
    def __init__(
        self,
        device: str,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        freeze_lm_embeddings: bool = False,
        patch_size: int = 16,
        **flamingo_kwargs,
    ):
        super().__init__(device)
        print(f"Flamingo Using device: {self.device}")

        # ── TS encoder (replaces CNNTokenizer) ────────────────────────────
        self.ts_encoder = TransformerCNNEncoder2D(patch_size=patch_size).to(device)
        self.patch_size  = patch_size
        enc_dim = ENCODER_OUTPUT_DIM  # 768

        # Thin wrapper that bridges TransformerCNNEncoder2D → Flamingo interface
        time_series_encoder = _EncoderFlamingoWrapper(self.ts_encoder)

        # ── MSE decoder path ───────────────────────────────────────────────
        self.decoder_projector = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, enc_dim),
        ).to(device)
        self.decoder = TransformerCNNDecoder2D(input_dim=enc_dim, patch_size=patch_size).to(device)

        text_tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )

        lang_encoder = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation="sdpa",
        )

        # add Flamingo special tokens to the tokenizer
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            text_tokenizer.pad_token = "<PAD>"

        # convert LM to FlamingoLM
        extend_instance(lang_encoder, FlamingoLMMixin)

        def _infer_decoder_layers_attr_name(model):
            __KNOWN_DECODER_LAYERS_ATTR_NAMES = {
                "opt": "model.decoder.layers",
                "gptj": "transformer.h",
                "gpt-j": "transformer.h",
                "pythia": "gpt_neox.layers",
                "llama": "model.layers",
                "gptneoxforcausallm": "gpt_neox.layers",
                "mpt": "transformer.blocks",
                "mosaicgpt": "transformer.blocks",
                "gemma": "model.layers",
                "gemma2": "model.layers",
                "gemma3": "model.layers",
                "medgemma": "model.layers",
                # Qwen3ForCausalLM matches none of the other substrings;
                # without this entry Flamingo cannot construct on a Qwen backbone.
                "qwen": "model.layers",
            }

            # Special handling for Gemma3 models with different architectures
            model_class_name = model.__class__.__name__
            if "gemma3" in model_class_name.lower():
                if "ConditionalGeneration" in model_class_name:
                    # Gemma3ForConditionalGeneration (multimodal 4B model) - layers are at language_model.layers
                    return "language_model.layers"
                else:
                    # Gemma3ForCausalLM (text-only 1B model) - layers are at standard model.layers
                    return "model.layers"

            # Original logic for non-Gemma3 models
            for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
                if k.lower() in model.__class__.__name__.lower():
                    return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

            raise ValueError(
                f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
            )

        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))

        # Fix compatibility for Gemma3Config which has hidden_size in text_config
        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = (
                    lang_encoder.config.text_config.hidden_size
                )

        model = TimeSeriesFlamingoWithTrainableEncoder(
            SimpleNamespace(visual=time_series_encoder),
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=enc_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            **flamingo_kwargs,
        )

        # TimeSeriesFlamingoWithTrainableEncoder._encode_vision_x calls
        # self.vision_encoder(x) directly, so the SimpleNamespace(visual=...)
        # wrapper used for construction must be swapped for the real module
        # before any forward ("SimpleNamespace object is not callable").
        model.vision_encoder = time_series_encoder

        # Freeze all parameters
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
            # TODO: investigate also training the output embeddings when untied

        # additonally unfreeze encoder. vision_encoder is the
        # SimpleNamespace(visual=<encoder>) wrapper built above, not a module,
        # so the unfreeze must reach the module inside it.
        getattr(model.vision_encoder, "visual", model.vision_encoder).requires_grad_(True)

        self.model = model
        self.llm = model
        self.text_tokenizer = text_tokenizer

    def pad_and_apply_batch(
        self, batch: List[Dict[str, any]], include_labels: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def pad_time_series(batch, max_length=None):
            """Pad time series to the same length (either max in batch or specified max)"""
            time_series = [item["time_series"] for item in batch]

            # Determine target length (either specified or max in batch)
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)

            padded_series = []
            for ts in time_series:
                current_length = ts.shape[1]
                if current_length < max_length:
                    # Pad with zeros to reach max_length
                    # Ensure padding has the same number of dimensions as the time series
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    # If already at or exceeding max_length, truncate
                    padded = ts[:, :max_length]

                padded_series.append(padded)

            return torch.stack(padded_series)

        cast_dtype = None
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
            "input_ids"
        ][-1]

        # Process time series data
        images = pad_time_series(batch).to(
            self.device, dtype=cast_dtype, non_blocking=True
        )
        images = images.unsqueeze(1)  # Add time dimension

        # Process text inputs WITH answers
        text_inputs = []
        # Process text inputs WITH answers
        text_inputs = []
        prompt_lengths = []

        for item in batch:
            # Build the prompt text without answer
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"

            if include_labels:
                text_inputs.append(prompt_text)
                continue

            # Store the prompt length in tokens
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))

            # Add the answer to create full text
            full_text = prompt_text + f" {item['answer']}"
            text_inputs.append(full_text)

        # Tokenize full text (prompt + answer)
        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        if include_labels:
            return input_ids, images, attention_mask, None

        # Create labels matrix (-100 for masked tokens)
        labels = torch.full_like(input_ids, -100)

        # Set labels for answer tokens using the stored prompt lengths
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding_indices = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding_indices[non_padding_indices >= prompt_length]

            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]

        return input_ids, images, attention_mask, labels

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True

        try:
            with torch.inference_mode():
                input_ids, images, attention_mask, _ = self.pad_and_apply_batch(
                    batch, include_labels=True
                )

                # open_flamingo's Flamingo.generate has a fixed signature with
                # no eos_token_id/pad_token_id (it manages stopping internally);
                # passing them raises TypeError. Decoded text is truncated at
                # EOS markers by the caller instead.
                # open_flamingo accepts only this fixed set; HF-style extras
                # (eos_token_id, repetition_penalty, ...) raise TypeError.
                _OF_OK = {"num_beams", "temperature", "top_k", "top_p",
                          "no_repeat_ngram_size", "prefix_allowed_tokens_fn",
                          "length_penalty", "num_return_sequences",
                          "do_sample", "early_stopping"}
                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in generate_kwargs.items() if k in _OF_OK},
                )

                # Remove input ids from generation
                answer_only_ids = gen_ids[:, input_ids.shape[1] :]

                return self.text_tokenizer.batch_decode(
                    answer_only_ids, skip_special_tokens=True
                )
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        input_ids, images, attention_mask, labels = self.pad_and_apply_batch(
            batch, include_labels=False
        )

        output = self.model(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output[0]

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    # ── Joint Phase-1 helpers ──────────────────────────────────────────────

    def _encode_chronos_for_mse(
        self, chronos_batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode Chronos time series with TransformerCNNEncoder2D.

        Returns:
            enc_embeds:        [N_ts, N_patches, enc_dim]
            targets_shifted:   [N_ts, N_patches, patch_size]  (next-patch targets)
            ts_patch_valid:    [N_ts, N_patches] bool (True where next patch is real)
        """
        device = self.device
        ts_list: List[torch.Tensor] = []
        ts_chan_id: List[int] = []
        ts_num_patches: List[int] = []

        for sample in chronos_batch:
            ts_tensor = torch.as_tensor(sample["time_series"], dtype=torch.float32)
            if ts_tensor.dim() == 1:
                ts_tensor = ts_tensor.unsqueeze(0)  # [1, T]
            ts_lengths = sample.get("time_series_lengths")
            ts_normed  = bool(sample.get("ts_already_normalized", False))
            num_series = ts_tensor.shape[0]

            for c in range(num_series):
                flat = ts_tensor[c]  # [T_padded]
                if ts_lengths is not None and len(ts_lengths) > c:
                    length_val = ts_lengths[c]
                    real_len = int(length_val.item() if torch.is_tensor(length_val) else length_val)
                    real_len = max(0, min(real_len, flat.size(0)))
                else:
                    real_len = flat.size(0)

                ts_num_patches.append((real_len + self.patch_size - 1) // self.patch_size)

                if ts_normed:
                    ts_proc = flat  # already z-normalised; trailing zeros are padding
                else:
                    flat_real = flat[:real_len]
                    m = flat_real.mean()
                    s = flat_real.std().clamp(min=1e-8)
                    ts_proc = torch.zeros_like(flat)
                    ts_proc[:real_len] = (flat_real - m) / s

                ts_list.append(ts_proc)
                ts_chan_id.append(c)

        if not ts_list:
            return None, None, None

        # Collate — each tensor is already padded to a patch_size multiple by the
        # collate fn; pad_sequence handles different lengths across samples.
        ts_stacked = pad_sequence(
            [ts.unsqueeze(-1) for ts in ts_list], batch_first=True
        ).squeeze(-1).to(device, non_blocking=True)  # [N_ts, T_max]

        # Ensure T_max is a multiple of patch_size (safety check)
        T_max = ts_stacked.size(1)
        rem = T_max % self.patch_size
        if rem:
            ts_stacked = F.pad(ts_stacked, (0, self.patch_size - rem))
            T_max = ts_stacked.size(1)

        max_patches = T_max // self.patch_size
        n_real_t = torch.tensor(ts_num_patches, device=device)  # [N_ts]
        ts_patch_mask = (
            torch.arange(max_patches, device=device).unsqueeze(0) < n_real_t.unsqueeze(1)
        )  # [N_ts, max_patches]

        # Next-patch prediction targets (shifted by one patch)
        ts_patches = ts_stacked.reshape(ts_stacked.size(0), max_patches, self.patch_size)
        ts_targets_shifted = torch.zeros_like(ts_patches)
        ts_targets_shifted[:, :-1, :] = ts_patches[:, 1:, :]

        ts_patch_valid = (
            torch.arange(max_patches, device=device).unsqueeze(0)
            < (n_real_t - 1).clamp(min=0).unsqueeze(1)
        )  # True only where the NEXT patch is real

        enc_embeds = self.ts_encoder(
            ts_stacked,
            chan_id=torch.tensor(ts_chan_id, device=device),
            attention_mask=ts_patch_mask,
        )  # [N_ts, max_patches, enc_dim]

        return enc_embeds, ts_targets_shifted, ts_patch_valid


    def compute_joint_phase2_loss(self, batch, alpha: float = 1.0,
                                  dynamic_weighting: bool = False):
        """CE-only adapter for joint_phase2_downstream on the official variant.

        FixPos2D adds an MSE reconstruction term through its forecasting
        decoder; this class has no decoder, and the downstream CoT corpora are
        CE-only, so the MSE term is identically zero. Returns the trainer's
        (total, ce, mse) triple.
        """
        import torch as _t
        ce = self.compute_loss(batch)
        zero = _t.zeros((), device=ce.device)
        return ce, ce, zero

    def compute_joint_phase1_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
        dynamic_weighting: bool = False,
        detach_llm_for_mse: bool = False,  # no-op for Flamingo; MSE never touches LLM
        ce_alpha: float = 1.0,  # trainer passes this for every JOINT_PHASE1 stage
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint Phase-1 loss: CE (TSQA/M4 via Flamingo pipeline) + MSE (Chronos
        via TransformerCNNEncoder2D → decoder, bypassing the LLM entirely).

        Gradient flows:
          CE path:  TS → ts_encoder → perceiver → gated_cross_attn → LLM (frozen) → CE loss
          MSE path: TS → ts_encoder → decoder_projector → decoder → MSE loss
                    (LLM is never involved in the MSE path)

        Args:
            alpha: fixed MSE weight when dynamic_weighting=False.
            dynamic_weighting: rescale MSE each step to match CE magnitude.
            detach_llm_for_mse: kept for API parity with OpenTSLMSPFixPos2D;
                always a no-op here since MSE never reaches the LLM.

        Returns:
            (total_loss, loss_ce, loss_tsqa, loss_m4, loss_mse)
        """
        zero = next(self.parameters()).new_zeros(())

        tsqa_batch    = [s for s in batch if s.get("type") == "tsqa"]
        m4_batch      = [s for s in batch if s.get("type") in ("m4", "m4_captioning")]
        ce_batch      = tsqa_batch + m4_batch
        chronos_batch = [s for s in batch if s.get("type") == "chronos"]

        # ── CE loss (Flamingo pipeline) ────────────────────────────────────
        if ce_batch:
            loss_ce = self.compute_loss(ce_batch)
        else:
            loss_ce = zero

        # Per-task CE breakdown for wandb logging.
        # With task-homogeneous sampling each batch is either all-tsqa OR all-m4.
        if tsqa_batch and not m4_batch:
            loss_tsqa = loss_ce.detach()
            loss_m4   = zero.detach()
        elif m4_batch and not tsqa_batch:
            loss_tsqa = zero.detach()
            loss_m4   = loss_ce.detach()
        else:
            loss_tsqa = zero.detach()
            loss_m4   = zero.detach()

        # ── MSE loss (encoder → decoder, no LLM) ──────────────────────────
        if not chronos_batch:
            return ce_alpha * loss_ce, loss_ce, loss_tsqa, loss_m4, zero

        enc_embeds, targets, valid = self._encode_chronos_for_mse(chronos_batch)
        if enc_embeds is None:
            return ce_alpha * loss_ce, loss_ce, loss_tsqa, loss_m4, zero

        ts_hidden = self.decoder_projector(enc_embeds.float())
        ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])
        ts_pred   = self.decoder(ts_hidden)  # [N_ts, N_patches, P]

        P = targets.shape[-1]
        targets_flat = targets.reshape(-1, P)
        pred_flat    = ts_pred.reshape(-1, P).to(targets_flat.dtype)
        valid_flat   = valid.reshape(-1)
        valid_count  = int(valid_flat.sum().item())

        loss_mse = (
            F.mse_loss(pred_flat[valid_flat], targets_flat[valid_flat])
            if valid_count > 0 else zero
        )

        if dynamic_weighting and ce_batch:
            mse_weight = (loss_ce.detach() / (loss_mse.detach() + 1e-8)).clamp(0.01, 100.0)
        else:
            mse_weight = alpha

        total = ce_alpha * loss_ce + mse_weight * loss_mse
        return total, loss_ce, loss_tsqa, loss_m4, loss_mse

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    def store_to_file(self, path: str = "best_model.pt"):
        # self.llm.state_dict() already includes ts_encoder weights (via vision_encoder wrapper).
        # decoder and decoder_projector live outside self.llm so are saved separately.
        state_dict = {
            "llm": self.llm.state_dict(),
            "decoder": self.decoder.state_dict(),
            "decoder_projector": self.decoder_projector.state_dict(),
        }
        torch.save(state_dict, path)
        print(f"Model saved to {path}")

    def load_from_file(self, path: str = "best_model.pt"):
        """
        Load model parameters with non-strict loading to handle Flamingo-specific layers.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if "llm" in checkpoint:
            model_state = checkpoint["llm"]
        elif "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            raise RuntimeError("No recognized model state key in checkpoint.")

        # Handle DDP (DistributedDataParallel) if needed
        if hasattr(self, "module"):
            model_state = {f"module.{k}": v for k, v in model_state.items()}

        # Remove 'model.' prefix if present in checkpoint keys
        if all(k.startswith("model.") for k in model_state.keys()):
            model_state = {
                k.replace("model.", "", 1): v for k, v in model_state.items()
            }

        # Load state dict with strict=False to handle missing/unexpected keys
        missing_keys, unexpected_keys = self.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"⚠️  Warning: Missing keys when loading checkpoint:")
            for key in missing_keys[:10]:
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... and {len(missing_keys) - 10} more keys")
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys when loading checkpoint:")
            for key in unexpected_keys[:10]:
                print(f"   - {key}")
            if len(unexpected_keys) > 10:
                print(f"   ... and {len(unexpected_keys) - 10} more keys")

        # Load decoder and decoder_projector if saved separately
        if "decoder" in checkpoint and hasattr(self, "decoder"):
            self.decoder.load_state_dict(checkpoint["decoder"], strict=False)
        if "decoder_projector" in checkpoint and hasattr(self, "decoder_projector"):
            self.decoder_projector.load_state_dict(checkpoint["decoder_projector"], strict=False)

        self.to(self.device)

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> str:
        """
        Evaluate a prompt and return the generated text.
        """
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            batch = [prompt.to_dict()]
            self.eval()
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                batch, normalize=normalize
            )
            print("Generating")
            output = self.generate(batch, max_new_tokens=max_new_tokens)
            print(f"Generated output: {output[0]}")
            return output[0]
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable
