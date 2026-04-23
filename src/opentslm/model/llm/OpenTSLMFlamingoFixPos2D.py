# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch._dynamo
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_flamingo.src.flamingo_lm import FlamingoLMMixin, FlamingoLayer
from open_flamingo.src.utils import extend_instance

from opentslm.model.decoder.TransformerCNNDecoder2D import TransformerCNNDecoder2D
from opentslm.model.encoder.TransformerCNNEncoder2D import TransformerCNNEncoder2D
from opentslm.model.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.model_config import PATCH_SIZE
from opentslm.prompt.full_prompt import FullPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


FlamingoLayer.attention_type = property(_attention_type_property)  # type: ignore


class _FlamingoPatchEncoderAdapter(nn.Module):
    """Adapt the causal TS encoder to the OpenFlamingo vision encoder interface."""

    def __init__(self, base_encoder: TransformerCNNEncoder2D):
        super().__init__()
        self.base_encoder = base_encoder
        self._current_lengths: Optional[torch.Tensor] = None

    def set_current_lengths(self, lengths: Optional[torch.Tensor]) -> None:
        self._current_lengths = None if lengths is None else lengths.detach()

    def clear_current_lengths(self) -> None:
        self._current_lengths = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected [N, L] time-series tensor, got {tuple(x.shape)}")

        device = x.device
        n_series, total_steps = x.shape
        if total_steps % self.base_encoder.patch_size != 0:
            raise ValueError(
                f"Sequence length {total_steps} not divisible by patch_size={self.base_encoder.patch_size}"
            )

        chan_id = torch.zeros(n_series, device=device, dtype=torch.long)
        attention_mask = None
        if self._current_lengths is not None:
            lengths = self._current_lengths.to(device=device, dtype=torch.long).reshape(-1)
            if lengths.numel() != n_series:
                raise ValueError(
                    f"Stored TS lengths ({lengths.numel()}) do not match encoder batch ({n_series})."
                )

            max_patches = total_steps // self.base_encoder.patch_size
            n_real = (lengths + self.base_encoder.patch_size - 1) // self.base_encoder.patch_size
            n_real = n_real.clamp(min=1, max=max_patches)
            attention_mask = (
                torch.arange(max_patches, device=device).unsqueeze(0)
                < n_real.unsqueeze(1)
            )

        return self.base_encoder(x, chan_id=chan_id, attention_mask=attention_mask)


class OpenTSLMFlamingoFixPos2D(TimeSeriesLLM):
    """Flamingo text-conditioning plus a dedicated forecasting decoder head.

    CE tasks follow the standard Flamingo path:
        TS encoder -> perceiver latents -> gated cross-attention -> frozen LLM

    Chronos forecasting follows the dedicated numerical path:
        TS encoder patch states (before perceiver) -> decoder_projector -> decoder

    This keeps Flamingo attention semantics intact for text generation while
    avoiding a text-generation bottleneck for numerical forecasting.
    """

    def __init__(
        self,
        args,
        device: str,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        cross_attn_every_n_layers: int = 1,
        freeze_lm_embeddings: bool = False,
        **flamingo_kwargs,
    ):
        super().__init__(device)
        self.args = args
        self._debug_step = 0

        self.text_tokenizer = AutoTokenizer.from_pretrained(
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

        self.text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.text_tokenizer.pad_token = "<PAD>"

        extend_instance(lang_encoder, FlamingoLMMixin)

        def _infer_decoder_layers_attr_name(model):
            known = {
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
            }

            model_class_name = model.__class__.__name__
            if "gemma3" in model_class_name.lower():
                if "ConditionalGeneration" in model_class_name:
                    return "language_model.layers"
                return "model.layers"

            for key, value in known.items():
                if key.lower() in model_class_name.lower():
                    return value
            raise ValueError(
                "Could not infer decoder layer attribute name for Flamingo language model."
            )

        lang_encoder.set_decoder_layers_attr_name(_infer_decoder_layers_attr_name(lang_encoder))
        lang_encoder.resize_token_embeddings(len(self.text_tokenizer))

        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = lang_encoder.config.text_config.hidden_size

        encoder_patch_size = int(getattr(args, "patch_size", PATCH_SIZE))
        self.patch_size = encoder_patch_size
        self.encoder = TransformerCNNEncoder2D(
            args=args,
            patch_size=encoder_patch_size,
        ).to(device)
        self.encoder_adapter = _FlamingoPatchEncoderAdapter(self.encoder).to(device)

        self.model = TimeSeriesFlamingoWithTrainableEncoder(
            SimpleNamespace(visual=self.encoder_adapter),
            lang_encoder,
            self.text_tokenizer.encode("<|endofchunk|>")[-1],
            self.text_tokenizer.encode("<image>")[-1],
            vis_dim=self.encoder.output_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            **flamingo_kwargs,
        )

        self.model.requires_grad_(False)
        assert sum(p.numel() for p in self.model.parameters() if p.requires_grad) == 0

        self.model.perceiver.requires_grad_(True)
        self.model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            self.model.lang_encoder.get_input_embeddings().requires_grad_(True)
        self.model.vision_encoder.requires_grad_(True)

        enc_dim = self.encoder.output_dim
        self.decoder_projector = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, enc_dim),
        ).to(device)
        self.decoder = TransformerCNNDecoder2D(
            input_dim=enc_dim,
            patch_size=self.patch_size,
        ).to(device)

        self.llm = self.model
        self.counter = 0

    def _flatten_image_lengths(self, image_lengths: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image_lengths is None:
            return None
        return image_lengths.reshape(-1)

    def _set_flamingo_lengths(self, image_lengths: Optional[torch.Tensor]) -> None:
        self.model.vision_encoder.set_current_lengths(self._flatten_image_lengths(image_lengths))

    def _clear_flamingo_lengths(self) -> None:
        self.model.vision_encoder.clear_current_lengths()

    def pad_and_apply_batch(
        self, batch: List[Dict[str, Any]], include_labels: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        def pad_time_series(batch_rows, max_length=None):
            time_series = [item["time_series"] for item in batch_rows]
            time_lengths = [item.get("time_series_lengths") for item in batch_rows]

            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)

            padded_series = []
            padded_lengths = []
            for ts, lengths in zip(time_series, time_lengths):
                ts = torch.as_tensor(ts, dtype=torch.float32)
                current_length = ts.shape[1]
                base_lengths = torch.full(
                    (ts.shape[0],),
                    min(current_length, max_length),
                    dtype=torch.long,
                )
                if current_length < max_length:
                    pad_shape = list(ts.shape)
                    pad_shape[1] = max_length - current_length
                    ts = torch.cat(
                        [ts, torch.zeros(pad_shape, device=ts.device, dtype=ts.dtype)],
                        dim=1,
                    )
                else:
                    ts = ts[:, :max_length]

                padded_series.append(ts)

                if lengths is None:
                    padded_lengths.append(base_lengths)
                else:
                    lengths_t = torch.as_tensor(lengths, dtype=torch.long).reshape(-1)
                    if lengths_t.numel() == 1 and ts.shape[0] > 1:
                        lengths_t = lengths_t.expand(ts.shape[0])
                    lengths_t = lengths_t[: ts.shape[0]].clamp(min=1, max=max_length)
                    padded_lengths.append(lengths_t)

            return torch.stack(padded_series), torch.stack(padded_lengths)

        tokenizer = self.text_tokenizer
        media_token = tokenizer.decode([tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]])
        eoc_token = tokenizer.decode([tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]])

        images, image_lengths = pad_time_series(batch)
        images = images.to(self.device, non_blocking=True).unsqueeze(1)
        image_lengths = image_lengths.to(self.device, non_blocking=True).unsqueeze(1)

        text_inputs: List[str] = []
        prompt_lengths: List[int] = []

        for item in batch:
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {media_token} {ts_text} {eoc_token}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"

            if include_labels:
                text_inputs.append(prompt_text)
                continue

            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))
            text_inputs.append(prompt_text + f" {item['answer']}")

        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        if include_labels:
            return input_ids, images, attention_mask, None, image_lengths

        labels = torch.full_like(input_ids, -100)
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding[non_padding >= prompt_length]
            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]

        return input_ids, images, attention_mask, labels, image_lengths

    def _run_flamingo_ce(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        image_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        self._set_flamingo_lengths(image_lengths)
        try:
            output = self.model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._clear_flamingo_lengths()
        return output[0]

    def compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        input_ids, images, attention_mask, labels, image_lengths = self.pad_and_apply_batch(
            batch, include_labels=False
        )
        return self._run_flamingo_ce(input_ids, images, attention_mask, labels, image_lengths)

    def _compute_forecast_components(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[torch.Tensor], List[int], torch.Tensor, List[Optional[torch.Tensor]]]:
        device = self.device
        ts_list: List[torch.Tensor] = []
        real_lens: List[int] = []
        futures: List[Optional[torch.Tensor]] = []

        for sample in batch:
            ts = torch.as_tensor(sample["time_series"], dtype=torch.float32, device=device)
            if ts.dim() == 1:
                ts = ts.unsqueeze(0)
            lengths = sample.get("time_series_lengths")
            if lengths is not None:
                lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
                real_len = int(lengths[0].item())
            else:
                real_len = ts.shape[1]
            real_len = max(1, min(real_len, ts.shape[1]))
            ts_list.append(ts[0, :real_len])
            real_lens.append(real_len)

            fut = sample.get("ts_future")
            if fut is None:
                futures.append(None)
            else:
                futures.append(
                    torch.as_tensor(fut, dtype=torch.float32, device=device)[: self.patch_size]
                )

        return ts_list, real_lens, torch.tensor(real_lens, device=device), futures

    def _forecast_next_patch(self, batch: List[Dict[str, Any]]) -> List[torch.Tensor]:
        device = self.device
        ts_list, real_lens, _, _ = self._compute_forecast_components(batch)
        if not ts_list:
            return []

        ts_padded = pad_sequence(ts_list, batch_first=True).to(device)
        rem = ts_padded.size(1) % self.patch_size
        if rem:
            pad_len = self.patch_size - rem
            ts_padded = torch.cat(
                [ts_padded, ts_padded.new_zeros(ts_padded.size(0), pad_len)],
                dim=1,
            )

        max_patches = ts_padded.size(1) // self.patch_size
        n_real_t = (
            (torch.tensor(real_lens, device=device) + self.patch_size - 1) // self.patch_size
        ).clamp(min=1, max=max_patches)
        patch_mask = (
            torch.arange(max_patches, device=device).unsqueeze(0) < n_real_t.unsqueeze(1)
        )

        h_enc = self.encoder(
            ts_padded,
            chan_id=torch.zeros(ts_padded.size(0), device=device, dtype=torch.long),
            attention_mask=patch_mask,
        )
        h_enc = self.decoder_projector(h_enc.float())
        h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
        pred_batch = self.decoder(h_enc)

        last_idx = n_real_t - 1
        pred_last = pred_batch[torch.arange(pred_batch.size(0), device=device), last_idx]
        return [pred_last[i].detach().cpu() for i in range(pred_last.size(0))]

    def _compute_chronos_forecast_mse(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        device = self.device
        zero = next(self.parameters()).new_zeros(())
        ts_list, real_lens, _, futures = self._compute_forecast_components(batch)
        if not ts_list:
            return zero

        ts_padded = pad_sequence(ts_list, batch_first=True).to(device)
        rem = ts_padded.size(1) % self.patch_size
        if rem:
            pad_len = self.patch_size - rem
            ts_padded = torch.cat(
                [ts_padded, ts_padded.new_zeros(ts_padded.size(0), pad_len)],
                dim=1,
            )

        patch_targets = ts_padded.reshape(ts_padded.size(0), -1, self.patch_size)
        n_samples, n_patches, _ = patch_targets.shape
        targets_shifted = torch.zeros_like(patch_targets)
        targets_shifted[:, :-1, :] = patch_targets[:, 1:, :]

        n_real_t = (
            (torch.tensor(real_lens, device=device) + self.patch_size - 1) // self.patch_size
        ).clamp(min=1, max=n_patches)
        patch_mask = (
            torch.arange(n_patches, device=device).unsqueeze(0) < n_real_t.unsqueeze(1)
        )
        patch_valid = (
            torch.arange(n_patches, device=device).unsqueeze(0)
            < (n_real_t - 1).clamp(min=0).unsqueeze(1)
        )

        h_enc = self.encoder(
            ts_padded,
            chan_id=torch.zeros(n_samples, device=device, dtype=torch.long),
            attention_mask=patch_mask,
        )
        h_enc = self.decoder_projector(h_enc.float())
        h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
        pred_batch = self.decoder(h_enc)

        pred_flat = pred_batch.reshape(-1, self.patch_size).to(targets_shifted.dtype)
        target_flat = targets_shifted.reshape(-1, self.patch_size)
        valid_flat = patch_valid.reshape(-1)
        loss_mse = (
            F.mse_loss(pred_flat[valid_flat], target_flat[valid_flat])
            if int(valid_flat.sum().item()) > 0
            else zero
        )

        future_preds, future_gts = [], []
        for idx, fut in enumerate(futures):
            if fut is None:
                continue
            last_patch_idx = int(n_real_t[idx].item()) - 1
            future_preds.append(pred_batch[idx, last_patch_idx])
            future_gts.append(fut.to(pred_batch.dtype))

        if future_preds:
            loss_mse = loss_mse + F.mse_loss(torch.stack(future_preds), torch.stack(future_gts))

        return loss_mse

    def compute_joint_phase1_loss(
        self,
        batch: List[Dict[str, Any]],
        alpha: float = 1.0,
        dynamic_weighting: bool = False,
        detach_llm_for_mse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del detach_llm_for_mse  # Forecasting already bypasses the LLM in this Flamingo variant.

        zero = next(self.parameters()).new_zeros(())
        tsqa_batch = [s for s in batch if s.get("type") == "tsqa"]
        m4_batch = [s for s in batch if s.get("type") in ("m4", "m4_captioning")]
        ce_batch = tsqa_batch + m4_batch
        chronos_batch = [s for s in batch if s.get("type") == "chronos"]

        loss_ce = self.compute_loss(ce_batch) if ce_batch else zero

        if tsqa_batch and not m4_batch:
            loss_tsqa = loss_ce.detach()
            loss_m4 = zero.detach()
        elif m4_batch and not tsqa_batch:
            loss_tsqa = zero.detach()
            loss_m4 = loss_ce.detach()
        elif tsqa_batch and m4_batch:
            with torch.no_grad():
                loss_tsqa = self.compute_loss(tsqa_batch).detach()
                loss_m4 = self.compute_loss(m4_batch).detach()
        else:
            loss_tsqa = zero.detach()
            loss_m4 = zero.detach()

        loss_mse = self._compute_chronos_forecast_mse(chronos_batch) if chronos_batch else zero

        if dynamic_weighting and ce_batch and chronos_batch:
            mse_weight = (loss_ce.detach() / (loss_mse.detach() + 1e-8)).clamp(0.01, 100.0)
        else:
            mse_weight = alpha

        total = loss_ce + mse_weight * loss_mse
        return total, loss_ce, loss_tsqa, loss_m4, loss_mse

    @torch.no_grad()
    def compute_loss_per_task(self, batch: List[Dict[str, Any]]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        task_types = list({row.get("type", "unknown") for row in batch})

        for task in task_types:
            sub = [row for row in batch if row.get("type") == task]
            if not sub:
                continue
            try:
                if task == "chronos":
                    loss = self._compute_chronos_forecast_mse(sub)
                else:
                    loss = self.compute_loss(sub)
                results[f"loss/{task}"] = float(loss.item())
            except Exception:
                results[f"loss/{task}"] = float("nan")

        tsqa_rows = [row for row in batch if row.get("type") == "tsqa"]
        if tsqa_rows:
            correct = 0
            for row in tsqa_rows:
                gold = row.get("answer", "").strip().lstrip("(").upper()[:1]
                if not gold:
                    continue
                try:
                    text_out = self.generate([row], max_new_tokens=5)
                    pred = text_out[0].strip().lstrip("(").upper()[:1] if text_out else ""
                    if pred == gold:
                        correct += 1
                except Exception:
                    pass
            results["accuracy/tsqa"] = correct / len(tsqa_rows)

        return results

    @torch.no_grad()
    def generate_phase1_samples(self, batch: List[Dict[str, Any]], max_new_tokens: int = 80) -> None:
        import sys
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        seen = {}
        for row in batch:
            task = row.get("type", "unknown")
            if task not in seen:
                seen[task] = row

        sep = "=" * 70
        lines = [f"\n{sep}", "[GENERATION SAMPLES]"]
        for task_type, row in seen.items():
            lines.append(f"\n--- {task_type.upper()} ---")
            lines.append(f"  PRE : {row.get('pre_prompt', '')[:120]}")
            lines.append(f"  POST: {row.get('post_prompt', '')[:120]}")
            lines.append(f"  GOLD: {row.get('answer', '')[:150]}")
            try:
                if task_type == "chronos":
                    preds = self._forecast_next_patch([row])
                    preview = preds[0][: min(10, preds[0].numel())].tolist() if preds else []
                    lines.append(f"  PRED PATCH: {preview}")
                else:
                    text_out = self.generate([row], max_new_tokens=max_new_tokens)
                    lines.append(f"  PRED: {text_out[0][:150] if text_out else '(empty)'}")
            except Exception as exc:
                lines.append(f"  PRED: ERROR — {exc}")
        lines.append(sep)
        print("\n".join(lines), flush=True)
        sys.stdout.flush()

    def generate(
        self, batch: List[Dict[str, Any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            with torch.inference_mode():
                input_ids, images, attention_mask, _, image_lengths = self.pad_and_apply_batch(
                    batch, include_labels=True
                )
                self._set_flamingo_lengths(image_lengths)
                try:
                    gen_ids = self.llm.generate(
                        vision_x=images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self.text_tokenizer.eos_token_id,
                        pad_token_id=self.text_tokenizer.pad_token_id,
                        **generate_kwargs,
                    )
                finally:
                    self._clear_flamingo_lengths()

                answer_only_ids = gen_ids[:, input_ids.shape[1] :]
                return self.text_tokenizer.batch_decode(
                    answer_only_ids, skip_special_tokens=True
                )
        finally:
            torch._dynamo.config.disable = original_disable

    def get_eos_token(self) -> str:
        return self.text_tokenizer.eos_token

    def eval_prompt(
        self, prompt: FullPrompt, max_new_tokens: int = 1000, normalize: bool = False
    ) -> str:
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            batch = [prompt.to_dict()]
            self.eval()
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=self.patch_size, normalize=normalize
            )
            output = self.generate(batch, max_new_tokens=max_new_tokens)
            return output[0]
        finally:
            torch._dynamo.config.disable = original_disable
