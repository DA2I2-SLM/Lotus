#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""Tiny fixed-window forecasting overfit diagnostic for OpenTSLM.

This script is intentionally separate from the curriculum trainer.  It answers
one focused question: can a forecasting path overfit a tiny, fixed set of
Chronos-style windows, or is the loss floor around ~0.7 caused by a code/path
issue?

The diagnostic compares three forecast paths:

  llm:
    encoder -> projector -> frozen LLM -> decoder_projector -> decoder

  projector:
    encoder -> projector -> decoder_projector -> decoder

  encoder:
    encoder -> mse_projector -> decoder

The target can be the in-window next-patch objective, the first future patch
after the history window, or both.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", f"matplotlib-{os.environ.get('USER', 'opentslm')}"),
)

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.ChronosDataset import (
    PREPROCESSED_SPLITS_DIR,
    _last_value_impute,
)
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


class FixedChronosForecastDataset(Dataset):
    """Deterministic Chronos next-patch/future-patch diagnostic set."""

    def __init__(
        self,
        *,
        split: str,
        num_samples: int,
        history_len: int,
        patch_size: int,
        seed: int,
        eos_token: str,
        window_position: str = "tail",
    ) -> None:
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if history_len % patch_size != 0:
            raise ValueError(
                f"history_len={history_len} must be divisible by patch_size={patch_size}"
            )
        if window_position not in {"head", "middle", "tail", "random"}:
            raise ValueError(f"Unknown window_position={window_position!r}")

        split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all", split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"Chronos split cache not found: {split_dir}\n"
                "Build Phase-1 Chronos splits first, or point PREPROCESSED_SPLITS_DIR "
                "at an existing cache."
            )

        raw = load_from_disk(split_dir, keep_in_memory=False)
        rng = random.Random(seed)
        indices = list(range(len(raw)))
        rng.shuffle(indices)

        needed_len = history_len + patch_size
        rows: List[Dict[str, Any]] = []
        for idx in indices:
            series = torch.as_tensor(raw[idx]["target"], dtype=torch.float32).reshape(-1)
            series = _last_value_impute(series)
            if series.numel() < needed_len:
                continue

            max_start = series.numel() - needed_len
            if window_position == "head":
                start = 0
            elif window_position == "middle":
                start = max_start // 2
            elif window_position == "tail":
                start = max_start
            else:
                start = rng.randint(0, max_start)

            history = series[start : start + history_len]
            future = series[start + history_len : start + history_len + patch_size]

            mean = history.mean()
            std = history.std().clamp(min=1e-8)
            history_norm = (history - mean) / std
            future_norm = (future - mean) / std

            mean_f = float(mean.item())
            std_f = float(std.item())
            rows.append(
                {
                    "type": "chronos",
                    "pre_prompt": "",
                    "time_series": [history_norm],
                    "time_series_text": [
                        f"This is the time series, it has mean {mean_f:.4f} "
                        f"and std {std_f:.4f}. <ts>"
                    ],
                    "post_prompt": eos_token,
                    "answer": eos_token,
                    "norm_mean": mean_f,
                    "norm_std": std_f,
                    "time_series_norm_stats": [(mean_f, std_f)],
                    "ts_already_normalized": True,
                    "n_patches": history_len // patch_size,
                    "ts_future": future_norm.to(torch.float32),
                    "_raw_idx": int(idx),
                    "_start": int(start),
                }
            )
            if len(rows) >= num_samples:
                break

        if not rows:
            raise RuntimeError(
                f"No Chronos rows in split={split!r} had at least {needed_len} points."
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return dict(self.rows[idx])


def _collate(batch: List[Dict[str, Any]], patch_size: int) -> List[Dict[str, Any]]:
    return extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=patch_size
    )


def _make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        is_pretraining=True,
        patch_size=args.patch_size,
        pe_type=args.pe_type,
        use_only_time_pe=args.use_only_time_pe,
        use_only_channel_pe=args.use_only_channel_pe,
        use_freq_embed=args.use_freq_embed,
    )


def _choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_compatible_state(module: torch.nn.Module, state: Dict[str, torch.Tensor], name: str) -> None:
    model_state = module.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for key, value in state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = module.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"[ckpt] {name}: skipped {len(skipped)} incompatible keys; first={skipped[:5]}")
    if missing:
        print(f"[ckpt] {name}: missing {len(missing)} keys after partial load; first={missing[:5]}")
    if unexpected:
        print(f"[ckpt] {name}: unexpected {len(unexpected)} keys; first={unexpected[:5]}")


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    if all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_checkpoint_flexible(model: OpenTSLMSPFixPos2D, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state" in checkpoint or "model_state_dict" in checkpoint:
        state = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
        state = _strip_module_prefix(state)
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        skipped = sorted(set(state) - set(compatible))
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"[ckpt] loaded full-model state with {len(compatible)} compatible keys; "
            f"skipped={len(skipped)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        return

    structured = {
        "encoder": "encoder_state",
        "projector": "projector_state",
        "decoder": "decoder_state",
        "decoder_projector": "decoder_projector_state",
        "mse_projector": "mse_projector_state",
    }
    for module_name, key in structured.items():
        if key in checkpoint:
            _load_compatible_state(getattr(model, module_name), checkpoint[key], module_name)

    if checkpoint.get("lora_enabled"):
        print("[ckpt] checkpoint contains LoRA adapters; diagnostic leaves LoRA disabled.")

    print(f"[ckpt] loaded structured checkpoint from epoch {checkpoint.get('epoch', '?')}")


def set_trainable_forecast_path(model: OpenTSLMSPFixPos2D, path: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    active_modules: List[torch.nn.Module]
    if path == "llm":
        active_modules = [model.encoder, model.projector, model.decoder_projector, model.decoder]
    elif path == "projector":
        active_modules = [model.encoder, model.projector, model.decoder_projector, model.decoder]
    elif path == "encoder":
        active_modules = [model.encoder, model.mse_projector, model.decoder]
    else:
        raise ValueError(f"Unknown path={path!r}")

    for module in active_modules:
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[trainable] path={path} trainable_params={trainable:,}")


def forward_forecast_path(
    model: OpenTSLMSPFixPos2D,
    batch: List[Dict[str, Any]],
    path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    (
        inputs_embeds,
        attention_mask,
        _token_ids,
        ts_mask,
        targets_shifted,
        ts_valid,
        _ts_chan_id,
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    if targets_shifted is None or targets_shifted.numel() == 0:
        raise RuntimeError("Batch did not produce any TS patch targets.")

    patch_counts = (ts_valid.sum(dim=1) + 1).tolist()
    patch_counts = [int(pc) for pc in patch_counts]

    if path == "llm":
        last_hidden = model._llm_last_hidden_only(inputs_embeds, attention_mask)
        flat_hidden = last_hidden[ts_mask].float()
        flat_hidden = model.decoder_projector(flat_hidden)
    elif path == "projector":
        flat_hidden = inputs_embeds[ts_mask].float()
        flat_hidden = model.decoder_projector(flat_hidden)
    elif path == "encoder":
        if model._last_ts_enc is None:
            raise RuntimeError("pad_and_apply_batch did not populate _last_ts_enc.")
        enc_chunks = [
            model._last_ts_enc[i, :pc, :] for i, pc in enumerate(patch_counts)
        ]
        flat_hidden = torch.cat(enc_chunks, dim=0).float()
        flat_hidden = model.mse_projector(flat_hidden)
    else:
        raise ValueError(f"Unknown path={path!r}")

    expected = int(sum(patch_counts))
    if flat_hidden.size(0) != expected:
        raise RuntimeError(
            f"Path {path} produced {flat_hidden.size(0)} hidden states, expected {expected}."
        )

    flat_hidden = F.layer_norm(flat_hidden, [flat_hidden.size(-1)])
    hidden_3d = pad_sequence(
        list(torch.split(flat_hidden, patch_counts)), batch_first=True
    )
    pred = model.decoder(hidden_3d)
    return pred, targets_shifted, ts_valid, patch_counts


def loss_from_predictions(
    model: OpenTSLMSPFixPos2D,
    batch: List[Dict[str, Any]],
    pred: torch.Tensor,
    targets_shifted: torch.Tensor,
    ts_valid: torch.Tensor,
    patch_counts: List[int],
    target: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = pred.device
    zero = pred.new_zeros(())
    losses: List[torch.Tensor] = []

    targets_flat = targets_shifted.reshape(-1, targets_shifted.size(-1))
    pred_flat = pred.reshape(-1, pred.size(-1)).to(targets_flat.dtype)
    valid_flat = ts_valid.reshape(-1)
    valid_count = int(valid_flat.sum().item())

    if valid_count > 0:
        window_mse = F.mse_loss(pred_flat[valid_flat], targets_flat[valid_flat])
    else:
        window_mse = zero

    future_preds: List[torch.Tensor] = []
    future_targets: List[torch.Tensor] = []
    ts_idx = 0
    for sample in batch:
        if sample.get("type") != "chronos":
            continue
        future = sample.get("ts_future")
        if future is not None:
            last_patch_idx = int(patch_counts[ts_idx]) - 1
            future_preds.append(pred[ts_idx, last_patch_idx, :])
            future_targets.append(
                torch.as_tensor(future, device=device, dtype=pred.dtype)[: model.patch_size]
            )
        ts_idx += 1

    if future_preds:
        future_pred = torch.stack(future_preds)
        future_target = torch.stack(future_targets).to(future_pred.dtype)
        future_mse = F.mse_loss(future_pred, future_target)
        pred_std = float(future_pred.std().detach().item())
        target_std = float(future_target.std().detach().item())
        pred_mean = float(future_pred.mean().detach().item())
        target_mean = float(future_target.mean().detach().item())
    else:
        future_mse = zero
        pred_std = math.nan
        target_std = math.nan
        pred_mean = math.nan
        target_mean = math.nan

    if target in {"window", "both"}:
        losses.append(window_mse)
    if target in {"future", "both"}:
        losses.append(future_mse)
    if not losses:
        raise ValueError(f"Unknown target={target!r}")

    total = torch.stack(losses).sum()
    stats = {
        "loss": float(total.detach().item()),
        "window_mse": float(window_mse.detach().item()),
        "future_mse": float(future_mse.detach().item()),
        "valid_window_positions": float(valid_count),
        "future_pred_std": pred_std,
        "future_target_std": target_std,
        "future_pred_mean": pred_mean,
        "future_target_mean": target_mean,
    }
    return total, stats


def compute_loss_and_stats(
    model: OpenTSLMSPFixPos2D,
    batch: List[Dict[str, Any]],
    path: str,
    target: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred, targets_shifted, ts_valid, patch_counts = forward_forecast_path(
        model, batch, path
    )
    return loss_from_predictions(
        model, batch, pred, targets_shifted, ts_valid, patch_counts, target
    )


def evaluate_fixed_loader(
    model: OpenTSLMSPFixPos2D,
    loader: DataLoader,
    path: str,
    target: str,
    max_batches: int | None = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue
            _, stats = compute_loss_and_stats(model, batch, path, target)
            for key, value in stats.items():
                if not math.isnan(value):
                    totals[key] = totals.get(key, 0.0) + value
            count += 1
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    if was_training:
        model.train()
    if count == 0:
        raise RuntimeError("Evaluation loader produced no batches.")
    return {key: value / count for key, value in totals.items()}


def write_history_header(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "step\ttrain_loss\ttrain_window_mse\ttrain_future_mse\t"
            "eval_loss\teval_window_mse\teval_future_mse\t"
            "future_pred_std\tfuture_target_std\n"
        )


def append_history(path: str, row: Dict[str, float]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            f"{int(row['step'])}\t"
            f"{row['train_loss']:.8f}\t"
            f"{row['train_window_mse']:.8f}\t"
            f"{row['train_future_mse']:.8f}\t"
            f"{row['eval_loss']:.8f}\t"
            f"{row['eval_window_mse']:.8f}\t"
            f"{row['eval_future_mse']:.8f}\t"
            f"{row['future_pred_std']:.8f}\t"
            f"{row['future_target_std']:.8f}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm_id", default="google/gemma-3-270m")
    parser.add_argument("--ckpt", default=None, help="Optional model checkpoint to warm-start.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--path", default="projector", choices=["llm", "projector", "encoder"])
    parser.add_argument("--target", default="both", choices=["window", "future", "both"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--history_len", type=int, default=480)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument(
        "--window_position",
        default="tail",
        choices=["head", "middle", "tail", "random"],
    )
    parser.add_argument("--out_dir", default="./eval_results/diagnose_forecasting_overfit")
    parser.add_argument("--pe_type", default="sincos")
    parser.add_argument("--use_only_time_pe", action="store_true")
    parser.add_argument("--use_only_channel_pe", action="store_true")
    parser.add_argument("--use_freq_embed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _choose_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[diagnose] creating model")
    model = OpenTSLMSPFixPos2D(
        args=_make_model_args(args),
        llm_id=args.llm_id,
        device=device,
    ).to(device)
    model.train()

    if args.ckpt:
        load_checkpoint_flexible(model, args.ckpt)

    set_trainable_forecast_path(model, args.path)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("[diagnose] building fixed Chronos windows")
    dataset = FixedChronosForecastDataset(
        split=args.split,
        num_samples=args.num_samples,
        history_len=args.history_len,
        patch_size=args.patch_size,
        seed=args.seed,
        eos_token=model.get_eos_token(),
        window_position=args.window_position,
    )
    print(
        f"[diagnose] fixed samples={len(dataset)} split={args.split} "
        f"history_len={args.history_len} patch_size={args.patch_size} "
        f"path={args.path} target={args.target}"
    )

    collate = lambda batch: _collate(batch, args.patch_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    history_path = os.path.join(args.out_dir, "loss_history.tsv")
    write_history_header(history_path)

    data_iter: Iterable[List[Dict[str, Any]]] = iter(loader)
    last_stats: Dict[str, float] = {}
    pbar = tqdm(range(1, args.steps + 1), desc="diagnose-overfit")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        if not batch:
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_stats = compute_loss_and_stats(model, batch, args.path, args.target)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
        optimizer.step()
        last_stats = train_stats

        pbar.set_postfix(
            loss=f"{train_stats['loss']:.4f}",
            future=f"{train_stats['future_mse']:.4f}",
            pred_std=f"{train_stats['future_pred_std']:.3f}",
        )

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            eval_stats = evaluate_fixed_loader(
                model,
                eval_loader,
                args.path,
                args.target,
                max_batches=args.eval_batches,
            )
            row = {
                "step": float(step),
                "train_loss": train_stats["loss"],
                "train_window_mse": train_stats["window_mse"],
                "train_future_mse": train_stats["future_mse"],
                "eval_loss": eval_stats["loss"],
                "eval_window_mse": eval_stats["window_mse"],
                "eval_future_mse": eval_stats["future_mse"],
                "future_pred_std": eval_stats.get("future_pred_std", math.nan),
                "future_target_std": eval_stats.get("future_target_std", math.nan),
            }
            append_history(history_path, row)
            print(
                "[diagnose] "
                f"step={step} "
                f"train_loss={row['train_loss']:.6f} "
                f"eval_loss={row['eval_loss']:.6f} "
                f"eval_window_mse={row['eval_window_mse']:.6f} "
                f"eval_future_mse={row['eval_future_mse']:.6f} "
                f"future_pred_std={row['future_pred_std']:.6f} "
                f"future_target_std={row['future_target_std']:.6f}"
            )

    final_stats = evaluate_fixed_loader(
        model, eval_loader, args.path, args.target, max_batches=args.eval_batches
    )
    summary = {
        "args": vars(args),
        "num_samples": len(dataset),
        "last_train_stats": last_stats,
        "final_eval_stats": final_stats,
        "history_path": history_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[diagnose] wrote {history_path}")
    print(f"[diagnose] wrote {summary_path}")


if __name__ == "__main__":
    main()
