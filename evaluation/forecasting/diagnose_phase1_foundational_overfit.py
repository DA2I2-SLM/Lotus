#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""Single-path overfit diagnostic for foundational Phase-1 Chronos MSE.

This script answers one narrow question:

Can the current foundational ``phase1_chronos_mse`` path overfit a tiny,
fixed Chronos window set when optimized exactly through
``compute_pretrain_loss(..., mse_only=True)``?

Unlike the broader forecasting diagnostic, this script:
- uses the exact foundational Phase-1 loss path for optimization
- trains only ``encoder + mse_projector + decoder``
- saves concrete predicted-vs-target patch snapshots so local collapse is easy
  to inspect
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from diagnose_overfit import (  # noqa: E402
    FixedChronosForecastDataset,
    _choose_device,
    _collate,
    _make_model_args,
    load_checkpoint_flexible,
)
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D  # noqa: E402


def set_trainable_foundational_path(model: OpenTSLMSPFixPos2D) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for module in [model.encoder, model.mse_projector, model.decoder]:
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[trainable] foundational path trainable_params={trainable:,}")


def forward_foundational_path(
    model: OpenTSLMSPFixPos2D,
    batch: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    (
        _inputs_embeds,
        _attention_mask,
        _token_ids,
        _ts_mask,
        targets_shifted,
        ts_valid,
        _ts_chan_id,
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    if targets_shifted is None or targets_shifted.numel() == 0:
        raise RuntimeError("Batch did not produce any TS patch targets.")
    if model._last_ts_enc is None:
        raise RuntimeError("Foundational path expected _last_ts_enc to be populated.")

    patch_counts = (ts_valid.sum(dim=1) + 1).tolist()
    patch_counts = [int(pc) for pc in patch_counts]

    enc_chunks = [model._last_ts_enc[i, :pc, :] for i, pc in enumerate(patch_counts)]
    flat_hidden = torch.cat(enc_chunks, dim=0).float()
    flat_hidden = model.mse_projector(flat_hidden)
    flat_hidden = F.layer_norm(flat_hidden, [flat_hidden.size(-1)])

    hidden_3d = pad_sequence(
        list(torch.split(flat_hidden, patch_counts)), batch_first=True
    )
    pred = model.decoder(hidden_3d)
    return pred, targets_shifted, ts_valid, patch_counts


def stats_from_predictions(
    model: OpenTSLMSPFixPos2D,
    batch: List[Dict[str, Any]],
    pred: torch.Tensor,
    targets_shifted: torch.Tensor,
    ts_valid: torch.Tensor,
    patch_counts: List[int],
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    device = pred.device
    zero = pred.new_zeros(())

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
    first_snapshot: Dict[str, Any] = {}
    chronos_ts_idx = 0
    for sample in batch:
        if sample.get("type") != "chronos":
            continue

        n_p = int(patch_counts[chronos_ts_idx])
        last_pred = pred[chronos_ts_idx, n_p - 1, :]
        future = sample.get("ts_future")

        if not first_snapshot:
            first_snapshot = {
                "sample_type": sample.get("type"),
                "n_patches": n_p,
                "norm_mean": float(sample.get("norm_mean", 0.0)),
                "norm_std": float(sample.get("norm_std", 1.0)),
                "history_pred_patch0": pred[chronos_ts_idx, 0, :].detach().cpu().tolist(),
                "history_target_patch0": targets_shifted[chronos_ts_idx, 0, :].detach().cpu().tolist(),
                "last_position_pred_patch": last_pred.detach().cpu().tolist(),
            }
            if n_p > 1:
                last_hist_target = targets_shifted[chronos_ts_idx, n_p - 2, :]
                first_snapshot["history_pred_last_valid"] = pred[
                    chronos_ts_idx, n_p - 2, :
                ].detach().cpu().tolist()
                first_snapshot["history_target_last_valid"] = last_hist_target.detach().cpu().tolist()

        if future is not None:
            gt_fut = torch.as_tensor(
                future, device=device, dtype=last_pred.dtype
            )[: model.patch_size]
            future_preds.append(last_pred)
            future_targets.append(gt_fut)
            if "future_target_patch" not in first_snapshot:
                first_snapshot["future_target_patch"] = gt_fut.detach().cpu().tolist()
                first_snapshot["future_patch_mse"] = float(
                    F.mse_loss(last_pred, gt_fut).detach().item()
                )
                first_snapshot["future_patch_mae"] = float(
                    F.l1_loss(last_pred, gt_fut).detach().item()
                )

        chronos_ts_idx += 1

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

    total = window_mse + future_mse
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
    return total, stats, first_snapshot


def compute_eval_stats(
    model: OpenTSLMSPFixPos2D,
    loader: DataLoader,
    max_batches: int | None = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    snapshot: Dict[str, Any] = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue
            pred, targets_shifted, ts_valid, patch_counts = forward_foundational_path(
                model, batch
            )
            _, stats, batch_snapshot = stats_from_predictions(
                model, batch, pred, targets_shifted, ts_valid, patch_counts
            )
            if not snapshot:
                snapshot = batch_snapshot
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
    return {key: value / count for key, value in totals.items()}, snapshot


def write_history_header(path: str) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            "step\ttrain_loss\ttrain_recon_loss\t"
            "eval_loss\teval_window_mse\teval_future_mse\t"
            "future_pred_std\tfuture_target_std\n"
        )


def append_history(path: str, row: Dict[str, float]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(
            f"{int(row['step'])}\t"
            f"{row['train_loss']:.8f}\t"
            f"{row['train_recon_loss']:.8f}\t"
            f"{row['eval_loss']:.8f}\t"
            f"{row['eval_window_mse']:.8f}\t"
            f"{row['eval_future_mse']:.8f}\t"
            f"{row['future_pred_std']:.8f}\t"
            f"{row['future_target_std']:.8f}\n"
        )


def save_snapshot(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm_id", default="google/gemma-3-270m")
    parser.add_argument("--ckpt", default=None, help="Optional model checkpoint to warm-start.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--history_len", type=int, default=480)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
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
    parser.add_argument("--out_dir", default="./eval_results/diagnose_phase1_foundational_overfit")
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

    print("[diagnose-foundational] creating model")
    model = OpenTSLMSPFixPos2D(
        args=_make_model_args(args),
        llm_id=args.llm_id,
        device=device,
    ).to(device)
    model.train()

    if args.ckpt:
        load_checkpoint_flexible(model, args.ckpt)

    set_trainable_foundational_path(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("[diagnose-foundational] building fixed Chronos windows")
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
        f"[diagnose-foundational] fixed samples={len(dataset)} split={args.split} "
        f"history_len={args.history_len} patch_size={args.patch_size}"
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
    last_train_stats: Dict[str, float] = {}

    pbar = tqdm(range(1, args.steps + 1), desc="foundational-overfit")
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
        loss, ntp_loss, recon_loss = model.compute_pretrain_loss(batch, mse_only=True)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
        optimizer.step()

        last_train_stats = {
            "loss": float(loss.detach().item()),
            "ntp_loss": float(ntp_loss.detach().item()),
            "recon_loss": float(recon_loss.detach().item()),
        }
        pbar.set_postfix(loss=f"{last_train_stats['loss']:.4f}")

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            eval_stats, snapshot = compute_eval_stats(
                model, eval_loader, max_batches=args.eval_batches
            )
            row = {
                "step": float(step),
                "train_loss": last_train_stats["loss"],
                "train_recon_loss": last_train_stats["recon_loss"],
                "eval_loss": eval_stats["loss"],
                "eval_window_mse": eval_stats["window_mse"],
                "eval_future_mse": eval_stats["future_mse"],
                "future_pred_std": eval_stats.get("future_pred_std", math.nan),
                "future_target_std": eval_stats.get("future_target_std", math.nan),
            }
            append_history(history_path, row)
            snapshot_payload = {
                "step": step,
                "train_stats": last_train_stats,
                "eval_stats": eval_stats,
                "sample_snapshot": snapshot,
            }
            save_snapshot(
                os.path.join(args.out_dir, f"snapshot_step_{step:06d}.json"),
                snapshot_payload,
            )
            print(
                "[diagnose-foundational] "
                f"step={step} "
                f"train_loss={row['train_loss']:.6f} "
                f"train_recon={row['train_recon_loss']:.6f} "
                f"eval_loss={row['eval_loss']:.6f} "
                f"eval_window_mse={row['eval_window_mse']:.6f} "
                f"eval_future_mse={row['eval_future_mse']:.6f} "
                f"future_pred_std={row['future_pred_std']:.6f} "
                f"future_target_std={row['future_target_std']:.6f}"
            )

    final_eval_stats, final_snapshot = compute_eval_stats(
        model, eval_loader, max_batches=args.eval_batches
    )
    summary = {
        "args": vars(args),
        "num_samples": len(dataset),
        "last_train_stats": last_train_stats,
        "final_eval_stats": final_eval_stats,
        "final_snapshot": final_snapshot,
        "history_path": history_path,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[diagnose-foundational] wrote {history_path}")
    print(f"[diagnose-foundational] wrote {summary_path}")


if __name__ == "__main__":
    main()
