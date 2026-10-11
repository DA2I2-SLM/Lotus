#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Phase-2 downstream forecasting evaluation — autoregressive only.

Evaluates a joint_phase2_downstream checkpoint on the standard long-horizon
benchmarks: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Traffic, Electricity.

The LLM generates tokens freely via forecast_phase2().  The decoder is invoked
at each generated <fp_k> token.  Matches deployment conditions exactly.

Usage
-----
  python eval_phase2_downstream.py \\
    --ckpt /scratch/.../joint_phase2_downstream/checkpoints/best_model.pt \\
    --llm_id google/gemma-3-270m \\
    --patch_size 32 \\
    --data_root /scratch/s223540177/TimeSeriesECM/dataset \\
    --all_datasets \\
    --pred_len 96 192 336 720

  # single dataset with explicit CSV:
  python eval_phase2_downstream.py \\
    --ckpt ...pt --llm_id ... --patch_size 32 \\
    --dataset etth1 --csv /path/ETTh1.csv
"""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.model.llm.OpenTSLMSPFixPos2DTimerXL import OpenTSLMSPFixPos2DTimerXL
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from fixed_inference import _generate_patches_direct


# ── Dataset configs ─────────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, dict] = {
    "etth1":       {"file": "ETTh1.csv",       "split": (8640, 11520, 14400),   "max_cols": None},
    "etth2":       {"file": "ETTh2.csv",       "split": (8640, 11520, 14400),   "max_cols": None},
    "ettm1":       {"file": "ETTm1.csv",       "split": (34560, 46080, 57600),  "max_cols": None},
    "ettm2":       {"file": "ETTm2.csv",       "split": (34560, 46080, 57600),  "max_cols": None},
    "weather":     {"file": "weather.csv",     "split": (0.7, 0.1, 0.2), "max_cols": None},
    "traffic":     {"file": "traffic.csv",     "split": (0.7, 0.1, 0.2), "max_cols": 16},
    "electricity": {"file": "electricity.csv", "split": (0.7, 0.1, 0.2), "max_cols": 16},
}

# Time-MoE context lengths (matches Phase2DownstreamDataset defaults)
_CONTEXT_BY_PRED: Dict[int, int] = {96: 512, 192: 512, 336: 672, 720: 720}


# ── Multivariate eval dataset ────────────────────────────────────────────────────

class DownstreamEvalDataset(Dataset):
    """Sliding-window eval dataset that exactly matches Phase2DownstreamDataset.

    One sample = one window across ALL columns (up to max_cols).  All channels
    are normalised using global mean/std from the training split, matching
    Phase2DownstreamDataset (standard LTSF benchmark convention).

    ts_future: Tensor[C, pred_len]  — normalised ground truth
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "test",
        patch_size: int = 32,
        pred_len: int = 96,
        history_len: Optional[int] = None,
        max_cols: Optional[int] = 16,
        stride: Optional[int] = None,
        split = (0.6, 0.2, 0.2),   # ratio (tr,va,te) or absolute (train_end, val_end)
        ds_name: str = "",
    ):
        if history_len is None:
            history_len = _CONTEXT_BY_PRED.get(pred_len, 512)
        if history_len % patch_size != 0:
            history_len = (history_len // patch_size) * patch_size

        self.history_len = history_len
        self.pred_len    = pred_len
        self.patch_size  = patch_size

        df = pd.read_csv(csv_path)
        df = df.select_dtypes(include=[float, int])
        if max_cols is not None and df.shape[1] > max_cols:
            df = df.iloc[:, :max_cols]

        N  = len(df)
        if len(split) == 3 and isinstance(split[0], int):
            tr, va, te = int(split[0]), int(split[1]), int(split[2])
        elif len(split) == 2 and isinstance(split[0], int):
            tr, va, te = int(split[0]), int(split[1]), N
        else:
            tr = int(N * split[0])
            va = int(N * (split[0] + split[1]))
            te = N
        s, e = {"train": (0, tr), "val": (tr, va), "test": (va, te)}.get(mode, (va, te))
        df_split  = df.iloc[s:e].reset_index(drop=True)
        T         = len(df_split)
        data      = torch.tensor(df_split.values, dtype=torch.float32)   # [T, C]
        col_names = list(df_split.columns)

        # Global stats from training split — matches Phase2DownstreamDataset convention
        train_data  = torch.tensor(df.iloc[:tr].values, dtype=torch.float32)
        global_mean = train_data.mean(dim=0)                   # [C]
        global_std  = train_data.std(dim=0).clamp(min=1e-5)    # [C]
        C           = data.shape[1]

        # Instance (per-window) normalization — must match the training-time flag
        # (Phase2DownstreamDataset._make_sample). PHASE2_INSTANCE_NORM=1 normalizes
        # each window by its own history mean/std (RevIN-style) instead of global
        # train stats, fixing the level-anchoring failure on shifted series (ETT).
        instance_norm = os.environ.get("PHASE2_INSTANCE_NORM") == "1"

        n_per_ch   = math.ceil(pred_len / patch_size)
        window_len = history_len + pred_len
        if stride is None:
            stride = pred_len if mode in ("val", "test") else 1

        self.samples: List[dict] = []
        for start in range(0, T - window_len + 1, stride):
            window = data[start : start + window_len]
            hist_r = window[:history_len]
            fut_r  = window[history_len:]

            time_series, time_series_text, futures = [], [], []
            for c in range(C):
                h    = hist_r[:, c]
                f    = fut_r[:, c]
                if instance_norm:
                    mean = h.mean().item()
                    std  = h.std().item()
                    std  = std if std > 1e-3 else 1e-3
                else:
                    mean = global_mean[c].item()
                    std  = global_std[c].item()
                time_series.append(((h - mean) / std).tolist())
                time_series_text.append(
                    f"This is the time series ({col_names[c]}), "
                    f"mean {mean:.4f} std {std:.4f}. <ts>"
                )
                futures.append((f - mean) / std)

            self.samples.append({
                "type":                  "chronos",
                "pre_prompt":            (
                    f"This is {ds_name} data. "
                    f"You are given {history_len} timesteps across {C} channels:"
                ),
                "post_prompt":           (
                    f". Predict the next {pred_len} timesteps "
                    f"as {n_per_ch} patch{'es' if n_per_ch != 1 else ''} per channel"
                ),
                "time_series":           time_series,
                "time_series_text":      time_series_text,
                "ts_future":             torch.stack(futures, dim=0),   # [C, pred_len]
                "ts_already_normalized": True,
                "n_channels":            C,
                "_n_channels":           C,
            })

        print(
            f"[DownstreamEvalDataset] {ds_name:12s} {mode}  "
            f"pred={pred_len:4d}  hist={history_len:4d}  C={C:3d}  "
            f"stride={stride:4d}  n={len(self.samples):6d}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ── Phase-2 forecasting via encoder→LLM→decoder patch loop ─────────────────────

def forecast_phase2(
    model: OpenTSLMSPFixPos2D,
    batch: list,
    pred_len: int,
    patch_size: int,
    **kwargs,   # absorbs unused max_new_tokens / force_template / logit_boost
) -> List[Optional[torch.Tensor]]:
    """Autoregressive multi-step forecasting for Phase-2 checkpoints.

    Uses _generate_patches_direct (encoder→LLM→decoder, re-run each step) to
    iteratively predict ``ceil(pred_len / patch_size)`` future patches, then
    trims to pred_len timesteps per channel.

    Returns a list of length B; each element is either:
      - Tensor[C, pred_len]  (normalised, same scale as ts_future), or
      - None if generation failed for that sample.
    """
    n_forecast_patches = math.ceil(pred_len / patch_size)
    device = model.device
    B = len(batch)

    try:
        # [B, n_ch, n_forecast_patches, patch_size]
        out = _generate_patches_direct(
            model, batch, n_forecast_patches, device, forecast_backbone="llm"
        )
    except Exception as exc:
        print(f"  [forecast_phase2] _generate_patches_direct failed: {exc}")
        return [None] * B

    results: List[Optional[torch.Tensor]] = []
    for i in range(B):
        try:
            pred_i = out[i]                                   # [n_ch, n_fp, patch_size]
            n_ch   = pred_i.size(0)
            pred_flat = pred_i.reshape(n_ch, -1)[:, :pred_len]  # [n_ch, pred_len]
            results.append(pred_flat)
        except Exception:
            results.append(None)
    return results


# ── Autoregressive evaluation ────────────────────────────────────────────────────

def _print_debug_sample(idx: int, sample: dict, pred: torch.Tensor, gt: torch.Tensor) -> None:
    """Print history, prediction, and ground truth for one window (channel 0 only)."""
    import numpy as np
    print(f"\n{'='*60}")
    print(f"[DEBUG] Window #{idx}")
    # history is stored as list-of-lists under "time_series" (already normalized)
    ts = sample.get("time_series")
    if ts is not None:
        h = np.array(ts[0])
        print(f"  history  ch0 (last 10): {np.round(h[-10:], 4)}")
        print(f"  history  ch0  min={h.min():.4f}  max={h.max():.4f}  mean={h.mean():.4f}  last={h[-1]:.4f}")
    p = pred[0].cpu().numpy() if pred.dim() > 1 else pred.cpu().numpy()
    g = gt[0].cpu().numpy()   if gt.dim()   > 1 else gt.cpu().numpy()
    print(f"  pred     ch0 (first 10): {np.round(p[:10], 4)}")
    print(f"  gt       ch0 (first 10): {np.round(g[:10], 4)}")
    print(f"  pred     ch0  min={p.min():.4f}  max={p.max():.4f}  mean={p.mean():.4f}")
    print(f"  gt       ch0  min={g.min():.4f}  max={g.max():.4f}  mean={g.mean():.4f}")
    sample_mse = float(((p - g) ** 2).mean())
    sample_mae = float(abs(p - g).mean())
    # persistence baseline: predict last history value for all future steps
    if ts is not None:
        last_val = float(np.array(ts[0])[-1])
        persist_mse = float(((last_val - g) ** 2).mean())
        persist_mae = float(abs(last_val - g).mean())
        print(f"  sample   ch0  mse={sample_mse:.4f}  mae={sample_mae:.4f}  "
              f"(persistence mse={persist_mse:.4f} mae={persist_mae:.4f})")
    else:
        print(f"  sample   ch0  mse={sample_mse:.4f}  mae={sample_mae:.4f}")
    print(f"{'='*60}")


@torch.no_grad()
def run_autoregressive_eval(
    model: OpenTSLMSPFixPos2D,
    loader: DataLoader,
    pred_len: int,
    patch_size: int = 32,
    max_new_tokens: int = 512,
    force_template: bool = False,
    logit_boost: float = 50.0,
    debug_n: int = 0,
) -> Tuple[float, float, int]:
    """Autoregressive evaluation via encoder→LLM→decoder patch loop.

    Returns (mean_mse, mean_mae, n_windows).
    """
    device  = model.device
    sum_mse = 0.0
    sum_mae = 0.0
    n       = 0
    n_debug = 0

    for batch in tqdm(loader, desc="  autoregressive ", leave=False):
        if not batch:
            continue

        preds = forecast_phase2(
            model, batch, pred_len=pred_len, patch_size=patch_size,
            max_new_tokens=max_new_tokens, force_template=force_template,
            logit_boost=logit_boost,
        )

        for i, pred in enumerate(preds):
            if pred is None:
                continue
            gt = batch[i].get("ts_future")
            if gt is None:
                continue
            gt = gt.to(device=device, dtype=torch.float32)
            if gt.dim() == 1:
                gt = gt.unsqueeze(0)

            pred_flat = pred.reshape(-1).float()
            gt_flat   = gt.reshape(-1).to(pred_flat.dtype)
            if pred_flat.shape != gt_flat.shape:
                continue

            if n_debug < debug_n:
                _print_debug_sample(n_debug, batch[i], pred.float(), gt)
                n_debug += 1

            sum_mse += F.mse_loss(pred_flat, gt_flat).item()
            sum_mae += F.l1_loss(pred_flat, gt_flat).item()
            n       += 1

    return (sum_mse / n, sum_mae / n, n) if n > 0 else (0.0, 0.0, 0)


# ── Model loading ────────────────────────────────────────────────────────────────

def load_model(args) -> OpenTSLMSPFixPos2D:
    cfg = SimpleNamespace(
        is_pretraining=True,
        use_only_time_pe=False,
        use_only_channel_pe=False,
        use_freq_embed=getattr(args, "use_freq_embed", False),
        patch_size=args.patch_size,
        pe_type=getattr(args, "pe_type", "sincos"),
        pe_base_c=getattr(args, "pe_base_c", None),
        enc_dec_type=getattr(args, "enc_dec_type", "transformer"),
        mlp_dec_hidden_dim=args.mlp_dec_hidden_dim,
        mlp_dec_ff_dim=args.mlp_dec_ff_dim,
        mlp_enc_hidden_dim=args.mlp_enc_hidden_dim,
        mlp_enc_ff_dim=args.mlp_enc_ff_dim,
    )
    model_cls = (
        OpenTSLMSPFixPos2DTimerXL
        if "TimerXL" in args.ckpt
        else OpenTSLMSPFixPos2D
    )
    model = model_cls(cfg, llm_id=args.llm_id, device=args.device)

    # Peek at the checkpoint to decide whether this is a LoRA or full-LLM checkpoint.
    _ckpt_keys = torch.load(args.ckpt, map_location="cpu", weights_only=False).keys()
    if "llm_state" in _ckpt_keys:
        print("Detected full-LLM checkpoint — skipping LoRA init.")
    elif args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print(f"LoRA enabled: r={args.lora_r}  alpha={args.lora_alpha}")

    model.load_from_file(args.ckpt)
    model.eval()
    print(f"Loaded: {args.ckpt}")
    print(f"patch_size={model.patch_size}  eos='{model.get_eos_token()}'")
    return model


# ── Per-dataset evaluation ───────────────────────────────────────────────────────

def eval_one_dataset(
    model: OpenTSLMSPFixPos2D,
    ds_name: str,
    csv_path: str,
    cfg_ds: dict,
    args,
) -> Dict[int, dict]:
    max_cols = cfg_ds.get("max_cols") or args.max_cols

    def _collate(batch):
        batch = [s for s in batch if s is not None]
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=args.patch_size
        ) if batch else []

    results: Dict[int, dict] = {}

    for pred_len in args.pred_len:
        ds = DownstreamEvalDataset(
            csv_path=csv_path,
            mode=args.eval_split,
            patch_size=args.patch_size,
            pred_len=pred_len,
            history_len=args.history_len,
            max_cols=max_cols,
            stride=args.stride,
            split=cfg_ds["split"],
            ds_name=ds_name,
        )

        if len(ds) == 0:
            print(f"    pred={pred_len}  [no windows — skipped]")
            continue

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=0,
        )

        mse, mae, n = run_autoregressive_eval(
            model, loader, pred_len,
            patch_size=args.patch_size,
            max_new_tokens=args.max_new_tokens,
            force_template=args.force_template,
            logit_boost=args.logit_boost,
            debug_n=getattr(args, "debug_n", 0),
        )

        results[pred_len] = {"mse": mse, "mae": mae, "n": n}
        print(f"    pred={pred_len:4d}  n={n:6d}  mse={mse:.4f}  mae={mae:.4f}")

    return results


# ── Argument parsing ─────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Phase-2 downstream forecasting evaluation (autoregressive)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt",       required=True,
                    help="Path to best_model.pt from joint_phase2_downstream.")
    ap.add_argument("--llm_id",     default="google/gemma-3-270m")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--patch_size", type=int, default=32)

    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset",      choices=list(DATASET_CONFIGS.keys()),
                     help="Single dataset to evaluate.")
    grp.add_argument("--all_datasets", action="store_true",
                     help="Evaluate all 7 standard datasets.")

    ap.add_argument("--data_root",   default="/scratch/s223540177/TimeSeriesECM/dataset",
                    help="Root dir containing the downstream CSV files.")
    ap.add_argument("--csv",         default=None,
                    help="Explicit CSV path (only with --dataset).")

    ap.add_argument("--pred_len",    type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--history_len", type=int, default=None,
                    help="Context length. Defaults to Time-MoE values per pred_len.")
    ap.add_argument("--max_cols",    type=int, default=16,
                    help="Max columns to load (should match training --max_cols_wide).")
    ap.add_argument("--eval_split",  default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--stride",      type=int, default=None,
                    help="Window stride. Defaults to pred_len for val/test splits.")

    ap.add_argument("--lora_r",             type=int, default=16)
    ap.add_argument("--lora_alpha",         type=int, default=32)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_dec_ff_dim",     type=int, default=None)
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_enc_ff_dim",     type=int, default=None)
    ap.add_argument("--enc_dec_type",       default="transformer")
    ap.add_argument("--pe_type",            default="sincos")
    ap.add_argument("--use_freq_embed", action="store_true",
                    help="freq-branch checkpoints (joint_phase1_extend_balanced_freq lineage)")
    ap.add_argument("--pe_base_c", type=float, default=None)

    ap.add_argument("--force_template", action="store_true",
                    help="Force the <ts>/<fp>/</ts> template instead of relying on "
                         "the LLM to generate it.")
    ap.add_argument("--logit_boost", type=float, default=50.0,
                    help="Add this scalar to <ts> and </ts> logits before argmax. "
                         "<fp> tokens are always forced deterministically.")
    ap.add_argument("--max_new_tokens", type=int, default=512)

    ap.add_argument("--out_dir", default="./eval_results/phase2_downstream")
    ap.add_argument("--debug_n", type=int, default=0,
                    help="Print history/pred/gt for the first N windows per dataset.")
    return ap.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    model = load_model(args)

    datasets = list(DATASET_CONFIGS.keys()) if args.all_datasets else [args.dataset]

    print(f"\nmode=autoregressive  split={args.eval_split}  "
          f"pred_lens={args.pred_len}  patch_size={args.patch_size}\n")

    all_results: Dict[str, Dict[int, dict]] = {}

    for ds_name in datasets:
        cfg_ds   = DATASET_CONFIGS[ds_name]
        csv_path = (
            args.csv
            if (args.csv and not args.all_datasets)
            else os.path.join(args.data_root, cfg_ds["file"])
        )
        if not os.path.exists(csv_path):
            print(f"[{ds_name}] not found at {csv_path} — skipping")
            continue

        print(f"{'='*60}")
        print(f"  {ds_name}  ({csv_path})")
        print(f"{'='*60}")

        res = eval_one_dataset(model, ds_name, csv_path, cfg_ds, args)
        all_results[ds_name] = res

        ds_dir = os.path.join(args.out_dir, ds_name)
        os.makedirs(ds_dir, exist_ok=True)
        out_path = os.path.join(ds_dir, f"results_{args.eval_split}.json")
        with open(out_path, "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f, indent=2)
        print(f"  → {out_path}")

    if not all_results:
        print("\nNo results produced.")
        return

    print(f"\n{'='*72}")
    print(f"  Summary  mode=autoregressive  split={args.eval_split}")
    print(f"{'='*72}")
    print(f"  {'dataset':>12}  {'pred':>6}  {'n':>6}  {'MSE':>10}  {'MAE':>10}")
    print(f"  {'-'*52}")
    for ds_name, res in sorted(all_results.items()):
        for pl, m in sorted(res.items()):
            print(
                f"  {ds_name:>12}  {pl:>6}  {m['n']:>6}  "
                f"{m['mse']:>10.4f}  {m['mae']:>10.4f}"
            )

    summary_path = os.path.join(args.out_dir, f"summary_autoregressive_{args.eval_split}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {ds: {str(k): v for k, v in res.items()} for ds, res in all_results.items()},
            f, indent=2,
        )
    print(f"\n  Full summary → {summary_path}")


if __name__ == "__main__":
    from types import SimpleNamespace
    main()
