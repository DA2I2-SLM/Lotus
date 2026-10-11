#!/usr/bin/env python3
"""
OpenTSLM rolling-window forecasting evaluation — self-contained baseline.

Loads a saved OpenTSLM checkpoint and evaluates it on a local parquet dataset.
Uses direct parquet loading with its own data processing pipeline;
no dependency on PREPROCESSED_SPLITS_DIR, Phase1JointDataset, or
SingleSourcePhase1EvalDataset.

Usage:
    python baseline_run/opentslm_prediction.py \
        --ckpt $FTS_RESULTS_ROOT/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse/checkpoints/best_model.pt \
        --llm-id google/gemma-3-270m \
        --dataset-path /scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets/monash_australian_electricity \
        --patch-size 32 \
        --history-len 256 \
        --pred-len 32 \
        --eval-split test \
        --batch-size 16 \
        --out-dir ./eval_results/opentslm_baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── project imports (read-only — nothing modified in the main tree) ─────────────
_PROJ_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ── Own parquet data loading ─────────────────────────────────────────────────────

def _load_series_from_parquet(
    parquet_path: str,
    id_col: str = "id",
    ts_col: str = "timestamp",
    val_col: str = "target",
) -> Dict[str, np.ndarray]:
    """Load parquet → dict {series_id: ndarray of shape [T] or [T, C]}.

    For multivariate datasets (e.g. monash_australian_electricity),
    each row's `target` column is a list of C-element vectors, giving [T, C].
    For univariate datasets, `target` is a flat list giving [T].

    No channel-explosion: multivariate series stay [T, C] so the model
    receives the same input format it was trained on.
    """
    df = pd.read_parquet(parquet_path)
    missing = {id_col, ts_col, val_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in parquet: {sorted(missing)}")

    series: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        sid = str(row[id_col])
        targets = list(row[val_col])
        arr = np.array(targets, dtype=np.float32)   # [T] or [T, C]
        if arr.ndim == 0 or arr.shape[0] == 0:
            continue
        series[sid] = arr
    return series


class ParquetOpenTSLMDataset(Dataset):
    """Eval windows loaded directly from parquet, formatted for OpenTSLM.

    Applies an 80/10/10 temporal split and builds sliding windows within the
    requested split segment.  Each sample is a dict compatible with
    ``extend_time_series_to_match_patch_size_and_aggregate`` and the
    autoregressive inference loop below.

    Multivariate series are kept as [history_len, C] tensors so the model
    receives the cross-channel input it was pretrained on.
    """

    def __init__(
        self,
        parquet_path: str,
        mode: str,
        eos_token: str,
        patch_size: int,
        history_len: int,
        pred_len: int,
        stride: int = 1,
    ) -> None:
        if history_len % patch_size != 0:
            raise ValueError(f"history_len={history_len} not divisible by patch_size={patch_size}")
        if pred_len % patch_size != 0:
            raise ValueError(f"pred_len={pred_len} not divisible by patch_size={patch_size}")

        self.patch_size  = patch_size
        self.history_len = history_len
        self.pred_len    = pred_len
        self.eos_token   = eos_token

        raw = _load_series_from_parquet(parquet_path)

        window_len = history_len + pred_len
        # List of (array, absolute_start_index) pairs
        self._samples: List[Tuple[np.ndarray, int]] = []

        for arr in raw.values():
            T = arr.shape[0]
            train_end = int(T * 0.8)
            val_end   = int(T * 0.9)

            if mode == "train":
                lo, hi = 0, train_end
            elif mode == "val":
                lo, hi = train_end, val_end
            else:
                lo, hi = val_end, T

            seg_len = hi - lo
            for rel_start in range(0, max(1, seg_len - window_len + 1), stride):
                abs_start = lo + rel_start
                if abs_start + window_len <= T:
                    self._samples.append((arr, abs_start))

        n_ch = next(iter(raw.values())).ndim
        n_series_ch = next(iter(raw.values())).shape[1] if n_ch == 2 else 1
        print(
            f"[ParquetOpenTSLMDataset] mode={mode} series={len(raw)} "
            f"windows={len(self._samples)} channels={n_series_ch} "
            f"patch_size={patch_size} history_len={history_len} pred_len={pred_len}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        arr, start = self._samples[idx]
        window = arr[start: start + self.history_len + self.pred_len]
        if len(window) < self.history_len + self.pred_len:
            return None

        history_np = window[:self.history_len]   # [T] or [T, C]
        future_np  = window[self.history_len:]   # [P] or [P, C]

        history = torch.tensor(history_np, dtype=torch.float32)
        future  = torch.tensor(future_np,  dtype=torch.float32)

        if history.dim() == 1:
            # Univariate — scalar z-norm stats
            mean = history.mean()
            std  = history.std().clamp(min=1e-8)
            history_norm = (history - mean) / std
            future_norm  = (future  - mean) / std
            n_ch = 1
            prompt_text = (
                f"This is the time series, it has mean {mean.item():.4f}"
                f" and std {std.item():.4f}. <ts>"
            )
        else:
            # Multivariate [T, C] — per-channel z-norm (matches Phase1JointDataset._znorm)
            mean = history.mean(dim=0, keepdim=True)          # [1, C]
            std  = history.std(dim=0, keepdim=True).clamp(min=1e-8)
            history_norm = (history - mean) / std             # [T, C]
            future_norm  = (future  - mean) / std             # [P, C]
            n_ch = history.shape[1]
            prompt_text  = f"This is a {n_ch}-channel multivariate time series. <ts>"

        return {
            "pre_prompt":            "",
            "time_series":           [history_norm],
            "time_series_text":      [prompt_text],
            "post_prompt":           self.eos_token,
            "answer":                self.eos_token,
            "ts_future":             future_norm,
            "ts_already_normalized": True,
        }


# ── Model loading ────────────────────────────────────────────────────────────────

def load_model(
    ckpt: str,
    llm_id: str,
    device: str,
    patch_size: int,
    enc_dec_type: str = "transformer",
) -> OpenTSLMSPFixPos2D:
    cfg = SimpleNamespace(
        is_pretraining=True,
        patch_size=patch_size,
        enc_dec_type=enc_dec_type,
        use_only_time_pe=False,
        use_only_channel_pe=False,
        use_freq_embed=False,
        mlp_enc_hidden_dim=None,
        mlp_enc_ff_dim=None,
        mlp_dec_hidden_dim=None,
        mlp_dec_ff_dim=None,
    )
    model = OpenTSLMSPFixPos2D(cfg, llm_id=llm_id, device=device)
    model.load_from_file(ckpt)
    model.eval()
    return model


# ── Autoregressive inference (own clean reimplementation, LLM backbone) ──────────

@torch.no_grad()
def generate_forecast_patches(
    model: OpenTSLMSPFixPos2D,
    batch: List[dict],
    n_forecast_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Run autoregressive patch-by-patch forecasting via encoder → LLM → decoder.

    At each step the full prompt (extended with predicted patches so far) is
    re-encoded through the complete model pipeline, mirroring the training path.

    Returns:
        Tensor of shape [B, n_ch, n_forecast_patches, patch_size] in normalised space.
    """
    B = len(batch)

    # Shallow-copy each sample; materialise time_series as mutable cloned tensors.
    work_batch = []
    for s in batch:
        s2 = dict(s)
        ts_entries = []
        for t in s2["time_series"]:
            if isinstance(t, torch.Tensor):
                ts_entries.append(t.detach().clone().to(device))
            else:
                ts_entries.append(torch.as_tensor(t, dtype=torch.float32, device=device))
        s2["time_series"] = ts_entries
        lens = s2.get("time_series_lengths")
        if lens is not None:
            s2["time_series_lengths"] = [
                int(l.item() if torch.is_tensor(l) else l) for l in lens
            ]
        work_batch.append(s2)

    # Gradient checkpointing interferes with output_hidden_states — disable it.
    gc_was_on = getattr(model.llm, "is_gradient_checkpointing", False)
    if gc_was_on:
        model.llm.gradient_checkpointing_disable()

    preds: List[torch.Tensor] = []
    n_ch: int = 1

    try:
        for step in range(n_forecast_patches):
            # 1) Encode entire current history into LLM input embeddings.
            inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
                work_batch, return_targets=True
            )
            N_ts = model._last_ts_raw.size(0)
            n_ch = N_ts // B

            # 2) Full LLM forward pass.
            lm_out = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            h_last = lm_out.hidden_states[-1]    # [B, L_seq, H]
            del lm_out

            # 3) Collect decoder history buffers for every TS segment.
            #    TS tokens are laid out time-major: [t0c0, t0c1, ..., t1c0, ...],
            #    so channel c of batch sample b lives at ts_mask positions c::n_ch.
            hist_list: List[torch.Tensor] = []
            for i in range(N_ts):
                b_idx = i // n_ch
                c     = i % n_ch
                patch_pos = ts_mask[b_idx].nonzero(as_tuple=True)[0]
                if len(patch_pos) == 0:
                    h_seg = h_last[b_idx, -1:, :]
                elif n_ch == 1:
                    h_seg = h_last[b_idx, patch_pos, :]
                else:
                    ch_pos = patch_pos[c::n_ch]
                    h_seg = (
                        h_last[b_idx, ch_pos, :]
                        if len(ch_pos) > 0
                        else h_last[b_idx, -1:, :]
                    )
                h_enc = model.decoder_projector(h_seg.float())
                h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])
                hist_list.append(h_enc)
            del h_last

            # 4) Decoder predicts the next patch from the last sequence position.
            dec_in  = pad_sequence(hist_list, batch_first=True)   # [N_ts, N, enc_dim]
            dec_out = model.decoder(dec_in)                       # [N_ts, N, patch_size]
            pred_patch = dec_out[:, -1, :].float()                # [N_ts, patch_size]
            preds.append(pred_patch)

            if step == n_forecast_patches - 1:
                break

            # 5) Append predicted patch to each sample's time_series for the next step.
            P = pred_patch.size(-1)
            for b_idx in range(B):
                ts_list = work_batch[b_idx]["time_series"]
                ts0      = ts_list[0]
                # pred_patch rows: [n_ch, P], one per channel for this batch item
                pred_rows = pred_patch[b_idx * n_ch:(b_idx + 1) * n_ch].to(
                    dtype=ts0.dtype, device=ts0.device
                )
                if ts0.dim() == 1:
                    ts_list[0] = torch.cat([ts0, pred_rows[0]], dim=0)
                else:
                    # [n_ch, P] → [P, n_ch] for the [T, C] layout
                    ts_list[0] = torch.cat([ts0, pred_rows.t()], dim=0)
                lens = work_batch[b_idx].get("time_series_lengths")
                if lens is not None:
                    lens[0] = int(lens[0]) + P
    finally:
        if gc_was_on:
            model.llm.gradient_checkpointing_enable()

    out = torch.stack(preds, dim=1)                              # [N_ts, n_fp, patch_size]
    out = out.reshape(B, n_ch, n_forecast_patches, -1)           # [B, n_ch, n_fp, patch_size]
    return out


# ── Evaluation loop ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: OpenTSLMSPFixPos2D,
    loader: DataLoader,
    n_forecast_patches: int,
    patch_size: int,
    device: torch.device,
) -> dict:
    """Evaluate the model on all batches in `loader`.

    Returns mean/median MSE and MAE (in normalised space, averaged across channels).
    """
    all_mse: List[float] = []
    all_mae: List[float] = []

    for bi, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if not batch:
            continue

        try:
            pred_patches = generate_forecast_patches(
                model, batch, n_forecast_patches, device
            )
        except Exception as exc:
            print(f"[batch {bi}] inference failed: {exc}")
            continue

        # pred_patches: [B, n_ch, n_fp, patch_size] → [B, n_ch, pred_len]
        B, n_ch, n_fp, P = pred_patches.shape
        pred_flat = pred_patches.reshape(B, n_ch, -1).float()   # [B, n_ch, pred_len]

        for b, sample in enumerate(batch):
            gt = sample.get("ts_future")
            if gt is None:
                continue
            gt = torch.as_tensor(gt, dtype=torch.float32)

            # Normalise gt to [n_ch, pred_len]
            if gt.dim() == 1:
                gt = gt.unsqueeze(0)    # [1, pred_len]
            else:
                gt = gt.t()             # [pred_len, C] → [C, pred_len]

            pred_b = pred_flat[b]       # [n_ch, pred_len]

            # Broadcast if needed (MV pred, UV gt)
            if gt.size(0) < pred_b.size(0):
                gt = gt.expand(pred_b.size(0), -1)

            L = min(pred_b.size(1), gt.size(1))
            p = pred_b[:, :L]
            g = gt[:, :L].to(p.device)

            all_mse.append(F.mse_loss(p, g).item())
            all_mae.append(F.l1_loss(p, g).item())

        if bi % 20 == 0 and all_mse:
            t = torch.tensor(all_mse)
            print(
                f"[{bi:05d} | n={len(all_mse):5d}]  "
                f"median_mse={t.median():.4f}  mean_mse={t.mean():.4f}"
            )

    if not all_mse:
        return {"mean_mse": float("nan"), "median_mse": float("nan"),
                "mean_mae": float("nan"), "median_mae": float("nan"), "n": 0}

    t_mse = torch.tensor(all_mse)
    t_mae = torch.tensor(all_mae)
    return {
        "mean_mse":   t_mse.mean().item(),
        "median_mse": t_mse.median().item(),
        "mean_mae":   t_mae.mean().item(),
        "median_mae": t_mae.median().item(),
        "n_exploded": int((t_mse > 100).sum().item()),
        "n":          len(all_mse),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="OpenTSLM parquet-based forecasting baseline.")

    # Checkpoint
    ap.add_argument("--ckpt", required=True,
                    help="Path to best_model.pt checkpoint.")
    ap.add_argument("--llm-id", default="google/gemma-3-270m",
                    help="HuggingFace model ID for the LLM backbone (must match training).")

    # Data
    ap.add_argument(
        "--dataset-path",
        default="/scratch/Projects/rl_timeseries/pretraining_data/multivariate/"
                "chronos_datasets/monash_australian_electricity",
    )
    ap.add_argument("--parquet-filename", default="train-00000-of-00001.parquet")
    ap.add_argument("--eval-split",  default="test", choices=["train", "val", "test"])
    ap.add_argument("--stride",      type=int, default=32,
                    help="Sliding-window stride (timesteps). Default = pred-len (non-overlapping).")

    # Model config (must match training)
    ap.add_argument("--patch-size",    type=int, default=32)
    ap.add_argument("--history-len",   type=int, default=256)
    ap.add_argument("--pred-len",      type=int, default=32)
    ap.add_argument("--enc-dec-type",  default="transformer", choices=["transformer", "mlp"])

    # Inference
    ap.add_argument("--batch-size",    type=int, default=16)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    ap.add_argument("--out-dir",       default="./eval_results/opentslm_baseline")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)
    n_forecast_patches = args.pred_len // args.patch_size

    print(f"Loading model from: {args.ckpt}")
    model = load_model(
        ckpt=args.ckpt,
        llm_id=args.llm_id,
        device=str(device),
        patch_size=args.patch_size,
        enc_dec_type=args.enc_dec_type,
    )
    eos_token = model.get_eos_token()

    parquet_path = Path(args.dataset_path) / args.parquet_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    print(f"Loading dataset from: {parquet_path}")

    stride = args.stride if args.stride > 0 else args.pred_len
    dataset = ParquetOpenTSLMDataset(
        parquet_path=str(parquet_path),
        mode=args.eval_split,
        eos_token=eos_token,
        patch_size=args.patch_size,
        history_len=args.history_len,
        pred_len=args.pred_len,
        stride=stride,
    )

    def _collate(batch):
        batch = [s for s in batch if s is not None]
        if not batch:
            return []
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=args.patch_size
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
    )

    print(
        f"\nEvaluating: split={args.eval_split}  windows={len(dataset)}"
        f"  n_forecast_patches={n_forecast_patches}"
        f"  patch_size={args.patch_size}  history_len={args.history_len}"
        f"  pred_len={args.pred_len}\n"
    )

    metrics = run_evaluation(
        model=model,
        loader=loader,
        n_forecast_patches=n_forecast_patches,
        patch_size=args.patch_size,
        device=device,
    )

    print(f"\n{'='*60}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Split      : {args.eval_split}")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print(f"{'='*60}\n")

    out_path = os.path.join(args.out_dir, f"metrics_{args.eval_split}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
