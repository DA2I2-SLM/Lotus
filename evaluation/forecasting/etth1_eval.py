#!/usr/bin/env python3
"""
Clean ETTh1 multivariate forecasting evaluator for OpenTSLMSPFixPos2D
checkpoints trained with joint_phase1_extend_balanced.

Design: feed ETTh1 in the SAME format training uses for multivariate
Chronos samples — one [L, C] tensor + one MV prompt — so the encoder's
2D positional encoding, the time-major token flatten, and the decoder
reorder all run on the path they were trained for.

Modes:
  --mode autoregressive  (default)
      Feed history only. Run encoder → LLM → decoder once per forecast
      patch, appending each predicted patch to the raw sequence and
      re-running. Honest end-to-end forecasting.

  --mode teacher_forced
      Feed [history + ground_truth_future] as a single sequence. The
      decoder predicts patch t+1 from LLM hidden state at position t.
      Only the FIRST future-patch prediction is leak-free; later patches
      condition on ground-truth future via causal attention — so this
      is an upper bound on decoder quality, not a realistic metric.

Outputs:
  out_dir/metrics.json       summary MSE/MAE (normalized + original units)
  out_dir/per_sample.csv     per-sample metrics
  out_dir/plots/*.png        forecast vs ground-truth plots (per-channel)
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


ETTH1_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — matches training MV Chronos format exactly
# ─────────────────────────────────────────────────────────────────────────────


class ETTh1MVDataset(Dataset):
    """Multivariate ETTh1 windows, formatted identically to training Chronos MV.

    Each sample dict has:
      - time_series:      [ tensor [H, C] ]   (single MV entry)
      - time_series_text: [ "This is a C-channel multivariate time series. <ts>" ]
      - pre_prompt / post_prompt / answer: empty strings (like training chronos rows)
      - ts_already_normalized: True
      - norm_mean_per_ch / norm_std_per_ch: per-channel history stats (for denorm)
      - future_norm:  [P, C] ground truth in normalized space
      - future_raw:   [P, C] ground truth in original units
      - history_raw:  [H, C] original history (for plotting)
    """

    def __init__(
        self,
        csv_path: str,
        split: str,
        patch_size: int,
        history_len: int,
        pred_len: int,
        stride: int = 1,
        split_ratios=(0.6, 0.2, 0.2),
    ):
        if history_len % patch_size != 0:
            raise ValueError(f"history_len={history_len} not divisible by patch_size={patch_size}")
        if pred_len % patch_size != 0:
            raise ValueError(f"pred_len={pred_len} not divisible by patch_size={patch_size}")

        self.patch_size = patch_size
        self.history_len = history_len
        self.pred_len = pred_len

        df = pd.read_csv(csv_path)
        N = len(df)
        tr_end = int(N * split_ratios[0])
        va_end = int(N * (split_ratios[0] + split_ratios[1]))
        if split == "train":
            df = df.iloc[:tr_end]
        elif split == "val":
            df = df.iloc[tr_end:va_end]
        elif split == "test":
            df = df.iloc[va_end:]
        else:
            raise ValueError(f"unknown split: {split}")

        cols = [c for c in ETTH1_COLS if c in df.columns]
        self._cols = cols
        self._series = torch.tensor(df[cols].values, dtype=torch.float32)  # [T, C]

        window = history_len + pred_len
        self._starts = list(range(0, self._series.size(0) - window + 1, stride))

        print(
            f"[ETTh1MVDataset] split={split} rows={len(df)} "
            f"windows={len(self._starts)} channels={cols} "
            f"(H={history_len} P={pred_len} patch={patch_size})"
        )

    def __len__(self):
        return len(self._starts)

    def __getitem__(self, idx: int) -> Dict:
        s = self._starts[idx]
        H, P = self.history_len, self.pred_len
        hist = self._series[s: s + H]          # [H, C]
        fut = self._series[s + H: s + H + P]   # [P, C]

        m = hist.mean(dim=0)                    # [C]
        sd = hist.std(dim=0).clamp(min=1e-8)   # [C]
        hist_n = (hist - m) / sd               # [H, C]
        fut_n = (fut - m) / sd                 # [P, C]

        C = hist.size(1)
        return {
            "type": "chronos",
            "pre_prompt": "",
            "time_series": [hist_n],
            "time_series_text": [f"This is a {C}-channel multivariate time series. <ts>"],
            "post_prompt": "",
            "answer": "",
            "ts_already_normalized": True,
            # Ch0 stats expected by pad_and_apply_batch; full per-channel stats kept separately.
            "norm_mean": float(m[0].item()),
            "norm_std": float(sd[0].item()),
            "time_series_norm_stats": [(float(m[0].item()), float(sd[0].item()))],
            # Eval-only extras (not consumed by pad_and_apply_batch):
            "norm_mean_per_ch": m,
            "norm_std_per_ch": sd,
            "future_norm": fut_n,
            "future_raw": fut,
            "history_raw": hist,
            "sample_idx": idx,
        }


def _collate(batch, patch_size):
    batch = [s for s in batch if s is not None]
    if not batch:
        return []
    return extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=patch_size
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared decoder step: LLM last_hidden → channel-first reorder → decoder
# (Mirrors compute_joint_phase1_loss exactly.)
# ─────────────────────────────────────────────────────────────────────────────


def _decode_from_hidden(
    model,
    last_hidden: torch.Tensor,   # [B, L, H]
    ts_mask: torch.Tensor,       # [B, L] bool
    n_ch: int,
    patch_counts_per_row: List[int],   # length = B*n_ch
) -> torch.Tensor:
    """Return decoder predictions [N_ts, N_patches, patch_size] (channel-first)."""
    B = ts_mask.size(0)
    ts_hidden = last_hidden[ts_mask]   # [B * n_ch * T, H]  time-major flat

    if n_ch > 1:
        # Reorder time-major → channel-first per sample (see training code).
        parts = []
        off = 0
        T_per_sample = [patch_counts_per_row[j * n_ch] for j in range(B)]
        for j in range(B):
            T_j = T_per_sample[j]
            chunk = ts_hidden[off: off + T_j * n_ch]          # [T*C, H]
            chunk = chunk.reshape(T_j, n_ch, -1).permute(1, 0, 2).reshape(T_j * n_ch, -1)
            parts.append(chunk)
            off += T_j * n_ch
        ts_hidden = torch.cat(parts, dim=0)

    assert ts_hidden.size(0) == int(sum(patch_counts_per_row)), (
        f"ts_hidden has {ts_hidden.size(0)} rows, expected {sum(patch_counts_per_row)}"
    )

    h = model.decoder_projector(ts_hidden.float())
    h = F.layer_norm(h, [h.size(-1)])
    h3d = pad_sequence(list(torch.split(h, patch_counts_per_row)), batch_first=True)
    return model.decoder(h3d)   # [N_ts, N_patches, patch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Teacher-forced evaluation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_teacher_forced(model, loader, patch_size: int, pred_len: int, device):
    """Feed [history + future] as the MV series; read decoder predictions at
    positions [Nh-1 ... Nh+Nf-2] to reconstruct future patches 0..Nf-1.

    NOTE: only the first future-patch prediction is leak-free; later positions
    condition on ground-truth future patches via causal attention.
    """
    n_fut = pred_len // patch_size
    results: List[Dict] = []

    for batch in tqdm(loader, desc="teacher_forced"):
        if not batch:
            continue

        # Build a DIFFERENT per-sample dict where time_series = [concat(hist, fut)].
        tf_batch = []
        for s in batch:
            hist = s["time_series"][0] if isinstance(s["time_series"], list) else s["time_series"][0]
            # After collate, time_series is stacked to [n_ts, L] or [n_ts, L, C].
            # With our dataset there's one MV entry → stacked shape [1, H, C].
            if hist.dim() == 3:
                hist = hist[0]       # [H, C]
            fut_n = s["future_norm"]  # [P, C]
            full = torch.cat([hist, fut_n], dim=0)  # [H+P, C]
            tf_batch.append({
                **s,
                "time_series": [full],
                "time_series_text": s["time_series_text"],
                "ts_already_normalized": True,
            })

        tf_batch = extend_time_series_to_match_patch_size_and_aggregate(
            tf_batch, patch_size=patch_size
        )

        (
            inputs_embeds,
            attn_mask,
            _, ts_mask,
            _, ts_valid,
            _,
        ) = model.pad_and_apply_batch(tf_batch, return_targets=True)

        out = model.llm(
            inputs_embeds=inputs_embeds.to(device),
            attention_mask=attn_mask.to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]

        n_ch = getattr(model, "_last_n_ch_chronos", 1)
        patch_counts = (ts_valid.sum(dim=1) + 1).tolist()

        dec = _decode_from_hidden(model, last_hidden, ts_mask, n_ch, patch_counts)
        # dec shape: [N_ts, N_patches, P]; N_ts = B*C; N_patches = (H+P)/patch_size.
        B = len(batch)
        n_hist = batch[0]["history_raw"].size(0) // patch_size

        pred_fut = dec[:, n_hist - 1: n_hist - 1 + n_fut, :]  # [B*C, n_fut, P]
        pred_fut = pred_fut.reshape(B, n_ch, n_fut, patch_size)

        for b, s in enumerate(batch):
            _accumulate_metrics(s, pred_fut[b], results, patch_size)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive evaluation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def eval_autoregressive(model, loader, patch_size: int, pred_len: int, device):
    """Feed history only. Predict patch, append, repeat."""
    n_fut = pred_len // patch_size
    results: List[Dict] = []

    for batch in tqdm(loader, desc="autoregressive"):
        if not batch:
            continue

        # Work on a copy we can mutate.
        cur_batch = []
        for s in batch:
            ts0 = s["time_series"][0] if isinstance(s["time_series"], list) else s["time_series"][0]
            if ts0.dim() == 3:
                ts0 = ts0[0]   # [H, C]
            cur_batch.append({**s, "time_series": [ts0.clone()], "ts_already_normalized": True})

        B = len(cur_batch)
        n_ch = cur_batch[0]["time_series"][0].size(1)
        pred_patches = torch.zeros(B, n_ch, n_fut, patch_size, device=device)

        for step in range(n_fut):
            coll = extend_time_series_to_match_patch_size_and_aggregate(
                [dict(s) for s in cur_batch], patch_size=patch_size
            )

            (
                inputs_embeds,
                attn_mask,
                _, ts_mask,
                _, ts_valid,
                _,
            ) = model.pad_and_apply_batch(coll, return_targets=True)

            out = model.llm(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=attn_mask.to(device),
                output_hidden_states=True,
                use_cache=False,
            )
            last_hidden = out.hidden_states[-1]
            n_ch_model = getattr(model, "_last_n_ch_chronos", 1)
            assert n_ch_model == n_ch, f"{n_ch_model} != {n_ch}"

            patch_counts = (ts_valid.sum(dim=1) + 1).tolist()
            dec = _decode_from_hidden(model, last_hidden, ts_mask, n_ch, patch_counts)
            # Prediction for next patch = decoder output at the LAST history position.
            # dec is channel-first: row c*B + b → sample b, channel c. Per-sample T is same.
            T_cur = patch_counts[0]   # assumes all samples equal-length (ETTh1 sliding windows)
            last_pred = dec[:, T_cur - 1, :]                     # [B*C, P]
            last_pred = last_pred.reshape(B, n_ch, patch_size)   # [B, C, P]
            pred_patches[:, :, step, :] = last_pred

            # Append prediction to each sample's history.
            for b in range(B):
                prev = cur_batch[b]["time_series"][0]             # [L_b, C]
                new_patch = last_pred[b].t().to(prev.dtype).cpu() # [P, C]
                cur_batch[b]["time_series"] = [torch.cat([prev, new_patch], dim=0)]

        for b, s in enumerate(batch):
            _accumulate_metrics(s, pred_patches[b], results, patch_size)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metric accumulation + plotting
# ─────────────────────────────────────────────────────────────────────────────


def _accumulate_metrics(sample, pred_per_ch_patched, results, patch_size):
    """pred_per_ch_patched: [C, n_fut, P] on device."""
    pred_norm = pred_per_ch_patched.reshape(pred_per_ch_patched.size(0), -1).float().cpu()   # [C, pred_len]
    gt_norm = sample["future_norm"].t()    # [C, P]
    L = min(pred_norm.size(1), gt_norm.size(1))
    pred_norm = pred_norm[:, :L]
    gt_norm = gt_norm[:, :L]

    m = sample["norm_mean_per_ch"].unsqueeze(1)   # [C, 1]
    sd = sample["norm_std_per_ch"].unsqueeze(1)   # [C, 1]
    pred_raw = pred_norm * sd + m
    gt_raw = sample["future_raw"].t()[:, :L]

    results.append({
        "sample_idx": int(sample.get("sample_idx", -1)),
        "mse_norm": float(F.mse_loss(pred_norm, gt_norm).item()),
        "mae_norm": float(F.l1_loss(pred_norm, gt_norm).item()),
        "mse_raw":  float(F.mse_loss(pred_raw, gt_raw).item()),
        "mae_raw":  float(F.l1_loss(pred_raw, gt_raw).item()),
        "pred_norm": pred_norm,
        "gt_norm":   gt_norm,
        "pred_raw":  pred_raw,
        "gt_raw":    gt_raw,
        "history_raw": sample["history_raw"],   # [H, C]
    })


def _plot_sample(rec, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(out_dir, exist_ok=True)
    C = rec["pred_norm"].size(0)
    H = rec["history_raw"].size(0)
    P = rec["pred_norm"].size(1)

    fig, axes = plt.subplots(C, 1, figsize=(10, 2.0 * C), sharex=True)
    if C == 1:
        axes = [axes]
    for c in range(C):
        ax = axes[c]
        hist_c = rec["history_raw"][:, c].numpy()
        gt_c = rec["gt_raw"][c].numpy()
        pr_c = rec["pred_raw"][c].numpy()
        ax.plot(range(H), hist_c, color="steelblue", lw=0.8, label="history")
        ax.plot(range(H, H + P), gt_c, color="green", lw=1.0, label="gt")
        ax.plot(range(H, H + P), pr_c, color="orange", lw=1.0, label="pred")
        ax.axvline(H - 0.5, color="grey", ls="--", lw=0.6)
        ax.set_ylabel(f"ch{c}")
        if c == 0:
            ax.set_title(f"{tag} — sample {rec['sample_idx']} (raw units)")
        if c == C - 1:
            ax.set_xlabel("timestep")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_sample_{rec['sample_idx']:06d}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _summarize_and_save(results, out_dir, tag, plot_every):
    if not results:
        print("[warn] no results to summarize")
        return

    mse_n = sum(r["mse_norm"] for r in results) / len(results)
    mae_n = sum(r["mae_norm"] for r in results) / len(results)
    mse_r = sum(r["mse_raw"] for r in results) / len(results)
    mae_r = sum(r["mae_raw"] for r in results) / len(results)

    print(f"\n=== {tag} ===")
    print(f"  N samples       : {len(results)}")
    print(f"  MSE (normalized): {mse_n:.4f}")
    print(f"  MAE (normalized): {mae_n:.4f}")
    print(f"  MSE (raw units) : {mse_r:.4f}")
    print(f"  MAE (raw units) : {mae_r:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "mode": tag, "n": len(results),
            "mse_norm": mse_n, "mae_norm": mae_n,
            "mse_raw": mse_r, "mae_raw": mae_r,
        }, f, indent=2)

    per_sample = [{k: v for k, v in r.items() if not isinstance(v, torch.Tensor)} for r in results]
    pd.DataFrame(per_sample).to_csv(os.path.join(out_dir, "per_sample.csv"), index=False)

    if plot_every > 0:
        plot_dir = os.path.join(out_dir, "plots")
        for i, r in enumerate(results):
            if i % plot_every == 0:
                _plot_sample(r, plot_dir, tag)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(args, device):
    model_args = argparse.Namespace(
        patch_size=args.patch_size,
        pe_type=args.pe_type,
        use_only_time_pe=False,
        use_only_channel_pe=False,
    )
    model = OpenTSLMSPFixPos2D(model_args, llm_id=args.llm_id, device=device)
    model.load_from_file(args.ckpt)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--etth1_csv", required=True)
    ap.add_argument("--patch_size", type=int, required=True,
                    help="MUST match the patch_size the checkpoint was trained with.")
    ap.add_argument("--history_len", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--eval_split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--mode", choices=["autoregressive", "teacher_forced", "both"],
                    default="autoregressive")
    ap.add_argument("--pe_type", default="sincos")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=-1,
                    help="Limit number of samples for quick sanity runs (-1 = all).")
    ap.add_argument("--plot_every", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    ds = ETTh1MVDataset(
        csv_path=args.etth1_csv,
        split=args.eval_split,
        patch_size=args.patch_size,
        history_len=args.history_len,
        pred_len=args.pred_len,
        stride=args.stride,
    )
    if args.max_samples > 0:
        ds._starts = ds._starts[:args.max_samples]
        print(f"[subset] using first {len(ds._starts)} windows for quick eval")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: _collate(b, patch_size=args.patch_size),
    )

    model = _load_model(args, device)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode in ("teacher_forced", "both"):
        res = eval_teacher_forced(model, loader, args.patch_size, args.pred_len, device)
        _summarize_and_save(res, os.path.join(args.out_dir, "teacher_forced"),
                            "teacher_forced", args.plot_every)

    if args.mode in ("autoregressive", "both"):
        res = eval_autoregressive(model, loader, args.patch_size, args.pred_len, device)
        _summarize_and_save(res, os.path.join(args.out_dir, "autoregressive"),
                            "autoregressive", args.plot_every)


if __name__ == "__main__":
    main()
