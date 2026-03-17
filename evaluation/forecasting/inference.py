#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Autoregressive forecasting evaluation for OpenTSLM.

Two modes:
  --mode autoregressive  (default)
      Runs model.generate() with n_forecast_patches. Each predicted patch is
      fed back as input to produce the next patch — identical to inference.
      Metrics: MSE / MAE on the full forecast horizon.

  --mode teacher_forced
      Feeds all ground-truth history to the LLM in one forward pass, then
      reads hidden states at each TS position and decodes via the MLP head.
      Measures decoder quality; not a realistic forecasting benchmark.

Usage:
  python inference.py --ckpt /path/best_model.pt --llm_id meta-llama/Llama-3.2-1B
  python inference.py --ckpt /path/best_model.pt --llm_id ... --mode teacher_forced
"""

import argparse
import json
import os
import re
import random
from types import SimpleNamespace
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.ChronosDataset import (
    BASE_DATA_PATH,
    PREPROCESSED_SPLITS_DIR,
    ChronosDataset,
    _last_value_impute,
)
from opentslm.time_series_datasets.Phase1JointDataset import (
    Phase1JointDataset,
    Phase1JointExtendDataset,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


# ── Dataset helpers (unchanged from original) ─────────────────────────────────

class ChronosOnlySubset(Dataset):
    """Wraps Phase1JointDataset / Phase1JointExtendDataset test split,
    exposing only the Chronos samples for forecasting evaluation."""

    def __init__(self, joint_ds):
        self._indices = [
            i for i, row in enumerate(joint_ds.rows)
            if isinstance(row, dict) and row.get("type") == "chronos"
        ]
        if not self._indices:
            raise RuntimeError("No chronos stubs found in the joint dataset test split.")
        self._joint_ds = joint_ds
        print(f"[ChronosOnlySubset] {len(self._indices)} chronos samples "
              f"(out of {len(joint_ds.rows)} total).")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._joint_ds[self._indices[idx]]


TEST_FRAC = 0.1
VAL_FRAC  = 0.1
TRAIN_ALIGNED_HISTORY_LENS = [96, 192, 384, 416]
TRAIN_ALIGNED_PRED_LEN     = 96


class TrainingAlignedChronosEvalDataset(ChronosDataset):
    """Chronos eval dataset that mirrors the joint-training forecasting prompt path.

    Each sample exposes:
      time_series  — normalized history patches (fed to encoder)
      ts_future    — normalized future values  (ground truth for MSE)
      answer       — 'Answer: <ts> v1 v2 ... </ts>' (for teacher-forced mode)
    """

    CACHE_SUFFIX = "_train_aligned_eval"

    def __init__(
        self,
        mode: str,
        EOS_TOKEN: str,
        dataset_name: str = "all",
        base_path: str = BASE_DATA_PATH,
        patch_size: int = 4,
        pred_len: int = TRAIN_ALIGNED_PRED_LEN,
        history_lens: List[int] | None = None,
    ):
        self.patch_size   = int(patch_size)
        self.pred_len     = int(pred_len)
        self.history_lens = [int(h) for h in (history_lens or TRAIN_ALIGNED_HISTORY_LENS)]
        self.valid_history_lens = [
            h for h in self.history_lens if h > 0 and h % self.patch_size == 0
        ]
        if not self.valid_history_lens:
            raise ValueError(
                f"No valid history lengths for patch_size={self.patch_size}: {self.history_lens}"
            )

        cache_key = (
            dataset_name, base_path,
            self.patch_size, self.pred_len,
            tuple(self.valid_history_lens),
        )
        if getattr(self.__class__, "_cache_key", None) != cache_key:
            for attr in ("loaded", "_train_rows", "_val_rows", "_test_rows"):
                if hasattr(self.__class__, attr):
                    delattr(self.__class__, attr)
            self.__class__._cache_key = cache_key

        super().__init__(
            mode=mode,
            EOS_TOKEN=EOS_TOKEN,
            dataset_name=dataset_name,
            base_path=base_path,
            max_seq_len=max(self.valid_history_lens) + self.pred_len,
        )

    def _load_splits(self):
        training_split_dir = os.path.join(PREPROCESSED_SPLITS_DIR, "all")
        done_flag = os.path.join(training_split_dir, "_DONE")
        if not os.path.exists(done_flag):
            raise RuntimeError(
                f"Training Chronos splits not found at {training_split_dir}. "
                "Run ChronosDataset with dataset_name='all' first."
            )
        train = load_from_disk(os.path.join(training_split_dir, "train"), keep_in_memory=False)
        val   = load_from_disk(os.path.join(training_split_dir, "val"),   keep_in_memory=False)
        test  = load_from_disk(os.path.join(training_split_dir, "test"),  keep_in_memory=False)
        return train, val, test

    def _process_row(self, row):
        series = torch.tensor(row["target"], dtype=torch.float32)
        T = len(series)

        candidates = [
            h for h in self.valid_history_lens if T >= (h + self.pred_len)
        ]
        if not candidates:
            raise ValueError(
                f"Series too short: len={T}, "
                f"need >= {min(self.valid_history_lens) + self.pred_len}"
            )

        history_len = (
            random.choice(candidates) if self.mode == "train" else max(candidates)
        )
        window_needed = history_len + self.pred_len
        start = (
            random.randint(0, T - window_needed) if self.mode == "train"
            else T - window_needed
        )

        window      = _last_value_impute(series[start : start + window_needed])
        history     = window[:history_len]
        future      = window[history_len:]

        mean = history.mean()
        std  = history.std().clamp(min=1e-8)
        history_norm = (history - mean) / std
        future_norm  = (future  - mean) / std

        return history_norm, future_norm, history_len // self.patch_size, self.pred_len // self.patch_size

    def _get_answer(self, row) -> str:
        _, future_norm, _, _ = self._process_row(row)
        future_str = " ".join(f"{v:.4f}" for v in future_norm.tolist())
        return f"Answer: <ts> {future_str} </ts>"

    def _get_pre_prompt(self, row) -> str:
        _, _, n_hist_patches, _ = self._process_row(row)
        return f"Given {n_hist_patches} time series patches"

    def _get_post_prompt(self, row) -> str:
        _, _, _, n_fut_patches = self._process_row(row)
        return f"</ts>, please predict the next {n_fut_patches} patches"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        history_norm, _, _, _ = self._process_row(row)
        return [TextTimeSeriesPrompt("<ts>", history_norm.tolist())]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Attach ts_future so both evaluation modes can access it without
        # parsing the answer string.
        row = self._get_split()[idx]
        _, future_norm, _, _ = self._process_row(row)
        sample["ts_future"] = future_norm
        return sample

    def _get_split(self):
        if self.mode == "train":
            return self._train_rows
        elif self.mode == "val":
            return self._val_rows
        return self._test_rows


# ── Autoregressive evaluation ─────────────────────────────────────────────────

def _parse_future_from_answer(answer: str) -> torch.Tensor | None:
    """Extract float values from 'Answer: <ts> v1 v2 ... </ts>'."""
    m = re.search(r'<ts>(.*?)</ts>', answer, re.DOTALL)
    if not m:
        return None
    try:
        vals = [float(v) for v in m.group(1).strip().split()]
        return torch.tensor(vals, dtype=torch.float32)
    except ValueError:
        return None


@torch.no_grad()
def _generate_patches_direct(model, batch, n_forecast_patches, device):
    """
    Direct patch-by-patch autoregressive forecasting.

    Bypasses text generation entirely — only n_forecast_patches LLM steps
    per sample instead of ~100 with model.generate().

    Pipeline per step:
      causal encoder (full growing context) → projector → LLM (KV-cache)
      → decoder_projector → MLP decoder → predicted next patch → feed back
    """
    B = len(batch)

    # ── 1. History prompt → LLM KV-cache ──────────────────────────────────────
    inputs_embeds, attn_mask, _, _, _, _, _ = model.pad_and_apply_batch(
        batch, return_targets=True
    )
    prompt_out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_kv     = prompt_out.past_key_values
    last_hidden = prompt_out.hidden_states[-1][:, -1, :]    # [B, H]

    h_dec = model.decoder_projector(last_hidden.float())    # [B, enc_dim]
    h_dec = F.layer_norm(h_dec, [h_dec.size(-1)])

    # ── 2. Raw-history buffer for causal encoder re-run ────────────────────────
    # After collation, time_series is a tensor [num_series, padded_len].
    # Before collation (edge case) it may still be a list of lists.
    raw_hist_list = []
    for b_idx in range(B):
        ts_segs = batch[b_idx].get("time_series", [])
        if isinstance(ts_segs, torch.Tensor) and ts_segs.numel() > 0:
            ts_b = ts_segs[0].float()                   # [padded_len] — already patch-aligned
        elif isinstance(ts_segs, (list, tuple)) and len(ts_segs) > 0:
            ts_b = torch.as_tensor(ts_segs[0], dtype=torch.float32)
            if ts_b.dim() > 1:
                ts_b = ts_b.squeeze(-1)
        else:
            raw_hist_list.append(torch.zeros(0, device=device))
            continue
        # Dataset already z-normalises; re-normalising is a near-no-op but keeps
        # the encoder input in the same space as during training.
        m_b, s_b = ts_b.mean(), ts_b.std().clamp(min=1e-8)
        ts_b = (ts_b - m_b) / s_b
        rem = ts_b.numel() % model.patch_size
        if rem:
            ts_b = torch.cat([ts_b, ts_b.new_zeros(model.patch_size - rem)])
        raw_hist_list.append(ts_b.to(device))

    L_max = max(t.numel() for t in raw_hist_list) if raw_hist_list else 0
    rem_h = L_max % model.patch_size
    if rem_h:
        L_max += model.patch_size - rem_h
    raw_hist      = torch.stack([F.pad(t, (0, L_max - t.numel())) for t in raw_hist_list]).to(device)
    generated_raw = raw_hist.new_zeros(B, 0, model.patch_size)

    # ── 3. Patch loop — n_forecast_patches LLM steps ──────────────────────────
    all_patches = []
    for step in range(n_forecast_patches):
        patch_pred = model.decoder(h_dec)                   # [B, patch_size]
        all_patches.append(patch_pred.detach().cpu())

        if step == n_forecast_patches - 1:
            break

        # Append predicted patch to growing buffer; re-run causal encoder
        generated_raw = torch.cat([generated_raw, patch_pred.detach().unsqueeze(1)], dim=1)
        full_seq      = torch.cat([raw_hist, generated_raw.reshape(B, -1)], dim=1)
        enc           = model.encoder(full_seq, chan_id=torch.zeros(B, device=device, dtype=torch.long))
        emb           = model.projector(enc[:, -1:, :]).to(inputs_embeds.dtype)  # [B, 1, H]

        # Single LLM KV-cache step
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones(B, 1)], dim=1)
        out = model.llm(
            inputs_embeds=emb,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv    = out.past_key_values
        new_hidden = out.hidden_states[-1][:, -1, :]        # [B, H]

        h_dec = model.decoder_projector(new_hidden.float())
        h_dec = F.layer_norm(h_dec, [h_dec.size(-1)])

    # [n_forecast_patches, B, patch_size] → [B, n_forecast_patches, patch_size]
    return torch.stack(all_patches, dim=0).transpose(0, 1)


@torch.no_grad()
def run_autoregressive(model, loader, n_forecast_patches, out_dir, save_plots_every, patch_size: int = 1):
    """
    Autoregressive forecasting via direct encoder→LLM→decoder pipeline.
    Only n_forecast_patches LLM forward steps per sample — no text generation.

    Reports mean AND median MSE/MAE.  Mean is sensitive to catastrophic
    failures on a few series; median reflects typical model behaviour.
    """
    device = model.device
    all_mse, all_mae = [], []
    all_results = []

    for bi, batch in enumerate(tqdm(loader, desc="Autoregressive")):
        pred_patches = _generate_patches_direct(model, batch, n_forecast_patches, device)
        # [B, n_forecast_patches, patch_size]

        for b, sample in enumerate(batch):
            pred_flat = pred_patches[b].reshape(-1).float()

            gt_flat = sample.get("ts_future", None)
            if gt_flat is None:
                gt_flat = _parse_future_from_answer(sample.get("answer", ""))
            if gt_flat is None:
                continue
            gt_flat = gt_flat.float()

            L      = min(pred_flat.size(0), gt_flat.size(0))
            pred_v = pred_flat[:L]
            gt_v   = gt_flat[:L].to(pred_v.device)

            # Extract history for plotting (already normalised by dataset)
            ts_segs = sample.get("time_series", None)
            if isinstance(ts_segs, torch.Tensor) and ts_segs.numel() > 0:
                hist_v = ts_segs[0].float().cpu().tolist()
            else:
                hist_v = []

            mse = F.mse_loss(pred_v, gt_v).item()
            mae = F.l1_loss(pred_v, gt_v).item()
            all_mse.append(mse)
            all_mae.append(mae)

            all_results.append({
                "mse": mse, "mae": mae,
                "pred": pred_v.cpu().tolist(),
                "gt":   gt_v.cpu().tolist(),
                "history": hist_v,
            })

        if bi % save_plots_every == 0 and all_results:
            _save_plot(all_results[-1], out_dir, step=bi, tag="autoregressive",
                       patch_size=patch_size)

        if bi % 20 == 0 and all_mse:
            t = torch.tensor(all_mse)
            print(f"[batch {bi:05d} | n={len(all_mse):6d}]  "
                  f"median_mse={t.median():.4f}  mean_mse={t.mean():.4f}  "
                  f"max_mse={t.max():.2e}")

    t_mse = torch.tensor(all_mse) if all_mse else torch.zeros(1)
    t_mae = torch.tensor(all_mae) if all_mae else torch.zeros(1)
    return {
        "mean_mse":   t_mse.mean().item(),
        "median_mse": t_mse.median().item(),
        "mean_mae":   t_mae.mean().item(),
        "median_mae": t_mae.median().item(),
        "n_exploded": int((t_mse > 100).sum().item()),   # series where model completely failed
        "n":          len(all_mse),
    }, all_results


# ── Teacher-forced evaluation (kept for comparison) ───────────────────────────

@torch.no_grad()
def run_teacher_forced(model, loader, out_dir, save_plots_every, patch_size: int = 1):
    """
    Teacher-forced next-patch prediction: feeds all ground-truth history to the
    LLM, extracts hidden states at TS positions, and decodes via the MLP head.
    Measures decoder quality in isolation — not a realistic forecasting setup.
    """
    total_mse = total_mae = total_n = 0
    all_results = []

    for bi, batch in enumerate(tqdm(loader, desc="Teacher-forced")):
        (
            inputs_embeds,
            attention_mask,
            _,
            ts_mask,
            ts_patch_targets_shifted,
            ts_patch_valid,
            _,
        ) = model.pad_and_apply_batch(batch, return_targets=True)

        if ts_patch_targets_shifted is None or ts_patch_targets_shifted.numel() == 0:
            continue

        outputs = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
        ts_hidden = outputs.hidden_states[-1][ts_mask]     # [N_ts_total*N_patches, H]

        h_enc     = model.decoder_projector(ts_hidden.float())
        h_enc     = F.layer_norm(h_enc, [h_enc.size(-1)])
        pred_flat = model.decoder(h_enc)                   # [N_ts_total*N_patches, P]
        pred      = pred_flat.reshape(N_ts_total, N_patches, P)
        tgt       = ts_patch_targets_shifted
        valid     = ts_patch_valid

        pred_v = pred.reshape(-1, P)[valid.reshape(-1)]
        tgt_v  = tgt.reshape(-1, P)[valid.reshape(-1)]
        n      = tgt_v.size(0)

        mse = F.mse_loss(pred_v, tgt_v).item()
        mae = F.l1_loss(pred_v, tgt_v).item()
        total_mse += mse * n
        total_mae += mae * n
        total_n   += n

        all_results.append({
            "mse": mse, "mae": mae,
            "pred": pred_v.cpu().tolist(),
            "gt":   tgt_v.cpu().tolist(),
        })

        if bi % save_plots_every == 0:
            _save_plot_patches(pred, tgt, valid, out_dir, step=bi, patch_size=patch_size)

        if bi % 10 == 0 and total_n > 0:
            print(f"[{bi:04d}] batch_mse={mse:.6f}  "
                  f"running avg mse={total_mse/total_n:.6f}  "
                  f"mae={total_mae/total_n:.6f}")

    return total_mse / max(1, total_n), total_mae / max(1, total_n), total_n, all_results


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _save_plot(result: dict, out_dir: str, step: int, tag: str, patch_size: int = 1):
    """
    Two-panel plot for autoregressive forecasting results.

    Panel 1 — Full view (history + forecast):
      · History shown in blue (last portion of the context fed to the model).
      · Ground-truth forecast in green, predicted forecast in orange, overlaid
        so the overlap is immediately visible.
      · Vertical dashed line marks the history / forecast boundary.

    Panel 2 — Forecast zoom (patch-level detail):
      · Ground truth vs predicted for the forecast horizon only.
      · Light vertical lines mark patch boundaries every *patch_size* steps.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)

        history = np.array(result.get("history", []), dtype=np.float32)
        pred    = np.array(result["pred"],    dtype=np.float32)
        gt      = np.array(result["gt"],      dtype=np.float32)
        L       = min(len(pred), len(gt))
        pred, gt = pred[:L], gt[:L]

        # x-axes
        hist_len  = len(history)
        hist_x    = np.arange(hist_len)
        fore_x    = np.arange(hist_len, hist_len + L)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                                 gridspec_kw={"height_ratios": [2, 1]})
        fig.suptitle(
            f"{tag}  |  step={step}  |  MSE={result['mse']:.4f}  MAE={result['mae']:.4f}",
            fontsize=11,
        )

        # ── Panel 1: full view ──────────────────────────────────────────────
        ax = axes[0]
        if hist_len > 0:
            ax.plot(hist_x, history, color="steelblue",  lw=1.2, label="history")
        ax.plot(fore_x, gt,   color="seagreen",  lw=1.5, label="ground truth")
        ax.plot(fore_x, pred, color="darkorange", lw=1.5, label="predicted", alpha=0.85)
        if hist_len > 0:
            ax.axvline(hist_len, color="grey", ls="--", lw=0.8, label="forecast start")
        ax.set_ylabel("value (normalised)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title("Context + Forecast", fontsize=9)

        # ── Panel 2: forecast zoom with patch boundaries ────────────────────
        ax2 = axes[1]
        ax2.plot(gt,   color="seagreen",  lw=1.5, label="ground truth")
        ax2.plot(pred, color="darkorange", lw=1.5, label="predicted", alpha=0.85)
        # Draw patch boundary lines
        if patch_size > 1:
            for boundary in range(0, L, patch_size):
                ax2.axvline(boundary, color="lightgrey", ls=":", lw=0.7)
        ax2.set_xlabel("forecast timestep")
        ax2.set_ylabel("value (normalised)")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.set_title("Forecast Zoom (patch boundaries shown)", fontsize=9)

        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"{tag}_step{step:06d}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
    except Exception:
        pass


def _save_plot_patches(pred, tgt, valid, out_dir, step, patch_size: int = 1, max_series: int = 2):
    """
    Teacher-forced plot: for each series, show gt-next-patch vs predicted-next-patch
    with patch boundaries marked.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        n = min(max_series, pred.size(0))
        for s in range(n):
            idx = valid[s].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            gt_1d = tgt[s][idx].reshape(-1).float().cpu().numpy()
            pr_1d = pred[s][idx].reshape(-1).float().cpu().numpy()
            L     = len(gt_1d)

            fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [2, 1]})
            fig.suptitle(f"Teacher-forced  step={step}  series={s}", fontsize=11)

            # full overlap plot
            ax = axes[0]
            ax.plot(gt_1d, color="seagreen",   lw=1.4, label="ground truth (next patch)")
            ax.plot(pr_1d, color="darkorange",  lw=1.4, label="predicted",  alpha=0.85)
            if patch_size > 1:
                for b in range(0, L, patch_size):
                    ax.axvline(b, color="lightgrey", ls=":", lw=0.7)
            ax.set_ylabel("value (normalised)")
            ax.legend(fontsize=8)
            ax.set_title("Ground Truth vs Predicted (all positions)", fontsize=9)

            # residual plot
            ax2 = axes[1]
            ax2.bar(range(L), gt_1d - pr_1d, color="indianred", alpha=0.7, width=1.0)
            ax2.axhline(0, color="black", lw=0.8)
            if patch_size > 1:
                for b in range(0, L, patch_size):
                    ax2.axvline(b, color="lightgrey", ls=":", lw=0.7)
            ax2.set_xlabel("timestep")
            ax2.set_ylabel("residual (gt − pred)")
            ax2.set_title("Residual", fontsize=9)

            fig.tight_layout()
            fig.savefig(
                os.path.join(out_dir, f"tf_step{step:06d}_s{s}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
    except Exception:
        pass


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(args) -> OpenTSLMSPFixPos2D:
    cfg = SimpleNamespace(
        is_pretraining=True,
        use_only_time_pe=args.use_only_time_pe,
        use_only_channel_pe=args.use_only_channel_pe,
    )
    model = OpenTSLMSPFixPos2D(cfg, llm_id=args.llm_id, device=args.device)
    if args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model.load_from_file(args.ckpt)
    model.eval()
    return model


def _resolve_eval_format(eval_format: str, ckpt_path: str) -> str:
    if eval_format != "auto":
        return eval_format
    if "joint_phase1_extend" in ckpt_path:
        return "joint_extend"
    if "joint_phase1" in ckpt_path:
        return "joint"
    if "stage2_all" in ckpt_path:
        return "train_aligned"
    return "chronos"


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="OpenTSLM forecasting evaluation")
    ap.add_argument("--ckpt",         required=True)
    ap.add_argument("--llm_id",       default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--mode",         default="autoregressive",
                    choices=["autoregressive", "teacher_forced"],
                    help="autoregressive: model.generate() patch-by-patch; "
                         "teacher_forced: one-pass hidden-state decoding")
    ap.add_argument("--dataset_name", default="all")
    ap.add_argument("--eval_format",  default="auto",
                    choices=["auto", "chronos", "train_aligned", "joint", "joint_extend"])
    ap.add_argument("--batch_size",   type=int,   default=32,
                    help="Batch size for inference. Larger = faster; limited by GPU memory.")
    ap.add_argument("--patch_size",   type=int,   default=None)
    ap.add_argument("--history_len",  type=int,   default=96,
                    help="History context length in time steps fed to the encoder. "
                         "Default 96 — symmetric with the default pred_len.")
    ap.add_argument("--pred_len",     type=int,   default=96,
                    help="Forecast horizon in time steps.")
    ap.add_argument("--max_batches",  type=int,   default=10_000_000)
    ap.add_argument("--out_dir",      default="./phase1_ts_eval")
    ap.add_argument("--save_plots_every", type=int, default=10)
    ap.add_argument("--use_only_time_pe",    action="store_true")
    ap.add_argument("--use_only_channel_pe", action="store_true")
    ap.add_argument("--lora_r",       type=int, default=0)
    ap.add_argument("--lora_alpha",   type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args)

    eval_patch_size = args.patch_size or getattr(model, "patch_size", None)
    if eval_patch_size is None:
        raise ValueError("Cannot infer patch_size. Pass --patch_size explicitly.")
    eval_patch_size = int(eval_patch_size)


    eos_str      = model.get_eos_token()
    eval_format  = _resolve_eval_format(args.eval_format, args.ckpt)
    n_forecast_patches = args.pred_len // eval_patch_size

    if eval_format in ("joint", "joint_extend"):
        ds_cls   = Phase1JointExtendDataset if eval_format == "joint_extend" else Phase1JointDataset
        joint_ds = ds_cls("test", eos_str, patch_size=eval_patch_size)
        test_ds  = ChronosOnlySubset(joint_ds)
    elif eval_format == "train_aligned":
        test_ds = TrainingAlignedChronosEvalDataset(
            mode="test", EOS_TOKEN=eos_str,
            dataset_name=args.dataset_name,
            patch_size=eval_patch_size,
            pred_len=args.pred_len,
            history_lens=[args.history_len],
        )
    else:
        test_ds = ChronosDataset(
            mode="test", EOS_TOKEN=eos_str, dataset_name=args.dataset_name,
        )

    print(f"Dataset: {len(test_ds)} series | mode={args.mode} | "
          f"patch_size={eval_patch_size} | pred_len={args.pred_len} "
          f"({n_forecast_patches} patches) | eval_format={eval_format}")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(
            b, patch_size=eval_patch_size
        ),
    )

    if args.mode == "autoregressive":
        metrics, _ = run_autoregressive(
            model, loader, n_forecast_patches, args.out_dir, args.save_plots_every,
            patch_size=eval_patch_size,
        )
    else:
        avg_mse, avg_mae, n, _ = run_teacher_forced(
            model, loader, args.out_dir, args.save_plots_every,
            patch_size=eval_patch_size,
        )
        metrics = {"mean_mse": avg_mse, "mean_mae": avg_mae, "n": n}

    metrics["mode"] = args.mode

    print(f"\n{'='*60}")
    print(f"Mode       : {args.mode}")
    print(f"Checkpoint : {args.ckpt}")
    for k, v in metrics.items():
        if k == "mode":
            continue
        print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print(f"{'='*60}")

    out_json = os.path.join(args.out_dir, f"metrics_{args.mode}.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
