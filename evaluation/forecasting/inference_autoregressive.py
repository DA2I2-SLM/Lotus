#!/usr/bin/env python3
"""
Autoregressive evaluation for OpenTSLM.

Instead of teacher-forced next-patch prediction (where the model always sees
real data), this script:
  1. Feeds the model a context window of real patches.
  2. Predicts the next patch from the last context position.
  3. Appends that predicted patch to the input sequence.
  4. Re-runs the model on the extended (partially synthetic) sequence.
  5. Repeats for the desired forecast horizon.
  6. Compares autoregressively generated patches against ground truth.

This measures how well the model forecasts when errors compound — a much
harder (and more realistic) test than teacher forcing.
"""

import os
import json
import copy
import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opentslm.time_series_datasets.ChronosDataset import ChronosDataset
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def stitch_patches(patches_2d: torch.Tensor) -> torch.Tensor:
    """[n_patches, P] -> [n_patches * P]"""
    return patches_2d.reshape(-1)


def _to_error(v: float) -> float:
    return -v if v < 0 else v


def _decode_ts_patch_predictions(
    model,
    last_hidden,
    ts_mask,
    ts_patch_targets_shifted,
    ts_patch_valid,
    ts_chan_id=None,
):
    """
    Rebuild the padded per-series patch layout expected by the decoder.

    `ts_mask` only marks real history patches, while targets/valid masks are
    padded to the batch-wide maximum number of patches.
    """
    N_ts_total, N_patches, _ = ts_patch_targets_shifted.shape
    ts_hidden = last_hidden[ts_mask]

    patch_counts = (ts_patch_valid.sum(dim=1) + 1).tolist()
    expected_flat = int(sum(patch_counts))
    if ts_hidden.size(0) != expected_flat:
        raise RuntimeError(
            f"ts_mask extracted {ts_hidden.size(0)} hidden vectors, "
            f"expected {expected_flat} real patch vectors."
        )

    ts_hidden_enc = model.decoder_projector(ts_hidden.float())
    ts_hidden_enc = F.layer_norm(ts_hidden_enc, [ts_hidden_enc.size(-1)])

    enc_dim = ts_hidden_enc.size(-1)
    ts_hidden_3d = ts_hidden_enc.new_zeros((N_ts_total, N_patches, enc_dim))

    offset = 0
    for s, count in enumerate(patch_counts):
        ts_hidden_3d[s, :count, :] = ts_hidden_enc[offset:offset + count]
        offset += count

    pred_patches = model.decoder(ts_hidden_3d, ts_chan_id)
    return pred_patches, patch_counts


# ---------------------------------------------------------------------------
# Single forward pass: get the predicted next patch at the *last* valid
# position.  This is the building block for autoregressive generation.
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_last_next_patch(model, batch):
    """
    Run one teacher-forced forward pass on `batch` and return the predicted
    next patch at the last valid position for every time series in the batch.

    Returns
    -------
    last_pred_patches : Tensor [N_ts_total, P]
        The predicted next patch at the last valid position for each series.
    N_patches : int
        Number of patches per series in this batch (needed for bookkeeping).
    ts_chan_id : list[int]
        Channel ids (passed through for convenience).

    Returns (None, None, None) if the batch has no reconstructable TS data.
    """
    (
        inputs_embeds,
        attention_mask,
        token_ids,
        ts_mask,
        ts_patch_targets_shifted,
        ts_patch_valid,
        ts_chan_id,
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    need_recon = (ts_patch_targets_shifted is not None) and (ts_patch_targets_shifted.numel() > 0)
    if not need_recon:
        return None, None, None

    N_ts_total, N_patches, P = ts_patch_targets_shifted.shape

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    pred_patches, patch_counts = _decode_ts_patch_predictions(
        model=model,
        last_hidden=outputs.hidden_states[-1],
        ts_mask=ts_mask,
        ts_patch_targets_shifted=ts_patch_targets_shifted,
        ts_patch_valid=ts_patch_valid,
        ts_chan_id=ts_chan_id,
    )

    # Predict from each series' last real context patch, not from a padded slot.
    row_idx = torch.arange(N_ts_total, device=pred_patches.device)
    last_pos = torch.tensor(
        [count - 1 for count in patch_counts],
        device=pred_patches.device,
        dtype=torch.long,
    )
    last_pred_patches = pred_patches[row_idx, last_pos, :]  # [N_ts_total, P]

    return last_pred_patches, N_patches, ts_chan_id


# ---------------------------------------------------------------------------
# Core: autoregressive generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def autoregressive_generate(
    model,
    context_samples,           # list of dataset items (context only)
    horizon_patches: int,
    patch_size: int,
    collate_fn,
):
    """
    Autoregressively generate `horizon_patches` future patches.

    Parameters
    ----------
    model : OpenTSLMSPFixPos2D
    context_samples : list
        A list of dataset items whose time-series values contain only the
        context (no future data).  These will be collated into a batch.
    horizon_patches : int
        How many patches to generate into the future.
    patch_size : int
        Number of raw time steps per patch (must match training config).
    collate_fn : callable
        The same collate function used during training / eval
        (extend_time_series_to_match_patch_size_and_aggregate).

    Returns
    -------
    generated_patches : list[Tensor]
        Length = horizon_patches.  Each element is [N_ts_total, P].
    """
    # We iteratively grow the time-series values inside each sample by
    # appending predicted patch values, then re-collate and re-run.

    current_samples = [copy.deepcopy(s) for s in context_samples]
    generated_patches = []

    for step in range(horizon_patches):
        batch = collate_fn(current_samples)

        last_pred, _, _ = predict_last_next_patch(model, batch)
        if last_pred is None:
            break

        generated_patches.append(last_pred)  # [N_ts_total, P]

        # -----------------------------------------------------------------
        # Append predicted patch values to each sample's raw time series
        # so the next iteration sees them as input.
        #
        # The key / attribute name depends on the dataset item format.
        # Common patterns:
        #   sample["values"]          — 1-D tensor of raw time-series values
        #   sample["time_series"]     — same
        #   sample.values             — if samples are SimpleNamespace / dataclass
        #
        # Below we assume each sample is a dict-like object with a field
        # that holds a 1-D tensor of time-series values.  Adjust as needed.
        # -----------------------------------------------------------------
        ts_idx = 0  # pointer into the N_ts_total dimension
        for sample in current_samples:
            # ---- adapt this block to your dataset format ----
            n_channels = _num_channels(sample)
            for ch in range(n_channels):
                pred_values = last_pred[ts_idx].cpu()   # [P]
                _append_values(sample, ch, pred_values)
                ts_idx += 1

    return generated_patches


# ---------------------------------------------------------------------------
# Dataset-format helpers — adapt these to your ChronosDataset item layout.
# ---------------------------------------------------------------------------

def _num_channels(sample) -> int:
    """
    Return the number of time-series channels in a single dataset item.

    Dataset-dependent.  Examples:
      - If each sample has one univariate series: return 1
      - If sample["values"] is a 2-D tensor [C, T]: return sample["values"].shape[0]
      - If sample has a list of channel tensors: return len(sample["channels"])
    """
    # Simple default: assume univariate
    return 1


def _ensure_list(sample, key_or_attr: str, is_dict: bool):
    """
    If the time_series field is a 2-D tensor [C, T], convert it to a Python
    list of 1-D tensors so individual channels can be extended in-place.
    Returns the (possibly converted) list/tensor.
    """
    if is_dict:
        ts = sample.get(key_or_attr)
    else:
        ts = getattr(sample, key_or_attr, None)
    if ts is None:
        return None
    if torch.is_tensor(ts) and ts.ndim == 2:
        ts = list(ts.unbind(0))
        if is_dict:
            sample[key_or_attr] = ts
        else:
            setattr(sample, key_or_attr, ts)
    return ts


def _append_values(sample, channel_idx: int, new_values: torch.Tensor):
    """
    Append `new_values` (a 1-D tensor of length P) to the raw time-series
    values for `channel_idx` inside `sample`.

    Uses the same key-resolution order as `_get_values` so the feedback loop
    is consistent across AR steps.  Converts a 2-D [C, T] tensor to a list of
    1-D tensors on first write so channels can grow independently.
    """
    if isinstance(sample, dict):
        ts = _ensure_list(sample, "time_series", is_dict=True)
        if ts is not None:
            ts[channel_idx] = torch.cat([ts[channel_idx], new_values], dim=0)
        else:
            sample["values"] = torch.cat([sample["values"], new_values], dim=0)
    else:
        ts = _ensure_list(sample, "time_series", is_dict=False)
        if ts is not None:
            ts[channel_idx] = torch.cat([ts[channel_idx], new_values], dim=0)
        else:
            sample.values = torch.cat([sample.values, new_values], dim=0)


def _set_values(sample, channel_idx: int, values: torch.Tensor):
    """Overwrite the raw 1-D time-series tensor for a given channel."""
    if isinstance(sample, dict):
        ts = _ensure_list(sample, "time_series", is_dict=True)
        if ts is not None:
            ts[channel_idx] = values
        else:
            sample["values"] = values
    else:
        ts = _ensure_list(sample, "time_series", is_dict=False)
        if ts is not None:
            ts[channel_idx] = values
        else:
            sample.values = values


def _get_values(sample, channel_idx: int) -> torch.Tensor:
    """Return the raw 1-D time-series tensor for a given channel."""
    if isinstance(sample, dict):
        ts = sample.get("time_series")
        if ts is not None:
            values = ts[channel_idx]
        else:
            values = sample["values"]
    else:
        ts = getattr(sample, "time_series", None)
        if ts is not None:
            values = ts[channel_idx]
        else:
            values = sample.values

    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    # values should already be 1-D (channel already selected above);
    # guard against a raw 2-D [C, T] tensor falling through without indexing.
    if values.ndim == 2:
        values = values[channel_idx]
    return values


# ---------------------------------------------------------------------------
# Split a dataset item into context + ground-truth horizon
# ---------------------------------------------------------------------------

def split_sample(sample, context_steps: int, horizon_steps: int):
    """
    Split a dataset item into a context-only copy and a ground-truth
    horizon tensor.

    Returns
    -------
    context_sample : dict/object
        A deep copy of `sample` with time-series truncated to `context_steps`.
    gt_horizon : Tensor [n_channels, horizon_steps]
        The ground-truth future values.
    """
    ctx = copy.deepcopy(sample)
    n_ch = _num_channels(sample)
    gt_channels = []

    for ch in range(n_ch):
        full_ts = _get_values(sample, ch)
        total_needed = context_steps + horizon_steps
        if full_ts.shape[0] < total_needed:
            return None, None  # series too short

        # Truncate context copy for every channel (not just ch 0).
        _set_values(ctx, ch, full_ts[:context_steps].clone())
        gt_channels.append(full_ts[context_steps : context_steps + horizon_steps].clone())

    gt_horizon = torch.stack(gt_channels, dim=0)  # [n_channels, horizon_steps]
    return ctx, gt_horizon


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_horizon_metrics(generated_patches, gt_horizon, patch_size):
    """
    Compare autoregressively generated patches against ground truth.

    Parameters
    ----------
    generated_patches : list[Tensor]
        Each element [N_ts_total, P] — one per horizon step.
    gt_horizon : Tensor [N_ts_total, horizon_steps]
        Raw ground-truth future values (horizon_steps = horizon_patches * P).
    patch_size : int

    Returns
    -------
    mse, mae : float
    per_step_mse, per_step_mae : list[float]
        Metrics broken down by forecast step (patch index).
    """
    horizon_patches = len(generated_patches)
    pred_stack = torch.stack(generated_patches, dim=1)  # [N_ts_total, H_patches, P]
    gt_patches = gt_horizon.reshape(gt_horizon.size(0), horizon_patches, patch_size)

    # Move to same device
    gt_patches = gt_patches.to(pred_stack.device)

    # Overall
    mse = F.mse_loss(pred_stack, gt_patches).item()
    mae = F.l1_loss(pred_stack, gt_patches).item()

    # Per-step (shows how errors grow with horizon)
    per_step_mse = []
    per_step_mae = []
    for h in range(horizon_patches):
        per_step_mse.append(F.mse_loss(pred_stack[:, h], gt_patches[:, h]).item())
        per_step_mae.append(F.l1_loss(pred_stack[:, h], gt_patches[:, h]).item())

    return mse, mae, per_step_mse, per_step_mae


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_ar_plots(generated_patches, gt_horizon, patch_size, context_values,
                  out_dir, batch_idx, max_series=2):
    """
    Plot autoregressive predictions vs ground truth, with context shown
    for visual reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    pred_stack = torch.stack(generated_patches, dim=1)  # [N_ts_total, H, P]
    n = min(max_series, pred_stack.size(0))

    for s in range(n):
        ctx = context_values[s].float().cpu().numpy()
        gt_1d = gt_horizon[s].float().cpu().numpy()
        pred_1d = stitch_patches(pred_stack[s]).float().cpu().numpy()

        t_ctx = range(len(ctx))
        t_future = range(len(ctx), len(ctx) + len(gt_1d))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_ctx, ctx, label="context (real)", color="steelblue")
        ax.plot(t_future, gt_1d, label="ground truth", color="tab:green")
        ax.plot(t_future, pred_1d[:len(gt_1d)], label="AR prediction",
                color="tab:red", alpha=0.7, linestyle="--")
        ax.axvline(len(ctx), color="gray", linestyle=":", label="forecast start")
        ax.set_title(f"Autoregressive forecast — batch {batch_idx}, series {s}")
        ax.legend()

        pth = os.path.join(out_dir, f"ar_pred_batch{batch_idx:04d}_series{s}.png")
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Autoregressive (non-teacher-forced) evaluation for OpenTSLM"
    )
    ap.add_argument("--ckpt", required=True, type=str,
                     help="Path to model checkpoint saved by store_to_file()")
    ap.add_argument("--dataset_name", default="ushcn_daily", type=str)
    ap.add_argument("--llm_id", default="google/gemma-3-270m", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--batch_size", default=2, type=int)
    ap.add_argument("--max_batches", default=200, type=int)
    ap.add_argument("--patch_size", default=4, type=int)
    ap.add_argument("--context_patches", default=None, type=int,
                     help="Number of context patches. If None, uses 75%% of "
                          "each series length.")
    ap.add_argument("--horizon_patches", default=12, type=int,
                     help="Number of patches to generate autoregressively.")
    ap.add_argument("--out_dir", default="./phase1_ar_eval", type=str)
    ap.add_argument("--save_plots_every", default=1, type=int)
    ap.add_argument("--use_only_time_pe", action="store_true")
    ap.add_argument("--use_only_channel_pe", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg = SimpleNamespace()
    cfg.is_pretraining = True
    cfg.use_only_time_pe = args.use_only_time_pe
    cfg.use_only_channel_pe = args.use_only_channel_pe

    model_cls = OpenTSLMSPFixPos2D
    model = model_cls(cfg, llm_id=args.llm_id, device=args.device)
    model.load_from_file(args.ckpt)
    model.eval()

    eos_str = model.get_eos_token()

    # ------------------------------------------------------------------
    # Dataset — we iterate over individual samples (not pre-batched)
    # ------------------------------------------------------------------
    test_ds = ChronosDataset(
        mode="test",
        EOS_TOKEN=eos_str,
        dataset_name=args.dataset_name,
    )
    print(f"Loaded test dataset with {len(test_ds)} series.")

    patch_size = args.patch_size
    horizon_patches = args.horizon_patches
    horizon_steps = horizon_patches * patch_size

    collate_fn = lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=patch_size
    )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    all_per_step_mse = [0.0] * horizon_patches
    all_per_step_mae = [0.0] * horizon_patches
    per_step_count = [0] * horizon_patches

    out_jsonl = os.path.join(args.out_dir, "ar_metrics.jsonl")

    sample_idx = 0
    batch_idx = 0

    with open(out_jsonl, "w") as f:
        while sample_idx < len(test_ds) and batch_idx < args.max_batches:
            # Collect a mini-batch of samples
            context_samples = []
            gt_horizons = []
            context_values_for_plot = []

            for _ in range(args.batch_size):
                if sample_idx >= len(test_ds):
                    break

                sample = test_ds[sample_idx]
                sample_idx += 1

                # Determine context length
                full_ts = _get_values(sample, 0)
                total_len = full_ts.shape[0]
                if args.context_patches is not None:
                    context_steps = args.context_patches * patch_size
                else:
                    # Default: use 75% of the series as context
                    context_steps = int(total_len * 0.75)
                    # Round down to a multiple of patch_size
                    context_steps = (context_steps // patch_size) * patch_size

                if total_len < context_steps + horizon_steps:
                    continue  # series too short, skip

                ctx_sample, gt_horizon = split_sample(
                    sample, context_steps, horizon_steps
                )
                if ctx_sample is None:
                    continue

                context_samples.append(ctx_sample)
                gt_horizons.append(gt_horizon)
                context_values_for_plot.append(full_ts[:context_steps])

            if len(context_samples) == 0:
                batch_idx += 1
                continue

            # ---- Autoregressive generation ----
            generated = autoregressive_generate(
                model=model,
                context_samples=context_samples,
                horizon_patches=horizon_patches,
                patch_size=patch_size,
                collate_fn=collate_fn,
            )

            if len(generated) == 0:
                batch_idx += 1
                continue

            # Stack ground truths: [N_ts_total, horizon_steps]
            # (for univariate, N_ts_total == len(context_samples))
            gt_stacked = torch.cat(gt_horizons, dim=0)  # [N_ts_total, horizon_steps]

            actual_horizon = len(generated)
            gt_for_eval = gt_stacked[:, :actual_horizon * patch_size]

            mse, mae, ps_mse, ps_mae = compute_horizon_metrics(
                generated, gt_for_eval, patch_size
            )

            n = gt_for_eval.size(0)  # number of series
            total_mse += mse * n
            total_mae += mae * n
            total_count += n

            for h in range(actual_horizon):
                all_per_step_mse[h] += ps_mse[h] * n
                all_per_step_mae[h] += ps_mae[h] * n
                per_step_count[h] += n

            row = {
                "batch": batch_idx,
                "mse": mse,
                "mae": mae,
                "n_series": n,
                "per_step_mse": ps_mse,
                "per_step_mae": ps_mae,
            }
            f.write(json.dumps(row) + "\n")

            # ---- Plots ----
            if (batch_idx % args.save_plots_every) == 0:
                save_ar_plots(
                    generated, gt_for_eval, patch_size,
                    context_values_for_plot,
                    out_dir=args.out_dir, batch_idx=batch_idx,
                )

            if (batch_idx % 10) == 0:
                avg_mse = total_mse / max(1, total_count)
                avg_mae = total_mae / max(1, total_count)
                print(
                    f"[{batch_idx:04d}] batch mse={mse:.6f} mae={mae:.6f} | "
                    f"running avg mse={avg_mse:.6f} mae={avg_mae:.6f}"
                )

            batch_idx += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    avg_mse = total_mse / max(1, total_count)
    avg_mae = total_mae / max(1, total_count)
    print(f"\nDone. Evaluated {total_count} series across {batch_idx} batches.")
    print(f"Overall AR MSE: {avg_mse:.6f}")
    print(f"Overall AR MAE: {avg_mae:.6f}")

    # Per-step breakdown (shows error growth over the horizon)
    print(f"\nPer-step metrics (error growth over horizon):")
    print(f"{'Step':>6} {'MSE':>12} {'MAE':>12}")
    for h in range(horizon_patches):
        if per_step_count[h] > 0:
            step_mse = all_per_step_mse[h] / per_step_count[h]
            step_mae = all_per_step_mae[h] / per_step_count[h]
            print(f"{h+1:>6} {step_mse:>12.6f} {step_mae:>12.6f}")

    print(f"\nWrote per-batch metrics to: {out_jsonl}")
    print(f"Plots saved under: {args.out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
