#!/usr/bin/env python3
import os
import json
import argparse
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# TODO: fix these imports to your repo paths
from opentslm.time_series_datasets.ChronosDataset import ChronosDataset
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


@torch.no_grad()
def stitch_patches(patches_2d: torch.Tensor) -> torch.Tensor:
    # patches_2d: [n_patches, P] -> [n_patches*P]
    return patches_2d.reshape(-1)


@torch.no_grad()
def predict_next_patches(model, batch):
    """
    Returns:
      pred_patches:  [N_ts_total, N_patches, P] predicted next patches at each patch position
      tgt_patches:   [N_ts_total, N_patches, P] ground truth next patches (shifted)
      valid_mask:    [N_ts_total, N_patches] boolean mask (last patch invalid)
    """
    (
        inputs_embeds,              # [B, L, H]
        attention_mask,             # [B, L]
        token_ids,                  # [B, L]
        ts_mask,                    # [B, L] True where TS patch embeddings are inserted
        ts_patch_targets_shifted,   # [N_ts_total, N_patches, P]
        ts_patch_valid,             # [N_ts_total, N_patches]
        ts_chan_id,                 # list[int] length N_ts_total
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    need_recon = (ts_patch_targets_shifted is not None) and (ts_patch_targets_shifted.numel() > 0)
    if not need_recon:
        return None, None, None

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
    last_hidden = outputs.hidden_states[-1]         # [B, L, H]
    ts_hidden = last_hidden[ts_mask]               # [N_ts_total*N_patches, H]

    expected = N_ts_total * N_patches
    if ts_hidden.size(0) != expected:
        raise RuntimeError(
            f"ts_mask extracted {ts_hidden.size(0)} hidden vectors, expected {expected}. "
            f"Check that all series are padded to the same patch count."
        )

    # Decode hidden states -> patches
    ts_hidden_enc = model.decoder_projector(ts_hidden.float())                 # [N_ts_total*N_patches, enc_dim]
    ts_hidden_3d = ts_hidden_enc.reshape(N_ts_total, N_patches, -1)            # [N_ts_total, N_patches, enc_dim]
    recon_flat = model.decoder(ts_hidden_3d, ts_chan_id)                       # [N_ts_total, T_padded]
    pred_patches = recon_flat.reshape(N_ts_total, N_patches, P)                # [N_ts_total, N_patches, P]

    return pred_patches, ts_patch_targets_shifted, ts_patch_valid


@torch.no_grad()
def batch_metrics(pred_patches, tgt_patches, valid_mask):
    """
    Computes MSE/MAE over valid patch positions only.
    Returns scalar floats.
    """
    # [N_ts_total, N_patches, P] -> [N_ts_total*N_patches, P]
    P = pred_patches.size(-1)
    pred_flat = pred_patches.reshape(-1, P)
    tgt_flat = tgt_patches.reshape(-1, P)
    valid_flat = valid_mask.reshape(-1)

    pred_v = pred_flat[valid_flat]
    tgt_v = tgt_flat[valid_flat]

    mse = F.mse_loss(pred_v, tgt_v).item()
    mae = F.l1_loss(pred_v, tgt_v).item()
    return mse, mae, int(valid_flat.sum().item())


@torch.no_grad()
def save_debug_plots(pred_patches, tgt_patches, valid_mask, out_dir, step, max_series=2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    n = min(max_series, pred_patches.size(0))
    for s in range(n):
        idx = valid_mask[s].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        gt = tgt_patches[s][idx]     # [n_valid, P]
        pr = pred_patches[s][idx]    # [n_valid, P]

        gt_1d = stitch_patches(gt).float().cpu().numpy()
        pr_1d = stitch_patches(pr).float().cpu().numpy()

        fig = plt.figure()
        plt.plot(gt_1d, label="gt_next_patch")
        plt.plot(pr_1d, label="pred_next_patch", alpha=0.35)
        plt.title(f"Next-patch prediction (teacher-forced) step={step} series={s}")
        plt.legend()
        pth = os.path.join(out_dir, f"ts_pred_step{step:06d}_series{s}.png")
        plt.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close(fig)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str, help="Path to model checkpoint saved by store_to_file()")
    ap.add_argument("--dataset_name", default="monash_fred_md", type=str)
    ap.add_argument("--llm_id", default="google/gemma-3-270m", type=str)
    ap.add_argument("--device", default="cuda", type=str)
    ap.add_argument("--batch_size", default=2, type=int)
    ap.add_argument("--max_batches", default=200, type=int)
    ap.add_argument("--patch_size", default=4, type=int)
    ap.add_argument("--out_dir", default="./phase1_ts_eval", type=str)
    ap.add_argument("--save_plots_every", default=50, type=int)
    ap.add_argument("--use_only_time_pe", action="store_true", help="Whether to use only time positional embeddings (no token type or value embeddings).")
    ap.add_argument("--use_only_channel_pe", action="store_true", help="Whether to use only channel positional embeddings (no token type or value embeddings).")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Minimal args object for your model (add encoder args if needed)
    cfg = SimpleNamespace()
    cfg.is_pretraining = True
    cfg.use_only_time_pe = args.use_only_time_pe
    cfg.use_only_channel_pe = args.use_only_channel_pe

    model = OpenTSLMSPFixPos2D(cfg, llm_id=args.llm_id, device=args.device)
    model.load_from_file(args.ckpt)
    model.eval()

    eos_str = model.get_eos_token()

    test_ds = ChronosDataset(
        mode="test",
        EOS_TOKEN=eos_str,
        dataset_name=args.dataset_name,
    )

    dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=args.patch_size
        ),
    )

    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    out_jsonl = os.path.join(args.out_dir, "ts_next_patch_metrics.jsonl")
    with open(out_jsonl, "w") as f:
        for bi, batch in enumerate(dl):
            if bi >= args.max_batches:
                break

            pred, tgt, valid = predict_next_patches(model, batch)
            if pred is None:
                continue

            mse, mae, count = batch_metrics(pred, tgt, valid)

            total_mse += mse * count
            total_mae += mae * count
            total_count += count

            row = {"batch": bi, "mse": mse, "mae": mae, "count": count}
            f.write(json.dumps(row) + "\n")

            if (bi % args.save_plots_every) == 0:
                save_debug_plots(pred, tgt, valid, out_dir=args.out_dir, step=bi, max_series=2)

            if (bi % 10) == 0:
                avg_mse = total_mse / max(1, total_count)
                avg_mae = total_mae / max(1, total_count)
                print(f"[{bi:04d}] batch mse={mse:.6f} mae={mae:.6f} | running avg mse={avg_mse:.6f} mae={avg_mae:.6f}")

    avg_mse = total_mse / max(1, total_count)
    avg_mae = total_mae / max(1, total_count)
    print(f"\nDone. Valid points={total_count}")
    print(f"Overall next-patch MSE: {avg_mse:.6f}")
    print(f"Overall next-patch MAE: {avg_mae:.6f}")
    print(f"Wrote per-batch metrics to: {out_jsonl}")
    print(f"Plots (every {args.save_plots_every} batches) saved under: {args.out_dir}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()