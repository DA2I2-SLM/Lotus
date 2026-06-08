#!/usr/bin/env python3
"""Diagnose where per-channel info is lost in an MV forecast.

Loads a trained checkpoint, picks one MV synthetic sample (C >= 3), runs it
through the model and prints per-channel similarity at each stage:
  encoder output → projector → LLM hidden at TS positions → decoder output.

If any stage shows near-identical vectors across channels, that stage is the
collapse point. If the encoder already collapses, the bug is upstream of LLM.
If the LLM collapses, the channel info doesn't survive the frozen LLM. If only
the decoder collapses, the issue is decoder reading from a degenerate position.

Usage:
  python scripts/diag_mv_collapse.py \
    --ckpt results/sincos_patch32/.../phase1_chronos_mse_freq/checkpoints/best_model.pt \
    --llm_id google/gemma-3-270m \
    --enc_dec_type transformer --use_freq_embed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from types import SimpleNamespace

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    na = (a * a).sum().sqrt().clamp(min=1e-8)
    nb = (b * b).sum().sqrt().clamp(min=1e-8)
    return float((a * b).sum() / (na * nb))


def per_channel_similarity(vecs, name):
    """vecs: [C, ...] tensor — pairwise cosine similarity matrix across channel dim 0."""
    C = vecs.size(0)
    print(f"\n=== {name} — shape {tuple(vecs.shape)} ===")
    sims = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            sims[i, j] = cosine(vecs[i], vecs[j])
    print("pairwise cosine similarity:")
    print("    " + "  ".join(f"c{j}  " for j in range(C)))
    for i in range(C):
        print(f"c{i}  " + "  ".join(f"{sims[i, j]:+.3f}" for j in range(C)))
    off = sims[~np.eye(C, dtype=bool)]
    print(f"off-diag mean={off.mean():+.4f}  max={off.max():+.4f}  min={off.min():+.4f}")
    # also report per-channel norm and per-pair MSE
    print(f"per-channel ||vec||: {[f'{vecs[c].float().norm().item():.3f}' for c in range(C)]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--cache_dir", default="_preprocessed_chronos_uv_synth_phase1_mse_freq")
    ap.add_argument("--enc_dec_type", default="transformer")
    ap.add_argument("--use_freq_embed", action="store_true")
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_enc_ff_dim",     type=int, default=None)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_dec_ff_dim",     type=int, default=None)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Build model with config matching checkpoint
    cfg = SimpleNamespace(
        patch_size=args.patch_size,
        pe_type="sincos",
        use_only_time_pe=False,
        use_only_channel_pe=False,
        is_pretraining=True,
        use_freq_embed=args.use_freq_embed,
        enc_dec_type=args.enc_dec_type,
        mlp_enc_hidden_dim=args.mlp_enc_hidden_dim,
        mlp_enc_ff_dim=args.mlp_enc_ff_dim,
        mlp_dec_hidden_dim=args.mlp_dec_hidden_dim,
        mlp_dec_ff_dim=args.mlp_dec_ff_dim,
    )
    print(f"Building model on {args.device}...")
    model = OpenTSLMSPFixPos2D(args=cfg, llm_id=args.llm_id, device=args.device)
    model.load_from_file(args.ckpt)
    model.to(args.device).eval()

    # Load an MV test sample with C >= 5 for clearest signal
    print(f"\nLoading from {args.cache_dir}/all/test_synthetic ...")
    ds = load_from_disk(
        os.path.join(args.cache_dir, "all", "test_synthetic"),
        keep_in_memory=False,
    )
    picked = None
    for i in range(len(ds)):
        tgt = np.asarray(ds[i]["target"], dtype=np.float32)
        if tgt.ndim == 2 and tgt.shape[1] >= 5 and tgt.shape[0] >= 768:
            picked = (i, tgt)
            break
    assert picked is not None, "No MV sample with C>=5 and T>=768 found."
    i, tgt = picked
    T, C = tgt.shape
    print(f"Picked test row {i}: T={T}, C={C}")

    # Build a single-sample "batch" the way Phase1PretrainEvalDataset would
    history_raw = torch.tensor(tgt[: T - args.patch_size], dtype=torch.float32)   # exclude last patch as "future"
    future_raw  = torch.tensor(tgt[T - args.patch_size: T], dtype=torch.float32)
    mean = history_raw.mean(dim=0, keepdim=True)
    std  = history_raw.std(dim=0, keepdim=True).clamp(min=1e-3)
    history_norm = ((history_raw - mean) / std).clamp(-20, 20)
    future_norm  = ((future_raw  - mean) / std).clamp(-20, 20)

    sample = {
        "type": "chronos",
        "time_series": [history_norm.to(args.device)],   # [T_hist, C]
        "time_series_text": [f"This is a {C}-channel multivariate time series. <ts>"],
        "pre_prompt": "",
        "post_prompt": model.get_eos_token(),
        "answer": model.get_eos_token(),
        "ts_already_normalized": True,
        "ts_future": future_norm.transpose(0, 1).to(args.device),   # [C, P]
        "norm_mean": float(mean[0, 0]),
        "norm_std":  float(std[0, 0]),
    }
    batch = [sample]

    # ── Step 1: encoder output ────────────────────────────────────────────
    with torch.no_grad():
        # Trigger pad_and_apply_batch in return_targets=True mode to populate
        # _last_ts_enc.
        _ = model.pad_and_apply_batch(batch, return_targets=True)
        enc_out = model._last_ts_enc        # [N_ts, T_patches, enc_dim]
        n_ch_per_outer = model._last_ts_n_ch_per_outer
        chan_id = model._last_ts_chan_id

    assert enc_out is not None
    n_ch = n_ch_per_outer[0]
    print(f"\n_last_ts_enc shape={tuple(enc_out.shape)} | n_ch_per_outer={n_ch_per_outer} | chan_id={chan_id.tolist()[:n_ch]}")
    per_channel_similarity(enc_out, "encoder output (per-channel)")

    # ── Step 2: projector output ──────────────────────────────────────────
    with torch.no_grad():
        proj_out = model.projector(enc_out.to(next(model.projector.parameters()).dtype))   # [N_ts, T_patches, llm_dim]
    per_channel_similarity(proj_out, "projector output (per-channel)")

    # ── Step 3: LLM hidden states at TS positions ─────────────────────────
    with torch.no_grad():
        inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
            batch, return_targets=True
        )
        out = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]   # [1, L, H]

    patch_positions = ts_mask[0].nonzero(as_tuple=True)[0]
    print(f"\npatch_positions len={patch_positions.numel()} (= n_ch * T_patches = {n_ch} * {enc_out.size(1)})")
    # Extract per-channel hidden as the eval does
    ch_hidden = []
    for c in range(n_ch):
        ch_pos = patch_positions[c::n_ch]
        h = last_hidden[0, ch_pos, :]    # [T_patches, H]
        ch_hidden.append(h)
    ch_hidden = torch.stack(ch_hidden, dim=0)   # [n_ch, T_patches, H]
    per_channel_similarity(ch_hidden, "LLM hidden at TS positions (per-channel)")

    # ── Step 4: decoder input + decoder output (per channel, at LAST position) ─
    with torch.no_grad():
        ch_dec_in = model.decoder_projector(ch_hidden.float())
        ch_dec_in = F.layer_norm(ch_dec_in, [ch_dec_in.size(-1)])
        dec_out = model.decoder(ch_dec_in)   # [n_ch, T_patches, P]
        pred_last = dec_out[:, -1, :]        # [n_ch, P]
    per_channel_similarity(ch_dec_in, "decoder_projector input (per-channel)")
    per_channel_similarity(dec_out, "decoder output (per-channel, all positions)")
    per_channel_similarity(pred_last, "decoder pred at LAST position (per-channel)")

    # ── Step 5: also dump decoder output at SECOND-TO-LAST position ───────
    pred_second_last = dec_out[:, -2, :]
    per_channel_similarity(pred_second_last, "decoder pred at SECOND-LAST position (per-channel)")

    # ── Step 6: compare to ground-truth future ────────────────────────────
    gt = sample["ts_future"]   # [C, P]
    print(f"\nground-truth ts_future shape={tuple(gt.shape)}")
    for c in range(n_ch):
        mse_last = float(F.mse_loss(pred_last[c], gt[c]).item())
        mse_2nd  = float(F.mse_loss(pred_second_last[c], gt[c]).item())
        print(f"  ch{c}: MSE pred_last vs gt = {mse_last:.4f}  |  pred_2nd_last vs gt = {mse_2nd:.4f}")

    print("\n[diag] Done. Look for the first stage where off-diag cosine > 0.95 — that's the collapse point.")


if __name__ == "__main__":
    main()
