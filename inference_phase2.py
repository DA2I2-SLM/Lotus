#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Phase-2 inference script for OpenTSLMSPFixPos2D.

Loads a joint_phase2_downstream (or any joint_phase2) checkpoint and runs
forecasting on arbitrary time series data supplied by the caller.

Two modes are provided:
  1. teacher_forced  (default, fast)
       Feeds the full prompt + a structured answer template
       (<ts><ch_sep><future_patch>...<future_patch>*N_per_ch * C</ts>)
       through the LLM in one forward pass.  Hidden states at <future_patch>
       positions are projected through decoder_projector → layer_norm → decoder.
       This exactly mirrors the output-patch branch of compute_joint_phase2_loss.

  2. autoregressive  (flag: autoregressive=True)
       Calls model.forecast_phase2(), which drives the LLM token-by-token
       and decodes via the numeric decoder at each generated <future_patch>.
       Slower but matches deployment conditions.

Normalization:
  History-only per-channel z-normalization (mean/std of each channel's context
  window). This is the RevIN convention used by Phase2DownstreamDataset.
  Predictions are returned in normalized space.  Pass `denormalize=True` to
  get back the original scale using the per-channel history stats.

Usage:
  python inference_phase2.py \\
      --ckpt $FTS_RESULTS_ROOT/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_joint_phase1_extend/checkpoints/best_model.pt \\
      --llm_id google/gemma-3-270m \\
      --patch_size 32 \\
      --pred_len 96

  # or call run_inference() from another script — see bottom of file.
"""

import argparse
import os
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import MAX_FUTURE_PATCH_PE_SIZE


# ---------------------------------------------------------------------------
# 1.  Model construction + checkpoint loading
# ---------------------------------------------------------------------------

def build_model(
    ckpt_path: str,
    llm_id: str = "google/gemma-3-270m",
    patch_size: int = 32,
    enc_dec_type: str = "transformer",
    mlp_enc_hidden_dim: int = 64,
    mlp_enc_ff_dim: int = 256,
    mlp_dec_hidden_dim: int = 64,
    mlp_dec_ff_dim: int = 256,
    pe_type: str = "sincos",
    device: str = "cuda",
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> OpenTSLMSPFixPos2D:
    """Construct OpenTSLMSPFixPos2D and load a phase-2 checkpoint.

    The cfg SimpleNamespace mirrors what CurriculumTrainer passes to the model
    constructor.  All non-obvious fields:

      is_pretraining    — kept True; controls gradient-checkpointing enable
                          inside __init__. Not used during inference.
      pe_type           — must match training; controls which PE branch is
                          taken inside MLPEncoder2D / TransformerCNNEncoder2D.
      enc_dec_type      — 'mlp' (your training config) or 'transformer'.
      mlp_{enc,dec}_{hidden,ff}_dim — MLP backbone dims; must match training.
    """
    cfg = SimpleNamespace(
        is_pretraining=True,
        use_only_time_pe=False,
        use_only_channel_pe=False,
        use_freq_embed=False,
        patch_size=patch_size,
        pe_type=pe_type,
        enc_dec_type=enc_dec_type,
        mlp_enc_hidden_dim=mlp_enc_hidden_dim,
        mlp_enc_ff_dim=mlp_enc_ff_dim,
        mlp_dec_hidden_dim=mlp_dec_hidden_dim,
        mlp_dec_ff_dim=mlp_dec_ff_dim,
    )

    model = OpenTSLMSPFixPos2D(cfg, llm_id=llm_id, device=device)

    if lora_r > 0:
        model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)

    model.load_from_file(ckpt_path)
    model.eval()

    print(
        f"[inference_phase2] Loaded checkpoint: {ckpt_path}\n"
        f"  patch_size={model.patch_size}  "
        f"enc_dec_type={model.enc_dec_type}  "
        f"device={model.device}"
    )
    return model


# ---------------------------------------------------------------------------
# 2.  Sample construction helpers
# ---------------------------------------------------------------------------

def _build_sample(
    history: torch.Tensor,       # [C, history_len]  already normalized
    pred_len: int,
    patch_size: int,
    ch_sep_token: str,
    fp_tokens: List[str],
    eos_token: str,
    ds_name: str = "",
    col_names: Optional[List[str]] = None,
    norm_stats: Optional[List[Tuple[float, float]]] = None,
) -> dict:
    """Build one sample dict in the exact format expected by pad_and_apply_batch.

    This format matches DownstreamEvalDataset (eval_phase2_downstream.py) and
    Phase2DownstreamDataset (training).  Each channel produces:
      - one entry in time_series    : list of length-history_len float tensors
      - one entry in time_series_text: "This is the ts (col), mean X std Y. <ts>"
      - the answer template        : "<ts><ch_sep><fp_0>…<fp_{N-1}> x C</ts><eos>"

    Args:
        history    : [C, T] normalized history (z-normed with per-channel stats).
        pred_len   : forecast horizon in time steps.
        patch_size : patch size; pred_len must be divisible by it.
        ch_sep_token, fp_tokens, eos_token : from model.tokenizer.
        norm_stats : list of (mean, std) per channel, used for the text prompt.
                     If None, the text will show mean=0 std=1.
    """
    C = history.shape[0]
    N_per_ch = pred_len // patch_size

    # One "<ch_sep><fp_0>…<fp_{N-1}>" block per channel
    ch_block = ch_sep_token + "".join(fp_tokens[p] for p in range(N_per_ch))
    answer   = "<ts>" + ch_block * C + "</ts>" + eos_token

    time_series      = []
    time_series_text = []
    for c in range(C):
        ch_hist = history[c].tolist()   # already normalized
        time_series.append(ch_hist)

        if col_names is not None and c < len(col_names):
            cname = col_names[c]
        else:
            cname = f"ch{c}"

        if norm_stats is not None and c < len(norm_stats):
            m, s = norm_stats[c]
        else:
            m, s = 0.0, 1.0

        time_series_text.append(
            f"This is the time series ({cname}), mean {m:.4f} std {s:.4f}. <ts>"
        )

    T = history.shape[1]
    return {
        "type":                  "chronos",
        "pre_prompt":            (
            f"{ds_name + ' ' if ds_name else ''}You are given {T} timesteps "
            f"across {C} channel{'s' if C != 1 else ''}:"
        ),
        "post_prompt":           (
            f". Predict the next {pred_len} timesteps "
            f"as {N_per_ch} patch{'es' if N_per_ch != 1 else ''} per channel"
        ),
        "answer":                answer,
        "time_series":           time_series,
        "time_series_text":      time_series_text,
        "ts_already_normalized": True,   # we pre-normalized above
        "n_channels":            C,
        "_n_channels":           C,
    }


def _normalize(
    history_raw: torch.Tensor,   # [C, T]  raw (un-normalized)
) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
    """Per-channel z-normalization matching Phase2DownstreamDataset convention.

    Returns:
        history_norm : [C, T] normalized
        stats        : [(mean_c, std_c), ...] — use to denormalize predictions
    """
    C = history_raw.shape[0]
    history_norm = torch.empty_like(history_raw)
    stats: List[Tuple[float, float]] = []
    for c in range(C):
        ch = history_raw[c]
        mean = ch.mean().item()
        std  = max(ch.std().item(), 1e-2)   # Phase2DownstreamDataset clips at 1e-2
        history_norm[c] = (ch - mean) / std
        stats.append((mean, std))
    return history_norm, stats


# ---------------------------------------------------------------------------
# 3.  Teacher-forced inference (recommended for benchmarking)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _teacher_forced_single_batch(
    model: OpenTSLMSPFixPos2D,
    batch: List[dict],
    patch_size: int,
) -> torch.Tensor:
    """Run one teacher-forced inference batch.

    Returns predicted patches for each sample.  Shape per sample:
        [C, N_per_ch, patch_size]  ← forecast patches in normalized space.

    This mirrors run_teacher_forced_eval in eval_phase2_downstream.py exactly.
    """
    device  = model.device
    _fp_id  = model.future_patch_token_id
    _sep_id = model.ch_sep_token_id
    bos_id  = model.tokenizer.bos_token_id

    # --- Prompt embeddings ---
    (inputs_embeds_prompt, attention_mask_prompt,
     _, ts_mask, _, _, _) = model.pad_and_apply_batch(batch, return_targets=True)

    B = len(batch)

    # --- Answer embeddings with sincos PE injected at <future_patch> positions ---
    answers = [s["answer"] for s in batch]
    ans_tok = model.tokenizer(
        answers, return_tensors="pt", padding=True, truncation=True, max_length=4096
    )
    ans_ids  = ans_tok.input_ids.to(device)
    ans_mask = ans_tok.attention_mask.to(device)
    ans_emb  = model.llm.get_input_embeddings()(ans_ids)   # [B, A, H]

    # Inject patch-index sincos PE (exact replica of training forward pass)
    if (ans_ids == _fp_id).any():
        _pe       = model.future_patch_pe.to(dtype=ans_emb.dtype)
        _pe_delta = torch.zeros_like(ans_emb)
        for i in range(B):
            pidx = 0
            for j in range(ans_ids.size(1)):
                t = ans_ids[i, j].item()
                if t == _sep_id:
                    pidx = 0
                elif t == _fp_id:
                    _pe_delta[i, j] = _pe[pidx]
                    pidx += 1
        ans_emb = ans_emb + _pe_delta

    # --- Concatenate prompt + answer per sample ---
    new_embeds, new_masks, valid_prompt_lens = [], [], []
    for i in range(B):
        valid_i   = int(attention_mask_prompt[i].sum().item())
        a_len     = int(ans_mask[i].sum().item())
        ans_start = (1 if bos_id is not None and a_len > 0 and
                     ans_ids[i, 0].item() == bos_id else 0)
        a_eff = a_len - ans_start
        new_embeds.append(torch.cat([
            inputs_embeds_prompt[i, :valid_i],
            ans_emb[i, ans_start:a_len],
        ], dim=0))
        new_masks.append(torch.cat([
            attention_mask_prompt[i, :valid_i],
            torch.ones(a_eff, device=device, dtype=attention_mask_prompt.dtype),
        ], dim=0))
        valid_prompt_lens.append(valid_i)

    inputs_comb  = pad_sequence(new_embeds, batch_first=True)
    attn_comb    = pad_sequence(new_masks,  batch_first=True)

    # --- Single LLM forward ---
    outputs = model.llm(
        inputs_embeds=inputs_comb,
        attention_mask=attn_comb,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1].float()   # [B, L, H]

    results = []
    for i, sample in enumerate(batch):
        C        = int(sample.get("n_channels", sample.get("_n_channels", 1)))
        N_per_ch = int(
            len([t for t in ans_ids[i].tolist() if t == _fp_id]) // C
        )
        if N_per_ch == 0:
            results.append(None)
            continue

        a_len       = int(ans_mask[i].sum().item())
        ans_start_i = (1 if bos_id is not None and a_len > 0 and
                       ans_ids[i, 0].item() == bos_id else 0)
        eff_ids     = ans_ids[i, ans_start_i:a_len]
        fp_mask     = (eff_ids == _fp_id)
        fp_in_eff   = fp_mask.nonzero(as_tuple=True)[0]
        if len(fp_in_eff) == 0:
            results.append(None)
            continue

        # Map positions in eff_ids → positions in combined sequence
        fp_in_comb = fp_in_eff + valid_prompt_lens[i]
        fp_in_comb = fp_in_comb[fp_in_comb < last_hidden.shape[1]]
        if len(fp_in_comb) != C * N_per_ch:
            print(
                f"[warn] sample {i}: expected {C * N_per_ch} fp positions, "
                f"found {len(fp_in_comb)} — skipping."
            )
            results.append(None)
            continue

        H_dim   = last_hidden.size(-1)
        h_fp    = last_hidden[i, fp_in_comb, :]       # [C*N_per_ch, H]
        h_fp_3d = h_fp.view(C, N_per_ch, H_dim)       # [C, N_per_ch, H]

        # Prepend history TS hidden states per channel
        ts_mask_i = ts_mask[i, :valid_prompt_lens[i]].bool()
        h_hist    = last_hidden[i, :valid_prompt_lens[i], :][ts_mask_i]   # [T*C, H]
        if h_hist.shape[0] > 0 and h_hist.shape[0] % C == 0:
            T_hist     = h_hist.shape[0] // C
            h_hist_ch  = h_hist.reshape(T_hist, C, H_dim).permute(1, 0, 2)   # [C, T, H]
            h_combined = torch.cat([h_hist_ch, h_fp_3d], dim=1)               # [C, T+N, H]
        else:
            T_hist     = 0
            h_combined = h_fp_3d

        h_enc_flat = model.decoder_projector(h_combined.reshape(-1, H_dim))
        h_enc_flat = F.layer_norm(h_enc_flat, [h_enc_flat.size(-1)])
        h_enc_3d   = h_enc_flat.view(C, T_hist + N_per_ch, -1)
        pred_full  = model.decoder(h_enc_3d)                         # [C, T+N, P]
        pred_3d    = pred_full[:, T_hist:, :]                        # [C, N_per_ch, P]
        results.append(pred_3d)                                      # [C, N_per_ch, patch_size]

    return results


# ---------------------------------------------------------------------------
# 4.  High-level run_inference() entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: OpenTSLMSPFixPos2D,
    histories: List[np.ndarray],
    pred_len: int,
    col_names: Optional[List[str]] = None,
    ds_name: str = "",
    batch_size: int = 4,
    autoregressive: bool = False,
    denormalize: bool = False,
) -> List[np.ndarray]:
    """Forecast from a list of multivariate history windows.

    Args:
        model        : loaded OpenTSLMSPFixPos2D (from build_model).
        histories    : list of numpy arrays, each shape [C, history_len].
                       C is the number of channels (can vary across samples).
                       history_len must be a multiple of model.patch_size.
                       Pass unnormalized data — normalization is applied here.
        pred_len     : forecast horizon in time steps; must be divisible by
                       model.patch_size.
        col_names    : optional list of C column names for the text prompt.
        ds_name      : optional dataset name for the text prompt.
        batch_size   : samples to process per LLM forward pass.
        autoregressive: if True, use model.forecast_phase2() instead of
                        the teacher-forced path.
        denormalize  : if True, scale predictions back to the input series'
                       original mean/std (per channel).

    Returns:
        list of numpy arrays, each shape [C, pred_len].
        If a sample fails to produce predictions (e.g. a mismatched token
        count), None is returned for that position.
    """
    patch_size = model.patch_size
    if pred_len % patch_size != 0:
        raise ValueError(
            f"pred_len={pred_len} must be divisible by patch_size={patch_size}"
        )

    eos_token   = model.get_eos_token()
    ch_sep_tok  = model.tokenizer.convert_ids_to_tokens(model.ch_sep_token_id)
    fp_tokens   = ["<future_patch>"] * MAX_FUTURE_PATCH_PE_SIZE

    all_preds: List[Optional[np.ndarray]] = []
    all_stats:  List[Optional[List[Tuple[float, float]]]] = []

    # -- Build sample dicts and collect norm stats ---------------------------
    samples: List[dict] = []
    for hist_np in histories:
        hist_t = torch.tensor(hist_np, dtype=torch.float32)   # [C, T]
        if hist_t.dim() == 1:
            hist_t = hist_t.unsqueeze(0)   # univariate [T] → [1, T]
        if hist_t.dim() != 2:
            raise ValueError(
                f"Each history must be 1-D (univariate) or 2-D [C, T]. "
                f"Got shape {tuple(hist_t.shape)}."
            )

        C, T = hist_t.shape
        if T % patch_size != 0:
            # right-pad with zeros so T is a multiple of patch_size
            pad = patch_size - (T % patch_size)
            hist_t = F.pad(hist_t, (0, pad), value=0.0)

        hist_norm, stats = _normalize(hist_t)   # per-channel z-norm
        all_stats.append(stats)

        samples.append(_build_sample(
            history=hist_norm,
            pred_len=pred_len,
            patch_size=patch_size,
            ch_sep_token=ch_sep_tok,
            fp_tokens=fp_tokens,
            eos_token=eos_token,
            ds_name=ds_name,
            col_names=col_names,
            norm_stats=stats,
        ))

    # -- Process in batches -------------------------------------------------
    for batch_start in range(0, len(samples), batch_size):
        batch  = samples[batch_start : batch_start + batch_size]
        stats_b = all_stats[batch_start : batch_start + batch_size]

        # extend_time_series_to_match_patch_size_and_aggregate pads to a common
        # multiple of patch_size within the batch and converts list → tensor.
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size
        )
        if not batch:
            all_preds.extend([None] * len(batch))
            continue

        if autoregressive:
            preds_batch = model.forecast_phase2(
                batch, pred_len=pred_len, max_new_tokens=512
            )
            # forecast_phase2 returns List[Optional[Tensor[C, pred_len]]]
            for i, pred in enumerate(preds_batch):
                if pred is None:
                    all_preds.append(None)
                    continue
                pred_np = pred.float().cpu().numpy()          # [C, pred_len]
                if denormalize and stats_b[i] is not None:
                    for c, (m, s) in enumerate(stats_b[i]):
                        if c < pred_np.shape[0]:
                            pred_np[c] = pred_np[c] * s + m
                all_preds.append(pred_np)

        else:
            results = _teacher_forced_single_batch(model, batch, patch_size)
            for i, pred_3d in enumerate(results):
                if pred_3d is None:
                    all_preds.append(None)
                    continue
                # pred_3d: [C, N_per_ch, patch_size]  → [C, pred_len]
                pred_np = pred_3d.reshape(pred_3d.shape[0], -1).float().cpu().numpy()
                if denormalize and stats_b[i] is not None:
                    for c, (m, s) in enumerate(stats_b[i]):
                        if c < pred_np.shape[0]:
                            pred_np[c] = pred_np[c] * s + m
                all_preds.append(pred_np)

    return all_preds


# ---------------------------------------------------------------------------
# 5.  CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase-2 inference for OpenTSLMSPFixPos2D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--ckpt", required=True,
        help="Path to the phase-2 best_model.pt checkpoint.",
    )
    ap.add_argument("--llm_id",      default="google/gemma-3-270m")
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--patch_size",  type=int, default=32)
    ap.add_argument("--enc_dec_type",default="transformer")
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_enc_ff_dim",     type=int, default=256)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_dec_ff_dim",     type=int, default=256)
    ap.add_argument("--pe_type",     default="sincos")
    ap.add_argument("--pred_len",    type=int, default=96)
    ap.add_argument("--history_len", type=int, default=512,
                    help="Number of context timesteps to feed. Must be a multiple "
                         "of patch_size.")
    ap.add_argument("--n_channels",  type=int, default=1,
                    help="Number of channels in the synthetic demo series.")
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--autoregressive", action="store_true",
                    help="Use forecast_phase2() instead of teacher-forced eval.")
    ap.add_argument("--denormalize", action="store_true",
                    help="Return predictions in the original (un-normalized) scale.")
    ap.add_argument("--lora_r",   type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(
        ckpt_path=args.ckpt,
        llm_id=args.llm_id,
        patch_size=args.patch_size,
        enc_dec_type=args.enc_dec_type,
        mlp_enc_hidden_dim=args.mlp_enc_hidden_dim,
        mlp_enc_ff_dim=args.mlp_enc_ff_dim,
        mlp_dec_hidden_dim=args.mlp_dec_hidden_dim,
        mlp_dec_ff_dim=args.mlp_dec_ff_dim,
        pe_type=args.pe_type,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # ── Synthetic demo data ─────────────────────────────────────────────────
    # Replace with real data: NumPy arrays of shape [C, history_len].
    rng = np.random.default_rng(42)
    C, T = args.n_channels, args.history_len
    # Three sample windows of (optionally multi-channel) random-walk data.
    demo_histories = [
        np.cumsum(rng.standard_normal((C, T)), axis=1).astype(np.float32)
        for _ in range(3)
    ]

    print(
        f"\nRunning {'autoregressive' if args.autoregressive else 'teacher-forced'} "
        f"inference on {len(demo_histories)} demo samples...\n"
        f"  history_len={T}  pred_len={args.pred_len}  C={C}\n"
    )

    # ── Inference ───────────────────────────────────────────────────────────
    preds = run_inference(
        model=model,
        histories=demo_histories,
        pred_len=args.pred_len,
        col_names=[f"col{c}" for c in range(C)],
        ds_name="demo",
        batch_size=args.batch_size,
        autoregressive=args.autoregressive,
        denormalize=args.denormalize,
    )

    # ── Print results ───────────────────────────────────────────────────────
    for idx, pred in enumerate(preds):
        if pred is None:
            print(f"  sample {idx}: FAILED (no prediction produced)")
            continue
        print(
            f"  sample {idx}: pred shape={pred.shape}  "
            f"mean={pred.mean():.4f}  std={pred.std():.4f}  "
            f"min={pred.min():.4f}  max={pred.max():.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
