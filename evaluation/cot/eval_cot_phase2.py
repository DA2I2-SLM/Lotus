#!/usr/bin/env python3
"""Evaluate OpenTSLMSPFixPos2D best_model.pt on stage 3/4/5 CoT datasets.

Stages:
  3 - HAR CoT (HARCoTQADataset)         → classification accuracy
  4 - Sleep CoT (SleepEDFCoTQADataset)  → classification accuracy
  5 - ECG-QA CoT (ECGQACoTQADataset)    → exact-match + ROUGE-L

Usage:
    python evaluation/cot/eval_cot_phase2.py \
        --ckpt $FTS_RESULTS_ROOT/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt \
        --tasks har sleep ecg \
        --max_samples 500 \
        --out_dir ./eval_results/phase2_cot/
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

# repo-relative import of the opentslm package (no personal paths)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# --- frozen-subset manifests -------------------------------------------------
# A capped eval (--max_samples N) silently takes the FIRST N samples in dataset
# order, so cross-model fairness rests on that order never changing. A manifest
# pins the exact subset: per sample, the dataset idx plus a sha1 of the gold
# answer (EOS-stripped, so it is tokenizer-agnostic). Evaluating with
# --sample_manifest runs exactly those idxs and HARD-FAILS on any hash
# mismatch — a mismatch means the dataset order/content changed and the
# comparison would be invalid.
# "<|im_end|>" is Qwen3's eos_token. Without it here, strip_eos() leaves it on
# every gold, gold_sha1() mismatches all 5000 pinned ECG rows, and the eval aborts.
_EOS_MARKERS = ("<eos>", "</s>", "<end_of_turn>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>")


def strip_eos(s: str) -> str:
    s = s.rstrip()
    changed = True
    while changed:
        changed = False
        for m in _EOS_MARKERS:
            if s.endswith(m):
                s = s[: -len(m)].rstrip()
                changed = True
    return s


def gold_sha1(gold: str) -> str:
    return hashlib.sha1(strip_eos(gold).encode("utf-8")).hexdigest()


def setup_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_geometry(ckpt: dict, args: argparse.Namespace, ckpt_path: str) -> None:
    """Compare the reconstructed architecture against the geometry the checkpoint
    was TRAINED with, and shout about any difference.

    This exists because a silent drift already happened. Training passes
    --pe_base_c 10 and the checkpoint records pe_base_c: 10.0, but this script
    built its Namespace with no pe_base_c field at all, so
    getattr(args, "pe_base_c", None) returned None and the encoder fell back to
    10000.0 (MLPEncoder2D.py:62-64). The channel positional encoding therefore
    differed between training and evaluation on every multi-channel task
    (HAR/ECG/PAMAP2; single-channel Sleep is immune, since c=0 gives sin 0 /
    cos 1 at any base). Nothing crashed and nothing warned. Loading is not
    verification — compare and complain.
    """
    geom = ckpt.get("encoder_geometry")
    if not geom:
        print("  [geometry] checkpoint records no encoder_geometry — cannot verify; "
              "confirm pe_type/pe_base_t/pe_base_c/patch_size/freq_mode by hand")
        return
    live = {
        "pe_type": getattr(args, "pe_type", None),
        "pe_base_t": float(getattr(args, "pe_base_t", None) or 10000.0),
        "pe_base_c": float(getattr(args, "pe_base_c", None) or 10000.0),
        "patch_size": int(getattr(args, "patch_size", 32)),
        "use_freq_embed": bool(getattr(args, "use_freq_embed", False)),
        "freq_mode": str(getattr(args, "freq_mode", "linear")),
    }
    drift = {k: (geom[k], live[k]) for k in live if k in geom and geom[k] != live[k]}
    if drift:
        print("  " + "!" * 68)
        print("  [geometry] TRAINED-VS-EVAL MISMATCH — predictions will be degraded:")
        for k, (was, now) in drift.items():
            print(f"      {k}: trained with {was!r}, evaluating with {now!r}")
        print(f"      checkpoint: {ckpt_path}")
        print("  " + "!" * 68)
    else:
        print(f"  [geometry] OK — matches training ({geom})")


def load_model(ckpt_path: str, llm_id: str, device: str,
               lora_r: int = 4, lora_alpha: int = 8, use_freq: bool = False,
               arch: dict | None = None, model_class: str = "fixpos2d"):
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

    if model_class == "official_fl":
        # OpenTSLM-Flamingo baseline from curriculum_learning (qwen3 arms).
        # Checkpoint = one full model_state dict (no LoRA; Flamingo trains
        # perceiver + gated cross-attn instead). patch_size from the ckpt's
        # own encoder_geometry. generate() returns List[str] -- wrapped below.
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ps = int((ckpt.get("encoder_geometry") or {}).get("patch_size", 32))
        print(f"Initialising OpenTSLMFlamingo (official baseline) LLM={llm_id} patch={ps}")
        model = OpenTSLMFlamingo(device=device, llm_id=llm_id, patch_size=ps).to(device)
        miss = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  model_state loaded (missing={len(miss.missing_keys)}, "
              f"unexpected={len(miss.unexpected_keys)})")
        assert not miss.unexpected_keys, miss.unexpected_keys[:5]
        print(f"  ckpt epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}")
        model.eval()
        _orig_gen = model.generate
        def _gen(batch, max_new_tokens=200, **kw):
            kw.pop("n_forecast_patches", None)
            # Flamingo's generate sets eos itself and forwards extra kwargs to
            # open_flamingo, whose generate() rejects these HF-style knobs.
            kw.pop("eos_token_id", None); kw.pop("pad_token_id", None)
            fixed = []
            for r in batch:
                r = dict(r)
                ts = r.get("time_series")
                if isinstance(ts, (list, tuple)):
                    r["time_series"] = [
                        t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32)
                        for t in ts]
                fixed.append(r)
            from opentslm.time_series_datasets.util import (
                extend_time_series_to_match_patch_size_and_aggregate as _pad)
            fixed = _pad(fixed, patch_size=ps)
            return _orig_gen(fixed, max_new_tokens=max_new_tokens, **kw), [None] * len(batch)
        model.generate = _gen
        return model

    if model_class == "official_sp":
        # OpenTSLM-SP baseline checkpoints trained through curriculum_learning
        # (qwen3 official arms). The class carries its own encoder geometry
        # (768-d, learned pos_embed), ignores the arch dict, takes no args, and
        # its generate() returns List[str] rather than the (texts, ts) tuple
        # this script unpacks -- wrap it. LoRA r/alpha come from the ckpt's own
        # lora_config, not the CLI defaults.
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
        print(f"Initialising OpenTSLMSP (official baseline) with LLM={llm_id}")
        model = OpenTSLMSP(llm_id=llm_id, device=device).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("lora_enabled", False) and "lora_state" in ckpt:
            lc = ckpt.get("lora_config") or {}
            r = int(lc.get("r", lora_r)); al = int(lc.get("lora_alpha", lora_alpha))
            print(f"  enabling LoRA r={r} alpha={al} (from ckpt lora_config)")
            model.enable_lora(lora_r=r, lora_alpha=al)
        for name in ("encoder_state", "projector_state"):
            getattr(model, name.replace("_state", "")).load_state_dict(ckpt[name])
        model.load_lora_state_from_checkpoint(ckpt, allow_missing=False)
        print(f"  ckpt epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}")
        model.eval()
        _orig_gen = model.generate
        def _gen(batch, max_new_tokens=200, **kw):
            kw.pop("n_forecast_patches", None)
            fixed = []
            for r in batch:
                # this class's pad_and_apply_batch requires torch tensors;
                # the eval datasets hand over numpy arrays
                r = dict(r)
                ts = r.get("time_series")
                if isinstance(ts, (list, tuple)):
                    r["time_series"] = [
                        t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32)
                        for t in ts]
                elif ts is not None and not torch.is_tensor(ts):
                    r["time_series"] = torch.as_tensor(ts, dtype=torch.float32)
                fixed.append(r)
            # the official encoder rejects lengths not divisible by patch_size;
            # apply the SAME padding collate the curriculum used at train time
            from opentslm.time_series_datasets.util import (
                extend_time_series_to_match_patch_size_and_aggregate as _pad)
            fixed = _pad(fixed, patch_size=32)
            return _orig_gen(fixed, max_new_tokens=max_new_tokens, **kw), [None] * len(batch)
        model.generate = _gen
        return model

    # Defaults reproduce this script's historical behaviour exactly, so every
    # existing gemma-3-270m MLP-encoder eval replays byte-identically.
    a = {
        "pe_type": "sincos", "pe_base_t": None, "pe_base_c": None,
        "patch_size": 32, "freq_mode": "linear",
        "enc_dec_type": "mlp",
        "mlp_enc_hidden_dim": 64, "mlp_enc_ff_dim": 256,
        "mlp_dec_hidden_dim": 64, "mlp_dec_ff_dim": 256,
        "trans_enc_dim": None, "trans_enc_num_layers": None,
        "trans_enc_num_heads": None, "trans_enc_ff_dim": None,
        "trans_enc_output_dim": None,
        "trans_dec_dim": None, "trans_dec_num_layers": None,
        "trans_dec_num_heads": None, "trans_dec_ff_dim": None,
    }
    a.update(arch or {})
    args = argparse.Namespace(
        use_only_time_pe=False,
        use_only_channel_pe=False,
        is_pretraining=False,
        gradient_checkpointing=False,
        use_freq_embed=use_freq,   # freq checkpoints (e.g. chronos_mse_freq) need this
        **a,
    )
    print(f"Initialising OpenTSLMSPFixPos2D with LLM={llm_id} "
          f"(enc_dec_type={a['enc_dec_type']}, use_freq_embed={use_freq})")
    model = OpenTSLMSPFixPos2D(args=args, llm_id=llm_id, device=device).to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check_geometry(ckpt, args, ckpt_path)

    # CRITICAL: enable LoRA BEFORE loading checkpoint. Without this, the call
    # to load_lora_state_from_checkpoint raises and gets swallowed by the
    # try/except below, leaving the base un-fine-tuned LLM (→ garbage output).
    if ckpt.get("lora_enabled", False) and "lora_state" in ckpt:
        print(f"  enabling LoRA r={lora_r} alpha={lora_alpha}")
        model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)

    # Resized LLM token embeddings — load BEFORE LoRA state so PEFT sees
    # the final embedding shape.
    if "llm_input_embeddings" in ckpt:
        model.llm.get_input_embeddings().weight.data.copy_(
            ckpt["llm_input_embeddings"].to(device)
        )
    if "llm_output_embeddings" in ckpt:
        out_emb = model.llm.get_output_embeddings()
        if out_emb is not None:
            out_emb.weight.data.copy_(ckpt["llm_output_embeddings"].to(device))

    # Encoder/projector/decoder/decoder_projector/mse_projector
    for name in (
        "encoder_state", "projector_state", "decoder_state",
        "decoder_projector_state", "mse_projector_state",
    ):
        if name in ckpt:
            getattr(model, name.replace("_state", "")).load_state_dict(ckpt[name])

    # Fully fine-tuned backbone, if this checkpoint came from the full-FT arm.
    # Must precede the LoRA call: a full-FT checkpoint has no adapter at all.
    if ckpt.get("llm_full_ft") and "llm_state" in ckpt:
        miss = model.llm.load_state_dict(ckpt["llm_state"], strict=False)
        print(f"  restored fully fine-tuned LLM (missing={len(miss.missing_keys)}, "
              f"unexpected={len(miss.unexpected_keys)})")

    # LoRA adapters (will raise if enable_lora wasn't called above).
    # allow_missing for full-FT: that arm legitimately has no adapter.
    model.load_lora_state_from_checkpoint(
        ckpt, allow_missing=bool(ckpt.get("llm_full_ft")))

    print(f"  ckpt epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}")
    model.eval()
    return model


# normalise() strips punctuation but "_" is a word char, so "<|im_end|>" survives as
# the token "im_end" in Qwen3 golds and can never be matched by a prediction decoded
# with skip_special_tokens=True -- a pure recall penalty on that backbone only.
_SPECIAL_TOK = {"eos", "bos", "pad", "unk", "ts", "im_end", "im_start",
                "endoftext", "end_of_turn", "eot_id"}


def normalise(s: str) -> str:
    """Lowercase + strip + collapse whitespace + remove punctuation + drop leaked
    special tokens (e.g. gold 'shifting seasonal<eos>' -> 'shifting seasonal')."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in _SPECIAL_TOK).strip()


_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*([^\n.<]+)", re.IGNORECASE)


def extract_answer_suffix(text: str) -> str:
    """Pull the post-'Answer:' suffix. Falls back to last short line."""
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    # fallback: last non-empty line, if it's short (likely a label)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 40:
        return lines[-1].rstrip(".").strip()
    return ""


def extract_label_from_text(text: str, candidates: List[str]) -> str:
    """First try `Answer: <label>` extraction, then fall back to substring match.

    LONGEST-FIRST PREFIX MATCH, and never return the raw suffix. _ANSWER_RE
    captures greedily up to a newline/period/'<', so a model that answers and
    then keeps generating with no separator produced
    "noEXAMINE THE ECG AND ANSWER: No" for gold "no" -- which then failed exact
    match. That penalised only models which fail to stop cleanly: our rows were
    unaffected while OpenTSLM-SP lost 28pp of ECG (9.72 -> 37.84 once fixed),
    so the ECG comparison was partly measuring stopping behaviour, in our
    favour. Longest-first matters so "none" wins over "no" on "noneEXAMINE...".
    Cross-checked against an independent LLM judge, which put SP-ECG at 37.50%.
    """
    if not text:
        return ""
    suffix = extract_answer_suffix(text)
    if suffix and candidates:
        n_suffix = normalise(suffix)
        by_len = sorted(candidates, key=lambda c: -len(normalise(c)))
        for c in by_len:
            if normalise(c) == n_suffix:
                return c
        for c in by_len:                       # answer runs into trailing text
            if n_suffix.startswith(normalise(c)):
                return c
        for c in by_len:
            if normalise(c) in n_suffix or n_suffix in normalise(c):
                return c
        return ""  # do NOT return the uncut suffix; it can never exact-match
    if suffix:
        return suffix
    # last resort: any candidate substring in the whole text
    t = normalise(text)
    for c in candidates:
        if normalise(c) in t:
            return c
    return ""


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 via LCS over whitespace tokens."""
    a = normalise(pred).split()
    b = normalise(gold).split()
    if not a or not b:
        return 0.0
    # LCS DP
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def collect_labels(dataset, label_field_keys=("answer", "rationale")) -> List[str]:
    """Walk a few samples to gather candidate classification labels.
    Heuristic: short, repeated answers ⇒ class names."""
    counts = defaultdict(int)
    for i, row in enumerate(dataset.dataset):
        if i >= 2000:
            break
        for k in label_field_keys:
            v = row.get(k) if isinstance(row, dict) else None
            if isinstance(v, str) and len(v) <= 30:
                counts[v.strip()] += 1
    # Keep candidates that appear at least 3 times (likely class labels).
    cands = sorted([k for k, c in counts.items() if c >= 3], key=lambda k: -counts[k])
    return cands[:30]


def compute_f1(label_golds: List[str], label_preds: List[str]) -> dict:
    """Per-class precision/recall/F1 + macro-F1 + accuracy."""
    classes = sorted({normalise(l) for l in label_golds if l})
    per_class = {}
    tp_total = fp_total = fn_total = 0
    for c in classes:
        tp = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) == c and normalise(p) == c)
        fp = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) != c and normalise(p) == c)
        fn = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) == c and normalise(p) != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for g in label_golds if normalise(g) == c)}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(per_class)) if per_class else 0.0
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    accuracy = sum(1 for g, p in zip(label_golds, label_preds)
                   if normalise(g) == normalise(p) and g) / max(len(label_golds), 1)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


def eval_task(model, dataset, task_name: str, label_candidates: List[str],
              max_samples: int, max_new_tokens: int, batch_size: int = 8,
              manifest: dict = None) -> dict:
    if manifest is not None:
        indices = [int(s["idx"]) for s in manifest["samples"]]
        expected_sha1 = [s["sha1"] for s in manifest["samples"]]
        n = len(indices)
    else:
        indices = None
        expected_sha1 = None
        n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    # Optional shard range via env (EVAL_SHARD_START/EVAL_SHARD_END) so slow tasks
    # (ECG 41k) can be split across jobs. Applied over the [0,n) index space.
    # ── series-reliance perturbations (Merrill-style) ───────────────────────
    # PERTURB_SERIES in {shuffle, gauss, zero, patchshuffle} destroys the
    # temporal signal at
    # inference while leaving the text prompt untouched. If accuracy under
    # perturbation matches the clean run, the model is answering from the
    # question text, not the series. Deterministic per (idx, mode) so reruns
    # and shards agree. Captions (which embed mean/std as text) are left as-is:
    # the test is "does the model USE the numbers it is given".
    _pmode = os.environ.get("PERTURB_SERIES", "")
    _PERTURB_MODES = {"shuffle", "gauss", "zero", "patchshuffle",
                      "bandstop", "phaserand", "specshuffle"}
    if _pmode:
        if _pmode not in _PERTURB_MODES:
            # an unknown mode would silently fall through every branch and
            # produce a CLEAN run mislabeled as perturbed
            raise ValueError(f"unknown PERTURB_SERIES mode {_pmode!r}; allowed: {sorted(_PERTURB_MODES)}")
        import numpy as _np
        print(f"  [perturb] PERTURB_SERIES={_pmode} — series destroyed at inference")
        def _perturb_row(row, ds_idx):
            row = dict(row)
            out = []
            for ci, ts in enumerate(row.get("time_series", []) or []):
                v = _np.asarray(ts, dtype=float).copy()
                rng = _np.random.default_rng(hash((ds_idx, ci, _pmode)) % (2**32))
                if _pmode == "shuffle":
                    rng.shuffle(v)
                elif _pmode == "gauss":
                    v = rng.standard_normal(v.shape) * (v.std() + 1e-6) + v.mean()
                elif _pmode == "zero":
                    v = _np.zeros_like(v)
                elif _pmode == "bandstop":
                    # Zero ONE frequency band, leave the rest intact. Sleep
                    # stages are DEFINED by band power (delta 0.5-2 = N3,
                    # spindles 11-16 = N2, alpha 8-12 = Wake), so a model that
                    # genuinely reads the spectrum should lose accuracy on the
                    # stages whose defining band was removed and not on others.
                    lo, hi = (float(x) for x in
                              os.environ.get("PERTURB_BAND", "0.5-2").split("-"))
                    fs = float(os.environ.get("PERTURB_FS", "50"))
                    F = _np.fft.rfft(v)
                    fr = _np.fft.rfftfreq(v.shape[0], d=1.0 / fs)
                    F[(fr >= lo) & (fr <= hi)] = 0
                    v = _np.fft.irfft(F, n=v.shape[0])
                elif _pmode == "phaserand":
                    # Randomise phase, PRESERVE the power spectrum exactly:
                    # destroys waveform morphology, leaves band power untouched.
                    F = _np.fft.rfft(v)
                    mag = _np.abs(F)
                    ph = rng.uniform(-_np.pi, _np.pi, size=mag.shape)
                    ph[0] = 0.0
                    if v.shape[0] % 2 == 0:
                        ph[-1] = 0.0
                    v = _np.fft.irfft(mag * _np.exp(1j * ph), n=v.shape[0])
                elif _pmode == "specshuffle":
                    # Mirror image of phaserand: permute the magnitude spectrum
                    # and keep phase, destroying band structure.
                    F = _np.fft.rfft(v)
                    mag = _np.abs(F).copy(); ph = _np.angle(F)
                    mag[1:] = mag[rng.permutation(mag.shape[0] - 1) + 1]
                    v = _np.fft.irfft(mag * _np.exp(1j * ph), n=v.shape[0])
                elif _pmode == "patchshuffle":
                    # permute whole encoder patches (32 samples, = --patch_size
                    # for every ckpt evaluated here), preserving within-patch
                    # content; a tail shorter than one patch stays in place so
                    # the patch count and pad mask are unchanged
                    P = 32
                    nb = v.shape[0] // P
                    if nb > 1:
                        perm = rng.permutation(nb)
                        v = _np.concatenate(
                            [v[: nb * P].reshape(nb, P)[perm].reshape(-1), v[nb * P:]])
                out.append(v)
            row["time_series"] = out
            return row

    gen_rp = float(os.environ.get("GEN_REP_PENALTY", "1.0"))
    gen_ng = int(os.environ.get("GEN_NO_REPEAT_NGRAM", "0"))
    if gen_rp != 1.0 or gen_ng:
        print(f"  [decoding] repetition_penalty={gen_rp} no_repeat_ngram_size={gen_ng}")
    _ss = int(os.environ.get("EVAL_SHARD_START", "0"))
    _se = int(os.environ.get("EVAL_SHARD_END", "-1"))
    _shard_lo = max(0, _ss)
    _shard_hi = n if _se < 0 else min(_se, n)
    records = []
    label_golds: List[str] = []
    label_preds: List[str] = []
    rouge_sum = 0.0
    manifest_mismatches: List[int] = []
    with torch.no_grad():
        with tqdm(total=_shard_hi - _shard_lo, desc=f"Eval {task_name}") as pbar:
            for start in range(_shard_lo, _shard_hi, batch_size):
                end = min(start + batch_size, _shard_hi)
                # positions [start,end) map to dataset idxs directly, or through
                # the manifest's pinned idx list when one is supplied.
                batch_rows = [dataset[indices[i] if indices is not None else i]
                              for i in range(start, end)]
                if _pmode:
                    batch_rows = [_perturb_row(r, indices[start + j] if indices is not None else start + j)
                                  for j, r in enumerate(batch_rows)]
                try:
                    text_outs, _ = model.generate(batch_rows, max_new_tokens=max_new_tokens,
                        repetition_penalty=gen_rp, no_repeat_ngram_size=gen_ng)
                except torch.cuda.OutOfMemoryError:
                    # Fall back to per-sample for this batch only.
                    torch.cuda.empty_cache()
                    text_outs = []
                    for r in batch_rows:
                        try:
                            t, _ = model.generate([r], max_new_tokens=max_new_tokens,
                        repetition_penalty=gen_rp, no_repeat_ngram_size=gen_ng)
                            text_outs.append(t[0] if t else "")
                        except Exception as e:
                            text_outs.append(f"<error: {type(e).__name__}>")
                except Exception as e:
                    # Print the FIRST traceback loudly — a systematic generate()
                    # failure otherwise degrades into silent all-zero metrics
                    # (every pred '<error: ...>'), which looks like a bad model
                    # instead of a broken harness.
                    if not getattr(eval_task, "_printed_gen_error", False):
                        eval_task._printed_gen_error = True
                        import traceback
                        print(f"\n⚠️  model.generate failed ({type(e).__name__}) — "
                              f"first traceback (further ones suppressed):")
                        traceback.print_exc()
                    text_outs = [f"<error: {type(e).__name__}>"] * len(batch_rows)
                for j, row in enumerate(batch_rows):
                    pos = start + j
                    ds_idx = indices[pos] if indices is not None else pos
                    pred = text_outs[j] if j < len(text_outs) and text_outs[j] is not None else ""
                    gold = row.get("answer", "")
                    if expected_sha1 is not None and gold_sha1(gold) != expected_sha1[pos]:
                        manifest_mismatches.append(ds_idx)
                    rl = rouge_l(pred, gold)
                    rouge_sum += rl
                    label_gold = (extract_answer_suffix(gold) or row.get("label") or "").strip()
                    label_pred = extract_label_from_text(pred, label_candidates)
                    label_golds.append(label_gold)
                    label_preds.append(label_pred)
                    records.append({
                        "idx": ds_idx,
                        "gold": gold,
                        "pred": pred,
                        "label_gold": label_gold,
                        "label_pred": label_pred,
                        "rouge_l": rl,
                    })
                pbar.update(end - start)
    if manifest_mismatches:
        raise RuntimeError(
            f"sample manifest mismatch for {task_name}: {len(manifest_mismatches)} of "
            f"{n} golds differ from the pinned sha1s (first idxs: "
            f"{manifest_mismatches[:10]}). Dataset order/content changed since the "
            f"manifest was frozen — results would NOT be comparable. Aborting."
        )
    cls_metrics = compute_f1(label_golds, label_preds)
    metrics = {
        "task": task_name,
        "n_evaluated": n,
        "sample_manifest": (manifest or {}).get("name"),  # None = plain first-N cap
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "micro_f1": cls_metrics["micro_f1"],
        "rouge_l_mean": rouge_sum / max(n, 1),
        "n_classes": len(cls_metrics["per_class"]),
        "n_label_candidates": len(label_candidates),
        "per_class": cls_metrics["per_class"],
    }
    return {"metrics": metrics, "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_model.pt")
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=["har", "sleep", "ecg"],
                    choices=["har", "sleep", "ecg", "pamap2", "tsqa", "timemqa"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max_samples", type=int, default=500,
                    help="cap per task; 0 = full split (slow)")
    ap.add_argument("--sample_manifest", default=None,
                    help="path to a frozen-subset manifest json; evaluates exactly "
                         "the pinned sample idxs and hard-fails if any gold's sha1 "
                         "differs (fair cross-model comparison). Overrides "
                         "--max_samples; requires a single matching --tasks entry.")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--model_class", default="fixpos2d",
                    choices=["fixpos2d", "official_sp", "official_fl"],
                    help="official_sp = OpenTSLM-SP baseline ckpts from curriculum_learning")
    # Decoding guards. OpenTSLMSPFixPos2D.generate() already implements both
    # (lines ~1620, ~1732) but defaults them to no-ops, and this script never
    # passed them -- so every eval so far ran pure greedy argmax. On ECG that
    # is 384 soft tokens of prefix with no repetition control, and ours-1b
    # collapsed into "TheTheThe..." loops on 32% of samples (0% on HAR/Sleep/
    # PAMAP2, whose prefixes are 12-47 tokens). Default OFF so prior runs
    # reproduce exactly; when enabled it MUST be enabled for every model in the
    # comparison, not just the one it helps.
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=0)
    ap.add_argument("--use_freq_embed", action="store_true",
                    help="set for freq checkpoints (e.g. chronos_mse_freq warm-start)")
    ap.add_argument("--out_dir",
                    default=os.environ.get("EVAL_OUT_DIR")
                    or os.path.join(os.environ.get("FTS_RESULTS_ROOT", "results"), "eval_phase2_cot"))

    # ── architecture ─────────────────────────────────────────────────────────
    # Every default below is the value this script hardcoded before these flags
    # existed, so omitting all of them reproduces prior runs exactly. Supply them
    # to evaluate a non-MLP / non-270m checkpoint (e.g. the gemma-3-1b run, which
    # needs --enc_dec_type transformer --trans_enc_dim 192 --trans_enc_num_layers 6
    # --trans_enc_num_heads 8 --trans_enc_ff_dim 896 --trans_dec_dim 192
    # --trans_dec_ff_dim 896 --pe_base_c 10 --lora_r 8 --lora_alpha 16).
    ap.add_argument("--enc_dec_type", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--pe_type", default="sincos")
    ap.add_argument("--pe_base_t", type=float, default=None,
                    help="None -> 10000.0. MUST match training: the encoder reads "
                         "this and a mismatch silently shifts the positional code.")
    ap.add_argument("--pe_base_c", type=float, default=None,
                    help="None -> 10000.0. Training scripts pass 10; see the "
                         "check_geometry() docstring for the drift this caused.")
    ap.add_argument("--freq_mode", default="linear", choices=["linear", "spectral"])
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_enc_ff_dim", type=int, default=256)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_dec_ff_dim", type=int, default=256)
    ap.add_argument("--trans_enc_dim", type=int, default=None)
    ap.add_argument("--trans_enc_num_layers", type=int, default=None)
    ap.add_argument("--trans_enc_num_heads", type=int, default=None)
    ap.add_argument("--trans_enc_ff_dim", type=int, default=None)
    ap.add_argument("--trans_enc_output_dim", type=int, default=None,
                    help="H4a arm: transformer at d=64 presenting the 768-wide interface")
    ap.add_argument("--trans_dec_dim", type=int, default=None)
    ap.add_argument("--trans_dec_num_layers", type=int, default=None)
    ap.add_argument("--trans_dec_num_heads", type=int, default=None)
    ap.add_argument("--trans_dec_ff_dim", type=int, default=None)
    args = ap.parse_args()

    manifest = None
    if args.sample_manifest:
        with open(args.sample_manifest) as f:
            manifest = json.load(f)
        if len(args.tasks) != 1 or args.tasks[0] != manifest["task"]:
            raise SystemExit(
                f"--sample_manifest is for task '{manifest['task']}' only; "
                f"got --tasks {args.tasks}. Run it as a single-task invocation.")
        if args.split != manifest["split"]:
            raise SystemExit(
                f"--sample_manifest was frozen on split '{manifest['split']}' "
                f"but --split is '{args.split}'.")
        print(f"Sample manifest: {manifest.get('name')} "
              f"({len(manifest['samples'])} pinned {manifest['task']}/{manifest['split']} samples)")

    os.environ.setdefault("GEN_REP_PENALTY", str(args.repetition_penalty))
    os.environ.setdefault("GEN_NO_REPEAT_NGRAM", str(args.no_repeat_ngram_size))
    device = setup_device()
    print(f"Device: {device}")
    model = load_model(args.ckpt, args.llm_id, device,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                       use_freq=args.use_freq_embed,
                       model_class=args.model_class,
                       arch={k: getattr(args, k) for k in (
                           "pe_type", "pe_base_t", "pe_base_c", "patch_size",
                           "freq_mode", "enc_dec_type",
                           "mlp_enc_hidden_dim", "mlp_enc_ff_dim",
                           "mlp_dec_hidden_dim", "mlp_dec_ff_dim",
                           "trans_enc_dim", "trans_enc_num_layers",
                           "trans_enc_num_heads", "trans_enc_ff_dim",
                           "trans_enc_output_dim",
                           "trans_dec_dim", "trans_dec_num_layers",
                           "trans_dec_num_heads", "trans_dec_ff_dim")})
    EOS = model.get_eos_token() if hasattr(model, "get_eos_token") else ""

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    for task in args.tasks:
        print(f"\n===== Stage: {task} =====")
        if task == "har":
            from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
            ds = HARCoTQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "sleep":
            from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
            ds = SleepEDFCoTQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "ecg":
            from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
            ds = ECGQACoTQADataset(split=args.split, EOS_TOKEN=EOS,
                                   preload_processed_data=False)
        elif task == "pamap2":
            from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
            ds = PAMAP2CoTQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "tsqa":
            from opentslm.time_series_datasets.tsqa.TSQAClsQADataset import TSQAClsQADataset
            ds = TSQAClsQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "timemqa":
            from opentslm.time_series_datasets.timemqa.TimeMQAOpenQADataset import TimeMQAOpenQADataset
            ds = TimeMQAOpenQADataset(split=args.split, EOS_TOKEN=EOS)
        else:
            raise ValueError(f"unknown task: {task}")

        # For classification tasks (HAR/Sleep), pull the canonical label list
        # from the dataset class itself. The earlier heuristic `collect_labels`
        # mined short `answer` strings but CoT `answer` is the full rationale
        # (hundreds of chars), yielding zero candidates. The QADataset subclasses
        # expose `get_labels()` returning the canonical N-class label list.
        label_candidates: List[str] = []
        if task == "har":
            from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
            label_candidates = HARCoTQADataset.get_labels()
        elif task == "sleep":
            from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
            label_candidates = SleepEDFCoTQADataset.get_labels()
        elif task == "pamap2":
            from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
            label_candidates = PAMAP2CoTQADataset.get_labels()
        elif task == "tsqa":
            from opentslm.time_series_datasets.tsqa.TSQAClsQADataset import TSQAClsQADataset
            label_candidates = TSQAClsQADataset.get_labels()
        if task in ("har", "sleep", "pamap2", "tsqa"):
            print(f"  label candidates ({len(label_candidates)}): {label_candidates}")

        result = eval_task(
            model, ds, task, label_candidates,
            args.max_samples, args.max_new_tokens,
            manifest=manifest,
        )
        summary[task] = result["metrics"]
        # Persist per-task predictions
        out_path = os.path.join(args.out_dir, f"{task}_{args.split}_predictions.jsonl")
        with open(out_path, "w") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec) + "\n")
        m = result["metrics"]
        print(f"  ✓ {task}: accuracy={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  (n={m['n_evaluated']})")
        print(f"  predictions → {out_path}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Summary: {summary_path}")
    print(f"\n{'Task':10s}  {'Accuracy':>10s}  {'F1':>8s}")
    print("-" * 32)
    for task, m in summary.items():
        print(f"{task:10s}  {m['accuracy']:>10.3f}  {m['macro_f1']:>8.3f}")


if __name__ == "__main__":
    main()
