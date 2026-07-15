#!/usr/bin/env python3
"""debug_embeds.py — pin down the remaining 80% degenerate-generation root cause.

Tests four hypotheses in one GPU job:

H1  embed-scale destruction: resize_token_embeddings(262145) replaces
    Gemma3TextScaledWordEmbedding (has embed_scale=sqrt(640)≈25.3) with a plain
    nn.Embedding → text token embeddings are 25x smaller than the LLM expects;
    projector must compensate or TS conditioning is miscalibrated.

H2  do_sample=True inherited: gemma-3-270m generation_config.json ships with
    do_sample=true, top_k=64, top_p=0.95.  When generate() is called without
    explicit do_sample, these are inherited → stochastic sampling → ~20% of
    draws happen to land on the coherent chain-of-thought token; the other 80%
    drift into garbage.  FIX: force do_sample=False.

H3  BOS injection: Gemma tokenizer has add_bos_token=True, so every segment in
    all_texts gets a spurious <bos> prepended.  Spurious BOS tokens inside the
    sequence might confuse the LLM.

H4  Projector / embed magnitude mismatch: compare text_embeds norm vs ts_proj
    norm to find conditioning imbalance.

Protocol:
  - Import encoder_compat (norm_first=True) BEFORE loading
  - Monkey-patch pad_and_apply_batch to capture norm stats
  - Run 8 Sleep samples with current settings (stochastic)
  - Run the SAME 8 samples with do_sample=False (greedy)
  - Run sample-0 twice to measure determinism
  - Print all diagnostics
"""
import getpass
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/hf_eval")

import torch
import torch.nn as nn

# ── apply compat shims BEFORE model construction ──────────────────────────────
import encoder_compat  # noqa: F401  (norm_first=True pre-norm fix)

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
N_SAMPLES = 8
MAX_NEW = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
model.eval()

# ── H1 / H4: inspect embed_tokens type and collect norm stats ─────────────────
print("\n" + "=" * 60)
print("H1: embed_tokens type + scale after resize_token_embeddings")
print("=" * 60)
raw_llm = model.llm
# unwrap PEFT wrapper to reach the base model
base = getattr(raw_llm, "base_model", raw_llm)
base_model = getattr(base, "model", base)
embed_tokens = getattr(base_model, "embed_tokens", None)
print(f"type(embed_tokens)        : {type(embed_tokens)}")
embed_scale = getattr(embed_tokens, "embed_scale", None)
print(f"embed_tokens.embed_scale  : {embed_scale}")
tokenizer_len = len(model.tokenizer)
print(f"len(tokenizer)            : {tokenizer_len}")
if embed_tokens is not None and hasattr(embed_tokens, "weight"):
    w = embed_tokens.weight.float()
    print(f"embed_tokens.weight.shape : {w.shape}")
    # row norms (one per vocabulary token)
    row_norms = w.norm(dim=-1)
    print(f"  row_norm  mean={row_norms.mean():.4f}  std={row_norms.std():.4f}  "
          f"min={row_norms.min():.4f}  max={row_norms.max():.4f}")

# H2: check generation config
print("\n" + "=" * 60)
print("H2: model.llm.generation_config defaults")
print("=" * 60)
gc = model.llm.generation_config
print(f"do_sample       : {getattr(gc, 'do_sample', None)}")
print(f"top_k           : {getattr(gc, 'top_k', None)}")
print(f"top_p           : {getattr(gc, 'top_p', None)}")
print(f"temperature     : {getattr(gc, 'temperature', None)}")
print(f"cache_impl      : {getattr(gc, 'cache_implementation', None)}")
print(f"transformers_ver: {getattr(gc, 'transformers_version', None)}")

# ── Instrument pad_and_apply_batch to capture embed norms ─────────────────────
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
_norm_log = []  # list of dicts
_orig_pab = OpenTSLMSP.pad_and_apply_batch


def _instrumented_pab(self, batch, return_targets=False):
    device = self.device

    # duplicate the text tokenisation step to capture raw embeds
    all_texts = []
    for sample in batch:
        all_texts.append(sample["pre_prompt"])
        all_texts.extend(sample["time_series_text"])
        all_texts.append(sample["post_prompt"])

    tok = self.tokenizer(all_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tok.input_ids.to(device)
    attn_mask = tok.attention_mask.to(device)
    text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N, P, H]

    # H3: BOS check — each segment should start with BOS token id
    bos_id = self.tokenizer.bos_token_id
    first_tokens = input_ids[:, 0].tolist()
    n_bos = sum(1 for t in first_tokens if t == bos_id)
    n_segs = len(all_texts)

    # run the real forward to get ts_proj
    out = _orig_pab(self, batch, return_targets=return_targets)
    inputs_embeds = out[0]  # [1, L_total, H]

    # compare magnitudes
    text_norm = text_embeds.float().norm(dim=-1)          # [N, P]
    ie_norm   = inputs_embeds.float().norm(dim=-1)        # [1, L]

    _norm_log.append({
        "text_embed_norm_mean": float(text_norm.mean()),
        "text_embed_norm_std" : float(text_norm.std()),
        "full_seq_norm_mean"  : float(ie_norm.mean()),
        "full_seq_norm_std"   : float(ie_norm.std()),
        "seq_len"             : int(inputs_embeds.shape[1]),
        "n_bos_in_segments"   : n_bos,
        "n_segments"          : n_segs,
        "bos_id"              : bos_id,
    })
    return out


OpenTSLMSP.pad_and_apply_batch = _instrumented_pab

# ── Load dataset ──────────────────────────────────────────────────────────────
PS = getattr(model, "patch_size", 4)
eos = model.get_eos_token()
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)
indices = list(range(N_SAMPLES))
print(f"\nUsing first {N_SAMPLES} test samples, patch_size={PS}", flush=True)


def collate(idx):
    b = [ds[idx]]
    return extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=PS)


def looks_garbage(t):
    low = (t or "").lower()
    bad = ("penurunan", "qualquer", "<unused", "attempt attempt",
           "тельное", "linklabel", "filedialog", "ramo ", "thụ",
           "chesterchester", "підви", "gradientpanel", "lastgen")
    return (not t) or any(b in low for b in bad)


def run_pass(tag, extra_kwargs):
    """Run N_SAMPLES through generate() and report coherence/correctness."""
    _norm_log.clear()
    coh, corr = 0, 0
    results = []
    with torch.no_grad():
        for idx in indices:
            batch = collate(idx)
            preds = model.generate(batch, max_new_tokens=MAX_NEW, **extra_kwargs)
            p = preds[0]
            s = batch[0]
            g = looks_garbage(p)
            has_ans = "answer:" in (p or "").lower()
            if not g and has_ans:
                coh += 1
            gold = (s.get("label") or "").strip().lower()
            pred_after = p.lower().split("answer:")[-1].strip() if "answer:" in p.lower() else ""
            if gold and gold in pred_after:
                corr += 1
            results.append((idx, g, has_ans, p[:100]))
    print(f"\n[{tag}]  coherent={coh}/{N_SAMPLES}  correct={corr}/{N_SAMPLES}", flush=True)
    for idx, g, has_ans, snippet in results:
        tag_ = "GARB" if g else ("OK" if has_ans else "NO-ANS")
        print(f"  idx={idx:3d} [{tag_}] {repr(snippet[:90])}", flush=True)
    if _norm_log:
        mean_text   = sum(d["text_embed_norm_mean"] for d in _norm_log) / len(_norm_log)
        mean_full   = sum(d["full_seq_norm_mean"]   for d in _norm_log) / len(_norm_log)
        mean_len    = sum(d["seq_len"]              for d in _norm_log) / len(_norm_log)
        n_bos_segs  = _norm_log[0]["n_bos_in_segments"]
        n_segs      = _norm_log[0]["n_segments"]
        bos_id      = _norm_log[0]["bos_id"]
        print(f"\n  [EMBED NORMS (avg across {len(_norm_log)} samples)]")
        print(f"  text_embeds norm   : {mean_text:.4f}")
        print(f"  full_seq norm      : {mean_full:.4f}")
        print(f"  sequence length    : {mean_len:.1f} tokens")
        print(f"  BOS id             : {bos_id}")
        print(f"  BOS count/segments : {n_bos_segs}/{n_segs} "
              f"({'ALL segments get BOS' if n_bos_segs == n_segs else 'some segments missing BOS'})",
              flush=True)
    return coh, corr


# ── H2: stochastic (current) vs greedy ───────────────────────────────────────
print("\n" + "=" * 60)
print("H2: stochastic (do_sample=True, inherited) vs greedy (do_sample=False)")
print("=" * 60)

coh_stoch, corr_stoch = run_pass("stochastic (current)", {})
coh_greedy, corr_greedy = run_pass("greedy (do_sample=False)", {"do_sample": False})

# ── H2-det: same-sample determinism test (sample 0 run twice) ────────────────
print("\n" + "=" * 60)
print("Determinism test: run sample 0 twice with stochastic settings")
print("=" * 60)
_norm_log.clear()
with torch.no_grad():
    batch0 = collate(0)
    pred0a = model.generate(batch0, max_new_tokens=MAX_NEW)[0]
    pred0b = model.generate(batch0, max_new_tokens=MAX_NEW)[0]
same = pred0a == pred0b
print(f"  Run A: {repr(pred0a[:100])}", flush=True)
print(f"  Run B: {repr(pred0b[:100])}", flush=True)
print(f"  Identical: {same}  ← if False, generation is stochastic", flush=True)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"stochastic : coherent={coh_stoch}/{N_SAMPLES}  correct={corr_stoch}/{N_SAMPLES}")
print(f"greedy     : coherent={coh_greedy}/{N_SAMPLES}  correct={corr_greedy}/{N_SAMPLES}")
print(f"embed_scale present    : {embed_scale is not None}")
if embed_scale is not None:
    print(f"  embed_scale value    : {embed_scale}")
else:
    print("  embed_scale LOST (resize_token_embeddings replaced Gemma3TextScaledWordEmbedding)")
print(f"do_sample (generation_config) : {getattr(gc, 'do_sample', 'N/A')}")
print("DONE_EMBEDS", flush=True)
