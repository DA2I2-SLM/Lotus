#!/usr/bin/env python3
"""debug_encoder_var.py

Root-cause the third failure mode: even with encoder_compat (pre-norm) + greedy decoding,
5/8 samples produce garbage AND 3 coherent outputs are WORD-FOR-WORD IDENTICAL.
That identical-text signature points to near-constant encoder output across samples —
the TS patches supply no discriminating conditioning to the LLM.

This script measures:
  A) Encoder output (before projector) per sample: mean per-position norm and
     inter-sample cosine similarity of the mean vector.
  B) ts_proj output per sample: same stats.
  C) Intra-sample variation: std of patch representations across the 375 positions.
  D) For reference: the label and ts_header mean/std text for each sample,
     to check whether the garbage/coherent split correlates with text features
     (which would indicate LLM is using text not TS patches).

If encoder inter-sample cosine similarity ≈ 1.0  →  near-constant output (the problem).
If intra-sample std ≈ 0 for all positions  →  uniform-attention averaging happening.
"""
import getpass, sys, math
USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/hf_eval")

import encoder_compat  # noqa: F401

import torch
import torch.nn.functional as F

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO  = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
N     = 8
MAX_NEW = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}", flush=True)

model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
model.eval()

PS  = getattr(model, "patch_size", 4)
eos = model.get_eos_token()
ds  = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)

def collate(i):
    b = [ds[i]]
    return extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=PS)

# ── Hook to capture raw encoder output and ts_proj ────────────────────────────
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

_enc_outs  = []  # list of [N_patches, enc_dim] tensors
_proj_outs = []  # list of [N_patches, 640] tensors
_orig_enc  = None
_orig_proj = None

def _hook_forward():
    # Wrap encoder and projector forward to capture output
    global _orig_enc, _orig_proj

    _orig_enc  = model.encoder.forward
    _orig_proj = model.projector.forward

    def enc_wrapper(x):
        out = _orig_enc(x)
        _enc_outs.append(out.detach().cpu())
        return out

    def proj_wrapper(x):
        out = _orig_proj(x)
        _proj_outs.append(out.detach().cpu())
        return out

    model.encoder.forward  = enc_wrapper
    model.projector.forward = proj_wrapper

_hook_forward()

# ── Run inference on N samples, greedy ────────────────────────────────────────
outputs = []  # (label, ts_header, pred_text)
_enc_outs.clear()
_proj_outs.clear()

with torch.no_grad():
    for i in range(N):
        batch = collate(i)
        sample = batch[0]
        preds = model.generate(batch, max_new_tokens=MAX_NEW, do_sample=False)
        label = (sample.get("label") or "?").strip()
        ts_texts = sample.get("time_series_text", ["?"])
        ts_hdr = ts_texts[0][:80] if ts_texts else "?"
        outputs.append((label, ts_hdr, preds[0] or ""))
        print(f"  idx={i}  label={label!r:<20s}  pred[:80]={repr((preds[0] or '')[:80])}", flush=True)

print(f"\nCaptured {len(_enc_outs)} encoder outputs, {len(_proj_outs)} projector outputs", flush=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("A) Encoder output stats per sample")
print("=" * 60)
enc_means = []  # mean vector across patches per sample [enc_dim]
for i, eo in enumerate(_enc_outs):
    # eo shape: [1, N_patches, enc_dim] or [N_patches, enc_dim]
    if eo.dim() == 3:
        eo = eo.squeeze(0)
    norm_per_patch = eo.float().norm(dim=-1)      # [N_patches]
    mean_vec = eo.float().mean(dim=0)             # [enc_dim]  global mean across patches
    intra_std = eo.float().std(dim=0).mean()      # avg std across patch positions
    enc_means.append(mean_vec)
    label = outputs[i][0]
    print(f"  idx={i}  label={label!r:<14s}  n_patches={eo.shape[0]}  "
          f"norm_mean={norm_per_patch.mean():.4f}  norm_std={norm_per_patch.std():.4f}  "
          f"intra_dim_std={intra_std:.5f}")

print("\nEncoder inter-sample cosine similarities (mean-vector):")
for i in range(N):
    row = []
    for j in range(N):
        c = F.cosine_similarity(enc_means[i].unsqueeze(0), enc_means[j].unsqueeze(0)).item()
        row.append(f"{c:+.4f}")
    print(f"  idx={i}: " + "  ".join(row))

print("\n" + "=" * 60)
print("B) ts_proj output stats per sample")
print("=" * 60)
proj_means = []
for i, po in enumerate(_proj_outs):
    if po.dim() == 3:
        po = po.squeeze(0)
    norm_per_patch = po.float().norm(dim=-1)
    mean_vec = po.float().mean(dim=0)
    intra_std = po.float().std(dim=0).mean()
    proj_means.append(mean_vec)
    label = outputs[i][0]
    print(f"  idx={i}  label={label!r:<14s}  n_patches={po.shape[0]}  "
          f"norm_mean={norm_per_patch.mean():.4f}  norm_std={norm_per_patch.std():.4f}  "
          f"intra_dim_std={intra_std:.5f}")

print("\nts_proj inter-sample cosine similarities (mean-vector):")
for i in range(N):
    row = []
    for j in range(N):
        c = F.cosine_similarity(proj_means[i].unsqueeze(0), proj_means[j].unsqueeze(0)).item()
        row.append(f"{c:+.4f}")
    print(f"  idx={i}: " + "  ".join(row))

print("\n" + "=" * 60)
print("C) Label and ts_header text for each sample (coherent vs garbage)")
print("=" * 60)
garbage_ids = set()
for i, (label, ts_hdr, pred) in enumerate(outputs):
    bad_words = ("penurunan", "qualquer", "<unused", "тельное", "linklabel",
                 "filedialog", "ramo ", "thụ", "підви", "lastgen")
    is_garb = any(b in pred.lower() for b in bad_words)
    status = "GARB" if is_garb else "OK  "
    if is_garb:
        garbage_ids.add(i)
    print(f"  idx={i} [{status}]  label={label!r:<14s}  ts_hdr={ts_hdr!r}")

print("\n" + "=" * 60)
print("D) Per-sample encoder/projector mean comparison: coherent vs garbage")
print("=" * 60)
ok_ids   = [i for i in range(N) if i not in garbage_ids]
garb_ids = list(garbage_ids)
print(f"  Coherent idx: {ok_ids}    Garbage idx: {garb_ids}")

if ok_ids and garb_ids:
    # Average mean-vector for coherent vs garbage
    ok_enc   = torch.stack([enc_means[i] for i in ok_ids]).mean(0)
    garb_enc = torch.stack([enc_means[i] for i in garb_ids]).mean(0)
    cos_enc  = F.cosine_similarity(ok_enc.unsqueeze(0), garb_enc.unsqueeze(0)).item()
    ok_proj   = torch.stack([proj_means[i] for i in ok_ids]).mean(0)
    garb_proj = torch.stack([proj_means[i] for i in garb_ids]).mean(0)
    cos_proj  = F.cosine_similarity(ok_proj.unsqueeze(0), garb_proj.unsqueeze(0)).item()
    print(f"  Encoder mean-vec cosine(ok vs garb): {cos_enc:+.6f}")
    print(f"  ts_proj mean-vec cosine(ok vs garb): {cos_proj:+.6f}")
    print(f"  (close to 1.0 → near-constant across groups → encoder not conditioning LLM)")
    print(f"  (close to 0.0 → meaningfully different → encoder IS conditioning LLM)")

    # Magnitude comparison
    ok_enc_norm   = ok_enc.norm().item()
    garb_enc_norm = garb_enc.norm().item()
    ok_proj_norm   = ok_proj.norm().item()
    garb_proj_norm = garb_proj.norm().item()
    print(f"\n  Encoder mean-vec norm: ok={ok_enc_norm:.4f}  garb={garb_enc_norm:.4f}")
    print(f"  ts_proj mean-vec norm: ok={ok_proj_norm:.4f}  garb={garb_proj_norm:.4f}")

print("\nDONE_ENC_VAR", flush=True)
