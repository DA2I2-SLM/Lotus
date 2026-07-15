#!/usr/bin/env python3
"""Does the model actually condition on the time series at inference?

For several Sleep samples, generate the predicted label with:
  (a) the real time series,
  (b) a ZEROED time series,
  (c) another sample's time series (different gold class).
If (a)==(b)==(c) for most samples, the model ignores the TS (conditioning
broken or model collapsed to a prior). If predictions follow the TS, the
conditioning works and low accuracy is a model-capacity issue.
"""
import getpass
import re
import sys
import copy

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/hf_eval")

import torch
import encoder_compat  # noqa: F401
from torch.utils.data import DataLoader
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
model.eval()
for ta in ("tokenizer", "text_tokenizer"):
    t = getattr(model, ta, None)
    if t is not None and hasattr(t, "padding_side"):
        t.padding_side = "right"
ps = getattr(model, "patch_size", 4)
eos = model.get_eos_token()
LABELS = SleepEDFCoTQADataset.get_labels()
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)


def after(t):
    low = (t or "").lower(); i = low.rfind("answer:")
    seg = t[i+7:] if i >= 0 else t
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", seg.split("\n")[0].lower())).strip()


def canon(t):
    nt = after(t) if "answer:" in (t or "").lower() else re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",(t or "").lower())).strip()
    for c in LABELS:
        cc = c.lower().replace("-"," ").replace("_"," ")
        if cc == nt or cc in nt or nt in cc:
            return c
    return nt[:20]


def collate1(sample):
    return extend_time_series_to_match_patch_size_and_aggregate([copy.deepcopy(sample)], patch_size=ps)


def gen_label(sample):
    batch = collate1(sample)
    with torch.no_grad():
        out = model.generate(batch, max_new_tokens=260, do_sample=False)
    return canon(out[0])


# pick samples with diverse gold classes
idxs = list(range(12))
samples = [ds[i] for i in idxs]
# find a 'Wake' and an 'N2' source TS to swap in
def gold_of(s): return canon(s.get("answer",""))
src_by_class = {}
for s in [ds[i] for i in range(60)]:
    g = gold_of(s)
    src_by_class.setdefault(g, s)
print("source classes available:", list(src_by_class.keys()), flush=True)

n_real_eq_zero = 0
n_real_eq_swap = 0
for k, s in enumerate(samples):
    g = gold_of(s)
    real = gen_label(s)
    # zeroed TS
    s_zero = copy.deepcopy(s)
    s_zero["time_series"] = [torch.zeros_like(torch.as_tensor(ts, dtype=torch.float32)) for ts in s["time_series"]]
    zero = gen_label(s_zero)
    # swapped TS from a different gold class
    other_class = next((c for c in src_by_class if c != g), None)
    s_swap = copy.deepcopy(s)
    if other_class is not None:
        s_swap["time_series"] = copy.deepcopy(src_by_class[other_class]["time_series"])
    swap = gen_label(s_swap)
    n_real_eq_zero += int(real == zero)
    n_real_eq_swap += int(real == swap)
    print(f"idx{idxs[k]} gold={g:16s} real={real:16s} zeroed={zero:16s} swap({other_class})={swap:16s}", flush=True)

N = len(samples)
print(f"\nreal==zeroed : {n_real_eq_zero}/{N}   real==swapped : {n_real_eq_swap}/{N}", flush=True)
print("If most are EQUAL, the model ignores the time series (no conditioning).", flush=True)
print("DONE_COND", flush=True)
