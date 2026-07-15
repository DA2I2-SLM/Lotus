#!/usr/bin/env python3
"""Test whether tokenizer.padding_side is the remaining bug.

pad_and_apply_batch extracts each text segment as sample_embeds[i, :length].
With LEFT padding the real tokens sit at the END, so [:length] grabs PAD
embeddings for every segment shorter than the longest -> the ts_header and
post_prompt text are silently dropped. With RIGHT padding [:length] grabs the
real tokens. If training used right-padding, left-padding at inference is a
train/inference MISMATCH that destroys the task prompt -> garbage.

Compares coherence + accuracy on Sleep test samples for padding_side in
{left, right}, with the norm_first encoder fix + greedy decoding.
"""
import getpass
import re
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/hf_eval")

import torch
import encoder_compat  # noqa: F401  (norm_first fix)
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
ps = getattr(model, "patch_size", 4)
print("default tokenizer.padding_side =", model.tokenizer.padding_side, flush=True)

eos = model.get_eos_token()
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)
LABELS = SleepEDFCoTQADataset.get_labels()


def norm(s):
    s = (s or "").lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def after_answer(t):
    low = (t or "").lower(); i = low.rfind("answer:")
    seg = t[i+7:] if i >= 0 else t
    return seg.split("\n")[0].strip().rstrip(".,;:").strip()


def to_canon(t):
    nt = norm(t)
    for c in LABELS:
        if norm(c) == nt:
            return c
    cands = [c for c in LABELS if norm(c) in nt or nt in norm(c)]
    return max(cands, key=lambda c: len(norm(c))) if cands else nt


def garbage(t):
    low = (t or "").lower()
    return (not t) or any(b in low for b in
        ("penurunan", "qualquer", "<unused", "linklabel", "filedialog", "thụ",
         "тельное", "besar", "这是", "sebelum", "adalah", "chesterchester"))


def run(side, n=16):
    model.tokenizer.padding_side = side
    sub = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=1, shuffle=False,
                        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=ps))
    coh = corr = 0
    sample_txt = None
    with torch.no_grad():
        for batch in loader:
            preds = model.generate(batch, max_new_tokens=300, do_sample=False)
            for s, p in zip(batch, preds):
                if not garbage(p):
                    coh += 1
                gold = to_canon(after_answer(s.get("answer", "")))
                pred = to_canon(after_answer(p))
                if gold and gold == pred:
                    corr += 1
                if sample_txt is None:
                    sample_txt = p[:160]
    print(f"PADDING={side:5s}  coherent={coh}/{n}  correct={corr}/{n}  sample={sample_txt!r}", flush=True)


for side in ("left", "right"):
    run(side)
print("DONE_PADDING", flush=True)
