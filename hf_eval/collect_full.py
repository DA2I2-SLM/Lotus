#!/usr/bin/env python3
"""Merge full-test shard outputs per cell and print the current table.
Handles partial completion (reports how many shards / samples are in so far)."""
import json, glob, re, os, getpass, collections

U = getpass.getuser()
R = f"/scratch/{U}/OpenTSLM_upstream/results"
CELLS = [  # (tag, task, expected_shards, checkpoint_label, model_type)
    ("sleep_sp", "sleep", 3,  "sleep-sp", "SP"),
    ("har_sp",   "har",   8,  "har-sp",   "SP"),
    ("ecg_sp",   "ecg",   20, "ecg-sp",   "SP"),
    ("sleep_fl", "sleep", 3,  "sleep-flamingo", "Flamingo"),
    ("har_fl",   "har",   8,  "har-flamingo",   "Flamingo"),
    ("ecg_fl",   "ecg",   20, "ecg-flamingo",   "Flamingo"),
]

def clean(s):
    s = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()
    return re.sub(r"\s*eos\s*$", "", s).strip()

def collect(tag, task, expected):
    files = sorted(glob.glob(f"{R}/full_{tag}.shard*.json"))
    recs, complete_shards = [], 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        recs += d.get("records", [])
        if d.get("complete"):
            complete_shards += 1
    if not recs:
        return None
    n = len(recs)
    correct = sum(1 for r in recs if r.get("correct"))
    acc = correct / n
    # macro-F1 over gold classes
    by = collections.defaultdict(lambda: [0, 0, 0])  # tp, fp, fn keyed by class
    classes = set(r["gold"] for r in recs)
    golds = [r["gold"] for r in recs]; preds = [r["pred"] for r in recs]
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    mf1 = sum(f1s)/len(f1s) if f1s else 0
    out = {"n": n, "shards_complete": f"{complete_shards}/{expected}",
           "accuracy": acc, "macro_f1": mf1}
    if task == "ecg":
        s = fp = 0
        for r in recs:
            g = clean(r["gold"]); p = clean(r["pred"])
            if g == p: s += 1
            gw = g.split()
            if (len(gw) == 1 and p.split()[:1] == gw) or (len(gw) > 1 and p.startswith(g)):
                fp += 1
        out["ecg_strict"] = s/n; out["ecg_lenient"] = fp/n
    return out

print(f"{'cell':16s} {'shards':>8s} {'N':>7s} {'acc%':>7s} {'macroF1%':>9s} {'ECGlenient%':>12s}")
print("-"*64)
for tag, task, exp, label, mt in CELLS:
    r = collect(tag, task, exp)
    if r is None:
        print(f"{label:16s} {'0/'+str(exp):>8s}  (no data yet)")
        continue
    ecg = f"{r.get('ecg_lenient',0)*100:12.2f}" if task == "ecg" else ""
    print(f"{label:16s} {r['shards_complete']:>8s} {r['n']:>7d} {r['accuracy']*100:>7.2f} {r['macro_f1']*100:>9.2f} {ecg:>12s}")
