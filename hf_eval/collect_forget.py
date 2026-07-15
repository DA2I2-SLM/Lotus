#!/usr/bin/env python3
"""Assemble the catastrophic-forgetting table (B): the POST-Sleep-CoT checkpoint of
each baseline, evaluated on BOTH HAR-CoT (forgetting) and Sleep-CoT (retention).
Reads the FIXED_* outputs from this session's runs (Flamingo uses the conditioning fix).
My method's numbers come from the user's own harness (eval_cot_phase2) and are hardcoded.
"""
import json, glob, getpass, collections

U = getpass.getuser()
R = f"/scratch/{U}/OpenTSLM_upstream/results"


def macro_f1(recs):
    golds = [r["gold"] for r in recs]; preds = [r["pred"] for r in recs]
    f1s = []
    for c in set(golds):
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
    return sum(f1s)/len(f1s) if f1s else 0.0


def load_single(path):
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return d.get("records", []), d.get("complete", False)


def load_sharded(base):
    files = sorted(glob.glob(f"{base}.shard*.json"))
    recs, done = [], 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        recs += d.get("records", [])
        done += int(bool(d.get("complete")))
    return recs, f"{done}/{len(files) if files else '?'}"


def cell(recs):
    if not recs:
        return None
    n = len(recs)
    acc = sum(1 for r in recs if r.get("correct")) / n
    coh = sum(1 for r in recs if r.get("text") and len(r["text"].strip()) > 0)
    return {"n": n, "acc": acc, "f1": macro_f1(recs),
            "coh": coh,
            "toppred": collections.Counter(r["pred"] for r in recs).most_common(3)}


CELLS = {
    "Flamingo  Sleep (retain)": ("single", f"{R}/FIXED_sleepckpt_flamingo_sleep.json"),
    "Flamingo  HAR  (forget)":  ("shard",  f"{R}/FIXED_sleepckpt_flamingo_har"),
    "SP        Sleep (retain)": ("single", f"{R}/FIXED_sleepckpt_sp_sleep.json"),
    "SP        HAR  (forget)":  ("shard",  f"{R}/FIXED_sleepckpt_sp_har"),
}

print("=== (B) Catastrophic-forgetting: POST-Sleep checkpoint on HAR + Sleep ===\n")
print(f"{'cell':28s} {'N':>6s} {'acc%':>7s} {'F1%':>7s} {'shards':>7s}  top-preds")
print("-" * 92)
for name, (kind, path) in CELLS.items():
    if kind == "single":
        loaded = load_single(path); recs = loaded[0] if loaded else []
        shards = "1" if loaded else "-"
    else:
        recs, shards = load_sharded(path)
    c = cell(recs)
    if c is None:
        print(f"{name:28s} {'-':>6s} {'(pending)':>7s}")
        continue
    tp = ", ".join(f"{p}:{n}" for p, n in c["toppred"])
    print(f"{name:28s} {c['n']:>6d} {c['acc']*100:>7.2f} {c['f1']*100:>7.2f} {shards:>7s}  {tp[:48]}")

print("\nMy method (OpenTSLMSPFixPos2D, sequential post-Sleep ckpt, user's eval_cot_phase2):")
print(f"{'  FixPos2D Sleep (retain)':28s} {930:>6d} {55.16:>7.2f} {31.55:>7.2f}")
print(f"{'  FixPos2D HAR  (forget)':28s} {1000:>6d} {9.70:>7.2f} {9.58:>7.2f}")
