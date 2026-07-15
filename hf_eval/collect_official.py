#!/usr/bin/env python3
"""Merge eval_official.py shard outputs -> full accuracy + macro-F1 (correct label set)."""
import json, glob, sys, collections
base = sys.argv[1]
files = sorted(glob.glob(f"{base}.shard*.json"))
recs, done = [], 0
for f in files:
    try: d = json.load(open(f))
    except Exception: continue
    recs += d.get("records", []); done += 1
if not recs:
    print("no records yet for", base); sys.exit(0)
n = len(recs); correct = sum(1 for r in recs if r.get("correct"))
golds = [r["gold"] for r in recs]; preds = [r["pred"] for r in recs]
f1s = []
for c in set(golds):
    tp = sum(1 for g,p in zip(golds,preds) if g==c and p==c)
    fp = sum(1 for g,p in zip(golds,preds) if g!=c and p==c)
    fn = sum(1 for g,p in zip(golds,preds) if g==c and p!=c)
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
mf1 = sum(f1s)/len(f1s) if f1s else 0
print(f"shards={done}/{len(files)}  N={n}  acc={correct/n*100:.2f}%  macroF1={mf1*100:.2f}%")
print("gold dist :", dict(collections.Counter(golds).most_common()))
print("pred dist :", dict(collections.Counter(preds).most_common(9)))
