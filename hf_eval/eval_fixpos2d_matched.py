#!/usr/bin/env python3
"""Evaluate the user's OpenTSLMSPFixPos2D checkpoint on Sleep-CoT with the SAME
generation + scoring as the reproduced Flamingo (eval_of201.py): max_new_tokens=500,
default decoding, canon()/get_labels() substring matching, sklearn acc + macro-F1.
This makes FixPos2D directly comparable to the reproduced paper Flamingo number.

Loads the LOCAL best_model.pt via the fork's OpenTSLMSPFixPos2D (patch32/dim768, LoRA r4/a8)
exactly like evaluation/cot/eval_cot_phase2.py:load_model.
"""
import sys, os, re, json, argparse, getpass, collections
USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")

import torch
from sklearn.metrics import accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task", default="sleep", choices=["sleep", "har"])
ap.add_argument("--llm_id", default="google/gemma-3-270m")
ap.add_argument("--lora_r", type=int, default=4)
ap.add_argument("--lora_alpha", type=int, default=8)
ap.add_argument("--n", type=int, default=0, help="0 = full test split")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--max_new_tokens", type=int, default=500)
ap.add_argument("--repetition_penalty", type=float, default=1.0, help=">1.0 penalizes repeated tokens (e.g. 1.3) to break degeneration loops")
ap.add_argument("--no_repeat_ngram_size", type=int, default=0, help=">0 blocks repeating any n-gram of this size (e.g. 3)")
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--out", default="")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(ckpt_path, llm_id):
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
    a = argparse.Namespace(pe_type="sincos", use_only_time_pe=False, use_only_channel_pe=False,
        is_pretraining=False, gradient_checkpointing=False, patch_size=32, enc_dec_type="mlp",
        mlp_enc_hidden_dim=64, mlp_enc_ff_dim=256, mlp_dec_hidden_dim=64, mlp_dec_ff_dim=256)
    model = OpenTSLMSPFixPos2D(args=a, llm_id=llm_id, device=device).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("lora_enabled", False) and "lora_state" in ckpt:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if "llm_input_embeddings" in ckpt:
        model.llm.get_input_embeddings().weight.data.copy_(ckpt["llm_input_embeddings"].to(device))
    if "llm_output_embeddings" in ckpt:
        oe = model.llm.get_output_embeddings()
        if oe is not None: oe.weight.data.copy_(ckpt["llm_output_embeddings"].to(device))
    for name in ("encoder_state","projector_state","decoder_state","decoder_projector_state","mse_projector_state"):
        if name in ckpt:
            getattr(model, name.replace("_state","")).load_state_dict(ckpt[name])
    model.load_lora_state_from_checkpoint(ckpt, allow_missing=False)
    print(f"  ckpt epoch={ckpt.get('epoch','?')} val_loss={ckpt.get('val_loss','?')}", flush=True)
    model.eval()
    return model

print(f"FixPos2D matched eval: ckpt={args.ckpt} tok={args.max_new_tokens}", flush=True)
model = load_model(args.ckpt, args.llm_id)
EOS = model.get_eos_token() if hasattr(model, "get_eos_token") else ""
if args.task == "sleep":
    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset as DS
else:
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset as DS
ds = DS(split="test", EOS_TOKEN=EOS)
LABELS = DS.get_labels()
print(f"task={args.task} labels ({len(LABELS)}): {LABELS}", flush=True)

# IDENTICAL scoring to eval_of201.py (the harness that reproduced Flamingo 68.49%)
def norm(s):
    s = (s or "").lower().replace("_"," ").replace("-"," "); return re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",s)).strip()
def after(t):
    low=(t or "").lower(); i=low.rfind("answer:"); return (t[i+7:] if i>=0 else t).split("\n")[0]
def canon(t):
    nt=norm(after(t))
    for c in LABELS:
        if norm(c)==nt: return c
    cs=[c for c in LABELS if norm(c) in nt or nt in norm(c)]
    return max(cs,key=lambda c:len(norm(c))) if cs else nt

total=len(ds); start=max(0,args.start); end=total if args.end<0 else min(args.end,total)
if args.n and args.n>0: end=min(end, start+args.n)
idxs=list(range(start,end)); n=len(idxs)
print(f"shard: total={total} start={start} end={end} n={n}", flush=True)

golds,preds,recs=[],[],[]
with torch.no_grad():
    for s0 in range(0, n, args.batch_size):
        rows=[ds[idxs[k]] for k in range(s0, min(s0+args.batch_size, n))]
        try:
            outs,_=model.generate(rows, max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size)
        except Exception as e:
            outs=[f"<error {type(e).__name__}>"]*len(rows)
        for row,p in zip(rows,outs):
            gl=canon(row.get("answer","")); pl=canon(p or "")
            golds.append(gl); preds.append(pl)
            recs.append({"gold":gl,"pred":pl,"correct":gl==pl,"text":(p or "")[:400]})
        done=len(recs)
        if done % 40 < args.batch_size:
            print(f"  {done}/{n} acc={sum(r['correct'] for r in recs)/len(recs):.3f}", flush=True)

acc=accuracy_score(golds,preds)
mf1=f1_score(golds,preds,labels=LABELS,average="macro",zero_division=0)
print(f"\nFIXPOS2D_RESULT {args.task}: n={n} acc={acc:.4f} macroF1={mf1:.4f}", flush=True)
print("pred dist:", dict(collections.Counter(preds).most_common(8)), flush=True)
if args.out:
    tmp=args.out+".tmp"
    json.dump({"model":"OpenTSLMSPFixPos2D","task":args.task,"n":n,"accuracy":acc,"macro_f1":mf1,"records":recs}, open(tmp,"w"))
    os.replace(tmp,args.out)
print("DONE_FIXPOS2D", flush=True)
