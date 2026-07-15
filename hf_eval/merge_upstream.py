import json, glob, sys, re, os, collections
from sklearn.metrics import f1_score, accuracy_score
base=sys.argv[1]  # e.g. /scratch/.../results/har_sp__har  (without .shardN.json)
task=sys.argv[2]
files=sorted(glob.glob(base+".shard*.json"), key=lambda f:int(re.search(r'shard(\d+)',f).group(1)))
recs=[]; 
for f in files: recs+=json.load(open(f))['records']
def clean(s):
    s=re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",(s or "").lower())).strip()
    return re.sub(r"\s*eos\s*$","",s).strip()
golds=[r['gold'] for r in recs]; preds=[r['pred'] for r in recs]
coh=sum(1 for r in recs if r.get('text'))  # placeholder; coherence stored per-shard
acc=accuracy_score(golds,preds)
labs=sorted(set(golds))
mf1=f1_score(golds,preds,labels=labs,average='macro',zero_division=0)
out={"task":task,"n":len(recs),"accuracy":acc,"macro_f1":mf1,"records_file":base+".merged.json"}
if task=="ecg":
    s=fp=0
    for r in recs:
        g=clean(r['gold']); p=clean(r['pred'])
        if g==p: s+=1
        gw=g.split()
        if (len(gw)==1 and p.split()[:1]==gw) or (len(gw)>1 and p.startswith(g)): fp+=1
    out["ecg_strict"]=s/len(recs); out["ecg_lenient"]=fp/len(recs)
json.dump({"n":len(recs),"records":recs}, open(base+".merged.json","w"))
print(json.dumps({k:v for k,v in out.items() if k!='records'}, indent=2))
