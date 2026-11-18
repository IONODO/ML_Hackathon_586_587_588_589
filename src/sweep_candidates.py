import numpy as np, pandas as pd
from data_loader import *; from geo import *
O=pd.read_pickle("../results/obs_labels.pkl"); O=O[O.partition=="train"]
cfgs=[(30,10),(50,15),(60,15),(60,25),(80,25),(80,40),(100,40)]
rows=[]; geos={}
agg={c:[0,0,0,0] for c in cfgs}  # miss_all, n, miss_clean, n_clean
bad={"00000033","00000054"}
for tid in list_trips("train"):
    tr,no,ar,ro=load_trip("train",tid); g=TripGeo(tr,no,ar); o=O[O.trip==tid]; lab=o.label_arc.values
    for (R,K) in cfgs:
        oi,ai,d,f=g.candidates(R,K); hit=np.zeros(len(tr),bool); hit[oi[ai==lab[oi]]]=True
        a=agg[(R,K)]; a[0]+=(~hit).sum(); a[1]+=len(hit); a[4:]=[] if False else a[4:]
        if tid not in bad: a[2]+=(~hit).sum(); a[3]+=len(hit)
        agg[(R,K)]=a
        rows.append(dict(trip=tid,R=R,K=K,miss=1-hit.mean(),avg_cand=len(oi)/len(tr)))
df=pd.DataFrame(rows); df.to_csv("../results/candidate_sweep_train.csv",index=False)
out=pd.DataFrame([dict(R=R,K=K,miss_all=a[0]/a[1],miss_excl_33_54=a[2]/a[3]) for (R,K),a in agg.items()])
out["avg_cand"]=[df[(df.R==R)&(df.K==K)].avg_cand.mean() for R,K in cfgs]
print(out.round(4).to_string(index=False))
