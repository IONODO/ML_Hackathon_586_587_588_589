import numpy as np, pandas as pd, time, sys
from data_loader import *; from geo import *; from labeling import *
rows=[]
trips=[("train",t) for t in list_trips("train")][:int(sys.argv[1])] if len(sys.argv)>1 else []
for part,tid in trips:
    t0=time.time()
    tr,no,ar,ro=load_trip(part,tid); g=TripGeo(tr,no,ar)
    js=align_to_route(g.P,g.t,g,ro); lab=ro[js]
    dlab,_=g.seg_dist(g.P,lab)
    mono=(np.diff(js)>=0).all()
    res=dict(trip=tid,n=len(tr),lab_med=np.median(dlab),lab_p95=np.percentile(dlab,95),lab_gt30=(dlab>30).mean(),t_align=time.time()-t0)
    for R,K in [(30,10),(50,15),(60,15),(80,20),(100,25)]:
        oi,ai,d,f=g.candidates(R,K)
        hit=np.zeros(len(tr),bool); m=ai==lab[oi]; hit[oi[m]]=True
        res[f"miss_R{R}_K{K}"]=1-hit.mean(); res[f"avgc_R{R}_K{K}"]=len(oi)/len(tr)
    rows.append(res); print(tid,{k:round(v,3) if isinstance(v,float) else v for k,v in res.items()},flush=True)
df=pd.DataFrame(rows); print(df.mean(numeric_only=True).round(4))
