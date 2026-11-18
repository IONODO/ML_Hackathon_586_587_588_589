import numpy as np, pandas as pd
from data_loader import *; from geo import *; from labeling import *
O=pd.read_pickle("../results/obs_labels.pkl")
for tid in ["00000054","00000033","00000072"]:
    tr,no,ar,ro=load_trip("train",tid); g=TripGeo(tr,no,ar)
    o=O[O.trip==tid]; ld=o.label_dist.values
    print("\n",tid,"n_obs",len(tr),"route",len(ro),"label_dist pct[10,50,90]",np.percentile(ld,[10,50,90]).round(1))
    # distance to ANY arc
    from shapely import points
    pr=g.tree().query(points(g.P[::10]),predicate="dwithin",distance=60)
    covered=len(set(pr[0]))/len(g.P[::10])
    print("  frac of sampled obs with ANY arc within 60m: %.3f"%covered)
    # nearest route arc distance regardless of order
    A=g.A[ro];AB=g.AB[ro];L2=(AB**2).sum(1)+1e-9
    dm=[]
    for i in range(0,len(g.P),10):
        f=np.clip(((g.P[i]-A)*AB).sum(1)/L2,0,1);C=A+f[:,None]*AB;dm.append(np.linalg.norm(g.P[i]-C,axis=1).min())
    print("  nearest route arc (any order) dist pct[10,50,90]:",np.percentile(dm,[10,50,90]).round(1))
    print("  route length m %.0f ; gps extent m"%g.L[ro].sum(), (g.P.max(0)-g.P.min(0)).round(0), "route extent", (np.r_[g.A[ro],g.B[ro]].max(0)-np.r_[g.A[ro],g.B[ro]].min(0)).round(0))
    print("  gps first/last", g.P[0].round(0), g.P[-1].round(0), "route first start/last end", g.A[ro[0]].round(0), g.B[ro[-1]].round(0))
    if tid=="00000072":
        m=o[~o.hit]; print("  72 misses: label_dist med",m.label_dist.median().round(1),"has_cand",m.has_cand.mean().round(2))
        print("  72 label arc lens (miss):",np.percentile(g.L[m.label_arc.values],[10,50,90]).round(1), " dup-geom?")
