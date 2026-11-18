import numpy as np, pandas as pd
from pyproj import Transformer
from data_loader import *
inv=pd.read_csv("../results/eda_inventory.csv")
print(inv.groupby(["partition"])[["lon_min","lat_min"]].agg(["min","max"]).round(2))
print(inv[["trip","partition","lon_min","lat_min","n_gps","n_arcs","n_route"]].head(8))
def proj(lon,lat,lon0,lat0):
    zone=int((lon0+180)//6)+1; epsg=(32600 if lat0>=0 else 32700)+zone
    return Transformer.from_crs(4326,epsg,always_xy=True).transform(lon,lat)
def pt_seg_dist(P,A,B):
    AB=B-A; L2=(AB**2).sum(1)+1e-12
    t=np.clip(((P[:,None,:]-A[None])*AB[None]).sum(2)/L2[None],0,1)
    C=A[None]+t[...,None]*AB[None]
    return np.sqrt(((P[:,None,:]-C)**2).sum(2))
for part,tid in [("train","00000048"),("train","00000001"),("student_test","00000056")]:
    tr,no,ar,ro=load_trip(part,tid)
    lon0,lat0=tr.lon.mean(),tr.lat.mean()
    x,y=proj(tr.lon.values,tr.lat.values,lon0,lat0); nx_,ny=proj(no.lon.values,no.lat.values,lon0,lat0)
    P=np.c_[x,y]; N=np.c_[nx_,ny]
    A=N[ar.u.values[ro]]; B=N[ar.v.values[ro]]
    d=pt_seg_dist(P,A,B)   # (n_gps, n_route)
    dmin=d.min(1); amin=d.argmin(1)
    seglen=np.linalg.norm(B-A,axis=1)
    print(part,tid,"n_gps",len(P),"n_route",len(ro),"route len m",seglen.sum().round(0),"gps path m",np.linalg.norm(np.diff(P,axis=0),axis=1).sum().round(0))
    print("  dist to nearest ROUTE arc: median %.1f p90 %.1f p99 %.1f max %.1f"%tuple(np.percentile(dmin,[50,90,99,100])))
    print("  nearest-route-arc index monotone frac:",(np.diff(amin)>=0).mean().round(3),"first,last:",amin[0],amin[-1],"of",len(ro))
    # all-arc distance
    A2=N[ar.u.values];B2=N[ar.v.values]
    dall=pt_seg_dist(P[::10],A2,B2).min(1)
    print("  dist to nearest ANY arc: median %.1f p99 %.1f"%tuple(np.percentile(dall,[50,99])))
