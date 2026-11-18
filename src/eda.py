import numpy as np, pandas as pd, networkx as nx
from data_loader import *
rows=[]
for part in ("train","student_test"):
    for tid in list_trips(part):
        tr,no,ar,ro=load_trip(part,tid)
        dt=np.diff(tr.t.values)
        G=nx.DiGraph(); G.add_nodes_from(range(len(no))); G.add_edges_from(ar.values.tolist())
        deg=np.array([G.degree(n) for n in G.nodes])
        rows.append(dict(partition=part,trip=tid,n_gps=len(tr),n_nodes=len(no),n_arcs=len(ar),n_route=len(ro),
          n_route_unique=len(set(ro)),nan_track=int(tr.isna().sum().sum()),dup_track=int(tr.duplicated().sum()),
          dt_eq1=float((dt==1).mean()),dt_max=float(dt.max()),dt_nonpos=int((dt<=0).sum()),
          lon_min=tr.lon.min(),lon_max=tr.lon.max(),lat_min=tr.lat.min(),lat_max=tr.lat.max(),
          self_loops=int(nx.number_of_selfloops(G)),dup_arcs=int(ar.duplicated().sum()),
          isolated=int((deg==0).sum()),weak_comps=nx.number_weakly_connected_components(G),
          strong_comps=nx.number_strongly_connected_components(G),
          route_arc_oob=int((ro>=len(ar)).sum()),
          route_connected=float(np.mean([ar.v.values[a]==ar.u.values[b] for a,b in zip(ro[:-1],ro[1:])])) if len(ro)>1 else np.nan))
df=pd.DataFrame(rows); df.to_csv("../results/eda_inventory.csv",index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(df.groupby("partition").agg(trips=("trip","count"),gps_mean=("n_gps","mean"),gps_min=("n_gps","min"),gps_max=("n_gps","max"),
  nodes_mean=("n_nodes","mean"),arcs_mean=("n_arcs","mean"),route_mean=("n_route","mean"),nan=("nan_track","sum"),dup=("dup_track","sum"),
  dt1=("dt_eq1","mean"),dtmax=("dt_max","max"),nonpos=("dt_nonpos","sum"),selfl=("self_loops","sum"),dupa=("dup_arcs","sum"),
  iso=("isolated","sum"),weak=("weak_comps","mean"),strong=("strong_comps","mean"),oob=("route_arc_oob","sum"),rconn=("route_connected","mean")).T)
print(df[["lon_min","lon_max","lat_min","lat_max"]].agg(["min","max"]))
print(df.sort_values("dt_max",ascending=False).head(5)[["trip","partition","dt_max","dt_eq1"]])
