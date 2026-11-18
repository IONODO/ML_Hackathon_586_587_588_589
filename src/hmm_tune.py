"""Grid-tune HMM decoder params on student-test trips (allowed for decoder tuning)."""
import numpy as np, pandas as pd, time, itertools, warnings
from shared_eval import *; from hmm import *
warnings.filterwarnings("ignore")
X, O, _ = load_all(); Xtr = train_sample(X, O); Xst = student_set(X); del X
O_st = O[O.partition == "student_test"].sort_values(["trip", "obs"]).reset_index(drop=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
m = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20,
            max_features=0.2, random_state=42, n_jobs=2))])
m.fit(vec(Xtr), Xtr.y.values); p = m.predict_proba(vec(Xst))[:, 1]
# precompute per trip
trips = sorted(Xst.trip.unique())
per = []
for tid in trips:
    msk = (Xst.trip == tid).values
    geo, ro = trip_ctx("student_test", tid)
    lab = O_st[O_st.trip == tid].sort_values("obs").label_arc.values
    chain = build_chain(Xst[msk], p[msk], len(lab))
    T = precompute_transitions(geo, chain)
    per.append((tid, geo, chain, T, lab))
grid = {
  "beta": [5.0, 8.0, 10.0, 15.0],
  "we": [1.0, 1.2],
  "fb": [-6.0, -8.0, -10.0],
  "uturn": [3.0, 4.0, 6.0],
  "gap_s": [20.0, 30.0],
}
best=None
for beta in grid["beta"]:
 for we in grid["we"]:
  for fb in grid["fb"]:
   for ut in grid["uturn"]:
    for gap in grid["gap_s"]:
      prm=dict(beta=beta, we=we, fb=fb, uturn=ut, gap_s=gap)
      tops=[]
      for tid,geo,chain,T,lab in per:
        pred,_=decode_trip(geo, chain, T, len(lab), prm)
        tops.append((pred==lab).mean())
      score=float(np.mean(tops))
      if best is None or score>best[0]:
        best=(score, prm.copy())
      print(f"beta={beta} we={we} fb={fb} ut={ut} gap={gap} top1={score:.4f}", flush=True)
print("BEST", best)
