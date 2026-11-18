import numpy as np, pandas as pd, time, warnings
from common_data import *; from hmm import *; from shared_eval import *
warnings.filterwarnings("ignore")
X, O, _ = load_all()
# full training (all obs), keep positive hits only
Xtr = X[(X.partition=="train")]
hit = O[(O.partition=="train") & O.hit][["tc","obs"]]
hit_key = set((hit.tc.values.astype(np.int64)*10**6 + hit.obs.values).tolist())
key = Xtr.tc.values.astype(np.int64)*10**6 + Xtr.obs.values
keep = np.fromiter((k in hit_key for k in key), bool, len(key))
Xtr = Xtr[keep].reset_index(drop=True)
Xst = student_set(X); del X
O_st = O[O.partition == "student_test"].sort_values(["trip","obs"]).reset_index(drop=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
m = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=10, max_features=0.25, random_state=42, n_jobs=2))])
t0=time.time(); m.fit(vec(Xtr), Xtr.y.values); print("trained RF full in %.1fs rows=%d" % (time.time()-t0, len(Xtr)))
p = m.predict_proba(vec(Xst))[:, 1]
tops_clf, tops_hmm = [], []
for tid in sorted(Xst.trip.unique()):
    msk = (Xst.trip == tid).values
    geo, ro = trip_ctx("student_test", tid)
    lab = O_st[O_st.trip == tid].sort_values("obs").label_arc.values
    pred_clf = top1_predictions(Xst[msk], p[msk], len(lab))
    chain = build_chain(Xst[msk], p[msk], len(lab))
    T = precompute_transitions(geo, chain)
    pred_hmm, _ = decode_trip(geo, chain, T, len(lab), {"beta":8.0,"we":1.0,"fb":-10.0,"uturn":6.0,"gap_s":30.0})
    tops_clf.append((pred_clf==lab).mean()); tops_hmm.append((pred_hmm==lab).mean())
print("FULL RF clf_top1=%.4f hmm_top1=%.4f" % (np.mean(tops_clf), np.mean(tops_hmm)))
