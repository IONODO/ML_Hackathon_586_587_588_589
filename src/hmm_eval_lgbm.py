import numpy as np, pandas as pd, time, warnings
from shared_eval import *; from hmm import *
warnings.filterwarnings("ignore")
X, O, _ = load_all(); Xtr = train_sample(X, O); Xst = student_set(X); del X
O_st = O[O.partition == "student_test"].sort_values(["trip", "obs"]).reset_index(drop=True)
from lightgbm import LGBMClassifier
m = LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20, random_state=42, n_jobs=2, verbose=-1)
t0=time.time(); m.fit(vec(Xtr), Xtr.y.values); print("trained LGBM in %.1fs" % (time.time()-t0))
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
print("LGBM clf_top1=%.4f hmm_top1=%.4f" % (np.mean(tops_clf), np.mean(tops_hmm)))
