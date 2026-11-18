"""HMM/Viterbi evaluation: classifier-only vs classifier+HMM on student-test trips."""
import numpy as np, pandas as pd, time, warnings, joblib
from shared_eval import *; from hmm import *
warnings.filterwarnings("ignore")
X, O, _ = load_all(); Xtr = train_sample(X, O); Xst = student_set(X); del X
O_st = O[O.partition == "student_test"].sort_values(["trip", "obs"]).reset_index(drop=True)
# train a RF (untuned baseline) for emission
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
m = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20,
            max_features=0.2, random_state=42, n_jobs=2))])
t0=time.time(); m.fit(vec(Xtr), Xtr.y.values); print("trained RF in %.1fs" % (time.time()-t0))
p = m.predict_proba(vec(Xst))[:, 1]
# per-trip decode
tops_clf, tops_hmm, cs, cr, lcs = [], [], [], [], []
for tid in sorted(Xst.trip.unique()):
    msk = (Xst.trip == tid).values
    geo, ro = trip_ctx("student_test", tid)
    lab = O_st[O_st.trip == tid].sort_values("obs").label_arc.values
    # classifier-only argmax
    pred_clf = top1_predictions(Xst[msk], p[msk], len(lab))
    # HMM
    chain = build_chain(Xst[msk], p[msk], len(lab))
    T = precompute_transitions(geo, chain)
    pred_hmm, filled = decode_trip(geo, chain, T, len(lab))
    tops_clf.append((pred_clf == lab).mean()); tops_hmm.append((pred_hmm == lab).mean())
    s, r = route_continuity(pred_hmm, geo); cs.append(s); cr.append(r)
    seq = sequence_metrics(pred_hmm, ro, geo); lcs.append(seq["lcs_f1"])
    print(f"{tid}: clf_top1={tops_clf[-1]:.3f} hmm_top1={tops_hmm[-1]:.3f} cont_strict={s:.3f} lcs_f1={seq['lcs_f1']:.3f} filled={filled.sum()}")
print("MEAN clf_top1=%.4f hmm_top1=%.4f cont_strict=%.4f lcs_f1=%.4f" % (np.mean(tops_clf), np.mean(tops_hmm), np.mean(cs), np.mean(lcs)))
