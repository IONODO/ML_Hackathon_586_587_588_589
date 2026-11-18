import numpy as np, pandas as pd, time
from sklearn.model_selection import GroupKFold
from common_data import *; from eval_utils import *; from features import FEATURES
def vec(Xf): return Xf[FEATURES].values.astype(np.float32)
def top1_hit_only(Xf, score):
    d = pd.DataFrame({"tc": Xf.tc.values, "obs": Xf.obs.values, "s": score, "y": Xf.y.values})
    idx = d.groupby(["tc", "obs"])["s"].idxmax(); return d.loc[idx, "y"].mean()
def group_cv(make, Xtr, n_splits=5):
    rows = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(Xtr, groups=Xtr.tc.values)):
        tr_, va_ = Xtr.iloc[a], Xtr.iloc[b]; m = make(); m.fit(vec(tr_), tr_.y.values); p = m.predict_proba(vec(va_))[:, 1]
        cm = candidate_metrics(va_.y.values, p); cm["top1"] = top1_hit_only(va_, p); cm["fold"] = k; rows.append(cm)
    return pd.DataFrame(rows)
def student_eval(make, Xtr, Xst, O_st):
    m = make(); t0 = time.time(); m.fit(vec(Xtr), Xtr.y.values); tt = time.time() - t0
    t0 = time.time(); p = m.predict_proba(vec(Xst))[:, 1]; it = time.time() - t0
    cm = candidate_metrics(Xst.y.values, p); tops, cs_, cr_, seq, w = [], [], [], [], []
    for tid in sorted(Xst.trip.unique()):
        msk = (Xst.trip == tid).values; geo, ro = trip_ctx("student_test", tid)
        lab = O_st[O_st.trip == tid].sort_values("obs").label_arc.values
        pred = top1_predictions(Xst[msk], p[msk], len(lab)); tops.append((pred == lab).mean())
        s, r = route_continuity(pred, geo); cs_.append(s); cr_.append(r); seq.append(sequence_metrics(pred, ro, geo)); w.append(len(lab))
    w = np.array(w) / sum(w)
    n_obs = Xst.groupby(["tc", "obs"]).ngroups
    return m, p, dict(**{f"st_{k}": v for k, v in cm.items()}, st_top1=float(np.dot(w, tops)), st_cont_strict=float(np.dot(w, cs_)),
        st_cont_relaxed=float(np.dot(w, cr_)), st_lcs_f1=float(np.mean([s["lcs_f1"] for s in seq])), st_len_f1=float(np.mean([s["len_f1"] for s in seq])),
        train_time_s=tt, infer_time_ms_per_obs=1000 * it / n_obs)
