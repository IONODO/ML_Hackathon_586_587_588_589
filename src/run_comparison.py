import numpy as np, pandas as pd, time, json, warnings, sys
from sklearn.model_selection import GroupKFold
from common_data import *; from model_zoo import *; from eval_utils import *
warnings.filterwarnings("ignore")
X, O, tcodes = load_all()
Xtr = train_sample(X, O); Xst = student_set(X); del X
print("train rows", len(Xtr), "obs", Xtr.groupby(["tc", "obs"]).ngroups, "| student rows", len(Xst), flush=True)
def top1_hit_only(Xf, score):
    d = pd.DataFrame({"tc": Xf.tc.values, "obs": Xf.obs.values, "s": score, "y": Xf.y.values})
    idx = d.groupby(["tc", "obs"])["s"].idxmax(); return d.loc[idx, "y"].mean()
def vec(Xf): return Xf[FEATURES].values.astype(np.float32)
def sub_for(Xf, s): return Xf if s == 1 else Xf[(Xf.obs // STRIDE) % s == 0]
O_st = O[O.partition == "student_test"]
cv_rows, res_rows = [], []
names = sys.argv[1:] or list(ZOO)
gkf = GroupKFold(n_splits=5)
for name in names:
    z = ZOO[name]; print("==", name, flush=True)
    # ---- group-aware 5-fold CV on the 80 training trips ----
    t_cv = time.time()
    for k, (a, b) in enumerate(gkf.split(Xtr, groups=Xtr.tc.values)):
        tr_, va_ = sub_for(Xtr.iloc[a], z["train_stride"]), Xtr.iloc[b]
        m = build(name); m.fit(vec(tr_), tr_.y.values); p = m.predict_proba(vec(va_))[:, 1]
        cm = candidate_metrics(va_.y.values, p); cm["top1"] = top1_hit_only(va_, p); cm.update(model=name, fold=k, n_val_trips=va_.tc.nunique())
        cv_rows.append(cm); print("  fold", k, {q: round(v, 4) for q, v in cm.items() if isinstance(v, float)}, flush=True)
    cv_time = time.time() - t_cv
    # ---- final fit on all 80 training trips (sample), eval on 10 student-test trips ----
    tr_all = sub_for(Xtr, z["train_stride"]); m = build(name)
    t0 = time.time(); m.fit(vec(tr_all), tr_all.y.values); train_time = time.time() - t0
    t0 = time.time(); p = m.predict_proba(vec(Xst))[:, 1]; inf_time = time.time() - t0
    cm = candidate_metrics(Xst.y.values, p)
    tops, cont_s, cont_r, seq, wts = [], [], [], [], []
    for tid in sorted(Xst.trip.unique()):
        msk = (Xst.trip == tid).values; Xs = Xst[msk]; pr = p[msk]
        geo, ro = trip_ctx("student_test", tid); lab = O_st[O_st.trip == tid].sort_values("obs").label_arc.values
        pred = top1_predictions(Xs, pr, len(lab)); tops.append((pred == lab).mean())
        cs, cr = route_continuity(pred, geo); cont_s.append(cs); cont_r.append(cr)
        seq.append(sequence_metrics(pred, ro, geo)); wts.append(len(lab))
    w = np.array(wts) / sum(wts)
    row = dict(model=name, family=z["family"], config=z["cfg"], seed=SEED, train_rows=len(tr_all),
               **{f"st_{k}": v for k, v in cm.items()},
               st_top1=float(np.dot(w, tops)), st_top1_macro_trip=float(np.mean(tops)),
               st_cont_strict=float(np.dot(w, cont_s)), st_cont_relaxed=float(np.dot(w, cont_r)),
               st_lcs_f1=float(np.mean([s["lcs_f1"] for s in seq])), st_len_f1=float(np.mean([s["len_f1"] for s in seq])),
               train_time_s=train_time, infer_time_ms_per_obs=1000 * inf_time / len(Xst.groupby(["tc", "obs"])), cv_wall_s=cv_time)
    res_rows.append(row); print("  STUDENT", {q: (round(v, 4) if isinstance(v, float) else v) for q, v in row.items() if q.startswith("st_") or q.endswith("_s") or q.startswith("infer")}, flush=True)
    pd.DataFrame([r for r in cv_rows if r["model"] == name]).to_csv(f"../results/cv_folds_{name}.csv", index=False)
    pd.DataFrame([row]).to_csv(f"../results/student_results_{name}.csv", index=False)
