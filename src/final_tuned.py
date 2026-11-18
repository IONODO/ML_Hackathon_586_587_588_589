"""Refit RandomForest with the Bayesian-best configuration and compare tuned vs untuned on IDENTICAL data/partitions/metrics."""
import json, joblib, numpy as np, pandas as pd, time, warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from shared_eval import *
warnings.filterwarnings("ignore")
best = json.load(open("../results/bayes_best.json"))["best_params"]
X, O, _ = load_all(); Xtr = train_sample(X, O); Xst = student_set(X); del X; O_st = O[O.partition == "student_test"]
def mk(p): return lambda: Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(**p, random_state=42, n_jobs=1))])
base = dict(n_estimators=40, max_depth=12, min_samples_leaf=20, max_features="sqrt")
variants = {"untuned (original table config, 2x thinner sample)": (mk(base), Xtr[(Xtr.obs // STRIDE) % 2 == 0]),
            "untuned (same config, full training sample)": (mk(base), Xtr),
            "Bayesian-tuned (full training sample)": (mk(best), Xtr)}
rows = []
for name, (make, data) in variants.items():
    cv = group_cv(make, data, 5); m, p, st = student_eval(make, data, Xst, O_st)
    rows.append(dict(variant=name, config=json.dumps(best if "tuned (full" in name and "untuned" not in name else base), n_train_rows=len(data), cv_macro_f1=cv.macro_f1.mean(), cv_macro_f1_std=cv.macro_f1.std(),
                     cv_top1=cv.top1.mean(), cv_top1_std=cv.top1.std(), cv_log_loss=cv.log_loss.mean(), **st)); print(name, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    if "Bayesian" in name:
        joblib.dump(m, "../artifacts/rf_tuned.joblib", compress=3)
        joblib.dump(dict(features=FEATURES, R=100.0, K=40, seed=42, params=best), "../artifacts/model_meta.joblib")
        np.save("../results/student_scores_tuned.npy", p); Xst[["trip", "tc", "obs", "arc", "y"]].to_pickle("../results/student_rows_index.pkl")
pd.DataFrame(rows).to_csv("../results/tuned_vs_untuned.csv", index=False)
