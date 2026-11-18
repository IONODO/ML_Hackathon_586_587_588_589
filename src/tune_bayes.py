"""Bayesian hyper-parameter optimisation (Optuna TPE) for RandomForest (model selected from CV evidence).
Objective: mean observation-level Top-1 accuracy over 3-fold GROUP (trip) CV on the 80 training trips ONLY
(student-test and evaluator data are never touched during the search)."""
import numpy as np, pandas as pd, optuna, json, time, warnings, sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from shared_eval import *
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40; SEED = 42
X, O, _ = load_all(); Xtr = train_sample(X, O); del X
Xs = Xtr[(Xtr.obs // STRIDE) % 3 == 0].reset_index(drop=True)   # search on a 3x thinner obs sample for speed (same 80 trips)
print("search rows", len(Xs), "trips", Xs.tc.nunique(), flush=True)
def mk(p): return lambda: Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(**p, random_state=SEED, n_jobs=1))])
def objective(trial):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 30, 120, step=10), max_depth=trial.suggest_int("max_depth", 6, 24),
             min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 100, log=True), max_features=trial.suggest_float("max_features", 0.15, 0.7),
             max_samples=trial.suggest_float("max_samples", 0.4, 1.0))
    t0 = time.time(); cv = group_cv(mk(p), Xs, n_splits=3)
    trial.set_user_attr("cv_top1_std", float(cv.top1.std())); trial.set_user_attr("cv_macro_f1", float(cv.macro_f1.mean()))
    trial.set_user_attr("cv_log_loss", float(cv.log_loss.mean())); trial.set_user_attr("runtime_s", time.time() - t0)
    return float(cv.top1.mean())
t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
# enqueue the untuned baseline config as trial 0 so the search can be compared against it on identical folds
study.enqueue_trial(dict(n_estimators=40, max_depth=12, min_samples_leaf=20, max_features=0.186, max_samples=1.0))
study.optimize(objective, n_trials=N_TRIALS, n_jobs=4, callbacks=[lambda s, t: print(f"trial {t.number} top1={t.value:.4f} {t.params}", flush=True)])
total = time.time() - t0
df = study.trials_dataframe(); df.to_csv("../results/bayes_trials.csv", index=False)
json.dump(dict(best_params=study.best_params, best_cv_top1=study.best_value, n_trials=N_TRIALS, total_runtime_s=total, sampler="TPE(seed=42, 8 random startup)",
               objective="mean 3-fold group-CV observation-level Top-1 on 80 train trips (3x thinner obs sample)"), open("../results/bayes_best.json", "w"), indent=2)
print("BEST", study.best_value, study.best_params, "runtime %.0fs" % total)
