"""Train-vs-CV gap for every model (evidence for the over/under-fitting question) + MLP epoch-wise loss curves."""
import numpy as np, pandas as pd, warnings, json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from shared_eval import *; from model_zoo import *
warnings.filterwarnings("ignore")
X, O, _ = load_all(); Xtr = train_sample(X, O); del X
rows = []
cvall = pd.read_csv("../results/cv_folds_all_models.csv")
ev = Xtr[(Xtr.obs // STRIDE) % 3 == 0]    # evaluate on a third of the train sample (cost control)
for name in ZOO:
    z = ZOO[name]; tr_ = Xtr if z["train_stride"] == 1 else Xtr[(Xtr.obs // STRIDE) % z["train_stride"] == 0]
    m = build(name); m.fit(vec(tr_), tr_.y.values); p = m.predict_proba(vec(ev))[:, 1]
    cm = candidate_metrics(ev.y.values, p); top1 = top1_hit_only(ev, p); c = cvall[cvall.model == name]
    rows.append(dict(model=name, train_top1=top1, cv_top1=c.top1.mean(), gap_top1=top1 - c.top1.mean(), train_macro_f1=cm["macro_f1"], cv_macro_f1=c.macro_f1.mean(),
                     gap_macro_f1=cm["macro_f1"] - c.macro_f1.mean(), train_log_loss=cm["log_loss"], cv_log_loss=c.log_loss.mean()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("../results/train_vs_cv_gap.csv", index=False)
# ---- MLP epoch-wise curves: train on 72 trips, validate on 8 held-out TRIPS (group split, seeded) ----
trips = np.array(sorted(Xtr.tc.unique())); rng = np.random.RandomState(SEED); val_t = rng.choice(trips, 8, replace=False)
trn = Xtr[~Xtr.tc.isin(val_t)]; val = Xtr[Xtr.tc.isin(val_t)]
sc = StandardScaler().fit(vec(trn)); A, B = sc.transform(vec(trn)), sc.transform(vec(val))
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, random_state=SEED)
from sklearn.metrics import log_loss
cur = []; idx = np.arange(len(A))
for ep in range(1, 31):
    rng.shuffle(idx)
    for s in range(0, len(idx), 512): mlp.partial_fit(A[idx[s:s + 512]], trn.y.values[idx[s:s + 512]], classes=[0, 1])
    cur.append(dict(epoch=ep, train_loss=log_loss(trn.y.values, np.clip(mlp.predict_proba(A)[:, 1], 1e-6, 1 - 1e-6)),
                    val_loss=log_loss(val.y.values, np.clip(mlp.predict_proba(B)[:, 1], 1e-6, 1 - 1e-6))))
pd.DataFrame(cur).to_csv("../results/mlp_epoch_curves.csv", index=False); print(pd.DataFrame(cur).round(4).iloc[[0, 4, 9, 14, 19, 24, 29]])
json.dump(dict(val_trips=[int(x) for x in val_t]), open("../results/mlp_curve_split.json", "w"))
