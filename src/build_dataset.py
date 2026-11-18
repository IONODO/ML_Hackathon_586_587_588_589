import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from data_loader import *; from geo import *; from labeling import *; from features import *
def work(args):
    part, tid = args
    tr, no, ar, ro = load_trip(part, tid); g = TripGeo(tr, no, ar)
    js = align_to_route(g.P, g.t, g, ro); lab = ro[js]
    dlab, _ = g.seg_dist(g.P, lab)
    if float(np.median(dlab)) > 50.0:
        print("SKIP outlier trip", tid, "med_dist", round(float(np.median(dlab)),1), flush=True)
        return None
    X = build_features(g)
    X["y"] = (X.arc.values == lab[X.obs.values]).astype(np.int8)
    # boundary alignment: keep only observations whose reference arc projection falls inside the segment
    pos = X[X.y == 1]
    good_obs = set(pos.loc[(pos.f >= 0) & (pos.f <= 1), "obs"].tolist())
    X = X[X.obs.isin(good_obs)].reset_index(drop=True)
    # recompute obs_labels hit accordingly (obs already in O)
    X["trip"] = tid; X["partition"] = part
    hit = np.zeros(len(tr), bool); hit[X.obs.values[X.y.values == 1]] = True
    has_c = np.zeros(len(tr), bool); has_c[X.obs.unique()] = True
    obs = pd.DataFrame({"trip": tid, "partition": part, "obs": np.arange(len(tr)), "label_arc": lab, "label_ridx": js,
                        "label_dist": dlab, "hit": hit, "has_cand": has_c})
    return X, obs
if __name__ == "__main__":
    jobs = [(p, t) for p in ("train", "student_test") for t in list_trips(p)]
    t0 = time.time()
    with Pool(max(1, __import__("os").cpu_count() - 0)) as pool: res = pool.map(work, jobs, chunksize=1)
    res = [r for r in res if r is not None]
    X = pd.concat([r[0] for r in res], ignore_index=True); O = pd.concat([r[1] for r in res], ignore_index=True)
    fc=[c for c in X.columns if X[c].dtype==np.float64]; X[fc]=X[fc].astype(np.float32); X.to_pickle("../results/cand_features.pkl"); O.to_pickle("../results/obs_labels.pkl")
    print("built in %.0fs" % (time.time() - t0), X.shape, O.shape)
    g = O.groupby("partition").agg(n_obs=("obs", "size"), miss_rate=("hit", lambda s: 1 - s.mean()), no_cand=("has_cand", lambda s: 1 - s.mean()),
                                   lab_med=("label_dist", "median"), lab_gt30=("label_dist", lambda s: (s > 30).mean()))
    print(g); print("pos rate", X.groupby("partition").y.mean()); print("cands/obs", X.groupby("partition").size() / O.groupby("partition").size())
    pt = O.groupby("trip").agg(miss=("hit", lambda s: 1 - s.mean()), gt30=("label_dist", lambda s: (s > 30).mean())).sort_values("miss", ascending=False)
    pt.to_csv("../results/candidate_miss_per_trip.csv"); print(pt.head(8))
