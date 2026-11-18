import numpy as np, pandas as pd
from data_loader import *; from geo import *
from features import FEATURES
STRIDE = 5   # training observations sub-sampled every 5th GPS point (1 CPU / 3 GB sandbox); documented
def load_all():
    X = pd.read_pickle("../results/cand_features.pkl"); O = pd.read_pickle("../results/obs_labels.pkl")
    tcodes = {t: i for i, t in enumerate(sorted(set(O.trip)))}
    X["tc"] = X.trip.map(tcodes).astype(np.int32); O["tc"] = O.trip.map(tcodes).astype(np.int32)
    return X, O, tcodes
def train_sample(X, O, stride=STRIDE):
    Xt = X[(X.partition == "train") & (X.obs % stride == 0)]
    hit = O[(O.partition == "train") & O.hit][["tc", "obs"]]
    hit_key = set((hit.tc.values.astype(np.int64) * 10**6 + hit.obs.values).tolist())
    key = Xt.tc.values.astype(np.int64) * 10**6 + Xt.obs.values
    keep = np.fromiter((k in hit_key for k in key), bool, len(key))
    return Xt[keep].reset_index(drop=True)
def student_set(X):
    return X[X.partition == "student_test"].reset_index(drop=True)
_geo_cache = {}
def trip_ctx(part, tid):
    if (part, tid) not in _geo_cache:
        tr, no, ar, ro = load_trip(part, tid); _geo_cache[(part, tid)] = (TripGeo(tr, no, ar), ro)
    return _geo_cache[(part, tid)]
