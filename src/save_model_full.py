import numpy as np, pandas as pd, time, joblib, os, warnings, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import load_all
from features import FEATURES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
X, O, _ = load_all()
Xtr = X[(X.partition=="train")]
hit = O[(O.partition=="train") & O.hit][["tc","obs"]]
hit_key = set((hit.tc.values.astype(np.int64)*10**6 + hit.obs.values).tolist())
key = Xtr.tc.values.astype(np.int64)*10**6 + Xtr.obs.values
keep = np.fromiter((k in hit_key for k in key), bool, len(key))
Xtr = Xtr[keep].reset_index(drop=True)
m = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=10, max_features=0.25, random_state=42, n_jobs=2))])
t0=time.time(); m.fit(Xtr[FEATURES].values.astype("float32"), Xtr.y.values)
print("trained full in %.1fs rows=%d" % (time.time()-t0, len(Xtr)))
os.makedirs("../artifacts", exist_ok=True)
joblib.dump(m, "../artifacts/rf_full.joblib")
print("saved")
