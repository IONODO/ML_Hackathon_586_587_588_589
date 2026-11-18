import numpy as np, pandas as pd, time, joblib, os, warnings, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_data import load_all, train_sample
from features import FEATURES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
X, O, _ = load_all()
Xtr = train_sample(X, O)
m = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20, max_features=0.2, random_state=42, n_jobs=2))])
t0=time.time(); m.fit(Xtr[FEATURES].values.astype("float32"), Xtr.y.values)
print("trained in %.1fs" % (time.time()-t0))
os.makedirs("../artifacts", exist_ok=True)
joblib.dump(m, "../artifacts/rf.joblib")
print("saved")
