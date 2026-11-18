"""Seven models from six algorithm families. All wrapped in Pipeline(StandardScaler -> clf);
scaler is fit on training folds only (leakage-safe)."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
SEED = 42
ZOO = {
 "LogReg": dict(family="Linear", cfg="C=1.0, L2, lbfgs, max_iter=300",
     make=lambda: LogisticRegression(C=1.0, max_iter=300, random_state=SEED), train_stride=1),
 "GaussianNB": dict(family="Probabilistic", cfg="var_smoothing=1e-9",
     make=lambda: GaussianNB(), train_stride=1),
 "KNN": dict(family="Instance-based", cfg="k=25, distance-weighted, euclid, train subsample x6",
     make=lambda: KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=1), train_stride=6),
 "RandomForest": dict(family="Tree ensemble (bagging)", cfg="n_estimators=40, max_depth=12, min_leaf=20, max_features=sqrt, train subsample x2",
     make=lambda: RandomForestClassifier(n_estimators=40, max_depth=12, min_samples_leaf=20, max_features="sqrt", n_jobs=1, random_state=SEED), train_stride=2),
 "LightGBM": dict(family="Gradient boosting", cfg="n_est=200, lr=0.1, num_leaves=31, min_child=20, subsample/colsample=1.0",
     make=lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20, random_state=SEED, n_jobs=1, verbose=-1), train_stride=1),
 "MLP": dict(family="Neural network", cfg="hidden=(64,32), relu, adam lr=1e-3, alpha=1e-4, bs=512, early_stop(val 10%, patience 5), max_iter=40",
     make=lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=SEED), train_stride=1),
 "LinearSVM": dict(family="Kernel/margin (linear SVM)", cfg="LinearSVC C=0.5 + sigmoid (Platt) calibration cv=3",
     make=lambda: CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=2000, random_state=SEED, dual=False), method="sigmoid", cv=3), train_stride=1),
}
def build(name):
    return Pipeline([("scaler", StandardScaler()), ("clf", ZOO[name]["make"]())])
