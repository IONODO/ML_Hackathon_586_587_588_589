"""FastAPI backend (Laptop 1). Server-side evaluation only; ground truth never leaves."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from data_loader import load_trip as _load_trip_dev
from geo import TripGeo
from features import build_features, FEATURES
from labeling import align_to_route
from hmm import build_chain, precompute_transitions, decode_trip
from eval_utils import top1_predictions
from viz import make_map
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
EVAL = "evaluator_test"
HMM_PARAMS = {"beta": 8.0, "we": 1.0, "fb": -10.0, "uturn": 6.0, "gap_s": 30.0}

def load_eval_trip(tid):
    p = os.path.join(DATA, EVAL, tid, tid)
    track = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(p + ".route", sep=r"\s+", header=None, names=["arc"])["arc"].values
    return track, nodes, arcs, route

# load pre-trained model (no retraining at startup)
import joblib
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "rf.joblib")
t0 = time.time()
try:
    model = joblib.load(MODEL_PATH)
    print("loaded model in %.1fs" % (time.time() - t0))
except Exception as e:
    print("model load failed:", e, "training fallback")
    from common_data import load_all, train_sample
    X, O, _ = load_all(); Xtr = train_sample(X, O)
    model = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=20, max_features=0.2, random_state=42, n_jobs=2))])
    model.fit(Xtr[FEATURES].values.astype("float32"), Xtr.y.values)
    joblib.dump(model, MODEL_PATH)
    print("trained+saved in %.1fs" % (time.time() - t0))

app = FastAPI(title="PathSnap API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class MatchReq(BaseModel):
    trip_id: str

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "version": "1.0.0", "n_eval_trips": len(os.listdir(os.path.join(DATA, EVAL))), "uptime_s": round(time.time() - t0, 1)}

@app.get("/trips")
def trips():
    out = []
    for tid in sorted(os.listdir(os.path.join(DATA, EVAL))):
        tr, no, ar, ro = load_eval_trip(tid)
        out.append({"trip_id": tid, "n_obs": len(tr), "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0])})
    return {"trips": out}

@app.post("/match")
def match(req: MatchReq):
    tid = req.trip_id
    if not os.path.isdir(os.path.join(DATA, EVAL, tid)):
        raise HTTPException(400, "unknown trip_id")
    t_start = time.time()
    track, nodes, arcs, route = load_eval_trip(tid)
    geo = TripGeo(track, nodes, arcs)
    Xf = build_features(geo)
    score = model.predict_proba(Xf[FEATURES].values.astype("float32"))[:, 1]
    lab = route[align_to_route(geo.P, geo.t, geo, route)]
    pred_clf = top1_predictions(Xf, score, len(track))
    classifier_top1 = float((pred_clf == lab).mean())
    chain = build_chain(Xf, score, len(track))
    T = precompute_transitions(geo, chain)
    pred, filled = decode_trip(geo, chain, T, len(track), HMM_PARAMS)
    # server-side metrics
    top1 = float((pred == lab).mean())
    from eval_utils import route_continuity, sequence_metrics, collapse
    s, r = route_continuity(pred, geo)
    seq = sequence_metrics(pred, route, geo)
    # folium map
    try:
        m = make_map(geo, track, ref_route=route, preds={"HMM route": (pred, "#d73027")})
        map_html = m.get_root().render()
    except Exception as e:
        map_html = ""
    latency = (time.time() - t_start) * 1000
    return {
        "trip_id": tid, "n_obs": len(track), "duration_s": float(track.t.iloc[-1] - track.t.iloc[0]),
        "latency_ms": round(latency, 1),
        "classifier_top1": classifier_top1, "hmm_top1": top1,
        "metrics": {"top1_accuracy": top1, "route_continuity_strict": float(s), "route_continuity_relaxed": float(r),
                    "lcs_f1": float(seq["lcs_f1"]), "len_f1": float(seq["len_f1"])},
        "predicted_arcs": [int(x) for x in pred], "filled_obs": [bool(x) for x in filled],
        "map_html": map_html
    }

@app.get("/map/{trip_id}", response_class=HTMLResponse)
def get_map(trip_id: str):
    # regenerate quickly
    track, nodes, arcs, route = load_eval_trip(trip_id)
    geo = TripGeo(track, nodes, arcs)
    Xf = build_features(geo)
    score = model.predict_proba(Xf[FEATURES].values.astype("float32"))[:, 1]
    lab = route[align_to_route(geo.P, geo.t, geo, route)]
    chain = build_chain(Xf, score, len(track))
    T = precompute_transitions(geo, chain)
    pred, _ = decode_trip(geo, chain, T, len(track), HMM_PARAMS)
    m = make_map(geo, track, ref_route=route, preds={"HMM route": (pred, "#d73027")})
    return HTMLResponse(m.get_root().render())
