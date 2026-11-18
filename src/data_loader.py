"""Central dataset loader enforcing partition separation.
Partitions: train (80), student_test (10). evaluator_test is NEVER loaded by dev code."""
import os, numpy as np, pandas as pd
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
def list_trips(partition):
    assert partition in ("train", "student_test"), "only train/student_test allowed in dev code"
    return sorted(os.listdir(os.path.join(DATA, partition)))
def load_trip(partition, tid, with_route=True):
    p = os.path.join(DATA, partition, tid, tid)
    track = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "t"])
    nodes = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["u", "v"])
    route = pd.read_csv(p + ".route", sep=r"\s+", header=None, names=["arc"])["arc"].values if with_route else None
    return track, nodes, arcs, route
