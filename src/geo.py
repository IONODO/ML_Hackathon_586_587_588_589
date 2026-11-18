"""Projection, arc geometry, spatial index, candidate generation."""
import numpy as np, shapely
from shapely import STRtree
from pyproj import Transformer

def utm_epsg(lon0, lat0):
    zone = int((lon0 + 180) // 6) + 1
    return (32600 if lat0 >= 0 else 32700) + zone

class TripGeo:
    """Per-trip projected network. Trips are located worldwide, so the UTM zone is chosen
    from each trip's own GPS centroid (documented in report)."""
    def __init__(self, track, nodes, arcs):
        lon0, lat0 = track.lon.mean(), track.lat.mean()
        self.epsg = utm_epsg(lon0, lat0)
        self.tf = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self.inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        x, y = self.tf.transform(track.lon.values, track.lat.values)
        self.P = np.c_[x, y]; self.t = track.t.values.astype(float)
        nx_, ny_ = self.tf.transform(nodes.lon.values, nodes.lat.values)
        self.N = np.c_[nx_, ny_]
        self.u = arcs.u.values.astype(np.int64); self.v = arcs.v.values.astype(np.int64)
        self.A = self.N[self.u]; self.B = self.N[self.v]
        self.AB = self.B - self.A
        self.L = np.linalg.norm(self.AB, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.bear = np.arctan2(self.AB[:, 1], self.AB[:, 0])  # math angle (rad) E=0
        self.n_nodes = len(nodes); self.n_arcs = len(arcs)
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        # reverse-arc existence (two-way street)
        key = {(a, b) for a, b in zip(self.u.tolist(), self.v.tolist())}
        self.has_rev = np.array([(b, a) in key for a, b in zip(self.u.tolist(), self.v.tolist())])
        self._tree = None
    def tree(self):
        if self._tree is None:
            lines = shapely.linestrings(np.stack([self.A, self.B], axis=1))
            self._tree = STRtree(lines)
        return self._tree
    def seg_dist(self, pts, arc_idx):
        """perpendicular distance + fraction along arc for paired (pts[i], arc_idx[i])."""
        A = self.A[arc_idx]; AB = self.AB[arc_idx]; L2 = (AB ** 2).sum(1) + 1e-9
        f = np.clip(((pts - A) * AB).sum(1) / L2, 0, 1)
        C = A + f[:, None] * AB
        return np.linalg.norm(pts - C, axis=1), f
    def candidates(self, R=60.0, K=15):
        """Return (obs_idx, arc_idx, dist, frac) with arcs within R of each obs, top-K nearest per obs."""
        pts = shapely.points(self.P)
        pairs = self.tree().query(pts, predicate="dwithin", distance=R)
        oi, ai = pairs[0], pairs[1]
        d, f = self.seg_dist(self.P[oi], ai)
        order = np.lexsort((d, oi)); oi, ai, d, f = oi[order], ai[order], d[order], f[order]
        # rank within obs
        starts = np.r_[0, np.flatnonzero(np.diff(oi)) + 1]
        counts = np.diff(np.r_[starts, len(oi)])
        rank = np.arange(len(oi)) - np.repeat(starts, counts)
        keep = rank < K
        return oi[keep], ai[keep], d[keep], f[keep]
