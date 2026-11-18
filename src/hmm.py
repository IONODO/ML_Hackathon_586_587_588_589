"""HMM / Viterbi map-matching decoder.
Hidden state at time t  = candidate road arc (arc_id) among the K nearest arcs within R metres of GPS fix t.
Emission   P(z_t|s_t)   ∝ classifier probability p(arc is reference | features)   (weighted in log space)
Transition P(s_t|s_t-1) = Newson-Krumm: (1/beta) exp(-|d_route - d_gps| / beta)  (+ U-turn penalty, fallback penalty,
                           speed plausibility, uninformative reset after large time gaps)."""
import heapq, numpy as np
from functools import lru_cache

class RouteDist:
    def __init__(self, geo):
        self.geo = geo
        adj = [[] for _ in range(geo.n_nodes)]
        for u, v, l in zip(geo.u.tolist(), geo.v.tolist(), geo.L.tolist()): adj[u].append((v, l))
        self.adj = adj
        self._cached = lru_cache(maxsize=4000)(self._dij)
    def _dij(self, src, cutoff):
        dist = {src: 0.0}; pq = [(0.0, src)]; adj = self.adj
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, 1e18): continue
            for m, l in adj[n]:
                nd = d + l
                if nd <= cutoff and nd < dist.get(m, 1e18): dist[m] = nd; heapq.heappush(pq, (nd, m))
        return dist
    def from_node(self, src, cutoff): return self._cached(int(src), float(cutoff))

def build_chain(X, score, n_obs):
    """Group candidate rows by observation. Returns chain list of dicts (only obs that have >=1 candidate)."""
    order = np.argsort(X["obs"].values, kind="stable")
    obs = X["obs"].values[order]; arc = X["arc"].values[order]; fr = X["f"].values[order].astype(float); sc = np.asarray(score)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]; ends = np.r_[starts[1:], len(obs)]
    return [dict(obs=int(obs[s]), arc=arc[s:e], f=fr[s:e], p=sc[s:e]) for s, e in zip(starts, ends)]

def precompute_transitions(geo, chain, gap_s=30.0, noise_back=15.0, rd=None):
    """For each consecutive chain pair: matrix of route distance (inf if unreachable within cutoff), u-turn flags, d_gps, dt."""
    rd = rd or RouteDist(geo); T = []
    for a, b in zip(chain[:-1], chain[1:]):
        dt = geo.t[b["obs"]] - geo.t[a["obs"]]; dgc = float(np.linalg.norm(geo.P[b["obs"]] - geo.P[a["obs"]]))
        if dt > gap_s:
            T.append(dict(dt=dt, dgc=dgc, dr=None, ut=None)); continue
        cutoff = max(250.0, 2.0 * dgc + 150.0); cutoff = np.ceil(cutoff / 250.0) * 250.0
        na, nb = len(a["arc"]), len(b["arc"]); dr = np.full((na, nb), np.inf); ut = np.zeros((na, nb), bool)
        for i in range(na):
            ai = a["arc"][i]; fa = a["f"][i]; La = geo.L[ai]
            dists = rd.from_node(geo.v[ai], cutoff)
            for j in range(nb):
                bj = b["arc"][j]; fb = b["f"][j]; Lb = geo.L[bj]
                if bj == ai:
                    diff = (fb - fa) * La
                    dr[i, j] = max(diff, 0.0) if diff >= -noise_back else np.inf
                else:
                    sp = dists.get(int(geo.u[bj]))
                    if sp is not None: dr[i, j] = (1 - fa) * La + sp + fb * Lb
                    ut[i, j] = (geo.u[bj] == geo.v[ai]) and (geo.v[bj] == geo.u[ai])
        T.append(dict(dt=dt, dgc=dgc, dr=dr, ut=ut))
    return T

DEFAULT = dict(beta=8.0, we=1.0, fb=-10.0, uturn=6.0, gap_s=30.0, vmax=50.0)
def viterbi(chain, T, prm=None):
    prm = {**DEFAULT, **(prm or {})}
    n = len(chain)
    if n == 0: return np.array([], dtype=np.int64), np.array([], dtype=int)
    logE = [prm["we"] * np.log(np.clip(c["p"], 1e-4, 1.0)) for c in chain]
    delta = logE[0].copy(); back = [None]
    for t in range(1, n):
        tr = T[t - 1]
        if tr["dr"] is None or tr["dt"] > prm["gap_s"]:
            # large GPS gap: transition carries no information -> restart from emission only (best previous score kept constant)
            logT = np.zeros((len(chain[t - 1]["arc"]), len(chain[t]["arc"])))
        else:
            dr = tr["dr"]; reach = np.isfinite(dr) & (dr <= prm["vmax"] * max(tr["dt"], 1.0) + 100.0)
            logT = np.where(reach, -np.abs(np.where(reach, dr, 0.0) - tr["dgc"]) / prm["beta"] - np.log(prm["beta"]), prm["fb"])
            logT = logT - prm["uturn"] * tr["ut"]
        sc = delta[:, None] + logT; bp = np.argmax(sc, axis=0)
        delta = sc[bp, np.arange(sc.shape[1])] + logE[t]; back.append(bp)
    j = int(np.argmax(delta)); sel = np.zeros(n, dtype=int); sel[-1] = j
    for t in range(n - 1, 0, -1): j = int(back[t][j]); sel[t - 1] = j
    return np.array([chain[t]["arc"][sel[t]] for t in range(n)], dtype=np.int64), sel

def decode_trip(geo, chain, T, n_obs, prm=None):
    """Return predicted arc per GPS observation (length n_obs). Observations with no candidate are forward-filled
    from the previous decoded arc (back-filled at the start) and flagged in `filled`."""
    arcs, _ = viterbi(chain, T, prm)
    pred = np.full(n_obs, -1, dtype=np.int64); filled = np.ones(n_obs, bool)
    for c, a in zip(chain, arcs): pred[c["obs"]] = a; filled[c["obs"]] = False
    last = -1
    for i in range(n_obs):
        if pred[i] >= 0: last = pred[i]
        elif last >= 0: pred[i] = last
    nxt = -1
    for i in range(n_obs - 1, -1, -1):
        if pred[i] >= 0: nxt = pred[i]
        elif nxt >= 0: pred[i] = nxt
    return pred, filled
