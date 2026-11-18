"""Per-(observation, candidate-arc) feature engineering. Label-agnostic: used identically in training and live API."""
import numpy as np, pandas as pd
R_DEFAULT, K_DEFAULT = 200.0, 60
FEATURES = ["d", "d_minus_min", "d_ratio_min", "rank_d", "n_cand", "f", "L", "dist_u", "dist_v",
            "cos_h2", "cos_h5", "abs_cos_h2", "align_rank", "lat_off", "speed", "moving", "dt",
            "has_rev", "outdeg_v", "indeg_u", "outdeg_u", "indeg_v",
            "d_prev3", "d_next3", "d_prev6", "d_next6", "d_ctx_mean", "turn_rate", "cos_x_moving"]

def _heading(P, w):
    n = len(P); i = np.arange(n)
    a = np.clip(i - w, 0, n - 1); b = np.clip(i + w, 0, n - 1)
    return P[b] - P[a]

def build_features(geo, R=R_DEFAULT, K=K_DEFAULT):
    oi, ai, d, f = geo.candidates(R, K)
    n = len(geo.P)
    if len(oi) == 0:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES)
    L = geo.L[ai]
    # rank / margins
    starts = np.r_[0, np.flatnonzero(np.diff(oi)) + 1]; counts = np.diff(np.r_[starts, len(oi)])
    rank = np.arange(len(oi)) - np.repeat(starts, counts)
    dmin = np.repeat(d[starts], counts); ncand = np.repeat(counts, counts)
    # kinematics
    t = geo.t; dt = np.r_[np.diff(t)[:1], np.diff(t)] if n > 1 else np.ones(1)
    dt = np.maximum(dt, 1.0)
    h2 = _heading(geo.P, 2); h5 = _heading(geo.P, 5)
    sp2 = np.linalg.norm(h2, axis=1); sp5 = np.linalg.norm(h5, axis=1)
    i = np.arange(n); a2 = np.clip(i - 2, 0, n - 1); b2 = np.clip(i + 2, 0, n - 1)
    tspan = np.maximum(t[b2] - t[a2], 1.0)
    speed = sp2 / tspan
    ang2 = np.arctan2(h2[:, 1], h2[:, 0]); ang5 = np.arctan2(h5[:, 1], h5[:, 0])
    dang = np.abs(np.angle(np.exp(1j * (ang5 - ang2))))
    bear = geo.bear[ai]
    cos2 = np.cos(ang2[oi] - bear); cos5 = np.cos(ang5[oi] - bear)
    moving = (sp2[oi] > 3.0).astype(float)          # >3 m over 4 steps -> heading is meaningful
    cos2 = np.where(moving > 0, cos2, 0.0); cos5 = np.where(sp5[oi] > 5.0, cos5, 0.0)
    # signed lateral offset (left +) of the observation relative to arc direction
    AB = geo.AB[ai]; A = geo.A[ai]; Pp = geo.P[oi]
    cross = (AB[:, 0] * (Pp[:, 1] - A[:, 1]) - AB[:, 1] * (Pp[:, 0] - A[:, 0])) / (geo.L[ai] + 1e-9)
    # rank among direction-aligned candidates
    aligned_d = np.where(cos2 > 0, d, 1e9)
    df = pd.DataFrame({"obs": oi, "arc": ai, "ad": aligned_d})
    align_rank = df.groupby("obs")["ad"].rank(method="first").values
    align_rank = np.where(cos2 > 0, align_rank, 99)
    # temporal context: distance of neighbouring GPS points to the SAME arc (continuity cue)
    def ctx(shift):
        j = np.clip(oi + shift, 0, n - 1)
        dd, _ = geo.seg_dist(geo.P[j], ai); return dd
    dp3, dn3, dp6, dn6 = ctx(-3), ctx(3), ctx(-6), ctx(6)
    out = pd.DataFrame({
        "obs": oi, "arc": ai, "d": d, "d_minus_min": d - dmin, "d_ratio_min": d / (dmin + 1.0),
        "rank_d": rank, "n_cand": ncand, "f": f, "L": L, "dist_u": f * L, "dist_v": (1 - f) * L,
        "cos_h2": cos2, "cos_h5": cos5, "abs_cos_h2": np.abs(cos2), "align_rank": align_rank,
        "lat_off": cross, "speed": speed[oi], "moving": moving, "dt": dt[oi],
        "has_rev": geo.has_rev[ai].astype(float),
        "outdeg_v": geo.out_deg[geo.v[ai]], "indeg_u": geo.in_deg[geo.u[ai]],
        "outdeg_u": geo.out_deg[geo.u[ai]], "indeg_v": geo.in_deg[geo.v[ai]],
        "d_prev3": dp3, "d_next3": dn3, "d_prev6": dp6, "d_next6": dn6,
        "d_ctx_mean": (d + dp3 + dn3 + dp6 + dn6) / 5.0, "turn_rate": dang[oi],
        "cos_x_moving": cos2 * moving})
    return out
