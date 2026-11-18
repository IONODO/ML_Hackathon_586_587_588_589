import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

def candidate_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    out = dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
               recall=recall_score(y, yh, zero_division=0), f1_pos=f1_score(y, yh, zero_division=0),
               macro_f1=f1_score(y, yh, average="macro", zero_division=0))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out["log_loss"] = log_loss(y, pc, labels=[0, 1])
    return out

def top1_predictions(X, score, n_obs):
    """argmax-score candidate arc per observation -> array (n_obs,), -1 when no candidate."""
    df = pd.DataFrame({"obs": X["obs"].values, "arc": X["arc"].values, "s": score})
    idx = df.groupby("obs")["s"].idxmax()
    pred = np.full(n_obs, -1, dtype=np.int64)
    pred[df.loc[idx, "obs"].values] = df.loc[idx, "arc"].values
    return pred

def obs_top1(pred, label, mask=None):
    ok = pred == label
    return ok.mean() if mask is None else ok[mask].mean()

def collapse(seq):
    seq = np.asarray(seq); seq = seq[seq >= 0]
    if len(seq) == 0: return seq
    return seq[np.r_[True, seq[1:] != seq[:-1]]]

def lcs_len(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0 or len(b) == 0: return 0
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for x in a:
        match = (b == x).astype(np.int32)
        tmp = np.maximum(prev[1:], prev[:-1] + match)
        cur = np.zeros_like(prev); cur[1:] = np.maximum.accumulate(tmp)
        prev = cur
    return int(prev[-1])

def sequence_metrics(pred_arcs, ref_route, geo):
    """Route-level: LCS-F1 on collapsed arc sequences + length-weighted arc-set precision/recall/F1."""
    pr = collapse(pred_arcs); rr = collapse(ref_route)
    l = lcs_len(pr, rr)
    lp = l / max(len(pr), 1); lr = l / max(len(rr), 1)
    lcs_f1 = 2 * lp * lr / (lp + lr + 1e-12)
    ps, rs = set(pr.tolist()), set(rr.tolist()); inter = ps & rs
    Lp = geo.L[list(ps)].sum() if ps else 0; Lr = geo.L[list(rs)].sum() if rs else 0
    Li = geo.L[list(inter)].sum() if inter else 0
    lp_m = Li / (Lp + 1e-9); lr_m = Li / (Lr + 1e-9)
    return dict(lcs_f1=lcs_f1, lcs_prec=lp, lcs_rec=lr, len_prec=lp_m, len_rec=lr_m,
                len_f1=2 * lp_m * lr_m / (lp_m + lr_m + 1e-12), n_pred_arcs=len(pr), n_ref_arcs=len(rr))

def route_continuity(pred, geo, hops=2):
    """Fraction of consecutive observation pairs whose predicted arcs are identical, directly connected
    (head-to-tail), or connected through <= `hops` intermediate arcs. Returns (strict, relaxed)."""
    out_by_node = {}
    for i, (u, v) in enumerate(zip(geo.u, geo.v)): out_by_node.setdefault(int(u), []).append(i)
    strict = relaxed = tot = 0
    cache = {}
    def reach(a):
        if a in cache: return cache[a]
        frontier = {int(geo.v[a])}; nodes = {int(geo.v[a])}
        for _ in range(hops):
            nxt = set()
            for n in frontier:
                for e in out_by_node.get(n, []): nxt.add(int(geo.v[e]))
            frontier = nxt - nodes; nodes |= nxt
        cache[a] = nodes; return nodes
    for a, b in zip(pred[:-1], pred[1:]):
        tot += 1
        if a < 0 or b < 0: continue
        if a == b or geo.v[a] == geo.u[b]: strict += 1; relaxed += 1
        elif int(geo.u[b]) in reach(a): relaxed += 1
    return strict / max(tot, 1), relaxed / max(tot, 1)
