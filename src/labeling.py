"""Derive per-observation reference arc by monotone alignment of GPS to the ordered .route.
The .route has no per-GPS-point index, so this alignment is a documented labelling assumption."""
import numpy as np
def align_to_route(P, t, geo, route, kmax=12, lam=0.5, gap_s=60.0):
    n, m = len(P), len(route)
    A = geo.A[route]; AB = geo.AB[route]; L2 = (AB ** 2).sum(1) + 1e-9
    INF = 1e12
    dp_prev = None
    bp = np.zeros((n, m), dtype=np.int16)
    def dist_row(i):
        f = np.clip(((P[i] - A) * AB).sum(1) / L2, 0, 1)
        C = A + f[:, None] * AB
        return np.linalg.norm(P[i] - C, axis=1)
    d0 = dist_row(0)
    dp_prev = d0.copy()
    for i in range(1, n):
        d = dist_row(i)
        big = (t[i] - t[i - 1]) > gap_s
        if big:  # unlimited forward advance after a gap
            cm = np.minimum.accumulate(dp_prev)
            best = cm.copy()
            # argmin position of running min
            idx = np.arange(m); arg = np.where(dp_prev == cm, idx, 0); arg = np.maximum.accumulate(arg)
            bp[i] = (idx - arg).clip(0, 32000)
        else:
            best = dp_prev.copy(); bk = np.zeros(m, dtype=np.int16)
            for k in range(1, kmax + 1):
                cand = np.full(m, INF); cand[k:] = dp_prev[:m - k] + lam * k
                better = cand < best; best = np.where(better, cand, best); bk = np.where(better, k, bk)
            bp[i] = bk
        dp_prev = best + d
    # backtrack: free end
    j = int(np.argmin(dp_prev)); js = np.zeros(n, dtype=np.int64); js[-1] = j
    for i in range(n - 1, 0, -1):
        j = j - int(bp[i, j]); js[i - 1] = j
    return js  # route index per obs
