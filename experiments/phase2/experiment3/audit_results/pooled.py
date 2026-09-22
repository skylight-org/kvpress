import numpy as np, pandas as pd, sys
sys.argv = ["x"]
exec(open("audit_stats.py").read().split("rows = []")[0])     # reuse loaders only
rng = np.random.default_rng(0)
for rn, r in ratios.items():
    ds = []
    for t in tasks:
        A, B = arm(t, "kvzip_bern", r), arm(t, "kvzip", r)
        if A is None or B is None: continue
        d = (A - B).groupby(level=0).mean().values
        ds.append(d / d.std(ddof=1) if d.std() > 0 else d)          # standardise per task so no task dominates
    x = np.concatenate(ds); obs = x.mean()
    null = (rng.choice([-1, 1], size=(200000, len(x))) * x).mean(1)
    print(rn, "tasks", len(ds), "contexts", len(x), "pooled std-diff %.3f" % obs, "two-sided sign-flip p %.5f" % (np.abs(null) >= abs(obs)).mean())
