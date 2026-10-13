"""Independent re-analysis of E3 from predictions.csv (does not import summarize.py)."""
import glob, itertools, json, math, re, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/rethinkingai-self/23b1835/phase2/experiment3")
from scbench_metrics import score_one

R = "/home/rethinkingai-self/23b1835/phase2/experiment3/results/"
def load(group, task, press, ratio):
    ds = glob.glob(f"{R}{group}/scbench__{task}__*__{press}__{ratio}")
    if not ds: return None
    ms = sorted(glob.glob(ds[0] + "/metrics.json") + glob.glob(ds[0] + "/[0-9]*/metrics.json"))
    if not ms:
        return "INCOMPLETE"
    assert len(ms) == 1, ms
    d = ms[0].rsplit("/", 1)[0]
    df = pd.read_csv(d + "/predictions.csv"); df["predicted_answer"] = df.predicted_answer.fillna("")
    df["s"] = [score_one(task, p, a) for p, a in zip(df.predicted_answer, df.answer)]
    return df

# --- 1. score cache equivalence: vcache (uncached compute) vs S1n10 (cache hit)
print("== cache equivalence ==")
for p in ("kvzip", "kvzip_bern_s1"):
    a = load("vcache", "summary", p, "0.88"); b = load("S1n10", "summary", p, "0.88")
    m = a.merge(b, on=["context_id", "turn"], suffixes=("_v", "_s"))
    print(p, "n", len(m), "identical", (m.predicted_answer_v == m.predicted_answer_s).sum())

# --- 2. paired stats with t, exact sign-flip permutation, Holm
tasks = ["prefix_suffix", "vt", "qa_eng", "summary"]
ratios = {"2x": "0.50", "4x": "0.75", "8x": "0.88", "16x": "0.94"}
def arm(task, a, r):
    if a == "kvzip_bern":
        fs = [load("S1n10", task, f"kvzip_bern_s{s}", r) for s in (1, 2, 3)]
        if any(f is None or isinstance(f, str) for f in fs): return None
        x = pd.concat([f.set_index(["context_id", "turn"]).s for f in fs], axis=1)
        return x.mean(axis=1)
    f = load("S1n10", task, a, r)
    return None if f is None or isinstance(f, str) else f.set_index(["context_id", "turn"]).s

def perm_p(x):
    x = np.asarray(x); obs = abs(x.mean())
    flips = np.array(list(itertools.product([1, -1], repeat=len(x))))
    return float((np.abs((flips * x).mean(1)) >= obs - 1e-12).mean())

rows = []
for comp, b in (("bern-topk", "kvzip"), ("bern-EA", "expected_attention"), ("bern-full", None), ("topk-full", None)):
    for t in tasks:
        for rn, r in ratios.items():
            A = arm(t, "kvzip" if comp == "topk-full" else "kvzip_bern", r)
            B = arm(t, "no_press", "0.00") if b is None else arm(t, b, r)
            if A is None or B is None: continue
            d = (A - B).groupby(level=0).mean()
            n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n)
            tp = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else float("nan")
            rows.append(dict(cmp=comp, task=t, ratio=rn, diff=m, se=se, z=m / se if se else np.nan,
                             t_p=tp, perm_p=perm_p(d.values), n_ctx=n, pos=(d > 0).sum(), neg=(d < 0).sum(),
                             ci_lo=m - stats.t.ppf(.975, n - 1) * se, ci_hi=m + stats.t.ppf(.975, n - 1) * se))
df = pd.DataFrame(rows)
for c, g in df.groupby("cmp"):
    ps = g.perm_p.values; order = np.argsort(ps); k = len(ps)
    holm = np.empty(k); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (k - i) * ps[j])); holm[j] = run
    df.loc[g.index, "holm_perm"] = holm
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.round(3).to_string())
df.to_csv("audit_stats.csv", index=False)

# --- 3. where does the prefix_suffix ceiling come from?
print("== prefix_suffix: no_press correct per context ==")
npz = load("S1n10", "prefix_suffix", "no_press", "0.00")
print(npz.groupby("context_id").s.apply(lambda x: int((x > 0).sum())).to_dict())
for r in ("0.50", "0.75", "0.88"):
    fs = [load("S1n10", "prefix_suffix", f"kvzip_bern_s{s}", r) for s in (1, 2, 3)]
    hits = [set(map(tuple, f[f.s > 0][["context_id", "turn"]].values)) for f in fs]
    base = set(map(tuple, npz[npz.s > 0][["context_id", "turn"]].values))
    print(r, "bern hits/seed", [len(h) for h in hits], "overlap with no_press hits", [len(h & base) for h in hits],
          "contexts with any hit", [len({c for c, _ in h}) for h in hits])
kz = load("S1n10", "prefix_suffix", "kvzip", "0.50")
print("kvzip 2x hits", int((kz.s > 0).sum()), "overlap w/ no_press", len(set(map(tuple, kz[kz.s > 0][["context_id", "turn"]].values)) & base))
