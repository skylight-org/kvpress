"""Experiment 5: turn the per-head parquet files into the pre-registered tables.

    python summarize.py                # every group under results/
    python summarize.py r32k_llama     # one group

CPU only, no GPU, no model: everything comes from head_stats.parquet. Writes out/REPORT.md and
out/cells.csv.

Conventions carried from the earlier experiments:
  * aggregates are median [p25, p75] over HEADS, never a pooled mean, and are split at layer 13;
  * standard errors cluster at the CONTEXT, never the query, because all turns of one context share
    one compression draw;
  * tests are small-sample (paired t on context means, 9 df) with Holm, never |z| > 2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RESULTS, OUT = HERE / "results", HERE / "out"
OUT.mkdir(exist_ok=True)
RATIO_NAME = {0.5: "2x", 0.75: "4x", 0.875: "8x", 0.9375: "16x",
              0.96875: "32x", 0.984: "64x", 0.995: "200x"}


def load_run(path: Path) -> pd.DataFrame:
    f = pq.ParquetFile(path)
    cfg = json.loads((f.schema_arrow.metadata or {}).get(b"kvpress_run_config", b"{}"))
    df = f.read().to_pandas()
    arm = cfg.get("press_name", "?")
    df["arm"] = arm
    # Order matters: perhead_bern_* must be tested BEFORE bernoulli_*/kvzip, or it falls through
    # to the baseline and is pooled with the arm it is supposed to be compared against.
    df["family"] = ("vkvwr" if arm.startswith("vkvwr") else
                    "perhead_bern" if arm.startswith("perhead_bern") else
                    "bernoulli" if arm.startswith("bernoulli") else "kvzip")
    df["seed"] = int(arm.rsplit("_s", 1)[1]) if "_s" in arm else 0
    df["ratio"] = float(cfg.get("compression_ratio", np.nan))
    df["ratio_name"] = df.ratio.map(RATIO_NAME).fillna(df.ratio.astype(str))
    df["task"] = cfg.get("data_dir")
    df["model"] = str(cfg.get("model", "")).split("/")[-1]
    df["dataset"] = cfg.get("dataset")
    df["gpu"] = cfg.get("gpu_name")
    df["group"] = path.parent.parent.name
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """The error columns. B_sparse == 0 (a head that kept nothing) is a relative error of 1."""
    lo = df.log_b_sparse.astype("float64")
    hi = df.log_b_dense.astype("float64")
    r = np.exp(lo - hi)
    r = np.where(np.isfinite(r), r, 0.0)          # log_b_sparse == -inf -> the estimate is zero
    df["den_signed"] = r - 1.0                     # signed relative error of the denominator
    df["den_err"] = np.abs(df.den_signed)
    df["kept_frac"] = df.n_kept_ctx / df.ctx_len
    df["deep"] = df.layer >= 13                    # the layer-13 boundary found in E5a and E1
    return df


def q(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(s):
        return "-"
    return f"{s.median():.4f} [{s.quantile(.25):.4f}, {s.quantile(.75):.4f}]"


def paired_by_context(df, metric, a, b):
    """Mean difference (a - b) over contexts, with a context-clustered SE and a 9-df paired t.

    Pairing is exact: same model, task, ratio, context, question, decode step, layer and head, with
    identical scores and budget, so only the selector differs.
    """
    keys = ["model", "task", "ratio", "context_idx", "question_idx", "decode_step", "layer", "query_head"]
    A = df[df.family == a].groupby(keys, as_index=False)[metric].mean()
    B = df[df.family == b].groupby(keys, as_index=False)[metric].mean()
    m = A.merge(B, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        return None
    m["d"] = m[f"{metric}_a"] - m[f"{metric}_b"]
    per_ctx = m.groupby(["model", "task", "context_idx"])["d"].mean()
    n = len(per_ctx)
    if n < 2:
        return dict(n=n, mean=float(per_ctx.mean()), se=float("nan"), t=float("nan"), rows=len(m))
    mean, se = float(per_ctx.mean()), float(per_ctx.std(ddof=1) / np.sqrt(n))
    return dict(n=n, mean=mean, se=se, t=mean / se if se else float("nan"), rows=len(m),
                win_frac=float((m.d < 0).mean()))


def holm(pvals):
    order = np.argsort(pvals)
    out = np.empty(len(pvals))
    for rank, i in enumerate(order):
        out[i] = min(1.0, pvals[i] * (len(pvals) - rank))
    return np.maximum.accumulate(out[order])[np.argsort(order)]


def main(groups):
    files = []
    for g in groups:
        files += sorted((RESULTS / g).rglob("head_stats.parquet"))
    assert files, f"no parquet under {[str(RESULTS / g) for g in groups]}"
    df = derive(pd.concat([load_run(p) for p in files], ignore_index=True))
    df.to_parquet(OUT / "all_rows.parquet", index=False)

    L = []
    w = L.append
    w("# Experiment 5 — upstream error of KV cache compression\n")
    w(f"Generated from {len(files)} runs, {len(df):,} (head, query) rows.\n")
    w(f"Models: {sorted(df.model.unique())}. Datasets: {sorted(df.dataset.unique())}. "
      f"GPUs: {sorted(set(df.gpu.dropna()))}.\n")

    w("\n## Realised memory (the x axis of every comparison)\n")
    w("| model | dataset | ratio | arm | kept fraction |")
    w("|---|---|---|---|---|")
    for k, g in df.groupby(["model", "dataset", "ratio_name", "family"], sort=False):
        w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {g.kept_frac.mean():.4f} |")

    for metric, label in (("den_err", "denominator |B_hat-B|/B"),
                          ("attn_err", "attention ||A/B_hat - A/B||/||A/B||"),
                          ("num_err", "numerator ||A_hat-A||/||A||"),
                          ("num_err_abar", "numerator vs Abar")):
        w(f"\n## {label}: median [p25, p75] over heads\n")
        w("| model | dataset | task | ratio | arm | all layers | layers 0-12 | layers 13+ |")
        w("|---|---|---|---|---|---|---|---|")
        for k, g in df.groupby(["model", "dataset", "task", "ratio_name", "family"], sort=False):
            w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {q(g[metric])} | "
              f"{q(g[~g.deep][metric])} | {q(g[g.deep][metric])} |")

    w("\n## Signed denominator error (is the estimator biased?)\n")
    w("Top-k drops mass and never reweights, so it must be negative. Bernoulli and VKvWR are "
      "unbiased by construction, so they should straddle zero.\n")
    w("| model | dataset | ratio | arm | mean B_hat/B | median signed err |")
    w("|---|---|---|---|---|---|")
    for k, g in df.groupby(["model", "dataset", "ratio_name", "family"], sort=False):
        w(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {float((g.den_signed + 1).mean()):.4f} | "
          f"{g.den_signed.median():+.4f} |")

    w("\n## Error versus REALISED memory (the comparison that is not confounded)\n")
    w("kvzip, bernoulli and perhead_bern realise their target exactly, so their nominal ratios are "
      "matched memory. VKvWR does not: its ratio only sizes topk_frac and the pilot, while the CLT "
      "top-up sets the rest. Read the arms off this table at equal kept fraction.\n")
    w("| model | dataset | task | arm | nominal | realised kept | denominator err | attention err |")
    w("|---|---|---|---|---|---|---|---|")
    pareto = (df.groupby(["model", "dataset", "task", "family", "ratio"], sort=False)
                .agg(kept=("kept_frac", "mean"), den=("den_err", "median"),
                     attn=("attn_err", "median")).reset_index()
                .sort_values(["model", "dataset", "task", "kept"]))
    for _, r in pareto.iterrows():
        w(f"| {r.model} | {r.dataset} | {r.task} | {r['family']} | {RATIO_NAME.get(r.ratio, r.ratio)} | "
          f"{r.kept:.4f} | {r.den:.4f} | {r.attn:.4f} |")
    pareto.to_csv(OUT / "pareto.csv", index=False)

    w("\n## C7: is the win the estimator or the allocation?\n")
    w("perhead_bern is the shipped Bernoulli estimator with a per-head budget instead of one global "
      "c. Against bernoulli it isolates the ALLOCATION; against vkvwr at matched realised memory it "
      "isolates the ESTIMATOR.\n")
    w("| model | dataset | ratio | comparison | mean diff | SE | t | contexts |")
    w("|---|---|---|---|---|---|---|---|")
    for rn in sorted(df.ratio_name.unique(), key=lambda s: df[df.ratio_name == s].ratio.iloc[0]):
        sub = df[df.ratio_name == rn]
        for a, b, lab in (("perhead_bern", "bernoulli", "allocation (perhead - global)"),
                          ("perhead_bern", "vkvwr", "estimator (perhead - vkvwr)")):
            r = paired_by_context(sub, "attn_err", a, b)
            if r is None or not np.isfinite(r.get("t", np.nan)):
                continue
            for k, g in sub.groupby(["model", "dataset"], sort=False):
                pass
            w(f"| {sub.model.iloc[0]} | {sub.dataset.iloc[0]} | {rn} | {lab} | {r['mean']:+.4f} | "
              f"{r['se']:.4f} | {r['t']:+.2f} | {r['n']} |")

    w("\n## C1 / C2: paired against top-k, clustered at the context\n")
    w("Negative means the sampled arm has LOWER error, i.e. it wins.\n")
    rows, pv = [], []
    from scipy import stats
    for metric in ("den_err", "attn_err", "num_err"):
        for fam in ("bernoulli", "perhead_bern", "vkvwr"):
            for rn in sorted(df.ratio_name.unique(), key=lambda s: df[df.ratio_name == s].ratio.iloc[0]):
                sub = df[df.ratio_name == rn]
                r = paired_by_context(sub, metric, fam, "kvzip")
                if r is None or not np.isfinite(r.get("t", np.nan)):
                    continue
                p = float(2 * stats.t.sf(abs(r["t"]), r["n"] - 1))
                rows.append(dict(metric=metric, arm=fam, ratio=rn, **r, p=p))
                pv.append(p)
    if rows:
        adj = holm(np.array(pv))
        w("| metric | arm | ratio | mean diff vs top-k | SE | t | p | Holm | contexts | head-rows | win rate |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r, a in zip(rows, adj):
            w(f"| {r['metric']} | {r['arm']} | {r['ratio']} | {r['mean']:+.4f} | {r['se']:.4f} | "
              f"{r['t']:+.2f} | {r['p']:.4g} | {a:.4g} | {r['n']} | {r['rows']:,} | "
              f"{r.get('win_frac', float('nan')):.2%} |")
        pd.DataFrame(rows).assign(holm=adj).to_csv(OUT / "cells.csv", index=False)

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT / 'REPORT.md'} and {OUT / 'cells.csv'}")


if __name__ == "__main__":
    gs = sys.argv[1:] or [p.name for p in sorted(RESULTS.iterdir()) if p.is_dir()]
    main(gs)
