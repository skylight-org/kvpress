"""Validate a run's head_stats.parquet against the invariants that must hold by construction.

    python check_run.py results/smoke/*/

These are the C0 checks. They are run against a finished run rather than as a pre-flight, because
every quantity they need is in the run's own output.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OK, BAD = "PASS", "FAIL"


def check(path: Path) -> bool:
    f = pq.ParquetFile(path)
    meta = f.schema_arrow.metadata or {}
    cfg = json.loads(meta.get(b"kvpress_run_config", b"{}"))
    df = f.read().to_pandas()
    arm, ratio = cfg.get("press_name", "?"), cfg.get("compression_ratio", float("nan"))
    print(f"\n=== {path.parent.name}")
    print(f"    arm={arm} ratio={ratio} model={cfg.get('model')} gpu={cfg.get('gpu_name')}")
    print(f"    rows={len(df)} layers={df.layer.nunique()} q_heads={df.query_head.nunique()} "
          f"contexts={df.context_idx.nunique()} questions={df.question_idx.nunique()} "
          f"steps={sorted(df.decode_step.unique())}")

    ok = True
    ratio_hat = np.exp(df.log_b_sparse.astype("float64") - df.log_b_dense.astype("float64"))
    kept = (df.n_kept_ctx / df.ctx_len).mean()
    print(f"    kept fraction {kept:.4f} (target {1 - ratio:.4f})")

    # T1: the estimator's direction. top-k drops mass and never reweights, so it must not exceed
    # the true denominator. A breach is a bug, not a result. Every reweighted arm ("bern" covers
    # bernoulli_* and perhead_bern_*) is unbiased instead, so it must straddle 1.
    reweighted = any(t in arm for t in ("bern", "vkvwr"))
    if not reweighted:
        n_over = int((df.log_b_sparse > df.log_b_dense + 1e-5).sum())
        print(f"    T1 top-k never over-estimates: {n_over} violations -> {OK if n_over == 0 else BAD}")
        ok &= n_over == 0
    else:
        m = float(ratio_hat[np.isfinite(ratio_hat)].mean())
        print(f"    T1 mean B_hat/B = {m:.4f} (unbiased estimator, want ~1)")

    # T2: nothing degenerate that would silently poison the aggregates
    n_nan = int(df[["log_b_dense", "attn_err", "num_err"]].isna().sum().sum())
    n_empty = int((~np.isfinite(df.log_b_sparse)).sum())
    print(f"    T2 NaNs {n_nan}, heads keeping nothing {n_empty} ({n_empty / len(df):.2%}) -> "
          f"{OK if n_nan == 0 else BAD}")
    ok &= n_nan == 0

    # T3: the compression is actually happening (the E2 trap: nine runs silently equalled no_press)
    moved = float((ratio_hat.sub(1).abs() > 1e-6).mean())
    print(f"    T3 rows where the estimate differs from the truth: {moved:.2%} -> "
          f"{OK if moved > 0.5 else BAD}")
    ok &= moved > 0.5

    for col in ("log_b_sparse", "log_b_dense", "attn_err", "num_err", "num_err_abar"):
        v = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"    {col:14s} median {v.median():+.4f}  p25 {v.quantile(.25):+.4f}  p75 {v.quantile(.75):+.4f}")
    return ok


if __name__ == "__main__":
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        paths += [p] if p.is_file() else sorted(p.rglob("head_stats.parquet"))
    assert paths, "no head_stats.parquet found"
    # not all(check(p) ...): `all` short-circuits and would skip every run after the first failure
    good = all([check(p) for p in paths])
    print(f"\n{'ALL CHECKS PASS' if good else 'CHECKS FAILED'} ({len(paths)} runs)")
    sys.exit(0 if good else 1)
