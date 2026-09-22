"""Phase 2, Experiment 1 (CPU): tables + figures from head_cert.csv and mc.csv -> out/REPORT.md.

    python report.py [--smoke]
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "smoke" if args.smoke else "")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

DELTA, EPS_TARGET = 0.05, 0.1
cert = pd.read_csv(os.path.join(OUT, "head_cert.csv"))
mc = pd.read_csv(os.path.join(OUT, "mc.csv"))
meta = json.load(open(os.path.join(OUT, "meta.json")))
TRIALS = 300
SE = math.sqrt(DELTA * (1 - DELTA) / TRIALS)
THR = DELTA + 3 * SE
L = []


def w(s=""):
    L.append(s)


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


w("# Phase 2, Experiment 1 — How vacuous is the denominator-only Bernoulli bound?")
w()
w(f"Bound: `|B̂ − B|/B ≤ u + √(u² + γu)` (derivation §10), δ = {DELTA}, {TRIALS} Bernoulli draws per "
  f"(head, budget). SE of a failure rate at δ is {SE:.3f}, so the pass threshold is δ + 3SE = **{THR:.3f}**.")
w()
w(f"Contexts: **{len(meta)}** (" + ", ".join(f"{m['task']} n={m['n']}" for m in meta) + ")."
  f" Certificates on **{len(cert)}** KV heads; Monte Carlo on **{mc.groupby(['ctx','layer','kv']).ngroups}** KV heads.")
w()

# ---------------------------------------------------------------- the score
w("## 0. What calibration produced")
w()
w("`T = Σᵢ τᵢ` is the derivation's total score: T ≈ 1 means the head is compressible, T ≈ n means it is not.")
w()
w("| | median | p10 | p90 | max |")
w("|---|---|---|---|---|")
for col, nm in (("T", "T"), ("T_over_n", "T/n")):
    q = cert[col].quantile([0.5, 0.1, 0.9, 1.0])
    w(f"| {nm} | {q[0.5]:.3g} | {q[0.1]:.3g} | {q[0.9]:.3g} | {q[1.0]:.3g} |")
w()
lo, hi = cert[cert.layer <= 12], cert[cert.layer >= 13]
w(f"By depth (median T/n): layers 0–12 **{med(lo.T_over_n):.3f}**, layers 13–31 **{med(hi.T_over_n):.3f}**.")
w()

# ---------------------------------------------------------------- certified memory
w("## 1. Certified memory (P4)")
w()
w("Fraction of the head's cache that must be kept to certify ε, before any question is seen. "
  "1.00 = no compression certifiable. `M` is expected memory, `M⁺` the high-probability bound (§11).")
w()
w("| ε | M median | M p10 | M p90 | M⁺ median | heads < 0.5n | heads < 0.25n |")
w("|---|---|---|---|---|---|---|")
for e in (0.1, 0.2, 0.3):
    c = cert[f"M_eps{e}"]
    w(f"| {e} | {med(c):.3f} | {c.quantile(.1):.3f} | {c.quantile(.9):.3f} | "
      f"{med(cert[f'Mhp_eps{e}']):.3f} | {(c < 0.5).mean()*100:.1f}% | {(c < 0.25).mean()*100:.1f}% |")
w()
m01 = med(cert["M_eps0.1"])
w(f"**P4 ({'PASS' if m01 < 0.5 else 'FAIL'}):** median certified memory at ε = 0.1 is "
  f"**{m01:.3f}n** (passes if < 0.5n).")
w()
w("By depth, median certified memory at ε = 0.1: layers 0–12 "
  f"**{med(lo['M_eps0.1']):.3f}n**, layers 13–31 **{med(hi['M_eps0.1']):.3f}n**.")
w()

# ---------------------------------------------------------------- validity
def block(sub, thr, col):
    """Failure-rate summary over rows of one budget."""
    if not len(sub):
        return None
    bad = (sub[col] > thr).mean()
    return {"n": len(sub), "share_bad": bad, "max": sub[col].max(), "med": med(sub[col])}


w("## 2. Does the bound hold? (P1, P2)")
w()
w("`fail_own` = Pr over draws that the real error exceeds **that budget's own certificate**. "
  "On calibration queries this is a theorem (§10); on future queries it is the transfer assumption (2.1).")
w()
w("| budget | M/n | ε certified | calib: share over δ+3SE | calib: max | future: share over δ+3SE | future: max |")
w("|---|---|---|---|---|---|---|")
order = ["comp2x", "comp4x", "eps0.1_M", "eps0.1_M/2", "eps0.1_M/4"]
for b in order:
    s = mc[mc.budget == b]
    if not len(s):
        continue
    ca, re = block(s[s.qset == "cal"], THR, "fail_own"), block(s[s.qset == "real"], THR, "fail_own")
    w(f"| `{b}` | {med(s.M_over_n):.3f} | {med(s.eps_cert):.4g} | {ca['share_bad']*100:.2f}% | {ca['max']:.3f} "
      f"| {re['share_bad']*100:.2f}% | {re['max']:.3f} |")
w()
cal_bad = (mc[mc.qset == "cal"].fail_own > THR).mean()
real_ok = 1 - (mc[mc.qset == "real"].fail_own > THR).mean()
w(f"**P1 ({'PASS' if cal_bad <= 0.001 else 'FAIL'}):** on calibration queries {cal_bad*100:.2f}% of "
  f"(head, budget, query) exceed δ+3SE. This is a theorem — anything above ~0 is a code bug.")
w(f"**P2 ({'PASS' if real_ok >= 0.95 else 'FAIL'}):** on future queries **{real_ok*100:.2f}%** of pairs "
  f"hold (passes if ≥ 95%).")
w()
r = mc[mc.qset == "real"]
w("Future-query failures by answer step (share over δ+3SE): " +
  ", ".join(f"step {int(s)}: {(g.fail_own > THR).mean()*100:.1f}%" for s, g in r.groupby("step")) + ".")
w()
w("By task: " + ", ".join(f"{t}: {(g.fail_own > THR).mean()*100:.1f}%" for t, g in r.groupby("task")) + ".")
w()

# ---------------------------------------------------------------- looseness
w("## 3. How loose? (P3)")
w()
w("Looseness = certified ε ÷ the p95 of the error actually measured. 1.0 would be perfectly tight.")
w()
w("| budget | M/n | ε certified | p95 error (median) | looseness (median) | p90 |")
w("|---|---|---|---|---|---|")
for b in order:
    s = r[(r.budget == b) & np.isfinite(r.eps_cert) & (r.err_p95 > 0)]
    if not len(s):
        continue
    ratio = s.eps_cert / s.err_p95
    w(f"| `{b}` | {med(s.M_over_n):.3f} | {med(s.eps_cert):.4g} | {med(s.err_p95):.4g} | "
      f"**{med(ratio):.1f}×** | {ratio.quantile(.9):.1f}× |")
w()
s2 = r[(r.budget == "comp2x") & np.isfinite(r.eps_cert) & (r.err_p95 > 0)]
loose2 = med(s2.eps_cert / s2.err_p95)
w(f"**P3 ({'PASS' if loose2 <= 5 else 'FAIL'}):** median looseness at 2× compression is "
  f"**{loose2:.1f}×** (passes if ≤ 5×).")
w()

# ---------------------------------------------------------------- the ladder
w("## 4. The ε = 0.1 ladder — spending less than the certificate asks for")
w()
w("Two yardsticks at each budget, as pre-registered. **Own certificate**: does the bound stay "
  "internally honest when re-certified at that budget (it should, everywhere). **ε = 0.1**: does the "
  "original promise survive (it should hold at M\\*, and break below it).")
w()
w("| budget | M/n | ε re-certified | future fail vs own cert | future fail vs ε=0.1 | median error p95 |")
w("|---|---|---|---|---|---|")
for b in ["eps0.1_M", "eps0.1_M/2", "eps0.1_M/4"]:
    s = r[r.budget == b]
    if not len(s):
        continue
    w(f"| `{b}` | {med(s.M_over_n):.3f} | {med(s.eps_cert):.4g} | {(s.fail_own > THR).mean()*100:.2f}% "
      f"| {(s['fail_eps0.1'] > THR).mean()*100:.2f}% | {med(s.err_p95):.4g} |")
w()
w("The gap between the last two columns is the whole point: the bound stays true about *its own* "
  "claim at every budget, while the ε = 0.1 promise degrades as the budget is cut.")
w()

# ---------------------------------------------------------------- verdict
w("## 5. Verdict")
w()
w("| | criterion | verdict |")
w("|---|---|---|")
w(f"| P1 | sound on calibration queries (theorem) | {'PASS' if cal_bad <= 0.001 else 'FAIL'} |")
w(f"| P2 | transfers to future queries (≥95%) | {'PASS' if real_ok >= 0.95 else 'FAIL'} |")
w(f"| P3 | not vacuous (looseness ≤ 5× at 2×) | {'PASS' if loose2 <= 5 else 'FAIL'} |")
w(f"| P4 | certifies real compression (median < 0.5n at ε=0.1) | {'PASS' if m01 < 0.5 else 'FAIL'} |")
w()

# ---------------------------------------------------------------- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for b in order:
        s = r[r.budget == b]
        if len(s):
            ax[0].scatter(s.M_over_n, s.fail_own, s=4, alpha=.3, label=b)
    ax[0].axhline(THR, color="r", ls="--", label=f"δ+3SE={THR:.3f}")
    ax[0].set(xlabel="stored fraction M/n", ylabel="Pr[error > own certificate]",
              title="1. Validity on future queries", yscale="symlog", ylim=(-0.002, 1.2))
    ax[0].legend(fontsize=6)

    for i, e in enumerate((0.1, 0.2, 0.3)):
        v = np.sort(cert[f"M_eps{e}"].values)
        ax[1].plot(v, np.linspace(0, 1, len(v)), label=f"ε={e}", color=f"C{i}")
    ax[1].axvline(0.5, color="k", ls=":", lw=.8)
    ax[1].set(xlabel="certified memory / n", ylabel="fraction of KV heads",
              title="2. Certified memory (CDF)")
    ax[1].legend(fontsize=7)

    ss = r[np.isfinite(r.eps_cert) & (r.err_p95 > 0)]
    bs = [b for b in order if len(ss[ss.budget == b])]
    ax[2].boxplot([(ss[ss.budget == b].eps_cert / ss[ss.budget == b].err_p95).values for b in bs],
                  tick_labels=bs, showfliers=False)
    ax[2].axhline(1, color="g", ls=":", lw=.8)
    ax[2].set(ylabel="certified ε ÷ real p95 error", title="3. Looseness", yscale="log")
    ax[2].tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "summary.png"), dpi=130)
    w("Figures: `out/figures/summary.png` (validity, certified memory, looseness).")
except Exception as e:                                                   # noqa: BLE001
    w(f"(figures skipped: {e})")

open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
