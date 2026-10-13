"""Phase 2, Experiment 3: figures -> out/figures/.

    python plots.py [--group S1n10]

Eight figures, in the order a reader should meet them:

  1_scores            raw score vs compression, per task, every arm
  2_degradation       paired drop from the uncompressed model  (the headline)
  3_paired_diff       sampling - top-k with context-clustered SE
  4_families          retrieval vs aggregation, E3 against E2's 4k values
  5_context_scatter   per-context sampling vs top-k, showing where the mean comes from
  6_tau_distribution  per-head T/n at 110k from the score caches, against E1 at 7k
  7_tau_by_layer      per-head T/n by depth -- where the incompressible heads live
  8_memory_honesty    realised vs target kept fraction, and the §11 high-probability bound

Statistics match summarize.py: paired on (context_id, turn), SEs clustered at the CONTEXT level
because a context's turns share one prefill and one compression draw.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import torch                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scbench_metrics import SCORERS, score_one     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--group", default="S1n10")
ap.add_argument("--skip-tau", action="store_true", help="skip figs 6-7 (they read 2.1 GB of score caches)")
args = ap.parse_args()
FIG = os.path.join(HERE, "out", "figures")
os.makedirs(FIG, exist_ok=True)

COMP = {0.0: "full", 0.5: 2, 0.75: 4, 0.88: 8, 0.94: 16}
RATIOS = [2, 4, 8, 16]
TASK_ORDER = ("prefix_suffix", "vt", "qa_eng", "summary")
FAMILY = {"prefix_suffix": "retrieval", "vt": "retrieval", "qa_eng": "aggregation", "summary": "aggregation"}
METRIC = {t: next(iter(SCORERS[t])) for t in SCORERS}
STYLE = {                       # colour, marker, label
    "no_press":           ("#333333", "", "no press (full cache)"),
    "kvzip":              ("#d1495b", "s", "KVzip (top-k)"),
    "kvzip_bern":         ("#2a9d8f", "o", "Bernoulli sampling (ours)"),
    "expected_attention": ("#e9c46a", "^", "ExpectedAttention"),
    "random":             ("#9aa0a6", "x", "random"),
}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote out/figures/{name}.png")


# ------------------------------------------------------------------ load runs
runs = []
for mj in sorted(glob.glob(os.path.join(HERE, "results", args.group, "**", "metrics.json"), recursive=True)):
    d = os.path.dirname(mj)
    name = os.path.basename(d)
    if re.fullmatch(r"[0-9]+", name):
        name = os.path.basename(os.path.dirname(d))
    p = name.split("__")
    task, press, ratio = p[1], p[3], round(float(p[4]), 2)
    pr = pd.read_csv(os.path.join(d, "predictions.csv"))
    pr["predicted_answer"] = pr["predicted_answer"].fillna("")
    pr["s"] = [score_one(task, a, g) for a, g in zip(pr.predicted_answer, pr.answer)]
    memf = os.path.join(d, "memory.json")
    runs.append({"task": task, "arm": re.sub(r"_s[123]$", "", press),
                 "seed": int(press[-1]) if re.search(r"_s[123]$", press) else None,
                 "comp": COMP.get(ratio, ratio), "score": float(pr.s.mean()), "pred": pr,
                 "mem": json.load(open(memf)) if os.path.exists(memf) else None})
TASKS = [t for t in TASK_ORDER if any(r["task"] == t for r in runs)]
print(f"loaded {len(runs)} runs across {TASKS}")


def per_query(task, arm, comp, seed=None):
    rs = [r for r in runs if r["task"] == task and r["arm"] == arm and r["comp"] == comp
          and (seed is None or r["seed"] == seed)]
    if not rs:
        return None
    return pd.concat([r["pred"].set_index(["context_id", "turn"]).s for r in rs], axis=1).mean(axis=1).rename("s")


def paired(a, b):
    """-> (mean diff, context-clustered SE, per-context diffs)"""
    if a is None or b is None:
        return None
    idx = a.index.intersection(b.index)
    d = (a[idx] - b[idx]).reset_index().groupby("context_id").s.mean()
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), d


def score(task, arm, comp):
    s = [r["score"] for r in runs if r["task"] == task and r["arm"] == arm and r["comp"] == comp]
    return (float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else 0.0) if s else None


def panels(n=4, w=11.5, h=6.4):
    fig, ax = plt.subplots(2, 2, figsize=(w, h))
    return fig, ax.ravel()[:n]


# ================================================================== 1: scores
fig, axes = panels()
for ax, t in zip(axes, TASKS):
    full = score(t, "no_press", "full")
    if full:
        ax.axhline(full[0], color=STYLE["no_press"][0], ls="--", lw=1.2, zorder=1)
        ax.text(16.6, full[0], f" full cache\n {full[0]:.1f}", va="center", fontsize=7.5, color="#333")
    for arm in ("kvzip", "kvzip_bern", "expected_attention", "random"):
        col, mk, lab = STYLE[arm]
        xs = [c for c in RATIOS if score(t, arm, c)]
        if not xs:
            continue
        ys = [score(t, arm, c)[0] for c in xs]
        es = [score(t, arm, c)[1] for c in xs]
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, ms=5, lw=1.8, capsize=3,
                    label=lab, zorder=3 if arm == "kvzip_bern" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATIOS)
    ax.set_xticklabels([f"{r}x" for r in RATIOS])
    ax.set_xlim(1.7, 19)
    ax.set_ylim(bottom=-1.5)
    ax.set_title(f"{t}  ({METRIC[t]}, {FAMILY[t]})", fontsize=10)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel(METRIC[t])
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("SCBench at ~110k tokens: task score vs compression ratio\n"
             "error bars = spread across 3 Bernoulli seeds", fontsize=11)
save(fig, "1_scores")

# ================================================================== 2: degradation
fig, axes = panels()
for ax, t in zip(axes, TASKS):
    full = per_query(t, "no_press", "full")
    width = 0.36
    for k, arm in enumerate(("kvzip", "kvzip_bern")):
        col, _, lab = STYLE[arm]
        xs, ys, es = [], [], []
        for i, c in enumerate(RATIOS):
            p = paired(per_query(t, arm, c), full)
            if p:
                xs.append(i + (k - 0.5) * width)
                ys.append(p[0])
                es.append(p[1])
        ax.bar(xs, ys, width, yerr=es, color=col, capsize=3, label=lab,
               error_kw={"lw": 1, "ecolor": "#444"})
    ax.axhline(0, color="#333", lw=1.2, ls="--")
    ax.set_xticks(range(len(RATIOS)))
    ax.set_xticklabels([f"{r}x" for r in RATIOS])
    ax.set_title(f"{t}  ({FAMILY[t]})", fontsize=10)
    ax.set_ylabel("points vs full cache")
    ax.set_xlabel("compression ratio")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Loss from compression: each arm paired against the uncompressed model on identical queries\n"
             "0 = lossless.  error bars = context-clustered SE (n = 10 contexts)", fontsize=11)
save(fig, "2_degradation")

# ================================================================== 3: paired difference
fig, axes = panels()
for ax, t in zip(axes, TASKS):
    xs, ys, es = [], [], []
    for i, c in enumerate(RATIOS):
        p = paired(per_query(t, "kvzip_bern", c), per_query(t, "kvzip", c))
        if p:
            xs.append(i)
            ys.append(p[0])
            es.append(p[1])
    cols = ["#2a9d8f" if y >= 0 else "#d1495b" for y in ys]
    ax.bar(xs, ys, 0.55, yerr=es, color=cols, capsize=3, error_kw={"lw": 1, "ecolor": "#444"})
    ax.margins(y=0.22)
    for x, y, e in zip(xs, ys, es):
        sig = abs(y) > 2 * e if e > 0 else False
        ax.annotate(f"{y:+.1f}" + ("*" if sig else ""), (x, y + (e if y >= 0 else -e)),
                    textcoords="offset points", xytext=(0, 5 if y >= 0 else -12),
                    ha="center", fontsize=8, fontweight="bold" if sig else "normal")
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(range(len(RATIOS)))
    ax.set_xticklabels([f"{r}x" for r in RATIOS])
    ax.set_title(f"{t}  ({FAMILY[t]})", fontsize=10)
    ax.set_ylabel("sampling - top-k (points)")
    ax.set_xlabel("compression ratio")
fig.suptitle("The controlled comparison: identical scores, identical budget, only the selection rule differs\n"
             "green = sampling wins.  * = more than 2 context-clustered SE", fontsize=11)
save(fig, "3_paired_diff")

# ================================================================== 4: families
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for fam, col in (("retrieval", "#264653"), ("aggregation", "#e76f51")):
    xs, ys, es = [], [], []
    for i, c in enumerate(RATIOS):
        ps = [paired(per_query(t, "kvzip_bern", c), per_query(t, "kvzip", c))
              for t in TASKS if FAMILY[t] == fam]
        ps = [p for p in ps if p]
        if ps:
            xs.append(i)
            ys.append(np.mean([p[0] for p in ps]))
            es.append(math.sqrt(sum(p[1] ** 2 for p in ps)) / len(ps))
    axL.errorbar(xs, ys, yerr=es, color=col, marker="o", ms=6, lw=2, capsize=4, label=fam)
axL.axhline(0, color="#333", lw=1.2, ls="--")
axL.set_xticks(range(len(RATIOS)))
axL.set_xticklabels([f"{r}x" for r in RATIOS])
axL.set_ylabel("sampling - top-k (points)")
axL.set_xlabel("compression ratio")
axL.set_title("E3: SCBench at ~110k", fontsize=10)
axL.legend(fontsize=9)
axL.text(0.02, 0.03, "retrieval at 2x / 4x / 16x is prefix_suffix alone;\nvt has only 8x so far",
         transform=axL.transAxes, fontsize=7.5, color="#666", va="bottom")
# E2's 4k numbers, from ../experiment2/CONCLUSION.md
e2 = {"aggregation (cwe)": {8: 13.0, 16: 72.0}, "retrieval (needles)": {8: 0.5, 16: 0.0}}
for (lab, d), col in zip(e2.items(), ("#e76f51", "#264653")):
    xs = [RATIOS.index(k) for k in d]
    axR.plot(xs, list(d.values()), color=col, marker="s", ms=6, lw=2, label=lab)
axR.axhline(0, color="#333", lw=1.2, ls="--")
axR.set_xticks(range(len(RATIOS)))
axR.set_xticklabels([f"{r}x" for r in RATIOS])
axR.set_xlabel("compression ratio")
axR.set_ylabel("sampling - top-k (points)")
axR.set_title("E2: RULER at 4k (for contrast)", fontsize=10)
axR.legend(fontsize=9)
fig.suptitle("E2 found the advantage only on aggregation. At 110k it appears on both families -- "
             "and retrieval is now the larger one.", fontsize=11)
save(fig, "4_families")

# ================================================================== 5: per-context scatter
fig, axes = panels()
for ax, t in zip(axes, TASKS):
    for c, mk, al in ((4, "^", 0.85), (8, "o", 0.85), (16, "s", 0.5)):
        a, b = per_query(t, "kvzip_bern", c), per_query(t, "kvzip", c)
        if a is None or b is None:
            continue
        idx = a.index.intersection(b.index)
        ca = a[idx].reset_index().groupby("context_id").s.mean()
        cb = b[idx].reset_index().groupby("context_id").s.mean()
        ax.scatter(cb, ca, marker=mk, s=34, alpha=al, label=f"{c}x",
                   color={4: "#8ab17d", 8: "#2a9d8f", 16: "#1d3557"}[c], edgecolor="white", lw=0.5)
    lim = 105
    ax.plot([0, lim], [0, lim], color="#999", lw=1, ls="--")
    ax.fill_between([0, lim], [0, lim], lim, color="#2a9d8f", alpha=0.05)
    ax.set_xlim(-4, lim)
    ax.set_ylim(-4, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("top-k score (per context)")
    ax.set_ylabel("sampling score (per context)")
    ax.set_title(f"{t}", fontsize=10)
axes[0].legend(fontsize=8, loc="lower right", title="ratio", title_fontsize=8)
fig.suptitle("Where the averages come from: one point per context (mean over its 5 turns)\n"
             "shaded half = sampling better", fontsize=11)
save(fig, "5_context_scatter")

# ================================================================== 6/7: tau statistics
if not args.skip_tau:
    task_n = {}
    for t in TASKS:
        r = [x for x in runs if x["task"] == t and x["arm"] == "kvzip_bern" and x["mem"]]
        if r:
            task_n[t] = [m["n"] for m in r[0]["mem"]]
    cache = {}
    for p in glob.glob(os.path.join(HERE, "out", "score_cache", "*.pt")):
        cache.setdefault(int(re.search(r"_n(\d+)\.pt", p).group(1)), []).append(p)

    tau = {}                                  # task -> (T/n [ctx, layer, kv], n)
    for t, ns in task_n.items():
        mats, used = [], set()
        for n in ns:
            for p in cache.get(n, []):
                if p in used:
                    continue
                used.add(p)
                s = torch.load(p, map_location="cpu").double()      # [NL, 1, NKV, n]
                mats.append((s.sum(-1)[:, 0, :] / s.shape[-1]).numpy())
                break
        if mats:
            tau[t] = np.stack(mats)
            print(f"  tau {t}: {tau[t].shape} from {len(mats)} contexts")

    e1 = pd.read_csv(os.path.join(HERE, "..", "experiment1", "out", "head_cert.csv"))

    fig, axes = panels(h=6.6)
    for ax, t in zip(axes, TASKS):
        v = tau.get(t)
        if v is None:
            continue
        x = np.sort(v.ravel())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color="#2a9d8f", lw=2,
                label=f"E3 {t} (~{int(np.mean(task_n[t])/1000)}k tokens)")
        e = np.sort(e1.T_over_n.to_numpy())
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color="#d1495b", lw=1.6, ls="--",
                label="E1 LongBench (~7k tokens)")
        ax.axvline(np.median(v), color="#2a9d8f", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1.2)
        ax.set_xlabel("T/n per KV head  (1 = incompressible, flat score)")
        ax.set_ylabel("fraction of heads")
        ax.set_title(f"{t}:  median {np.median(v):.5f},  p90 {np.quantile(v, 0.9):.3f},  "
                     f"max {v.max():.3f}", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Compressibility per head at 110k against E1 at 7k.  The typical head gets much more\n"
                 "compressible; a tail approaches T/n = 1, where no budget compresses without loss.",
                 fontsize=11)
    save(fig, "6_tau_distribution")

    fig, axes = panels(h=6.4)
    for ax, t in zip(axes, TASKS):
        v = tau.get(t)
        if v is None:
            continue
        per_layer = v.reshape(v.shape[0], v.shape[1], -1)          # [ctx, layer, kv]
        L = per_layer.shape[1]
        med = np.median(per_layer, axis=(0, 2))
        p90 = np.quantile(per_layer, 0.9, axis=(0, 2))
        mx = per_layer.max(axis=(0, 2))
        ax.fill_between(range(L), med, p90, color="#2a9d8f", alpha=0.25, label="median - p90")
        ax.plot(range(L), med, color="#2a9d8f", lw=2, label="median")
        ax.plot(range(L), mx, color="#d1495b", lw=1.4, ls="--", label="worst head")
        ax.axhline(1.0, color="#333", lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 2)
        ax.set_xlabel("layer")
        ax.set_ylabel("T/n")
        ax.set_title(f"{t}", fontsize=10)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Where the incompressible heads live.  Under a single global budget the dashed line -- "
                 "the worst head --\nsets certified memory for every head in its layer.", fontsize=11)
    save(fig, "7_tau_by_layer")

# ================================================================== 8: memory honesty
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2))
rows = []
for t in TASKS:
    for c in RATIOS:
        ms = [m for r in runs if r["task"] == t and r["arm"] == "kvzip_bern" and r["comp"] == c
              and r["mem"] for m in r["mem"]]
        for m in ms:
            rows.append({"task": t, "comp": c, "kept": m["kept_frac"], "target": m["expected_frac"],
                         "hp": m["M_hp_frac"]})
M = pd.DataFrame(rows)
for t, col in zip(TASKS, ("#264653", "#2a9d8f", "#e9c46a", "#e76f51")):
    g = M[M.task == t]
    axL.scatter(g.target, 100 * (g.kept - g.target) / g.target, s=22, alpha=0.7, color=col, label=t)
axL.axhline(0, color="#333", lw=1.2, ls="--")
axL.axhspan(-1, 1, color="#2a9d8f", alpha=0.08)
axL.set_xscale("log", base=2)
axL.set_xticks([0.0625, 0.125, 0.25, 0.5])
axL.set_xticklabels(["16x", "8x", "4x", "2x"])
axL.set_xlabel("target kept fraction")
axL.set_ylabel("realised - target  (% of target)")
axL.set_title("C4: the budget is met exactly\nshaded band = the +/-1% criterion", fontsize=10)
axL.legend(fontsize=8)
slack = 100 * (M.hp - M.kept) / M.target
axR.hist(slack, bins=40, color="#2a9d8f", alpha=0.85)
axR.axvline(0, color="#d1495b", lw=1.5, ls="--")
axR.set_xlabel("headroom under the §11 high-probability bound  (% of target)")
axR.set_ylabel("contexts")
axR.set_title(f"C4: every draw stays inside the bound\n{int((slack >= 0).sum())}/{len(slack)} inside, "
              f"median headroom {slack.median():.2f}%", fontsize=10)
fig.suptitle("Memory accounting: the sampled cache is the size the derivation says it is", fontsize=11)
save(fig, "8_memory_honesty")

print(f"\n[{len(glob.glob(os.path.join(FIG, '*.png')))} figures in out/figures/]")
