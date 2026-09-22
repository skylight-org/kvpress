"""Phase 2, Experiment 3: the single headline figure -> out/figures/0_headline.png

One panel, averaged over the three SCBench tasks that finished all 25 runs
(prefix_suffix, qa_eng, summary). `vt` is excluded: it has only 8x so far, and mixing a
one-ratio task into the average would make the curve mean a different thing at each x.

The three tasks use different metrics (substring match, token F1, ROUGE-L), so this is a
macro-average over tasks -- each task contributes equally, as RULER's headline number does.

The shaded band is the **range across the 3 Bernoulli seeds**, and only the Bernoulli arm has one:
it is the only arm with run-to-run randomness. Every other arm here is deterministic given the
context, so a band on it would be inviting a reading that does not exist. The statistical
uncertainty of the comparison itself (context-clustered, paired) is in out/REPORT.md, not here.
"""

import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scbench_metrics import score_one    # noqa: E402

GROUP = "S1n10"
TASKS = ("prefix_suffix", "qa_eng", "summary")
RATIOS = [2, 4, 8, 16]
COMP = {0.0: "full", 0.5: 2, 0.75: 4, 0.88: 8, 0.94: 16}
ARMS = [                                   # arm, colour, marker, label, z
    ("kvzip_bern",         "#0f8b7e", "o", "Bernoulli sampling (ours)",     5),
    ("kvzip",              "#d1495b", "s", "KVzip (top-k, same scores)",    4),
    ("expected_attention", "#d9a441", "^", "ExpectedAttention",             3),
    ("random",             "#9aa0a6", "x", "random (floor)",                2),
]

# ---------------------------------------------------------------- load
runs = []
for mj in sorted(glob.glob(os.path.join(HERE, "results", GROUP, "**", "metrics.json"), recursive=True)):
    d = os.path.dirname(mj)
    name = os.path.basename(d)
    if re.fullmatch(r"[0-9]+", name):
        name = os.path.basename(os.path.dirname(d))
    p = name.split("__")
    task, press, ratio = p[1], p[3], round(float(p[4]), 2)
    if task not in TASKS:
        continue
    pr = pd.read_csv(os.path.join(d, "predictions.csv"))
    pr["predicted_answer"] = pr["predicted_answer"].fillna("")
    pr["s"] = [score_one(task, a, g) for a, g in zip(pr.predicted_answer, pr.answer)]
    runs.append({"task": task, "arm": re.sub(r"_s[123]$", "", press), "press": press,
                 "comp": COMP.get(ratio, ratio), "pred": pr})


def macro(arm, comp, press=None):
    """Task macro-average: each of the 3 tasks contributes equally.

    `press` pins one specific run (e.g. kvzip_bern_s2) instead of averaging the seeds."""
    means = []
    for t in TASKS:
        rs = [r for r in runs if r["task"] == t and r["comp"] == comp
              and (r["press"] == press if press else r["arm"] == arm)]
        if not rs:
            return None
        means.append(float(np.mean([r["pred"].s.mean() for r in rs])))
    return float(np.mean(means))


def seed_band(comp):
    """(min, max) of the macro score over the 3 Bernoulli seeds -- the only real run-to-run spread."""
    v = [macro(None, comp, press=f"kvzip_bern_s{s}") for s in (1, 2, 3)]
    v = [x for x in v if x is not None]
    return (min(v), max(v)) if v else None


# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans",
                     "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.7,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.linewidth": 0.9,
                     "xtick.major.size": 0, "ytick.major.size": 0})

fig, ax = plt.subplots(figsize=(9.2, 6.0))
fig.subplots_adjust(left=0.095, right=0.72, top=0.80, bottom=0.145)

full = macro("no_press", "full")
ax.axhline(full, color="#444444", ls=(0, (5, 3)), lw=1.4, zorder=1)
ax.text(17.6, full, "  full cache", va="center", ha="left", fontsize=10,
        color="#444444", fontweight="normal")
ax.text(17.6, full - 1.35, f"  {full:.1f}", va="center", ha="left", fontsize=9, color="#777")

ends = []
for arm, col, mk, lab, z in ARMS:
    pts = [(r, macro(arm, r)) for r in RATIOS]
    pts = [(r, m) for r, m in pts if m is not None]
    xs = [r for r, _ in pts]
    ys = np.array([m for _, m in pts])
    if arm == "kvzip_bern":                  # the only arm with run-to-run randomness
        bands = [seed_band(r) for r in xs]
        ax.fill_between(xs, [b[0] for b in bands], [b[1] for b in bands],
                        color=col, alpha=0.20, lw=0, zorder=z)
    ax.plot(xs, ys, color=col, marker=mk, ms=6.5, lw=2.4, zorder=z + 10,
            markeredgecolor="white", markeredgewidth=1.1)
    ends.append((ys[-1], col, lab))

# labels at the right-hand end of each line, nudged apart so they never collide
ends.sort(reverse=True)
prev = None
for y, col, lab in ends:
    y_lab = y if prev is None else min(y, prev - 2.6)
    prev = y_lab
    ax.annotate(lab, xy=(16, y), xytext=(17.6, y_lab), fontsize=10, color=col,
                fontweight="bold", va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.45,
                                shrinkA=2, shrinkB=2))

ax.annotate("lossless through 4x", xy=(4, macro("kvzip_bern", 4)), xytext=(4.15, 34.0),
            fontsize=9.5, color="#0f8b7e", ha="left",
            arrowprops=dict(arrowstyle="->", color="#0f8b7e", lw=1.1, alpha=0.8))
ax.annotate("top-k falls off a cliff\nbetween 2x and 4x", xy=(4.06, macro("kvzip", 4) + 0.5), xytext=(4.55, 21.0),
            fontsize=9.5, color="#d1495b", ha="left",
            arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1.1, alpha=0.8))

ax.set_xscale("log", base=2)
ax.set_xticks(RATIOS)
ax.set_xticklabels([f"{r}x" for r in RATIOS])
ax.set_xlim(1.82, 17.6)
ax.set_ylim(-1.5, 37)
ax.set_yticks(range(0, 35, 5))
ax.set_xlabel("KV cache compression ratio", labelpad=9, fontsize=11)
ax.set_ylabel("SCBench score  (mean of 3 tasks)", labelpad=9, fontsize=11)
ax.tick_params(axis="both", pad=6)

fig.text(0.095, 0.935, "Bernoulli KV sampling holds the full-cache score to 4x at ~110k context",
         fontsize=14, fontweight="bold", ha="left", va="top", color="#1a1a1a")
fig.text(0.095, 0.882,
         "Llama-3.1-8B-Instruct on SCBench, untruncated. Mean of prefix_suffix, qa_eng and summary\n"
         "(10 contexts x 5 turns each). Shading is the range over the 3 Bernoulli seeds \u2014 the only arm\n"
         "with run-to-run randomness. KVzip reads the identical scores; only the selection rule differs.",
         fontsize=9.5, ha="left", va="top", color="#5a5a5a", linespacing=1.5)

out = os.path.join(HERE, "out", "figures", "0_headline.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
for arm, _, _, lab, _ in ARMS:
    print(f"  {lab:32s}", " ".join(f"{r}x {macro(arm, r):5.1f}" for r in RATIOS))
print(f"  {'full cache':32s} {full:.1f}")
print("  seed range          ", " ".join(f"{r}x [{seed_band(r)[0]:.1f}, {seed_band(r)[1]:.1f}]"
                                         for r in RATIOS))
