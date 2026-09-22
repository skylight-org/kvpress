"""Experiment 4 figures -> out/figures/. Imports summarize.py and writes out/REPORT.md with it, so the
plots and the report are computed from the same loaded runs and the same statistics.

    python plots.py

0_headline.png   2 x 2: rows = benchmark (LongBench, SCBench), columns = scorer (KVzip, Compactor).
                 Each panel: Bernoulli sampling vs top-k on the same scores, benchmark mean over 4 tasks,
                 value printed at every point. The shaded band on the sampling line is mean ± 1 standard
                 deviation (n - 1) across its 3 seeds (top-k is deterministic, so it has none). Dotted line = full cache.
1_effects.png    forest plot: sampling minus top-k per task, 4x and 8x, 95% clustered CIs.
2_tasks.png      score vs compression for each of the 8 tasks, all four arms.
3_nobias.png     8x, seed 1: same draws with and without the log(1/r) correction.

Colours are from the Okabe-Ito colour-blind-safe palette (no red/green pairs).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import summarize as S  # noqa: E402

FIG = S.OUT / "figures"
FIG.mkdir(exist_ok=True)
X = {"2x": 2, "4x": 4, "8x": 8, "16x": 16}
XS = list(X.values())

# Okabe-Ito
BLUE, SKY, ORANGE, PURPLE, GREY = "#0072B2", "#56B4E9", "#E69F00", "#7B61A8", "#555555"
DARK_ORANGE = "#9A6A00"                              # orange text readable on white
SEL = {"bern": BLUE, "topk": ORANGE}                 # headline: colour = selection rule
SCORER = {"kvzip": BLUE, "compactor": ORANGE}        # per-task plot: colour = scorer, style = selection
NAME = {"kvzip": "KVzip", "compactor": "Compactor"}

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "savefig.bbox": "tight",
})


def seed_scores(task, arm, comp):
    return [r["score"] for r in S.get(task, arm, comp)]


def sd(v):
    """Sample standard deviation across seeds (n - 1); 0 for a single run."""
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def bench_curve(tasks, arm):
    """Benchmark mean per ratio, and mean -/+ 1 SD across seeds of the seed-level benchmark mean."""
    mean, lo, hi = [], [], []
    for c in X:
        per_task = [seed_scores(t, arm, c) for t in tasks]
        by_seed = [np.mean(v) for v in zip(*per_task)]      # seed i across tasks (runs load sorted by name)
        m = float(np.mean(by_seed))
        mean.append(m)
        lo.append(m - sd(by_seed))
        hi.append(m + sd(by_seed))
    return np.array(mean), np.array(lo), np.array(hi)


def signed(v):
    """+1.2 / -1.2, and 0.0 instead of -0.0."""
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}"


def style_x(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(XS, [f"{v}x" for v in XS])
    ax.minorticks_off()
    ax.margins(x=0.08)
    ax.grid(alpha=0.3, lw=0.8)


# ================================================================ 0 headline (2 x 2)
BENCHES = (("LongBench", S.LB_TASKS, "~5-15k-token contexts"), ("SCBench", S.SC_TASKS, "~110k-token contexts"))
fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey="row")
for i, (bench, tasks, blurb) in enumerate(BENCHES):
    full = np.mean([S.get(t, "no_press", "full")[0]["score"] for t in tasks])
    curves = {s: (bench_curve(tasks, f"{s}_bern"), bench_curve(tasks, s)[0]) for s in ("kvzip", "compactor")}
    lo_all = min(min(b[1].min(), t.min()) for b, t in curves.values())
    hi_all = max(max(b[2].max(), t.max()) for b, t in curves.values())
    hi_all = max(hi_all, full)
    span = hi_all - lo_all
    for j, scorer in enumerate(("kvzip", "compactor")):
        ax = axes[i, j]
        (mb, lob, hib), mt = curves[scorer]
        ax.axhline(full, color=GREY, lw=1.8, ls=":", zorder=1)
        ax.annotate(f"full cache {full:.1f}", (XS[-1], full), textcoords="offset points", xytext=(8, 6),
                    fontsize=11, color=GREY, ha="right")
        ax.fill_between(XS, lob, hib, color=SEL["bern"], alpha=0.18, lw=0, zorder=2)
        ax.plot(XS, mt, color=SEL["topk"], ls="--", marker="s", ms=9, lw=3, mfc="white", mew=2.5, zorder=3)
        ax.plot(XS, mb, color=SEL["bern"], ls="-", marker="o", ms=9, lw=3, zorder=4)
        for x, b, t in zip(XS, mb, mt):
            up = b >= t
            ax.annotate(f"{b:.1f}", (x, b), textcoords="offset points", xytext=(0, 12 if up else -20),
                        ha="center", fontsize=11.5, color=SEL["bern"], fontweight="bold")
            ax.annotate(f"{t:.1f}", (x, t), textcoords="offset points", xytext=(0, -20 if up else 12),
                        ha="center", fontsize=11.5, color=DARK_ORANGE, fontweight="bold")
        style_x(ax)
        ax.set_ylim(lo_all - 0.14 * span, hi_all + 0.14 * span)
        ax.set_title(f"{bench}  |  {NAME[scorer]} scores", loc="left")
        if j == 0:
            ax.set_ylabel(f"score, mean of 4 tasks\n({blurb})")
        if i == 1:
            ax.set_xlabel("compression  (2x = 50% of the context cache removed, 16x = 94%)")
handles = [Line2D([], [], color=SEL["bern"], lw=3, marker="o", ms=9, label="Bernoulli sampling + log(1/r) correction (ours)"),
           plt.Rectangle((0, 0), 1, 1, color=SEL["bern"], alpha=0.18, label="± 1 standard deviation across the 3 sampling seeds"),
           Line2D([], [], color=SEL["topk"], lw=3, ls="--", marker="s", ms=9, mfc="white", mew=2.5,
                  label="top-k on the same scores (baseline)"),
           Line2D([], [], color=GREY, lw=1.8, ls=":", label="full cache (no compression)")]
fig.suptitle("Does Bernoulli sampling beat top-k selection?\n"
             "Llama-3.1-8B-Instruct, 4 tasks x 50 questions per benchmark", fontsize=16, fontweight="bold")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(FIG / "0_headline.png", dpi=170)
plt.close(fig)

# ================================================================ 1 forest
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
labels = S.TASKS
RATIO_STYLE = {"4x": (SKY, "o", -0.17), "8x": (BLUE, "s", 0.17)}
for ax, (name, scorer) in zip(axes, (("D_KV", "kvzip"), ("D_CP", "compactor"))):
    ax.axhspan(3.5, len(labels) - 0.5, color="#EFEFEF", zorder=0)
    for comp, (col, mk, off) in RATIO_STYLE.items():
        for i, t in enumerate(labels):
            m = [c for c in S.cells if c["contrast"] == name and c["task"] == t and c["comp"] == comp]
            if not m:
                continue
            d, se = m[0]["diff"], m[0]["se"]
            ax.errorbar(d, i + off, xerr=1.96 * se, fmt=mk, color=col, ms=10, lw=2.5, capsize=5, capthick=2,
                        zorder=3, label=comp if i == 0 else None)
            ax.annotate(signed(d), (d + 1.96 * se, i + off), textcoords="offset points", xytext=(6, -5),
                        fontsize=11, color=col, fontweight="bold")
    ax.axvline(0, color=GREY, lw=1.5)
    ax.set_title(f"{NAME[scorer]} scores: sampling minus top-k", loc="left")
    ax.set_xlabel("paired score difference, points (95% CI)\n<-- top-k better      sampling better -->")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(title="compression", frameon=True, facecolor="white", edgecolor="#CCCCCC", loc="upper right")
    ax.text(0.01, 0.03, "shaded rows: SCBench (~110k tokens)", transform=ax.transAxes, fontsize=11, color=GREY)
axes[0].set_yticks(range(len(labels)), [f"{t}\n({S.BENCH[t]})" for t in labels])
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig(FIG / "1_effects.png", dpi=170)
plt.close(fig)

# ================================================================ 2 per task
fig, axes = plt.subplots(2, 4, figsize=(21, 10.5))
for ax, t in zip(axes.flat, S.TASKS):
    npv = S.get(t, "no_press", "full")
    if npv:
        ax.axhline(npv[0]["score"], color=GREY, lw=1.6, ls=":")
    for scorer in ("kvzip", "compactor"):
        col = SCORER[scorer]
        for arm, ls, mfc in ((scorer, "--", "white"), (f"{scorer}_bern", "-", col)):
            pts = [(X[c], np.mean(v), np.mean(v) - sd(v), np.mean(v) + sd(v))
                   for c in X for v in [seed_scores(t, arm, c)] if v]
            if not pts:
                continue
            xs, m, lo, hi = map(np.array, zip(*pts))
            ax.plot(xs, m, color=col, ls=ls, marker="o" if scorer == "kvzip" else "s", ms=8, lw=2.6, mfc=mfc, mew=2)
            if arm.endswith("_bern"):
                ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
    style_x(ax)
    ax.set_title(f"{t}\n", loc="left")
    ax.text(0.0, 1.02, f"{S.BENCH[t]}  |  {S.METRIC_NAME[t]}", transform=ax.transAxes, fontsize=11,
            color=GREY, va="bottom")
for ax in axes[:, 0]:
    ax.set_ylabel("score (0-100)")
for ax in axes[1]:
    ax.set_xlabel("compression")
handles = [Line2D([], [], color=BLUE, lw=2.6, marker="o", ms=8, label="KVzip scores + sampling"),
           Line2D([], [], color=BLUE, lw=2.6, ls="--", marker="o", ms=8, mfc="white", mew=2, label="KVzip top-k"),
           Line2D([], [], color=ORANGE, lw=2.6, marker="s", ms=8, label="Compactor scores + sampling"),
           Line2D([], [], color=ORANGE, lw=2.6, ls="--", marker="s", ms=8, mfc="white", mew=2, label="Compactor top-k"),
           Line2D([], [], color=GREY, lw=1.6, ls=":", label="full cache")]
fig.suptitle("Per task: score vs compression  (shaded = ± 1 SD across the 3 sampling seeds)",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(FIG / "2_tasks.png", dpi=170)
plt.close(fig)

# ================================================================ 3 nobias
rows = [c for c in S.cells if c["contrast"] == "D2_bias" and c["comp"] == "8x"]
if rows:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ys = np.arange(len(rows))
    cols = [SKY if S.BENCH[r["task"]] == "LongBench" else PURPLE for r in rows]
    ax.barh(ys, [r["diff"] for r in rows], xerr=[1.96 * r["se"] for r in rows], color=cols, height=0.6,
            error_kw={"lw": 2, "capsize": 5, "ecolor": GREY})
    for y, r in zip(ys, rows):
        edge = r["diff"] + (1 if r["diff"] >= 0 else -1) * 1.96 * r["se"]
        ax.annotate(signed(r["diff"]), (edge, y), textcoords="offset points",
                    xytext=(8 if r["diff"] >= 0 else -8, -5), ha="left" if r["diff"] >= 0 else "right",
                    fontsize=12, fontweight="bold", color=GREY)
    ax.axvline(0, color=GREY, lw=1.5)
    lefts = [r["diff"] - 1.96 * r["se"] for r in rows]
    rights = [r["diff"] + 1.96 * r["se"] for r in rows]
    ax.set_xlim(min(lefts) - 5, max(rights) + 5)             # room for the value labels on both sides
    ax.set_yticks(ys, [r["task"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("score with the correction minus without, points (95% CI)")
    pooled, k = S.pool("D2_bias", "8x")
    ax.set_title(f"8x, same draws: does the log(1/r) correction help?\n"
                 f"pooled over {k} tasks: {pooled['diff']:+.1f} ± {pooled['se']:.1f} points "
                 f"(one-sided p = {S.fmt_p(pooled['p_one'])})", loc="left")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SKY, label="LongBench"),
                       plt.Rectangle((0, 0), 1, 1, color=PURPLE, label="SCBench")], frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "3_nobias.png", dpi=170)
    plt.close(fig)

S.write_report()
print("figures and REPORT.md written to", FIG.parent)
