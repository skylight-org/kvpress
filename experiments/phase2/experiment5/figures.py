"""Experiment 5: the figures. CPU only, regenerates entirely from out/.

    python figures.py

Every figure is a full grid: both models x all five tasks x all four methods.

Conventions carried from summarize.py: aggregates are median over HEADS (never a pooled mean --
the per-head distribution is heavy-tailed and a mean is dominated by a few large-norm heads, which
is exactly what a per-layer Frobenius norm hides); the layer-13 boundary from E5a/E1 is marked;
VKvWR is always placed at its REALISED kept fraction, never its nominal compression_ratio.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {"kvzip": "#d62728", "bernoulli": "#ff7f0e", "perhead_bern": "#1f77b4", "vkvwr": "#2ca02c"}
LBL = {"kvzip": "KVzip (top-k)", "bernoulli": "Bernoulli (global c)",
       "perhead_bern": "Bernoulli (per-head)", "vkvwr": "VKvWR (prof.)"}
ORDER = ["kvzip", "bernoulli", "perhead_bern", "vkvwr"]
SHORT = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen3-4B-Instruct-2507": "Qwen3-4B"}
TASK = {"scbench_prefix_suffix": "scbench prefix_suffix"}
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})


def grid(models, tasks, w=3.05, h=3.05, **kw):
    fig, axes = plt.subplots(len(models), len(tasks),
                             figsize=(w * len(tasks), h * len(models)), squeeze=False, **kw)
    return fig, axes


def decorate(ax, i, j, m, t, nrow, ylab, xlab):
    if i == 0:
        ax.set_title(TASK.get(t, t), fontsize=9)
    if j == 0:
        ax.set_ylabel(f"{SHORT.get(m, m)}\n{ylab}", fontsize=8)
    if i == nrow - 1:
        ax.set_xlabel(xlab, fontsize=8)


def legend(fig, fams=ORDER, y=-.01):
    fig.legend(handles=[Line2D([], [], color=C[f], marker="o", lw=2, label=LBL[f]) for f in fams],
               loc="lower center", ncol=len(fams), frameon=False, bbox_to_anchor=(.5, y))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", (OUT / f"{name}.png").relative_to(HERE))


def ratio_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([.0625, .125, .25])
    ax.set_xticklabels(["16×", "8×", "4×"])
    ax.xaxis.set_minor_formatter(NullFormatter())


# ============================================================ 1. the Pareto frontier
def fig1_pareto(p, models, tasks):
    """Error vs REALISED memory -- the only comparison not confounded by VKvWR's
    compression_ratio setting no budget."""
    for met, short, ylab in (("attn", "attention", "attention error"),
                             ("den", "denominator", "denominator error"),
                             ("num", "numerator", "numerator error")):
        if met not in p.columns:
            continue
        fig, axes = grid(models, tasks, sharex=True, sharey="row")
        for i, m in enumerate(models):
            for j, t in enumerate(tasks):
                ax = axes[i][j]
                g = p[(p.model == m) & (p.task == t)]
                for fam in ORDER:
                    s = g[g.family == fam].sort_values("kept")
                    if s.empty:
                        continue
                    ax.plot(s.kept, s[met], "o-", color=C[fam], ms=4, lw=1.7)
                    if fam == "vkvwr":
                        ax.axvline(s.kept.min(), color=C[fam], ls=":", lw=1, alpha=.7)
                ax.set_yscale("log"); ratio_axis(ax)
                decorate(ax, i, j, m, t, len(models), ylab, "realised kept fraction")
        fig.suptitle(f"E5 — {ylab} vs realised memory  (median over query heads)\n"
                     "x is what each method ACTUALLY keeps; dotted line = VKvWR's memory floor",
                     fontsize=11)
        legend(fig)
        fig.tight_layout(rect=[0, .04, 1, .93])
        save(fig, f"fig1_pareto_{short}")


# ============================================================ 2. the headline, matched memory
def fig2_headline(p, models, tasks, c7):
    """Attention error at a matched REALISED kept fraction of 0.125. VKvWR has no point there,
    so its curve is interpolated onto the same memory."""
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    cells = [(m, t) for m in models for t in tasks]
    x = np.arange(len(cells)); w = .21
    interp = {(r.model_full, r.task): r.vkvwr_attn for r in c7.itertuples()}
    for k, fam in enumerate(ORDER):
        vals = []
        for m, t in cells:
            if fam == "vkvwr":
                vals.append(interp.get((m, t), np.nan))
            else:
                s = p[(p.model == m) & (p.task == t) & (p.family == fam)
                      & (np.abs(p.kept - .125) < 2e-3)]
                vals.append(float(s.attn.iloc[0]) if len(s) else np.nan)
        b = ax.bar(x + (k - 1.5) * w, vals, w, color=C[fam],
                   label=LBL[fam] + (" (interp.)" if fam == "vkvwr" else ""))
        ax.bar_label(b, fmt="%.3f", fontsize=5.8, padding=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SHORT.get(m, m)}\n{TASK.get(t, t)}" for m, t in cells],
                       fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("attention error (median over query heads)")
    ax.set_title("E5 — attention error at matched memory (realised kept fraction 0.125 ≈ 8×)\n"
                 "every sampled arm beats top-k in every cell", fontsize=11)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    save(fig, "fig2_headline_8x")


# ============================================================ 3. why per-head measurement matters
def fig3_perhead_tail(df, models, tasks):
    fig, axes = grid(models, tasks, sharex=True, sharey=True)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.task == t)]
            for fam in ORDER:
                s = g[g.family == fam]
                if s.empty:
                    continue
                v = s.groupby(["layer", "query_head"])["attn_err"].median().values
                v = np.sort(v[np.isfinite(v)])
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=C[fam], lw=1.7)
            ax.set_xscale("log")
            ax.axhline(.5, color="k", lw=.6, ls="--", alpha=.4)
            decorate(ax, i, j, m, t, len(models), "fraction of heads ≤ x", "per-head attention error")
    fig.suptitle("E5 — distribution of error ACROSS query heads, at matched memory ≈0.125\n"
                 "top-k's curve lies to the right at EVERY quantile — it is stochastically "
                 "dominated, not merely worse on average, which a per-layer norm cannot show",
                 fontsize=11)
    legend(fig)
    fig.tight_layout(rect=[0, .04, 1, .92])
    save(fig, "fig3_perhead_distribution")


# ============================================================ 4. where the damage lives
def fig4_layers(df, models, tasks):
    fig, axes = grid(models, tasks, sharex=False, sharey="row")
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.task == t)]
            for fam in ORDER:
                s = g[g.family == fam]
                if s.empty:
                    continue
                med = s.groupby("layer")["attn_err"].median()
                q1 = s.groupby("layer")["attn_err"].quantile(.25)
                q3 = s.groupby("layer")["attn_err"].quantile(.75)
                ax.plot(med.index, med.values, color=C[fam], lw=1.6)
                ax.fill_between(med.index, q1.values, q3.values, color=C[fam], alpha=.12, lw=0)
            ax.axvline(13, color="k", ls="--", lw=.9, alpha=.55)
            ax.set_yscale("log")
            decorate(ax, i, j, m, t, len(models), "attention error", "layer")
    fig.suptitle("E5 — error by layer at matched memory ≈0.125  (median, IQR band over heads)\n"
                 "dashed line = the layer-13 boundary found in E5a and E1", fontsize=11)
    legend(fig)
    fig.tight_layout(rect=[0, .04, 1, .92])
    save(fig, "fig4_layer_profile")


# ============================================================ 5. bias vs variance
def fig5_signed(df, models, tasks):
    """The pre-registered honesty note, drawn: top-k drops positive mass and never reweights, so
    its signed error is <= 0 by construction and the |.| metric flatters the reweighted arms."""
    fig, axes = grid(models, tasks, sharex=True, sharey=True)
    bins = np.linspace(-1, .6, 160)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.task == t)]
            for fam in ORDER:
                v = g[g.family == fam].den_signed.values
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                ax.hist(v, bins=bins, histtype="step", color=C[fam], lw=1.6, density=True)
                ax.axvline(np.median(v), color=C[fam], ls=":", lw=1.1, alpha=.85)
            ax.axvline(0, color="k", lw=.9)
            ax.set_yscale("log")
            decorate(ax, i, j, m, t, len(models), "density", "(B̂ − B) / B")
    fig.suptitle("E5 — SIGNED denominator error at matched memory ≈0.125  (dotted = median)\n"
                 "top-k is ≤ 0 everywhere (biased low, never reweighted); the reweighted arms "
                 "straddle zero — which is why C5, the attention ratio, is the criterion that matters",
                 fontsize=11)
    legend(fig)
    fig.tight_layout(rect=[0, .04, 1, .90])
    save(fig, "fig5_signed_denominator")


# ============================================================ 6. WHY top-k fails
def fig6_budget_spread(df, models, tasks):
    fig, axes = grid(models, tasks, sharey=True)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.task == t)]
            data, cols, ticks = [], [], []
            for fam in ORDER:
                s = g[g.family == fam]
                if s.empty:
                    continue
                v = s.groupby(["layer", "kv_head"])["kept_frac"].mean().values
                v = v[np.isfinite(v)]
                data.append(np.clip(v, 1e-5, None)); cols.append(C[fam])
                ticks.append(f"{v.max() / max(v.min(), 1e-9):,.0f}×")
            bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False, whis=(1, 99))
            for patch, c in zip(bp["boxes"], cols):
                patch.set_facecolor(c); patch.set_alpha(.5); patch.set_edgecolor(c)
            for key in ("medians", "whiskers", "caps"):
                for ln in bp[key]:
                    ln.set_color("0.25")
            ax.set_yscale("log")
            ax.set_xticklabels(ticks, fontsize=6.5)
            decorate(ax, i, j, m, t, len(models), "kept fraction of one (layer, KV head)",
                     "spread = max ÷ min")
    fig.suptitle("E5 — how each method spends its budget across heads, at matched memory ≈0.125\n"
                 "box = IQR, whiskers = 1st–99th pct.  top-k starves some heads to ≈0: "
                 "this is the mechanism behind fig 3's tail", fontsize=11)
    legend(fig)
    fig.tight_layout(rect=[0, .04, 1, .91])
    save(fig, "fig6_budget_spread")


# ============================================================ 7. C7, allocation vs estimator
def fig7_c7(p, c7, models, tasks):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for m in models:
        for t in tasks:
            g = p[(p.model == m) & (p.task == t) & p.family.isin(["bernoulli", "perhead_bern"])]
            g = g.assign(budget=g.kept.round(3))   # the two arms differ in the 6th decimal
            piv = g.pivot_table(index="budget", columns="family", values="attn").dropna().sort_index()
            if piv.empty:
                continue
            ax.plot(piv.index, piv.bernoulli / piv.perhead_bern, "o-" if "Llama" in m else "o--",
                    color=C["perhead_bern"], alpha=.75, ms=4, lw=1.3)
    ax.axhline(1, color="k", lw=.9)
    ratio_axis(ax)
    ax.set_xlabel("realised kept fraction"); ax.set_ylabel("global-c error ÷ per-head error")
    ax.set_title("ALLOCATION: per-head budget vs one global c\n"
                 "above 1 ⇒ per-head wins; the gain grows as memory tightens", fontsize=9.5)
    ax.legend(handles=[Line2D([], [], color=C["perhead_bern"], ls="-", label="Llama-3.1-8B"),
                       Line2D([], [], color=C["perhead_bern"], ls="--", label="Qwen3-4B")],
              frameon=False, fontsize=8)

    ax = axes[1]
    ok = c7[~c7.extrapolated].reset_index(drop=True)
    x = np.arange(len(ok))
    ax.bar(x - .2, ok.vkvwr_attn, .4, color=C["vkvwr"], label="VKvWR (interp. to same memory)")
    ax.bar(x + .2, ok.ph_attn, .4, color=C["perhead_bern"], label="Bernoulli (per-head)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.model}\n{TASK.get(r.task, r.task)}" for r in ok.itertuples()],
                       fontsize=6.5, rotation=35, ha="right")
    ax.set_yscale("log"); ax.set_ylabel("attention error")
    ax.set_title(f"ESTIMATOR: at matched realised memory (kept 0.125)\n"
                 f"per-head Bernoulli wins {int(ok.ph_wins_attn.sum())}/{len(ok)}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("E5 / C7 — the win is the ALLOCATION, not the estimator", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .91])
    save(fig, "fig7_c7_allocation_vs_estimator")


# ============================================================ 8. the E6 cancellation
def fig8_abar(df, models, tasks):
    """E6 found ||A|| ~= 0.4 Abar because the value vectors cancel, so an error that looks
    acceptable against Abar is ~2.5x worse against the quantity that actually matters."""
    fig, axes = grid(models, tasks, sharey="row")
    x = np.arange(len(ORDER)); w = .38
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i][j]
            g = df[(df.model == m) & (df.task == t)]
            for k, (col, hatch) in enumerate([("num_err_abar", ""), ("num_err", "//")]):
                vals = [g[g.family == f][col].median() if len(g[g.family == f]) else np.nan
                        for f in ORDER]
                ax.bar(x + (k - .5) * w, vals, w, color=[C[f] for f in ORDER],
                       alpha=1 - .45 * k, hatch=hatch, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(["top-k", "bern", "per-head", "VKvWR"], fontsize=6.3)
            ax.set_yscale("log")
            decorate(ax, i, j, m, t, len(models), "numerator error", "")
    fig.legend(handles=[Patch(facecolor="0.6", label="vs Ā = Σ wᵢ‖vᵢ‖"),
                        Patch(facecolor="0.6", hatch="//", label="vs ‖A‖  (the honest one)")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("E5 — numerator error against Ā vs against ‖A‖, at matched memory ≈0.125\n"
                 "E6: ‖A‖ ≈ 0.4 Ā because the value vectors cancel, so the hatched bar is the "
                 "one that counts", fontsize=11)
    fig.tight_layout(rect=[0, .04, 1, .91])
    save(fig, "fig8_numerator_abar")



# ============================================================ 9. the allocation trade-off
def fig9_allocation_tradeoff(df_all, models):
    """Per-head budgeting helps the TYPICAL head at every ratio, but at a loose budget it hurts
    the worst heads: forcing every head to keep exactly the same fraction takes budget away from
    the heads that genuinely need more. The crossover is the finding."""
    d = df_all[df_all.family.isin(["bernoulli", "perhead_bern"]) & df_all.task.ne("scbench_prefix_suffix")]
    stats = [("median head", "median", "-", "o"), ("p95 head", lambda x: x.quantile(.95), "--", "s"),
             ("p99 head (worst)", lambda x: x.quantile(.99), ":", "^")]
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.2 * (len(models) + 1), 4.2))
    for ax, m in zip(axes, models):
        g = d[d.model == m]
        for lab, fn, ls, mk in stats:
            a = g[g.family == "perhead_bern"].groupby("ratio").attn_err.agg(fn)
            b = g[g.family == "bernoulli"].groupby("ratio").attn_err.agg(fn)
            ratio = (b / a).sort_index()
            ax.plot(1 - ratio.index, ratio.values, ls, marker=mk, color=C["perhead_bern"],
                    ms=5, lw=1.6, label=lab)
        ax.axhline(1, color="k", lw=1)
        ratio_axis(ax)
        ax.set_xlabel("realised kept fraction")
        ax.set_ylabel("global-c error ÷ per-head error")
        ax.set_title(f"{SHORT.get(m, m)}\nabove 1 ⇒ per-head budgeting wins", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8)

    ax = axes[-1]                                  # win rate over (layer, head) cells
    for m in models:
        g = d[d.model == m]
        k = ["task", "layer", "query_head"]
        fr = []
        for r in sorted(g.ratio.unique()):
            h = g[g.ratio == r]
            a = h[h.family == "perhead_bern"].groupby(k).attn_err.median()
            b = h[h.family == "bernoulli"].groupby(k).attn_err.median()
            j = pd.concat([a.rename("ph"), b.rename("gl")], axis=1).dropna()
            fr.append((1 - r, (j.ph < j.gl).mean()))
        fr = np.array(fr)
        ax.plot(fr[:, 0], fr[:, 1] * 100, "o-" if "Llama" in m else "o--",
                color=C["perhead_bern"], ms=5, lw=1.6, label=SHORT.get(m, m))
    ax.axhline(50, color="k", lw=1)
    ratio_axis(ax)
    ax.set_xlabel("realised kept fraction")
    ax.set_ylabel("% of (layer, head) cells where per-head wins")
    ax.set_title("win rate across heads\nrises as the budget tightens", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("E5 — the allocation trade-off: per-head budgeting helps the typical head at every "
                 "ratio,\nbut at a LOOSE budget (4×) it makes the worst heads worse — "
                 "equalising takes budget from heads that need more", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .89])
    save(fig, "fig9_allocation_tradeoff")


def main():
    p = pd.read_csv(HERE / "out" / "pareto.csv")
    models = sorted(p.model.unique())
    tasks = sorted(p.task.unique())
    print(f"models={[SHORT.get(m, m) for m in models]}\ntasks={tasks}")

    # Matched-realised-memory comparison. summarize.py's C7 row pairs at NOMINAL ratio, where
    # VKvWR silently keeps ~60% more memory; this puts both on the realised axis instead.
    rows = []
    for (m, t), g in p.groupby(["model", "task"]):
        v = g[g.family == "vkvwr"].sort_values("kept")
        ph = g[(g.family == "perhead_bern") & (np.abs(g.kept - .125) < 2e-3)]
        if ph.empty or v.empty:
            continue
        rows.append(dict(model_full=m, model=SHORT.get(m, m), task=t,
                         vkvwr_attn=float(np.exp(np.interp(np.log(.125), np.log(v.kept),
                                                           np.log(v.attn)))),
                         ph_attn=float(ph.attn.iloc[0]),
                         extrapolated=not (v.kept.min() <= .125 <= v.kept.max())))
    c7 = pd.DataFrame(rows)
    c7["ph_wins_attn"] = c7.ph_attn < c7.vkvwr_attn
    c7.to_csv(HERE / "out" / "c7_matched_memory.csv", index=False)

    print("\nfrom pareto.csv:")
    fig1_pareto(p, models, tasks)
    fig2_headline(p, models, tasks, c7)
    fig7_c7(p, c7, models, tasks)

    cols = ["family", "model", "task", "layer", "query_head", "kv_head", "ratio",
            "attn_err", "num_err", "num_err_abar", "den_signed", "kept_frac"]
    df = pd.read_parquet(HERE / "out" / "all_rows.parquet", columns=cols)
    # Budgeted arms at nominal 8x realise exactly 0.125. For VKvWR pick, per (model, task), the
    # ratio whose REALISED kept fraction is closest to 0.125 -- never its nominal 8x, which keeps
    # ~0.20 and would hand it 60% more memory than everything it is drawn beside.
    keep = df.family.isin(["kvzip", "bernoulli", "perhead_bern"]) & np.isclose(df.ratio, .875)
    print("\nVKvWR ratio chosen per cell (closest realised kept to 0.125):")
    for (m, t), g in df[df.family == "vkvwr"].groupby(["model", "task"]):
        km = g.groupby("ratio")["kept_frac"].mean()
        best = (km - .125).abs().idxmin()
        print(f"  {SHORT.get(m, m):13s} {t:22s} ratio={best:<7} realised={km[best]:.4f}")
        keep |= (df.family == "vkvwr") & (df.model == m) & (df.task == t) & np.isclose(df.ratio, best)
    df_all = df[df.family.isin(["bernoulli", "perhead_bern"])]   # every ratio, for fig 9
    df = df[keep]
    print(f"  row-level subset: {len(df):,} rows")

    print("\nfrom all_rows.parquet:")
    fig3_perhead_tail(df, models, tasks)
    fig4_layers(df, models, tasks)
    fig5_signed(df, models, tasks)
    fig6_budget_spread(df, models, tasks)
    fig8_abar(df, models, tasks)
    fig9_allocation_tradeoff(df_all, models)


if __name__ == "__main__":
    main()
