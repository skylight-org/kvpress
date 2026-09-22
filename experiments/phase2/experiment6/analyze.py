"""Experiment 6: downstream accuracy on E5's settings. CPU only; reads results/ and E5's out/pareto.csv.

    python analyze.py            -> out/*.csv, out/figures/*.png, out/REPORT.md

Per-sample scores are recomputed with RULER's own rule (qa: any gold present; else the fraction of golds
present, after RULER's control-character strip) and must reproduce every run's metrics.json exactly.
All arms answer the same 30 samples (df.sample(30, random_state=42)), so comparisons are paired per sample.
VKvWR's compression_ratio sets no budget, so it is placed at its REALISED kept fraction measured in E5.
"""
import json, re, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
E5 = Path.home() / "phase2/experiment5/out/pareto.csv"
SHORT = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen3-4B-Instruct-2507": "Qwen3-4B"}
FAM = lambda a: ("no_press" if a == "no_press" else "vkvwr" if a.startswith("vkvwr") else
                 "perhead_bern" if a.startswith("perhead") else "bernoulli" if a.startswith("bernoulli") else "kvzip")
C = {"no_press": "#555555", "kvzip": "#d62728", "bernoulli": "#ff7f0e", "perhead_bern": "#1f77b4", "vkvwr": "#2ca02c"}
LBL = {"no_press": "no compression", "kvzip": "KVzip (top-k)", "bernoulli": "Bernoulli (global c)",
       "perhead_bern": "Bernoulli (per-head)", "vkvwr": "VKvWR (prof.)"}
CTRL = re.compile(r"[\x00-\x1f]")
L = []; w = L.append


def golds(s):
    return [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", str(s))]


def sample_score(task, pred, ans):
    pred = CTRL.sub("", str(pred).strip()).strip().lower()
    g = golds(ans)
    hits = [1.0 if x.lower() in pred else 0.0 for x in g]
    return 100.0 * (max(hits) if task.split("_")[0] == "qa" else sum(hits) / len(hits))


# ------------------------------------------------------------------ load + verify
runs, rows, mismatches = [], [], []
for cfgp in sorted(glob.glob(str(HERE / "results/r32k_*/*/config.yaml"))):
    d = Path(cfgp).parent
    cfg = yaml.safe_load(open(cfgp))
    task, arm, ratio = cfg["data_dir"], cfg["press_name"], float(cfg["compression_ratio"])
    model = SHORT[cfg["model"].split("/")[-1]]
    reported = list(list(json.load(open(d / "metrics.json")).values())[0].values())[0]
    df = pd.read_csv(d / "predictions.csv")
    df["score"] = [sample_score(task, p, a) for p, a in zip(df.predicted_answer, df.answer)]
    mine = round(df.score.mean(), 2)
    if abs(mine - reported) > 0.011:
        mismatches.append((d.name, mine, reported))
    seed = int(arm.rsplit("_s", 1)[1]) if "_s" in arm else 0
    runs.append(dict(model=model, task=task, arm=arm, family=FAM(arm), seed=seed, ratio=ratio, acc=reported,
                     n=len(df), gpu=json.load(open(d / "cost.json"))["gpu"] if (d / "cost.json").exists() else ""))
    for i, r in df.iterrows():
        rows.append(dict(model=model, task=task, family=FAM(arm), seed=seed, ratio=ratio,
                         sid=f"{r.question}||{r.answer}", score=r.score))
R, S = pd.DataFrame(runs), pd.DataFrame(rows)
R.to_csv(OUT / "runs.csv", index=False)
print(f"{len(R)} runs, {len(S)} sample-scores; metrics.json reproduced for {len(R) - len(mismatches)}/{len(R)}")
assert not mismatches, mismatches
assert (R.n == 30).all()

# realised kept fraction: budgeted arms hit 1-ratio exactly (E5, C0); VKvWR from E5's measurement
P = pd.read_csv(E5); P["model"] = P.model.map(SHORT)
vk = P[(P.family == "vkvwr") & (P.dataset == "ruler32k")][["model", "task", "ratio", "kept"]]
R["kept"] = np.where(R.family == "no_press", 1.0, 1 - R.ratio)
R = R.merge(vk.rename(columns={"kept": "kept_vk"}), on=["model", "task", "ratio"], how="left")
qa_fill = vk.groupby(["model", "ratio"]).kept.mean().rename("kept_vk_mean").reset_index()
R = R.merge(qa_fill, on=["model", "ratio"], how="left")
R["kept_source"] = "budget"
m = R.family == "vkvwr"
R.loc[m, "kept"] = R.loc[m, "kept_vk"].fillna(R.loc[m, "kept_vk_mean"])
R.loc[m, "kept_source"] = np.where(R.loc[m, "kept_vk"].notna(), "E5 measured", "E5 mean of other tasks")
R = R.drop(columns=["kept_vk", "kept_vk_mean"])
R.to_csv(OUT / "runs.csv", index=False)

# ------------------------------------------------------------------ 1. the score table
A = R.groupby(["model", "task", "family", "ratio"], as_index=False).agg(acc=("acc", "mean"), kept=("kept", "mean"))
tasks = ["qa_1", "niah_single_3", "niah_multikey_2", "vt", "cwe"]
models = ["Llama-3.1-8B", "Qwen3-4B"]
w("# Experiment 6: downstream accuracy on E5's settings\n")
w(f"{len(R)} runs, 30 samples each, every per-sample score recomputed and matched to metrics.json "
  f"({len(R)}/{len(R)}).\n")
w("## Accuracy (string_match, %) at the budgeted ratios\n")
w("Bernoulli = mean of seeds s1 and s2. no_press is the uncompressed ceiling.\n")
for ratio, name in [(0.75, "4x (kept 0.25)"), (0.875, "8x (kept 0.125)"), (0.9375, "16x (kept 0.0625)")]:
    w(f"\n### {name}\n")
    w("| model | task | no_press | KVzip | Bernoulli (global) | Bernoulli (per-head) |")
    w("|---|---|---|---|---|---|")
    for mo in models:
        for t in tasks:
            g = A[(A.model == mo) & (A.task == t)]
            v = lambda f, r: g[(g.family == f) & np.isclose(g.ratio, r)].acc.mean()
            w(f"| {mo} | {t} | {v('no_press', 0.0):.0f} | {v('kvzip', ratio):.0f} | {v('bernoulli', ratio):.0f} | "
              f"{v('perhead_bern', ratio):.0f} |")

w("\n## VKvWR ladder (placed at its realised kept fraction)\n")
w("| model | task | " + " | ".join(f"cr {r}" for r in [0.875, 0.9375, 0.96875, 0.984, 0.995]) + " |")
w("|---|---|" + "---|" * 5)
for mo in models:
    for t in tasks:
        g = A[(A.model == mo) & (A.task == t) & (A.family == "vkvwr")].sort_values("ratio")
        w(f"| {mo} | {t} | " + " | ".join(f"{r.acc:.0f} @ {r.kept:.3f}" for r in g.itertuples()) + " |")
w("\n(qa_1 has no E5 memory measurement; its VKvWR kept fraction is the mean of the other four tasks.)\n")

# ------------------------------------------------------------------ 2. paired tests vs KVzip
w("## Paired against KVzip at the same budget (per sample, pooled over tasks and models)\n")
w("Difference in per-sample score (percentage points); positive = the arm beats KVzip. Tests: paired t over "
  "samples and an exact sign test on the non-tied samples; Holm over the whole family.\n")
Sm = S.groupby(["model", "task", "family", "ratio", "sid"], as_index=False).score.mean()   # averages bernoulli seeds
tests = []
for fam in ["bernoulli", "perhead_bern"]:
    for ratio in [0.75, 0.875, 0.9375]:
        for scope in ["all", *models]:
            a = Sm[(Sm.family == fam) & np.isclose(Sm.ratio, ratio)]
            b = Sm[(Sm.family == "kvzip") & np.isclose(Sm.ratio, ratio)]
            if scope != "all":
                a, b = a[a.model == scope], b[b.model == scope]
            j = a.merge(b, on=["model", "task", "sid"], suffixes=("_a", "_b"))
            d = j.score_a - j.score_b
            t = stats.ttest_1samp(d, 0.0)
            wins, losses = int((d > 0).sum()), int((d < 0).sum())
            ps = stats.binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
            tests.append(dict(arm=fam, ratio=ratio, scope=scope, n=len(d), mean=d.mean(),
                              ci=1.96 * d.std(ddof=1) / np.sqrt(len(d)), wins=wins, losses=losses,
                              p_t=t.pvalue, p_sign=ps))
T = pd.DataFrame(tests)
fam_mask = T.scope == "all"
order = np.argsort(T.loc[fam_mask, "p_t"].values)
p = T.loc[fam_mask, "p_t"].values; k = len(p); holm = np.empty(k); run = 0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (k - rank) * p[i])); holm[i] = run
T.loc[fam_mask, "holm"] = holm
T.to_csv(OUT / "paired_vs_kvzip.csv", index=False)
w("| arm | ratio | scope | n | mean diff | 95% CI | wins/losses | p (t) | Holm | p (sign) |")
w("|---|---|---|---|---|---|---|---|---|---|")
for r in T.itertuples():
    w(f"| {r.arm} | {r.ratio} | {r.scope} | {r.n} | {r.mean:+.1f} | ±{r.ci:.1f} | {r.wins}/{r.losses} | "
      f"{r.p_t:.3g} | {'' if pd.isna(r.holm) else f'{r.holm:.3g}'} | {r.p_sign:.3g} |")

# per task, 8x, bernoulli vs kvzip
w("\n### Per task at 8x: Bernoulli (global) minus KVzip\n")
w("| model | task | KVzip | Bernoulli | diff | wins/losses |")
w("|---|---|---|---|---|---|")
for mo in models:
    for t in tasks:
        a = Sm[(Sm.model == mo) & (Sm.task == t) & (Sm.family == "bernoulli") & np.isclose(Sm.ratio, 0.875)]
        b = Sm[(Sm.model == mo) & (Sm.task == t) & (Sm.family == "kvzip") & np.isclose(Sm.ratio, 0.875)]
        j = a.merge(b, on="sid", suffixes=("_a", "_b")); d = j.score_a - j.score_b
        w(f"| {mo} | {t} | {j.score_b.mean():.0f} | {j.score_a.mean():.0f} | {d.mean():+.1f} | "
          f"{int((d > 0).sum())}/{int((d < 0).sum())} |")

# ------------------------------------------------------------------ 3. VKvWR at matched memory
w("\n## VKvWR at matched realised memory\n")
w("VKvWR's accuracy is interpolated (linear in log kept) onto the budgeted kept fractions. Out of range = its "
  "memory floor is above that budget.\n")
w("| model | task | budget | KVzip | Bernoulli | VKvWR (interp.) |")
w("|---|---|---|---|---|---|")
vm = []
for mo in models:
    for t in tasks:
        g = A[(A.model == mo) & (A.task == t)]
        v = g[g.family == "vkvwr"].sort_values("kept")
        for kept, ratio in [(0.25, 0.75), (0.125, 0.875), (0.0625, 0.9375)]:
            inr = v.kept.min() <= kept <= v.kept.max()
            val = float(np.interp(np.log(kept), np.log(v.kept), v.acc)) if inr else np.nan
            kz = g[(g.family == "kvzip") & np.isclose(g.ratio, ratio)].acc.mean()
            be = g[(g.family == "bernoulli") & np.isclose(g.ratio, ratio)].acc.mean()
            vm.append(dict(model=mo, task=t, kept=kept, kvzip=kz, bernoulli=be, vkvwr=val))
            w(f"| {mo} | {t} | {kept} | {kz:.0f} | {be:.0f} | {'out of range' if not inr else f'{val:.0f}'} |")
VM = pd.DataFrame(vm); VM.to_csv(OUT / "vkvwr_matched.csv", index=False)
ok = VM.dropna()
w(f"\nIn range: {len(ok)} cells. VKvWR >= KVzip in {int((ok.vkvwr >= ok.kvzip).sum())}, "
  f">= Bernoulli in {int((ok.vkvwr >= ok.bernoulli).sum())}. Mean VKvWR - KVzip {(ok.vkvwr - ok.kvzip).mean():+.1f}, "
  f"VKvWR - Bernoulli {(ok.vkvwr - ok.bernoulli).mean():+.1f} points.\n")

# ------------------------------------------------------------------ 4. does E5's upstream error predict accuracy?
Pj = P[P.dataset == "ruler32k"].copy()
Pj["family"] = Pj.family.replace({"bernoulli": "bernoulli"})
J = A.merge(Pj[["model", "task", "family", "ratio", "attn", "den"]], on=["model", "task", "family", "ratio"])
ceil = A[A.family == "no_press"][["model", "task", "acc"]].rename(columns={"acc": "ceiling"})
J = J.merge(ceil, on=["model", "task"]); J["drop"] = J.ceiling - J.acc
J.to_csv(OUT / "upstream_vs_downstream.csv", index=False)
w("## Does E5's upstream error predict downstream accuracy?\n")
w("Cells = (model, task, arm, ratio) measured in both experiments (qa_1 has no E5 errors). Spearman rank "
  "correlation between E5's median attention error and the accuracy drop from the uncompressed ceiling.\n")
w("| subset | cells | Spearman rho (attention err vs drop) | p |")
w("|---|---|---|---|")
for name, sub in [("all arms", J), ("without per-head", J[J.family != "perhead_bern"]),
                  *[(f"within {f}", J[J.family == f]) for f in ["kvzip", "bernoulli", "perhead_bern", "vkvwr"]]]:
    rho, pv = stats.spearmanr(sub.attn, sub["drop"])
    w(f"| {name} | {len(sub)} | {rho:+.2f} | {pv:.3g} |")
by = J.groupby("family").agg(attn=("attn", "median"), drop=("drop", "mean")).reindex(
    ["kvzip", "bernoulli", "perhead_bern", "vkvwr"])
w("\n| arm | median E5 attention error | mean accuracy drop (points) |")
w("|---|---|---|")
for f, r in by.iterrows():
    w(f"| {LBL[f]} | {r.attn:.3f} | {r['drop']:.1f} |")

(OUT / "REPORT.md").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 170, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})
fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharex=True, sharey=True)
for i, mo in enumerate(models):
    for j, t in enumerate(tasks):
        ax = axes[i][j]; g = A[(A.model == mo) & (A.task == t)]
        ax.axhline(g[g.family == "no_press"].acc.mean(), color=C["no_press"], ls="--", lw=1.2)
        for f in ["kvzip", "bernoulli", "perhead_bern", "vkvwr"]:
            s = g[g.family == f].sort_values("kept")
            ax.plot(s.kept, s.acc, "o-", color=C[f], ms=4, lw=1.7)
        ax.set_xscale("log"); ax.set_xticks([.0625, .125, .25]); ax.set_xticklabels(["16×", "8×", "4×"])
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_ylim(-5, 105)
        if i == 0: ax.set_title(t, fontsize=9.5)
        if j == 0: ax.set_ylabel(f"{mo}\naccuracy (%)")
        if i == 1: ax.set_xlabel("realised kept fraction")
fig.legend(handles=[plt.Line2D([], [], color=C[f], marker="o" if f != "no_press" else None,
                               ls="--" if f == "no_press" else "-", label=LBL[f]) for f in C],
           loc="lower center", ncol=5, frameon=False)
fig.suptitle("E6: downstream accuracy vs realised memory (30 samples per point; dashed = uncompressed)", fontsize=11)
fig.tight_layout(rect=[0, .05, 1, .95]); fig.savefig(OUT / "figures/fig1_accuracy_vs_memory.png", bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 5))
for f in ["kvzip", "bernoulli", "perhead_bern", "vkvwr"]:
    s = J[J.family == f]; ax.scatter(s.attn, s["drop"], color=C[f], s=28, alpha=.8, label=LBL[f])
ax.set_xscale("log"); ax.set_xlabel("E5 median attention error (upstream)"); ax.set_ylabel("accuracy drop vs uncompressed (points)")
ax.set_title("E5 upstream error vs E6 downstream accuracy drop\n(one point per model, task, arm, ratio)", fontsize=10)
ax.legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(OUT / "figures/fig2_upstream_vs_downstream.png", bbox_inches="tight"); plt.close(fig)
print((OUT / "REPORT.md").read_text())
