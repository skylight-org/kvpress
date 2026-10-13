"""E6 SCBench (prefix_suffix, E5's 25-question slice, 8x budget). Run after analyze.py; appends to out/REPORT.md.

Per-question scores use SCBench's own scorer (substring_match on as_list(answer)) and must reproduce metrics.json.
VKvWR is placed at the realised kept fraction E5 measured on this exact slice; its cr 0.96875 run keeps ~0.123,
i.e. matched to the 0.125 budget, so it is paired directly.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy import stats
from evaluation.benchmarks.scbench.calculate_metrics import substring_match, as_list

HERE = Path(__file__).resolve().parent; OUT = HERE / "out"
SHORT = {"Llama-3.1-8B-Instruct": "Llama-3.1-8B", "Qwen3-4B-Instruct-2507": "Qwen3-4B"}
FAM = lambda a: ("no_press" if a == "no_press" else "vkvwr" if a.startswith("vkvwr") else
                 "perhead_bern" if a.startswith("perhead") else "bernoulli" if a.startswith("bernoulli") else "kvzip")
P = pd.read_csv(Path.home() / "phase2/experiment5/out/pareto.csv"); P = P[P.dataset == "scbench"]
P["model"] = P.model.map(SHORT)

runs, rows, bad = [], [], []
for cfgp in sorted(glob.glob(str(HERE / "results/scb_*/*/config.yaml"))):
    d = Path(cfgp).parent; cfg = yaml.safe_load(open(cfgp))
    arm, ratio, model = cfg["press_name"], float(cfg["compression_ratio"]), SHORT[cfg["model"].split("/")[-1]]
    reported = list(list(json.load(open(d / "metrics.json")).values())[0].values())[0]
    df = pd.read_csv(d / "predictions.csv")
    df["score"] = [substring_match("" if pd.isna(p) else p, as_list(g)) for p, g in zip(df.predicted_answer, df.answer)]
    if abs(round(df.score.mean(), 2) - reported) > 0.011:
        bad.append((d.name, df.score.mean(), reported))
    fam = FAM(arm)
    kept = 1.0 if fam == "no_press" else 1 - ratio
    if fam == "vkvwr":
        kept = float(P[(P.model == model) & (P.family == "vkvwr") & np.isclose(P.ratio, ratio)].kept.iloc[0])
    runs.append(dict(model=model, family=fam, ratio=ratio, kept=kept, acc=reported, n=len(df)))
    rows += [dict(model=model, family=fam, ratio=ratio, qid=f"{q}||{a}", score=s)
             for q, a, s in zip(df.question, df.answer, df.score)]
R, S = pd.DataFrame(runs), pd.DataFrame(rows)
print(f"{len(R)} SCBench runs; metrics.json reproduced for {len(R) - len(bad)}/{len(R)}"); assert not bad, bad
R.to_csv(OUT / "scbench_runs.csv", index=False)

L = ["\n## SCBench prefix_suffix (8x budget, 25 questions = 5 shared contexts x 5 turns)\n",
     f"{len(R)} runs; every per-question score reproduces metrics.json ({len(R)}/{len(R)}).\n",
     "| model | no_press | KVzip | Bernoulli | per-head | VKvWR ladder (acc @ realised kept) |", "|---|---|---|---|---|---|"]
for mo in ["Llama-3.1-8B", "Qwen3-4B"]:
    g = R[R.model == mo]; v = lambda f: g[g.family == f].acc.mean()
    lad = ", ".join(f"{r.acc:.0f} @ {r.kept:.3f}" for r in g[g.family == "vkvwr"].sort_values("kept").itertuples())
    L.append(f"| {mo} | {v('no_press'):.0f} | {v('kvzip'):.0f} | {v('bernoulli'):.0f} | {v('perhead_bern'):.0f} | {lad} |")

L += ["\nPaired per question against KVzip (VKvWR = its cr 0.96875 run, realised kept ~0.123):\n",
      "| model | arm | mean diff (points) | wins/losses | exact sign p |", "|---|---|---|---|---|"]
for mo in ["Llama-3.1-8B", "Qwen3-4B"]:
    kz = S[(S.model == mo) & (S.family == "kvzip")]
    for fam, ratio in [("bernoulli", 0.875), ("perhead_bern", 0.875), ("vkvwr", 0.96875), ("no_press", 0.0)]:
        a = S[(S.model == mo) & (S.family == fam) & np.isclose(S.ratio, ratio)]
        j = a.merge(kz, on="qid", suffixes=("_a", "_b")); d = j.score_a - j.score_b
        wn, ls = int((d > 0).sum()), int((d < 0).sum())
        p = stats.binomtest(wn, wn + ls).pvalue if wn + ls else 1.0
        L.append(f"| {mo} | {fam} | {d.mean():+.1f} | {wn}/{ls} | {p:.3g} |")
with open(OUT / "REPORT.md", "a") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
