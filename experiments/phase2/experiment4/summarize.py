"""Experiment 4 results -> out/REPORT.md, out/cells.csv (pre-registered criteria in DESIGN.md).

    python summarize.py

Sources
  experiment4/results/longbench   every LongBench arm (new)
  experiment4/results/scbench     SCBench: compactor, compactor_bern_s*, kvzip_bern_nobias_s1 (new)
  experiment3/results/S1n10       SCBench: no_press, kvzip, kvzip_bern_s* (REUSED, same contexts)

Per-question scores are recomputed from predictions.csv and must agree with each run's metrics.json.
LongBench rows are keyed by `_id` (gold answers re-read from the dataset parquet, because
predictions.csv stores them as numpy reprs); SCBench rows by (context_id, turn).

Clusters: SCBench = context (5 turns share one compression draw), LongBench = sample.
Sign-flip tests are exact when a task has <= 16 clusters, Monte Carlo (200k draws) otherwise.
"""

import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
HOME = Path.home()
sys.path.insert(0, str(HOME / "phase2" / "experiment3"))
sys.path.insert(0, str(HOME / "src" / "kvpress" / "evaluation"))
from scbench_metrics import SCORERS, score_one  # noqa: E402
from benchmarks.longbench.calculate_metrics import dataset2metric  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

LB_TASKS = ["qasper", "hotpotqa", "passage_retrieval_en", "gov_report"]
SC_TASKS = ["prefix_suffix", "qa_eng", "summary", "vt"]
TASKS = LB_TASKS + SC_TASKS
BENCH = {**{t: "LongBench" for t in LB_TASKS}, **{t: "SCBench" for t in SC_TASKS}}
COMP = {0.0: "full", 0.5: "2x", 0.75: "4x", 0.88: "8x", 0.94: "16x"}
COMPS = ["2x", "4x", "8x", "16x"]
GPU = {"qasper": "A40", "hotpotqa": "A40", "passage_retrieval_en": "A40", "gov_report": "A40",
       "prefix_suffix": "L40S", "summary": "L40S", "qa_eng": "A100", "vt": "A100"}
# the new SCBench arms all run on the A100 (compactor_bern needs >45 GB at 124k)
GPU_NEW_SC = "A100"
SC_REUSED = {"no_press", "kvzip", "kvzip_bern"}
METRIC_NAME = {"qasper": "token F1", "hotpotqa": "token F1", "passage_retrieval_en": "retrieval score",
               "gov_report": "ROUGE-L F", "prefix_suffix": "substring match", "qa_eng": "token F1",
               "summary": "ROUGE-L F1", "vt": "fraction of variables found"}
SC_METRIC = {t: next(iter(SCORERS[t])) for t in SCORERS}

# ------------------------------------------------------------------ LongBench gold answers
_lb_gold = {}


def lb_gold(task):
    if task not in _lb_gold:
        files = glob.glob(str(HOME / "hf/hub/datasets--Xnhyacinth--LongBench/snapshots/*" / Path(task) / "test*.parquet"))
        assert files, f"LongBench parquet for {task} not found"
        df = pd.concat([pd.read_parquet(f) for f in files])
        _lb_gold[task] = {r["_id"]: (list(r["answers"]), r["all_classes"]) for _, r in df.iterrows()}
    return _lb_gold[task]


def lb_score(task, pred, gold):
    answers, all_classes = gold
    pred = "" if pred is None or (isinstance(pred, float) and math.isnan(pred)) else str(pred)
    return 100 * max(dataset2metric[task](pred.lstrip(), gt, all_classes=all_classes) for gt in answers)


# ------------------------------------------------------------------ load runs
def run_folders(root):
    """One folder per run: the latest of base / numbered subfolders that holds a metrics.json."""
    by_name = {}
    for mj in Path(root).rglob("metrics.json"):
        d = mj.parent
        name = d.parent.name if re.fullmatch(r"[0-9]+", d.name) else d.name
        rank = int(d.name) if re.fullmatch(r"[0-9]+", d.name) else 0
        if name not in by_name or rank > by_name[name][0]:
            by_name[name] = (rank, d)
    return {n: d for n, (_, d) in by_name.items()}


def load(root, source, keep=lambda task, arm: True):
    out = []
    for name, d in sorted(run_folders(root).items()):
        parts = name.split("__")
        dataset, task, press, ratio = parts[0], parts[1], parts[3], round(float(parts[4]), 2)
        if task not in TASKS:
            continue
        arm = re.sub(r"_s[123]$", "", press)
        if not keep(task, arm):
            continue
        seed = int(press[-1]) if re.search(r"_s[123]$", press) else None
        pr = pd.read_csv(d / "predictions.csv")
        met = json.load(open(d / "metrics.json"))
        if dataset == "longbench":
            gold = lb_gold(task)
            pr["key"] = pr["_id"]
            pr["cluster"] = pr["_id"]
            pr["s"] = [lb_score(task, p, gold[i]) for p, i in zip(pr.predicted_answer, pr._id)]
            official = float(met) if isinstance(met, (int, float)) else float(met[task])
        else:
            pr["predicted_answer"] = pr["predicted_answer"].fillna("")
            pr["key"] = list(zip(pr.context_id, pr.turn))
            pr["cluster"] = pr.context_id
            pr["s"] = [score_one(task, p, a) for p, a in zip(pr.predicted_answer, pr.answer)]
            official = float(met[task][SC_METRIC[task]])
        assert abs(pr.s.mean() - official) < 0.02, f"{name}: per-question {pr.s.mean():.3f} != metrics.json {official}"
        cost = json.load(open(d / "cost.json")) if (d / "cost.json").exists() else {}
        mem = json.load(open(d / "memory.json")) if (d / "memory.json").exists() else None
        out.append({"bench": BENCH[task], "task": task, "press": press, "arm": arm, "seed": seed,
                    "comp": COMP.get(ratio, str(ratio)), "ratio": ratio, "score": official,
                    "pred": pr, "cost": cost, "mem": mem, "source": source, "dir": str(d)})
    return out


RUNS = (load(HERE / "results" / "longbench", "E4")
        + load(HERE / "results" / "scbench", "E4", keep=lambda t, a: a not in SC_REUSED)
        + load(HOME / "phase2/experiment3/results/S1n10", "E3", keep=lambda t, a: a in SC_REUSED))
assert RUNS, "no results yet"


def get(task, arm, comp):
    return [r for r in RUNS if r["task"] == task and r["arm"] == arm and r["comp"] == comp]


def per_q(task, arm, comp):
    """Per-question score (mean over seeds), indexed by key, plus the cluster of each key."""
    rs = get(task, arm, comp)
    if not rs:
        return None
    frames = [r["pred"].set_index("key") for r in rs]
    for f in frames[1:]:
        assert set(f.index) == set(frames[0].index), f"question sets differ between seeds: {task} {arm} {comp}"
    s = pd.concat([f.s for f in frames], axis=1).mean(axis=1)
    return pd.DataFrame({"s": s, "cluster": frames[0].cluster.reindex(s.index)})


def cluster_diffs(task, a, b, comp):
    """Mean paired difference per cluster (a - b), or None if an arm is missing."""
    A = per_q(task, a, comp if a != "no_press" else "full")
    B = per_q(task, b, comp if b != "no_press" else "full")
    if A is None or B is None:
        return None
    assert set(A.index) == set(B.index), f"{task} {comp}: {a} and {b} answered different questions"
    d = A.s - B.s.reindex(A.index)
    return d.groupby(A.cluster).mean().to_numpy()


def signs(n, draws=200_000):
    if n <= 16:
        return np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    return RNG.choice([-1, 1], size=(draws, n))


def one_task(x):
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    flips = signs(n) @ x / n
    p = float((flips >= x.mean() - 1e-12).mean())                 # one-sided, H1: mean > 0
    return {"diff": float(x.mean()), "se": float(se), "n": n, "p_one": p}


def pooled(xs, draws=200_000):
    """Stratified sign-flip: statistic = mean over tasks of the task means; flips within each task."""
    obs = np.mean([x.mean() for x in xs])
    null = np.zeros(draws)
    for x in xs:
        null += (RNG.choice([-1, 1], size=(draws, len(x))) @ x) / len(x)
    null /= len(xs)
    se = math.sqrt(sum((x.std(ddof=1) / math.sqrt(len(x))) ** 2 for x in xs)) / len(xs)
    return {"diff": float(obs), "se": se, "p_one": float((null >= obs - 1e-12).mean()), "k": len(xs),
            "ci95": (float(obs - 1.96 * se), float(obs + 1.96 * se))}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


CONTRASTS = {"D_KV": ("kvzip_bern", "kvzip"), "D_CP": ("compactor_bern", "compactor"),
             "D2_bias": ("kvzip_bern_s1", "kvzip_bern_nobias_s1"), "D3_kv_vs_cp": ("kvzip_bern", "compactor_bern")}


def contrast(name, task, comp):
    a, b = CONTRASTS[name]
    if name == "D2_bias":                                           # seed 1 only, same draws
        A = [r for r in get(task, "kvzip_bern", comp) if r["seed"] == 1]
        B = get(task, "kvzip_bern_nobias", comp)
        if not A or not B:
            return None
        pa, pb = A[0]["pred"].set_index("key"), B[0]["pred"].set_index("key")
        d = pa.s - pb.s.reindex(pa.index)
        return d.groupby(pa.cluster).mean().to_numpy()
    return cluster_diffs(task, a, b, comp)


# ------------------------------------------------------------------ cells
cells = []
for comp in COMPS:
    for task in TASKS:
        for name in CONTRASTS:
            x = contrast(name, task, comp)
            if x is not None and len(x):
                cells.append({"comp": comp, "task": task, "bench": BENCH[task], "contrast": name, **one_task(x), "_x": x})
C = pd.DataFrame(cells)
if len(C):
    C.drop(columns="_x").to_csv(OUT / "cells.csv", index=False)


def xs(name, comp, bench=None):
    rows = [c for c in cells if c["contrast"] == name and c["comp"] == comp and (bench is None or c["bench"] == bench)]
    return rows


def pool(name, comp, bench=None):
    rows = xs(name, comp, bench)
    return (pooled([r["_x"] for r in rows]), len(rows)) if rows else (None, 0)


# ------------------------------------------------------------------ report
ARMS = ["kvzip", "kvzip_bern", "compactor", "compactor_bern"]
EXPECTED = [("no_press", "full")] + [(p, c) for c in COMPS for p in
            ("kvzip", "kvzip_bern_s1", "kvzip_bern_s2", "kvzip_bern_s3",
             "compactor", "compactor_bern_s1", "compactor_bern_s2", "compactor_bern_s3")] + [("kvzip_bern_nobias_s1", "8x")]


def plural(k, word="task"):
    return f"{k} {word}" + ("" if k == 1 else "s")


def missing_runs():
    have = {(r["task"], r["press"], r["comp"]) for r in RUNS}
    return [(t, p, c) for t in TASKS for p, c in EXPECTED if (t, p, c) not in have]


def arm_mean(task, arm, comp):
    """(mean, SD across seeds with n - 1, number of seeds); SD is 0 for single-run arms."""
    s = [r["score"] for r in get(task, arm, comp)]
    return (float(np.mean(s)), float(np.std(s, ddof=1)) if len(s) > 1 else 0.0, len(s)) if s else None


def fmt_score(m):
    if m is None:
        return "-"
    mean, sd, n = m
    return f"{mean:.1f}" + (f" ± {sd:.1f}" if n > 1 else "")


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def repetitive(ans):
    w = str(ans).split()
    g = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return bool(g) and max(g.count(x) for x in set(g)) >= 3


def build_report():
    L = []
    w = L.append
    miss = missing_runs()
    complete = not miss

    # ---------------- criteria first (needed for the verdict at the top)
    P = {(comp, n): pool(n, comp) for comp in ("8x", "4x") for n in ("D_KV", "D_CP")}
    keys = [k for k in P if P[k][0]]
    for k, pa in zip(keys, holm([P[k][0]["p_one"] for k in keys])):
        P[k][0]["p_holm"] = pa
    crit = {}
    for n, cid in (("D_KV", "C1"), ("D_CP", "C2")):
        p8, k = P[("8x", n)]
        crit[cid] = None if k < 8 else bool(p8["diff"] > 0 and p8["p_one"] < 0.05)
    c3 = {(n, b): (sum(r["diff"] >= 0 for r in xs(n, "8x", b)), len(xs(n, "8x", b)))
          for n in ("D_KV", "D_CP") for b in ("LongBench", "SCBench")}
    crit["C3"] = None if any(v[1] < 4 for v in c3.values()) else all(v[0] >= 3 for v in c3.values())
    status = lambda c: "PASS" if crit[c] else "FAIL"
    c3_detail = ", ".join(f"{n} {b} {v[0]}/{v[1]}" for (n, b), v in c3.items())
    if None in (crit["C1"], crit["C2"], crit["C3"]):
        verdict = "INCOMPLETE: not every task has every arm at 8x yet."
    elif crit["C1"] and crit["C2"] and crit["C3"]:
        verdict = "**Sampling helps both scorers** (C1, C2 and C3 pass)."
    elif crit["C1"] != crit["C2"]:
        who = "KVzip" if crit["C1"] else "Compactor"
        verdict = (f"**Sampling helps one scorer: {who}** (C1 {status('C1')}, C2 {status('C2')}, "
                   f"C3 {status('C3')}; C3 counts: {c3_detail}).")
    elif crit["C1"] and crit["C2"]:
        verdict = ("**Pre-registered label: no consistent help.** On average sampling helps both scorers at 8x "
                   "(C1 PASS, C2 PASS), but not on at least 3 of 4 tasks in each benchmark "
                   f"(C3 FAIL; counts of tasks with a non-negative effect: {c3_detail}).")
    else:
        verdict = f"**No consistent help** (C1 {status('C1')}, C2 {status('C2')}, C3 {status('C3')})."

    w("# Experiment 4: does Bernoulli sampling help both scorers?\n")
    w("Llama-3.1-8B-Instruct, kvpress v0.5.4. LongBench: 4 tasks x 50 questions (E8's subset). SCBench: "
      "4 tasks x 10 untruncated contexts x 5 turns (E3's set; qa_eng has 53 questions). Scores are 0-100 per "
      "task metric. Bernoulli arms average 3 seeds per question; in the score tables, `± x` after a Bernoulli score is the standard deviation across its 3 seeds (n - 1), not an error bar. Paired "
      "differences are clustered (SCBench: context, LongBench: question); `p` is a one-sided sign-flip test "
      "of 'sampling helps', unadjusted unless marked Holm. Definitions: `DESIGN.md`, section "
      "'Measurement breakdown'.\n")
    w(f"**Verdict:** {verdict}\n")
    w(f"**Completeness:** {len(RUNS)} runs loaded, "
      + ("all 272 expected runs present." if complete else f"**{len(miss)} of 272 expected runs missing** (listed under C0)."))
    w("")
    w("*Memory is counted, not freed:* every Bernoulli arm (and kvzip, via kvpress's head-wise masking) keeps "
      "dropped tokens in GPU memory behind a -inf bias, so \"8x\" means the model behaves as if 7/8 of the "
      "context cache were gone. Only plain `compactor` actually prunes. Wall-clock and memory savings are not "
      "measured here.\n")

    # ---------------- key numbers
    p8kv, p8cp = P[("8x", "D_KV")][0], P[("8x", "D_CP")][0]
    p2, _ = pool("D2_bias", "8x")
    p3, _ = pool("D3_kv_vs_cp", "8x")
    w("## Key numbers (8x, pooled over all 8 tasks)\n")
    w("| quantity | effect (points) | one-sided p |")
    w("|---|---|---|")
    if p8kv:
        w(f"| sampling vs top-k on KVzip scores (D_KV) | {p8kv['diff']:+.2f} ± {p8kv['se']:.2f} | {fmt_p(p8kv['p_one'])} (Holm {fmt_p(p8kv['p_holm'])}) |")
    if p8cp:
        w(f"| sampling vs top-k on Compactor scores (D_CP) | {p8cp['diff']:+.2f} ± {p8cp['se']:.2f} | {fmt_p(p8cp['p_one'])} (Holm {fmt_p(p8cp['p_holm'])}) |")
    if p2:
        w(f"| log(1/r) correction on vs off (D2, seed 1) | {p2['diff']:+.2f} ± {p2['se']:.2f} | {fmt_p(p2['p_one'])} |")
    if p3:
        w(f"| KVzip scores vs Compactor scores, both sampled (D3) | {p3['diff']:+.2f} ± {p3['se']:.2f} | two-sided {fmt_p(min(1, 2 * min(p3['p_one'], 1 - p3['p_one'])))} |")
    w("")

    # ---------------- setup
    w("## Setup\n")
    w("| benchmark | task | metric | contexts | questions | context tokens, median [min, max] | GPU (new arms) | GPU (reused E3 arms) |")
    w("|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        npv = get(t, "no_press", "full")
        nq = len(npv[0]["pred"]) if npv else 0
        nc = npv[0]["pred"].cluster.nunique() if npv else 0
        lens = []
        for r in get(t, "compactor_bern", "8x")[:1]:
            lens = [m["n"] for m in (r["mem"] or [])]
        ln = f"{int(np.median(lens)):,} [{min(lens):,}, {max(lens):,}]" if lens else "-"
        new_gpu = sorted({str(r["cost"].get("gpu", "")).replace("NVIDIA ", "") for r in RUNS
                          if r["task"] == t and r["source"] == "E4" and r["cost"]})
        old_gpu = sorted({str(r["cost"].get("gpu", "")).replace("NVIDIA ", "") for r in RUNS
                          if r["task"] == t and r["source"] == "E3" and r["cost"]})
        w(f"| {BENCH[t]} | `{t}` | {METRIC_NAME[t]} | {nc} | {nq} | {ln} | {', '.join(new_gpu) or '-'} | {', '.join(old_gpu) or '-'} |")
    w("\nContext tokens are Llama-3.1 tokens of each compressed context (from the Bernoulli arms' memory logs). "
      "LongBench has one question per context; SCBench answers all of a context's turns from one compression.\n")

    # ---------------- figures
    w("## Figures\n")
    w("Generated by `plots.py` into `out/figures/`.\n")
    for fn, cap in (("0_headline", "**The headline**, 2 x 2: rows = benchmark, columns = scorer. In each panel, blue solid = "
                                   "Bernoulli sampling, orange dashed = top-k on the same scores, dotted = full cache; values "
                                   "printed at each point. The blue band is ± 1 standard deviation across the 3 sampling "
                                   "seeds, not an error bar."),
                    ("1_effects", "The controlled comparisons: sampling minus top-k on the same scores, per task, at 4x (light "
                                  "blue) and 8x (dark blue), with 95% clustered CIs. Shaded rows are SCBench."),
                    ("2_tasks", "Score vs compression for each task: blue = KVzip scores, orange = Compactor scores; solid = "
                                "sampling (band ± 1 SD across seeds), dashed with hollow markers = top-k; dotted = full cache."),
                    ("3_nobias", "8x, seed 1: the same Bernoulli draws with and without the log(1/r) correction, per task, "
                                 "with 95% clustered CIs.")):
        if (OUT / "figures" / f"{fn}.png").exists():
            w(f"![{fn}](figures/{fn}.png)\n")
            w(f"*{cap}*\n")

    # ---------------- headline table
    w("## Benchmark averages\n")
    w("Unweighted mean over the benchmark's 4 tasks of each task score. Only rows where every task has the arm.\n")
    for bench, tasks in (("LongBench", LB_TASKS), ("SCBench", SC_TASKS)):
        nps = [get(t, "no_press", "full") for t in tasks]
        w(f"### {bench}" + (f" (full cache = {np.mean([x[0]['score'] for x in nps]):.1f})" if all(nps) else "") + "\n")
        w("| arm | " + " | ".join(COMPS) + " |")
        w("|---|" + "---|" * len(COMPS))
        for a in ARMS:
            row = []
            for c in COMPS:
                ms = [arm_mean(t, a, c) for t in tasks]
                row.append("-" if None in ms else f"{np.mean([m[0] for m in ms]):.1f}")
            w(f"| `{a}` | " + " | ".join(row) + " |")
        w("")

    # ---------------- per task
    w("## Scores per task\n")
    for t in TASKS:
        npv = get(t, "no_press", "full")
        w(f"### {BENCH[t]}: `{t}`" + (f" (full cache = {npv[0]['score']:.1f})" if npv else " (no no_press run)") + "\n")
        w("| arm | " + " | ".join(COMPS) + " |")
        w("|---|" + "---|" * len(COMPS))
        for a in ARMS + ["kvzip_bern_nobias"]:
            row = [fmt_score(arm_mean(t, a, c)) for c in COMPS]
            if a == "kvzip_bern_nobias" and all(v == "-" for v in row):
                continue
            w(f"| `{a}` | " + " | ".join(row) + " |")
        w("")

    # ---------------- effects
    w("## Sampling effect per task\n")
    w("Paired difference in points, ± clustered SE, one-sided p (unadjusted). Positive = sampling beats top-k "
      "on the same scores.\n")
    for n, title in (("D_KV", "D_KV = kvzip_bern - kvzip"), ("D_CP", "D_CP = compactor_bern - compactor")):
        w(f"### {title}\n")
        w("| task | " + " | ".join(COMPS) + " |")
        w("|---|" + "---|" * len(COMPS))
        for t in TASKS:
            row = []
            for c in COMPS:
                m = [x for x in cells if x["contrast"] == n and x["task"] == t and x["comp"] == c]
                row.append(f"{m[0]['diff']:+.2f} ± {m[0]['se']:.2f} (p={m[0]['p_one']:.3f})" if m else "-")
            w(f"| {t} | " + " | ".join(row) + " |")
        w("")

    # ---------------- criteria
    w("## Criteria (pre-registered)\n")
    for n, cid in (("D_KV", "C1"), ("D_CP", "C2")):
        p8, k = P[("8x", n)]
        res = {None: "INCOMPLETE", True: "**PASS**", False: "**FAIL**"}[crit[cid]]
        w(f"- **{cid}**, {n} at 8x pooled over {plural(k)}: "
          + (f"{p8['diff']:+.2f} ± {p8['se']:.2f}, p = {fmt_p(p8['p_one'])} -> " if p8 else "") + res)
    res = {None: "INCOMPLETE", True: "**PASS**", False: "**FAIL**"}[crit["C3"]]
    w("- **C3**, 8x, >= 0 on at least 3 of 4 tasks per benchmark: "
      + "; ".join(f"{n} {b} {v[0]}/{v[1]}" for (n, b), v in c3.items()) + f" -> {res}")
    w("- **C4**, pooled tests with Holm over {4x, 8x} x {KV, CP}:")
    for (comp, n) in (("8x", "D_KV"), ("8x", "D_CP"), ("4x", "D_KV"), ("4x", "D_CP")):
        p, k = P[(comp, n)]
        if p:
            w(f"  - {comp} {n}: {p['diff']:+.2f} ± {p['se']:.2f}, p = {fmt_p(p['p_one'])}, Holm {fmt_p(p['p_holm'])} ({plural(k)})")
    for n in ("D_KV", "D_CP"):
        p2, k = pool(n, "2x")
        if p2:
            ok = ("PASS" if p2["ci95"][0] > -2 else "FAIL") if k == 8 else "INCOMPLETE"
            w(f"  - 2x {n} (no harm if the 95% CI stays above -2): {p2['diff']:+.2f}, "
              f"CI [{p2['ci95'][0]:+.2f}, {p2['ci95'][1]:+.2f}] ({plural(k)}) -> {ok}")
    for n in ("D_KV", "D_CP"):
        p16, k = pool(n, "16x")
        if p16:
            w(f"  - 16x {n} (reported only): {p16['diff']:+.2f} ± {p16['se']:.2f}, p = {fmt_p(p16['p_one'])} ({plural(k)})")
    w(f"\n**Verdict:** {verdict}\n")

    # ---------------- descriptive
    w("## Descriptive (no pass/fail)\n")
    w("**D1, by benchmark** (pooled paired effect, ± SE, one-sided p):\n")
    w("| effect | ratio | LongBench | SCBench |")
    w("|---|---|---|---|")
    for n in ("D_KV", "D_CP"):
        for c in COMPS:
            row = []
            for b in ("LongBench", "SCBench"):
                p, k = pool(n, c, b)
                row.append(f"{p['diff']:+.2f} ± {p['se']:.2f} (p={fmt_p(p['p_one'])}, {plural(k)})" if p else "-")
            w(f"| {n} | {c} | " + " | ".join(row) + " |")
    w("")
    p, k = pool("D2_bias", "8x")
    if p:
        w(f"**D2, is the log(1/r) correction what helps?** kvzip_bern_s1 - kvzip_bern_nobias_s1 at 8x (same draws), "
          f"pooled over {plural(k)}: {p['diff']:+.2f} ± {p['se']:.2f}, one-sided p = {fmt_p(p['p_one'])}.\n")
        w("| task | with - without correction |")
        w("|---|---|")
        for x in xs("D2_bias", "8x"):
            w(f"| {x['task']} | {x['diff']:+.2f} ± {x['se']:.2f} (p={x['p_one']:.3f}) |")
        w("")
    w("**D3, which scorer is better under sampling?** kvzip_bern - compactor_bern, pooled over tasks "
      "(two-sided p):\n")
    for c in COMPS:
        p, k = pool("D3_kv_vs_cp", c)
        if p:
            w(f"- {c}: {p['diff']:+.2f} ± {p['se']:.2f}, p ~ {fmt_p(min(1, 2 * min(p['p_one'], 1 - p['p_one'])))} ({plural(k)})")
    w("")
    w("**D4, each arm against the full cache** (benchmark mean of task score minus no_press, points):\n")
    w("| benchmark | arm | " + " | ".join(COMPS) + " |")
    w("|---|---|" + "---|" * len(COMPS))
    for bench, tasks in (("LongBench", LB_TASKS), ("SCBench", SC_TASKS)):
        for a in ARMS:
            row = []
            for c in COMPS:
                d = []
                for t in tasks:
                    m, npv = arm_mean(t, a, c), get(t, "no_press", "full")
                    if m is None or not npv:
                        break
                    d.append(m[0] - npv[0]["score"])
                row.append(f"{np.mean(d):+.1f}" if len(d) == len(tasks) else "-")
            w(f"| {bench} | `{a}` | " + " | ".join(row) + " |")
    w("")

    # ---------------- memory accounting
    w("## Memory accounting\n")
    w("Realised kept fraction of the context cache, per compressed context (mean [min, max] over all contexts "
      "and seeds). Top-k arms keep exactly the target by construction. KVzip-Bernoulli splits the budget "
      "globally (the total is on target, per head it varies); Compactor-Bernoulli per head.\n")
    w("| benchmark | arm | 2x (0.5) | 4x (0.25) | 8x (0.125) | 16x (0.0625) |")
    w("|---|---|---|---|---|---|")
    for bench in ("LongBench", "SCBench"):
        for a in ("kvzip_bern", "compactor_bern", "kvzip_bern_nobias"):
            row = []
            for c in COMPS:
                k = [m["kept_frac"] for r in RUNS if r["bench"] == bench and r["arm"] == a and r["comp"] == c
                     for m in (r["mem"] or [])]
                row.append(f"{np.mean(k):.4f} [{min(k):.3f}, {max(k):.3f}]" if k else "-")
            w(f"| {bench} | `{a}` | " + " | ".join(row) + " |")
    w("")

    # ---------------- cost
    w("## Cost\n")
    w("Wall-clock seconds per run (one run = all questions of a task at one ratio) and peak GPU memory, "
      "mean over the run's ratios and seeds. Reused E3 runs are included for SCBench.\n")
    w("| task | GPU | " + " | ".join(["no_press"] + ARMS + ["kvzip_bern_nobias"]) + " |")
    w("|---|---|" + "---|" * (len(ARMS) + 2))
    tot_new = tot_old = 0.0
    for t in TASKS:
        row = []
        for a in ["no_press"] + ARMS + ["kvzip_bern_nobias"]:
            rs = [r for r in RUNS if r["task"] == t and r["arm"] == a and r["cost"]]
            row.append(f"{np.mean([r['cost']['seconds'] for r in rs]):.0f} s / {max(r['cost']['peak_gpu_gb'] for r in rs):.0f} GB"
                       if rs else "-")
        gpus = sorted({str(r["cost"].get("gpu", "")).replace("NVIDIA ", "").split("-")[0] for r in RUNS
                       if r["task"] == t and r["cost"]})
        w(f"| {t} | {'/'.join(gpus)} | " + " | ".join(row) + " |")
    for r in RUNS:
        if r["cost"]:
            if r["source"] == "E4":
                tot_new += r["cost"]["seconds"]
            else:
                tot_old += r["cost"]["seconds"]
    w(f"\nTotal GPU time in runs: **{tot_new / 3600:.1f} h new** (E4) + {tot_old / 3600:.1f} h reused from E3 "
      "(excludes model loading and self-tests, a few minutes per job). Bernoulli runs are slower than top-k "
      "because the SDPA bias path builds a full bias mask at every decoding step.\n")

    # ---------------- C0
    w("## C0: harness checks\n")
    w("- **Completeness:** " + ("all 272 expected runs present." if complete else
      f"{len(miss)} missing: " + ", ".join(f"{t}/{p}/{c}" for t, p, c in miss)))
    bad_mem = []
    for r in RUNS:
        if r["mem"] and r["comp"] != "full":
            tgt = {"2x": 0.5, "4x": 0.25, "8x": 0.125, "16x": 0.0625}[r["comp"]]
            kf = float(np.mean([m["kept_frac"] for m in r["mem"]]))
            if abs(kf - tgt) > 0.01:
                bad_mem.append(f"{r['task']} {r['press']} {r['comp']}: {kf:.4f}")
    n_mem = sum(1 for r in RUNS if r["mem"])
    w(f"- **Kept fraction** ({n_mem} Bernoulli runs): " + ("all within 1% of target." if not bad_mem else "OUT OF RANGE: " + "; ".join(bad_mem)))
    bad_gpu = sorted({f"{r['task']} {r['press']} {r['comp']}: {r['cost'].get('gpu')}" for r in RUNS
                      if r["cost"] and (GPU_NEW_SC if (r["bench"] == "SCBench" and r["source"] == "E4") else GPU[r["task"]])
                      not in str(r["cost"].get("gpu"))})
    w("- **GPU types:** " + ("as planned (LongBench A40; new SCBench arms A100; reused E3 arms on E3's GPUs)."
                             if not bad_gpu else "MISMATCH: " + "; ".join(bad_gpu)))
    fb = {}
    for r in RUNS:
        v = r["cost"].get("cholesky_fallbacks") if r["cost"] else None
        if v:
            fb[r["task"]] = fb.get(r["task"], 0) + v
    w("- **Compactor Cholesky fallback** (layer calls rescued, summed over runs): "
      + (", ".join(f"{t} {v}" for t, v in fb.items()) if fb else "never used") + ".")
    tests = []
    for sj in sorted((HERE / "results").glob("*/selftest*.json")):
        tests.append(f"{sj.parent.name}/{sj.stem}: {'pass' if json.load(open(sj)).get('pass') else 'FAIL'}")
    w("- **Self-tests** (last job per group): " + "; ".join(tests))
    w("- **Repetitive answers**: share of answers containing a 3-word phrase 3+ times, all ratios pooled. "
      "Long summaries repeat phrases naturally, so compare each arm with no_press on the same task:\n")
    w("| task | no_press | " + " | ".join(ARMS) + " |")
    w("|---|---|" + "---|" * len(ARMS))
    for t in TASKS:
        row = []
        for a in ["no_press"] + ARMS:
            ans = [x for r in RUNS if r["task"] == t and r["arm"] == a for x in r["pred"].predicted_answer]
            row.append(f"{100 * np.mean([repetitive(x) for x in ans]):.0f}%" if ans else "-")
        w(f"| {t} | " + " | ".join(row) + " |")
    w("\n**LongBench reproduction against E8** (same 50 samples, same GPU type; identical answers):\n")
    e8 = HOME / "e8/results/dev"
    for t in LB_TASKS[:3]:
        for arm, comp, ratio in (("no_press", "full", "0.00"), ("kvzip", "2x", "0.50"), ("kvzip", "4x", "0.75"), ("kvzip", "8x", "0.88")):
            mine = [r for r in RUNS if r["task"] == t and r["press"] == arm and r["comp"] == comp]
            old = e8 / f"longbench__{t}__meta-llama--Llama-3.1-8B-Instruct__{arm}__{ratio}__fraction0.250/predictions.csv"
            if not mine or not old.exists():
                continue
            o = pd.read_csv(old).set_index("_id").predicted_answer.fillna("")
            m = mine[0]["pred"].set_index("_id").predicted_answer.fillna("")
            same = (m == o.reindex(m.index)).mean()
            eo = json.load(open(old.parent / "metrics.json"))
            w(f"- {t} {arm} {comp}: {same * 100:.0f}% identical, score {mine[0]['score']:.2f} vs E8 {float(eo):.2f}"
              + ("" if same >= 0.9 else " **(below 90%)**"))

    w("\n**vt reproduction inside E3** (8x arms run twice, hours apart, by jobs 311762 and 312032: `S1n10`, "
      "used here, against `S1`):\n")
    s1 = run_folders(HOME / "phase2/experiment3/results/S1")
    for name, d in sorted(s1.items()):
        if "__vt__" not in name:
            continue
        press = name.split("__")[3]
        other = run_folders(HOME / "phase2/experiment3/results/S1n10").get(name)
        if other is None:
            continue
        a = pd.read_csv(d / "predictions.csv").set_index(["context_id", "turn"]).predicted_answer.fillna("")
        b = pd.read_csv(other / "predictions.csv").set_index(["context_id", "turn"]).predicted_answer.fillna("")
        w(f"- {press} {name.split('__')[4]}: {(a == b.reindex(a.index)).mean() * 100:.0f}% identical")
    return "\n".join(L) + "\n"


def write_report():
    text = build_report()
    (OUT / "REPORT.md").write_text(text)
    return text


if __name__ == "__main__":
    print(write_report())
