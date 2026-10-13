"""Phase 2, Experiment 2: results -> out/REPORT.md + out/figures/ (B0-B6 verdicts).

    python summarize.py [--groups P1 L40S] [--out out]

Statistics
----------
Every run scores the SAME 780 RULER samples (same fraction, same seed), so arms are compared PAIRED,
sample by sample, not just through their means:
  * per-sample RULER scores are recomputed from predictions.csv with kvpress's own metric, and checked
    against each run's metrics.json (the job fails loudly if they disagree);
  * rows are paired by order, and the question text is asserted equal row by row;
  * a difference is averaged within each subtask, then across the 13 subtasks (the same weighting as
    RULER's headline number), with a stratified standard error sqrt(sum_t var_t / n_t) / T.
Bernoulli arms are first averaged over their 3 seeds per sample, so the paired SE also absorbs seed noise
only through the per-sample mean; the across-seed spread of the headline score is reported separately.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--groups", nargs="*", default=["P1", "L40S"])
ap.add_argument("--out", default="out")
args = ap.parse_args()
OUT = os.path.join(HERE, args.out)
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

COMP = {0.0: "full", 0.5: "2x", 0.75: "4x", 0.88: "8x", 0.875: "8x", 0.94: "16x", 0.9375: "16x"}
ORDER = ("no_press", "kvzip", "kvzip_bern", "kvzip_bern_nobias", "bern_tau", "snapkv", "expected_attention",
         "streaming_llm", "random")
CTRL = re.compile(r"[\x00-\x1f]")


# ---------------------------------------------------------------- RULER per-sample metric (kvpress's)
def parse_refs(s):
    """The answer column is a numpy-array repr, "['a' 'b']". Each element is a Python string literal, with
    quotes escaped inside (e.g. 'Royce da 5\\'9"'), so decode every quoted token with ast.literal_eval.
    A regex that only strips the outer quotes leaves the backslashes in and mis-scores those samples
    (caught by the metrics.json cross-check: qa_2 off by one sample)."""
    import ast
    toks = re.findall(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", str(s))
    return [ast.literal_eval(t) for t in toks] or [str(s)]


def sample_score(task, pred, refs):
    pred = CTRL.sub("", str(pred).strip()).strip().lower() if isinstance(pred, str) else ""
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return 100.0 * (max(hits) if task.split("_")[0] == "qa" else sum(hits) / len(hits))


# ---------------------------------------------------------------- load runs
runs = []
for grp in args.groups:
    root = Path(HERE) / "results" / grp
    if not root.exists():
        continue
    for mj in sorted(root.rglob("metrics.json")):
        d = mj.parent
        name = d.name if not re.fullmatch(r"[0-9]+", d.name) else d.parent.name
        parts = name.split("__")
        press, ratio = parts[3], round(float(parts[4]), 2)
        arm = re.sub(r"_s[123]$", "", press)
        seed = int(press[-1]) if re.search(r"_s[123]$", press) else None
        hw = "L40S" if grp == "L40S" else "A40"
        pr = pd.read_csv(d / "predictions.csv")
        pr["s"] = [sample_score(t, p, parse_refs(a)) for t, p, a in zip(pr.task, pr.predicted_answer, pr.answer)]
        m = json.load(open(mj))
        official = {t: float(np.mean([x for x in v.values()])) if isinstance(v, dict) else float(v) for t, v in m.items()}
        mine = pr.groupby("task").s.mean()
        worst = max(abs(mine[t] - official[t]) for t in official)
        assert worst < 0.02, f"per-sample metric disagrees with metrics.json for {name}: {worst:.3f}"
        mem = json.load(open(d / "memory.json")) if (d / "memory.json").exists() else None
        runs.append({"group": grp, "hw": hw, "press": press, "arm": arm, "seed": seed, "ratio": ratio,
                     "comp": COMP.get(ratio, str(ratio)), "score": float(np.mean(list(official.values()))),
                     "official": official, "pred": pr, "mem": mem})

if not runs:
    raise SystemExit("no results found")
R = pd.DataFrame([{k: v for k, v in r.items() if k not in ("pred", "official", "mem")} for r in runs])
L = []


def w(s=""):
    L.append(s)


def get(arm, comp, hw="A40"):
    return [r for r in runs if r["arm"] == arm and r["comp"] == comp and r["hw"] == hw]


def per_sample(arm, comp, hw="A40", seed=None):
    """Per-sample score vector (mean over seeds unless one is given) + task/question columns."""
    rs = [r for r in get(arm, comp, hw) if seed is None or r["seed"] == seed]
    if not rs:
        return None
    base = rs[0]["pred"][["task", "question"]].reset_index(drop=True)
    for r in rs[1:]:
        assert (r["pred"].question.reset_index(drop=True) == base.question).all(), \
            f"row order differs between seeds of {arm} {comp}"
    base["s"] = np.mean([r["pred"].s.to_numpy() for r in rs], axis=0)
    return base


def paired(a, b):
    """Stratified paired difference a - b (subtask-mean weighting) -> (diff, se, n)."""
    if a is None or b is None:
        return None
    assert len(a) == len(b) and (a.question.to_numpy() == b.question.to_numpy()).all(), "samples not aligned"
    d = pd.DataFrame({"task": a.task, "d": a.s.to_numpy() - b.s.to_numpy()})
    g = d.groupby("task").d
    means, var, n = g.mean(), g.var(ddof=1).fillna(0.0), g.count()
    T = len(means)
    return float(means.mean()), float(math.sqrt((var / n).sum()) / T), int(len(d))


def fmt(p):
    if p is None:
        return "—"
    diff, se, _ = p
    z = diff / se if se > 0 else float("inf") if diff != 0 else 0.0
    star = " **(>2 SE)**" if abs(z) > 2 else ""
    return f"{diff:+.2f} ± {se:.2f}{star}"


def arm_score(arm, comp, hw="A40"):
    rs = get(arm, comp, hw)
    if not rs:
        return None
    s = [r["score"] for r in rs]
    return float(np.mean(s)), (float(np.std(s, ddof=1) / math.sqrt(len(s))) if len(s) > 1 else None), len(s)


# ================================================================ report
w("# Phase 2, Experiment 2 — Bernoulli sampling vs the published methods (RULER 4096)")
w()
nsamp = len(runs[0]["pred"])
w(f"Llama-3.1-8B-Instruct, kvpress v0.5.4, RULER 4096, **{nsamp} samples per run** (13 subtasks). Scores are "
  "RULER's metric (0–100), averaged over subtasks. All arms score the same samples, so comparisons are paired "
  "per sample (± = stratified SE). Bernoulli arms average 3 seeds.")
w()
w("*Memory is counted, not freed:* dropped tokens are masked, so \"8×\" means the model behaves as if 7/8 of "
  "the cache were gone. GPU memory is not reduced.")
w()

comps = [c for c in ("full", "2x", "4x", "8x", "16x") if c in set(R.comp)]
w("## Scores")
w()
w("| arm | GPU | " + " | ".join(comps) + " |")
w("|---|---|" + "---|" * len(comps))
for arm in ORDER:
    for hw in ("A40", "L40S"):
        if not any(r["arm"] == arm and r["hw"] == hw for r in runs):
            continue
        cells = []
        for c in comps:
            s = arm_score(arm, c, hw)
            cells.append("—" if s is None else (f"{s[0]:.2f} (seed SE {s[1]:.2f})" if s[1] is not None else f"{s[0]:.2f}"))
        w(f"| `{arm}` | {hw} | " + " | ".join(cells) + " |")
w()

# ---------------------------------------------------------------- B0 hardware
np_a, np_l = arm_score("no_press", "full", "A40"), arm_score("no_press", "full", "L40S")
b0_hw = None
if np_a and np_l:
    hwd = paired(per_sample("no_press", "full", "L40S"), per_sample("no_press", "full", "A40"))
    b0_hw = abs(hwd[0]) < 1.0
    w("## B0 — is the harness trustworthy?")
    w()
    w(f"- **Hardware check.** `no_press` on L40S vs A40: {fmt(hwd)} points. The P2 baselines ran on L40S; "
      f"{'the hardware difference is negligible, so they can be compared with the A40 arms.' if b0_hw else 'this is NOT negligible, so treat cross-hardware comparisons with caution.'}")
rnd = {c: arm_score("random", c) for c in ("2x", "4x", "8x") if arm_score("random", c)}
if rnd:
    w("- **Random floor:** " + ", ".join(f"{c} {v[0]:.1f}" for c, v in rnd.items()) +
      ". The harness punishes bad token selection.")
    have = [c for c in rnd if arm_score("kvzip_bern", c)]
    if have:
        beats = all(arm_score("kvzip_bern", c)[0] > rnd[c][0] + 10 for c in have)
        w(f"- **B0: {'PASS' if beats else 'FAIL'}** — the Bernoulli arm beats `random` by >10 points at "
          f"{', '.join(have)}" + ("" if len(have) == len(rnd) else " (other ratios not yet run)") + ".")
    else:
        w("- **B0:** no Bernoulli results yet — verdict withheld.")
w()

# ---------------------------------------------------------------- B1
w("## B1 — sampling vs top-k, everything else identical (the central claim)")
w()
w("`kvzip` and `kvzip_bern` share their scores exactly (same chunked repeat pass, same sinks, same budget). "
  "The only difference is how tokens are selected: deterministic top-k, or Bernoulli sampling with a log(1/r) bias.")
w()
w("| compression | kvzip (top-k) | kvzip_bern (sampling, 3 seeds) | paired difference | per-seed differences |")
w("|---|---|---|---|---|")
b1, b1x = {}, {}
for c, hw in (("2x", "A40"), ("4x", "A40"), ("8x", "A40"), ("16x", "L40S")):
    k, b = arm_score("kvzip", c, hw), arm_score("kvzip_bern", c, hw)
    if not (k and b):
        continue
    p = paired(per_sample("kvzip_bern", c, hw), per_sample("kvzip", c, hw))
    seeds = [paired(per_sample("kvzip_bern", c, hw, seed=s), per_sample("kvzip", c, hw)) for s in (1, 2, 3)]
    (b1x if c == "16x" else b1)[c] = p
    label = f"{c} *(exploratory, L40S)*" if c == "16x" else c
    w(f"| {label} | {k[0]:.2f} | {b[0]:.2f} | {fmt(p)} | " + ", ".join(f"{x[0]:+.2f}" for x in seeds if x) + " |")
w()
rep = None
if arm_score("kvzip", "8x", "L40S") and arm_score("kvzip_bern", "8x", "L40S"):
    rep = paired(per_sample("kvzip_bern", "8x", "L40S"), per_sample("kvzip", "8x", "L40S"))
    ka, kb = arm_score("kvzip", "8x", "L40S"), arm_score("kvzip_bern", "8x", "L40S")
    w(f"**Independent-hardware replication of the 8× pair (L40S):** `kvzip` {ka[0]:.2f}, `kvzip_bern` "
      f"{kb[0]:.2f}, difference {fmt(rep)} — against {fmt(b1['8x']) if '8x' in b1 else 'n/a'} on A40.")
    w()
if b1x:
    p = b1x["16x"]
    w(f"*16× was added overnight and is not part of the pre-registered B1 verdict.* Both arms ran on L40S, so the "
      f"pair is within one hardware type. Difference: {fmt(p)}.")
    w()
if b1:
    nonneg = all(p[0] >= -2 * p[1] for p in b1.values())        # not significantly worse anywhere
    wins = [c for c, p in b1.items() if p[1] > 0 and p[0] > 2 * p[1]]
    losses = [c for c, p in b1.items() if p[1] > 0 and p[0] < -2 * p[1]]
    verdict = "PASS" if nonneg else "FAIL"
    w(f"**B1: {verdict}.** Sampling is " + ("significantly better than top-k at " + ", ".join(wins) if wins else
      "not significantly better than top-k at any ratio") +
      ("; significantly worse at " + ", ".join(losses) if losses else "") + ".")
w()

# ---------------------------------------------------------------- B2 / B3
w("## B2 / B3 — against the field")
w()
w("| compression | " + " | ".join(f"`{a}`" for a in ORDER if a not in ("no_press", "random")) + " |")
w("|---|" + "---|" * len([a for a in ORDER if a not in ("no_press", "random")]))
for c in ("2x", "4x", "8x"):
    cells = []
    for a in ORDER:
        if a in ("no_press", "random"):
            continue
        s = arm_score(a, c, "A40") or arm_score(a, c, "L40S")
        cells.append("—" if s is None else f"{s[0]:.2f}")
    w(f"| {c} | " + " | ".join(cells) + " |")
w()
fieldbest = {}
for c in ("2x", "4x"):
    cands = [(a, (arm_score(a, c, "A40") or arm_score(a, c, "L40S"))) for a in ("snapkv", "expected_attention", "streaming_llm")]
    cands = [(a, s) for a, s in cands if s]
    ours = max([s for s in (arm_score("kvzip_bern", c), arm_score("bern_tau", c)) if s], key=lambda s: s[0], default=None)
    if cands and ours:
        best = max(cands, key=lambda x: x[1][0])
        fieldbest[c] = (ours[0], best)
if fieldbest:
    ok = all(o >= b[1][0] - 1 for o, b in fieldbest.values())
    w(f"**B2: {'PASS' if ok else 'FAIL'}** — best Bernoulli arm vs best non-KVzip baseline: " +
      "; ".join(f"{c}: {o:.2f} vs `{b[0]}` {b[1][0]:.2f}" for c, (o, b) in fieldbest.items()) + ".")
npa = np_a[0] if np_a else None
if npa and arm_score("kvzip", "8x") and arm_score("kvzip_bern", "8x"):
    dk, db = npa - arm_score("kvzip", "8x")[0], npa - arm_score("kvzip_bern", "8x")[0]
    w(f"**B3: {'PASS' if db <= dk else 'FAIL'}** — drop from the full cache at 8×: `kvzip` {dk:.2f}, `kvzip_bern` {db:.2f}.")
w()

# ---------------------------------------------------------------- B5 / B6
if any(r["arm"] == "kvzip_bern_nobias" for r in runs):
    w("## B5 — does the log(1/r) bias matter?")
    w()
    w("Same seeds, so the same tokens are kept. The only difference is the Horvitz–Thompson correction.")
    w()
    w("| compression | with bias | without bias | paired difference (bias − no bias) |")
    w("|---|---|---|---|")
    b5 = {}
    for c in ("4x", "8x"):
        a, b = arm_score("kvzip_bern", c), arm_score("kvzip_bern_nobias", c)
        if a and b:
            b5[c] = paired(per_sample("kvzip_bern", c), per_sample("kvzip_bern_nobias", c))
            w(f"| {c} | {a[0]:.2f} | {b[0]:.2f} | {fmt(b5[c])} |")
    w()
    if b5:
        w(f"**B5: {'PASS' if all(p[0] >= -2 * p[1] for p in b5.values()) else 'FAIL'}** — "
          + ("the bias helps significantly at " + ", ".join(c for c, p in b5.items() if p[0] > 2 * p[1])
             if any(p[0] > 2 * p[1] for p in b5.values()) else "no significant effect of the bias on task score") + ".")
    w()

if any(r["arm"] == "bern_tau" for r in runs):
    w("## B6 — chunked vs full-context calibration")
    w()
    w("Both arms sample by Bernoulli with a log(1/r) bias. `kvzip_bern` uses KVzip's chunked (2048-token) "
      "repeat pass; `bern_tau` (E9's press) re-reads the whole context.")
    w()
    w("| compression | kvzip_bern (chunked) | bern_tau (full context) | paired difference (full − chunked) |")
    w("|---|---|---|---|")
    b6 = {}
    for c in ("2x", "4x", "8x"):
        a, b = arm_score("kvzip_bern", c), arm_score("bern_tau", c)
        if a and b:
            b6[c] = paired(per_sample("bern_tau", c), per_sample("kvzip_bern", c))
            w(f"| {c} | {a[0]:.2f} | {b[0]:.2f} | {fmt(b6[c])} |")
    w()
    if b6:
        big = [c for c, p in b6.items() if p[0] > 1 and p[0] > 2 * p[1]]
        w(f"**B6:** full-context calibration " + (f"is worth more than 1 point, significantly, at {', '.join(big)}."
                                                    if big else "is not worth a significant +1 point at any ratio."))
    w()

# ---------------------------------------------------------------- B4 memory
w("## B4 — is the memory accounting honest?")
w()
w("| arm | compression | target kept | realised kept (mean) | worst sample | samples within §11 bound M⁺ |")
w("|---|---|---|---|---|---|")
b4_worst, b4_hp = 0.0, []
for arm in ("kvzip_bern", "kvzip_bern_nobias", "bern_tau"):
    for c in ("2x", "4x", "8x"):
        mems = [r["mem"] for r in get(arm, c) if r["mem"]]
        if not mems:
            continue
        tgt = 1 - {"2x": 0.5, "4x": 0.75, "8x": 0.875}[c]
        kf = np.array([s["kept_frac"] for m in mems for s in m])
        err = abs(kf.mean() - tgt) / tgt
        b4_worst = max(b4_worst, err)
        hp = [s["kept_frac"] <= s["M_hp_frac"] + 1e-12 for m in mems for s in m if "M_hp_frac" in s]
        b4_hp += hp
        w(f"| `{arm}` | {c} | {tgt:.4f} | {kf.mean():.4f} ({100 * (kf.mean() - tgt) / tgt:+.2f}%) | "
          f"{kf.max():.4f} | {(f'{100 * np.mean(hp):.2f}%' if hp else 'n/a (E9 press)')} |")
w()
if b4_hp:
    w(f"**B4: {'PASS' if b4_worst < 0.01 and np.mean(b4_hp) >= 0.95 else 'FAIL'}** — mean kept fraction within "
      f"{100 * b4_worst:.2f}% of target; §11's high-probability bound holds for {100 * np.mean(b4_hp):.2f}% of samples.")
w()
w("Note: the target is set on the whole cache, so expected memory is exact by construction. The realised count "
  "per sample fluctuates, and the §11 bound says by how much.")
w()

# ---------------------------------------------------------------- per-subtask at 8x
w("## Per-subtask scores at 8×")
w()
tasks = sorted(runs[0]["official"].keys())
w("| arm | " + " | ".join(tasks) + " |")
w("|---|" + "---|" * len(tasks))
for arm in ORDER:
    rs = get(arm, "8x") or get(arm, "8x", "L40S")
    if rs:
        w(f"| `{arm}` | " + " | ".join(f"{np.mean([r['official'][t] for r in rs]):.0f}" for t in tasks) + " |")
w()

# ---------------------------------------------------------------- figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = {"2x": 2, "4x": 4, "8x": 8, "16x": 16}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, arm in enumerate(a for a in ORDER if a not in ("no_press", "random")):
        pts = [(xs[c], (arm_score(arm, c) or arm_score(arm, c, "L40S"))) for c in xs]
        pts = [(x, s) for x, s in pts if s]
        if pts:
            ax[0].errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                           yerr=[(p[1][1] or 0) for p in pts], marker="o", capsize=3, label=arm, color=f"C{i}")
    if npa:
        ax[0].axhline(npa, color="k", ls="--", lw=1, label="no_press")
    ax[0].set(xscale="log", xticks=[2, 4, 8, 16], xticklabels=["2×", "4×", "8×", "16×"], xlabel="compression",
              ylabel="RULER 4k score", title="Score vs compression")
    ax[0].legend(fontsize=7)
    allb = {**b1, **b1x}
    cs = [c for c in ("2x", "4x", "8x", "16x") if c in allb]
    ax[1].bar(range(len(cs)), [allb[c][0] for c in cs], yerr=[2 * allb[c][1] for c in cs], capsize=4,
              color=["C2" if c != "16x" else "C7" for c in cs])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xticks=range(len(cs)), xticklabels=[c.replace("x", "×") for c in cs],
              ylabel="kvzip_bern − kvzip (points, ±2 SE)", title="B1: sampling minus top-k, same scores")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "scores.png"), dpi=130)
    w("Figure: `out/figures/scores.png` (score vs compression per arm; B1 paired difference with ±2 SE).")
except Exception as e:                                                   # noqa: BLE001
    w(f"(figure skipped: {e})")

open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L) + "\n")
R.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
print("\n".join(L))
