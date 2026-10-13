# Experiment 4: result audit

**Date:** 2026-09-17. **Scope:** `DESIGN.md`, `CONCLUSION.md`, `out/REPORT.md`, figures, all code on the run path
(`compactor_bern.py`, `run_sweep.py`, `data.py`, `selftests4.py`, `summarize.py`, E2/E3 presses, kvpress v0.5.4
Compactor/Leverage/NonCausalAttn/ScorerPress/BasePress), all 272 runs, job logs and `sacct`.
**Scripts:** `audit_results/` (`a0_load.py` first, about 5 min; then `a1`-`a5`, each under a minute).

**Note:** `summarize.py`, `plots.py`, `CONCLUSION.md`, `REPORT.md` and the figures were being edited by another
session during this audit (11:30-11:46). The audit read and did not modify any of them. The only changes were
presentation (seed SD instead of range) and Monte Carlo noise in p-values.

## Summary

| claim | verdict |
|---|---|
| C1: sampling helps KVzip at 8x (+4.90 ± 1.38, p 0.0005) | **Holds.** Robust to dropping any task, to each single seed, and to re-clustering. |
| KVzip gain concentrated at ~110k (SCBench +8.2 at 8x, +7.9 at 4x) | **Holds** (numbers reproduce). |
| C2: sampling helps Compactor at 8x (+2.49, p 0.033) | **Fragile.** One task carries it. Its pass/fail status, and with it the verdict label, depends on analysis choices. |
| C3 fails, label "no consistent help" | Holds as computed, but the label is **not stable** (see 2). |
| D2: the log(1/r) correction is worth +2.8 (p 0.027), "same draws" | **Not defensible as stated.** The draws are not the same on the task that drives it. |
| D3: KVzip scores beat Compactor scores by 4-10 points | Holds. |
| vt: "top-k answers degenerate 60-82%" | True as pooled, but **misleading**: ratio-dependent. |
| Harness (C0) | Mostly holds. Two statements are overstated (see 6). |

## What was verified and holds

- **Reproducibility.**
  - Re-running `summarize.py` on the saved runs gives a **byte-identical `cells.csv`**.
  - All 272 per-question rescorings match `metrics.json` (the script asserts this).
  - The regenerated report differs from the published one only in formatting and in the 4th digit of
    Monte Carlo p-values.
- **Run folders.**
  - E3's `S1n10` has two crashed-then-rerun folders (vt kvzip 8x and 16x). In both, the base folder is empty and
    `/1` is complete.
  - `run_folders` picks the correct one.
  - No duplicate runs exist in E4.
- **Configs.** E3 and E4 `config.yaml` files are identical except for press name and output dir: same
  `max_new_tokens` (dataset default), seed 42, no truncation. vt's 16x/4x/2x arms from job 312108 also match.
- **Same contexts.** For every SCBench context, E3's `kvzip_bern` and E4's `compactor_bern` logged the **same
  token length**. Question keys match (the script asserts this too).
- **E8 reproduction.** 10 of 12 LongBench runs are identical to E8, and 2 are 98% identical.
- **Pairing within the selection pairs.**
  - Pairing is sound by construction: kvzip/kvzip_bern read the same score-cache file.
  - compactor/compactor_bern use the same `(context crc, layer)` sketch seed through the same
    `SeededCompactorPress.score`.
  - Each pair ran on one GPU type.
- **Budgets.** Kept fractions are on target in all 200 Bernoulli runs. `keep_probabilities` is correct:
  - per-head bisection;
  - forced tokens paid from the same budget;
  - kvpress's sink padding with the max does not affect the per-head minimum.
- **Self-tests.** Every recorded T7-T10 value is sensible:
  - mask vs prune: 1.0% relative difference, argmax equal;
  - budget error: 0;
  - the fallback path agrees with stock to 3e-6.
- **Cholesky fallback.** Used 10 times per vt run (1 layer per context x 10 contexts x 16 Compactor-family runs
  = 160). It was never used on other tasks.

## Problems, most serious first

### 1. D2 is not a same-draw comparison on prefix_suffix and summary (`a1`, `a3`)

On SCBench:
- `kvzip_bern_s1` is **reused from E3**, which ran prefix_suffix and summary on **L40S**;
- `kvzip_bern_nobias_s1` is **new, on A100**.

The E3 audit already found that the same seed gives different draws on the two GPU types. Measured here:

| task | GPUs (bias / no bias) | solved `c` equal | kept_frac identical | identical answers |
|---|---|---|---|---|
| qasper, hotpotqa, passage_retrieval_en, gov_report | A40 / A40 | yes | 45/45, 50/50, 50/50, 50/50 | 20-70% (gov_report 0%) |
| qa_eng, vt | A100 / A100 | yes | 10/10, 10/10 | 11%, 34% |
| **prefix_suffix** | **L40S / A100** | **no** | **0/10** | 0% |
| **summary** | **L40S / A100** | **no** | **0/10** | 0% |

The pooled D2 hinges on exactly the unpaired task:

| D2 at 8x (with − without correction) | effect | one-sided p |
|---|---|---|
| as reported, 8 tasks | +2.80 ± 1.31 | 0.027 |
| leave out prefix_suffix | +0.91 ± 1.32 | 0.25 |
| **6 tasks with truly identical draws** | **+1.00 ± 1.49** | **0.26** |
| of which LongBench (4) | −1.39 ± 1.77 | 0.78 |
| of which qa_eng + vt | +5.78 ± 2.72 | 0.018 (uncorrected, 2 tasks) |

**What is defensible:**
- **prefix_suffix.** The single no-correction draw scored 0 in all 10 contexts. The three corrected seeds hit in
  6-7 contexts each (8-10 hits of 50; no_press 10). That is a large effect, but it is an unpaired, single-draw
  comparison across GPU types.
- **On long contexts where the pairing is valid** (qa_eng, vt), the correction helps by about 6 points.
- **On LongBench**, no benefit is detectable.

**Must change:**
- CONCLUSION Finding 4 ("same draws", "+2.80, p = 0.027", "answers part of the question the E3 audit left
  open");
- the Trust/GPU paragraph ("Each selection pair is on one GPU type. Only the KVzip-vs-Compactor and vs-no_press
  comparisons mix types"): D2 also mixes GPU types;
- the `3_nobias.png` title ("same draws");
- the REPORT's D2 heading.

**Cheap fix:** rerun `kvzip_bern_nobias_s1` at 8x for prefix_suffix and summary **on L40S**. E3's `kvzip_bern` peaked
at 44.2-44.4 GB there, so it fits. Then check that `c` and kept_frac equal E3's before using it. Two runs, about
20 min each.

### 2. C2 rests on one task, and the verdict label flips with analysis choices (`a3`)

| D_CP at 8x | effect | one-sided p |
|---|---|---|
| as reported | +2.49 ± 1.32 | 0.033 |
| leave out passage_retrieval_en | **+0.28 ± 1.19** | **0.41** |
| seed 1 / 2 / 3 alone | +2.48 / +2.63 / +2.37 | 0.059 / 0.043 / 0.080 |
| tasks with a positive effect | 4 of 8 | |

C1, by contrast, stays significant with any task left out (worst: +3.03, p 0.016, without prefix_suffix) and
with every single seed.

**The label is on a knife edge:**
- **The pre-registration is ambiguous.** DESIGN.md says "Holm within each family", and C4 puts 8x
  {KV, CP} in the Holm family. The C2 row, however, states a raw p < 0.05.
  - With Holm, C2 = 0.066 and **fails**.
  - The verdict rule then gives **"helps one scorer: KVzip"**, because that label does not require C3.
- **Clustering qasper correctly (item 3) moves D_CP's Holm p to about 0.048**, and C2 passes again.
- **The verdict rule is not monotone.** "Both pass but C3 fails" gets a *weaker* label ("no consistent help")
  than "only C1 passes" ("helps one scorer").

**Recommendation:** don't let either label carry weight. Say: **KVzip effect established; Compactor effect not
established** (it rests on passage_retrieval_en; p 0.033 raw, 0.066 Holm).

### 3. qasper has shared contexts, so its clusters are wrong (`a2`, `a3`)

- **The data.** qasper's 50 questions come from **45 contexts**. kvpress groups rows by context and answers them
  from one compression, which is also one Bernoulli draw.
- **The design is wrong here.** DESIGN.md says "Each row has its own context and one question".
  `summarize.py` clusters LongBench by `_id`. The other three LongBench tasks do have 50 distinct contexts.
- **Effect of clustering by context:**
  - qasper D_KV at 8x: −0.43 → −0.59;
  - **qasper D_CP at 8x: −1.14 → +0.40 (the sign flips)**, so C3's Compactor LongBench count becomes 3/4;
  - pooled D_CP at 8x: p 0.033 → 0.024.
- **Consequence.** C3 still fails, because the KVzip LongBench count stays at 2/4.
- **Fix.** In `summarize.load`, cluster LongBench by a hash of the context (from the parquet), not by `_id`.

### 4. C3's "non-negative" count includes a floor tie

prefix_suffix D_CP = 0.00 counts as non-negative, but both Compactor arms score 0 at every ratio (top-k scores 2.0
at 2x). So SCBench's "3/4" is really 2 of 3 informative tasks. This does not change the verdict.

### 5. vt: the degeneracy finding needs its ratio split, and the metric can be gamed (`a4`)

Share of degenerate answers (a 3-word phrase repeated 3+ times):

| arm | full | 2x | 4x | 8x | 16x |
|---|---|---|---|---|---|
| no_press | 34% | | | | |
| kvzip (top-k) | | 36% | 32% | **76%** | **94%** |
| kvzip_bern | | 37% | 35% | 31% | 33% |
| compactor (top-k) | | 82% | 80% | 94% | 72% |
| compactor_bern | | 38% | 38% | 29% | 36% |
| kvzip_bern_nobias | | | | 56% | |

**KVzip top-k.** It is as well-formed as the full cache at 2x and 4x. "Top-k makes vt degenerate" is true for
KVzip only at 8x and above. For Compactor it holds at every ratio.

**The vt metric** ("fraction of gold variables found") does not penalise extra names:
- Compactor's 16x answers are often chains such as `VAR A = VAR B = VAR C ...`, which score up to 80.
- That explains its 16x score (33.2, above its own 2x-8x).
- It therefore also drives vt D_CP 16x = −10.9, and through it "Compactor turns negative at 16x on SCBench":
  −3.46 pooled, but **−0.97 without vt**.
- CONCLUSION calls this "partly an artefact". The mechanism is now identified.

### 6. Harness statements that are overstated

**"Every job passed T1-T10" is not true for `lb_A` (312116) and `lb_B` (312117).**
- Both started at 01:56:23. `selftests4.py` gained T10 at 02:08:22, so their logs show T1-T9 only.
- `results/longbench/selftest_compactor.json` shows T10 only because `lb_B2` (started 02:35) overwrote it.
- The "last job per group" line in C0 hides this.
- **Impact is low:**
  - they did import the fallback version of `compactor_bern.py` (edited at 01:57:17; their `config.yaml`
    names `FallbackLeverageScorePress`);
  - the fallback never fired on LongBench.

**Self-tests (and T7's mask = prune check) run on a synthetic 1.9k-token context.** Mask vs prune at 110k
is untested. D_CP compares a pruned top-k with a masked sampler, so a numerical difference between the two paths
is not excluded at SCBench length.

### 7. Factual slips in CONCLUSION.md / DESIGN.md

| text | actual |
|---|---|
| "Bernoulli runs took about 1.5-2x as long as top-k" | compactor_bern / compactor = **1.16-3.93x** (above 2x on gov_report and every SCBench task). kvzip vs kvzip_bern is not like-for-like: the top-k run went first and computed the score cache (E3 vt kvzip 8x: 6,990 s). |
| "KVzip pays ... 2-3x the prefill" | Not measured in E4. E3's probe measured 55 s vs 259 s at 124k (~4.7x). |
| "prefix_suffix scores 0 for Compactor with either rule, at every ratio including 2x" | compactor top-k at 2x = **2.0** (1 hit of 50). |
| D3 "all p < 0.001" | 2x p ≈ 0.0010. |
| DESIGN.md length table (e.g. qa_eng 69,644-130,810) | Stale. The measured contexts are qa_eng 87,882-130,839; REPORT's table is right. |
| "slightly more repetitive answers than plain Compactor on qasper (3% against 2%)" | A handful of answers, pooled over ratios and seeds. Not evidence of anything. |

### 8. Process

- **`summarize.py` still writes `out/cells.csv` on import** (line 217), and `plots.py` imports it. This is the same
  trap that overwrote REPORT.md earlier: move the write into `write_report()`.
- **Pooled p-values are Monte Carlo from a shared RNG stream.** They change with the entry point: D2 reads 0.0264,
  0.0268, 0.0271 and 0.0272 in different places.
  - Harmless at this precision, but give each test its own seeded generator, so a number reads the same in the
    report and the figure.

## Corrected one-paragraph summary (for the professor)

On 8 tasks, Bernoulli sampling on KVzip's scores beats KVzip's top-k by about 5 points at 8x (p 0.0005). The result
holds with any task removed and for each seed alone, and it is concentrated at ~110k tokens (+8). On Compactor's
scores the average gain (+2.5, p 0.033 raw, 0.066 Holm) comes almost entirely from passage_retrieval_en and is not
established. Removing the log(1/r) correction:
- cost everything on prefix_suffix (0/50, but in an unpaired comparison run on a different GPU type);
- cost about 6 points on qa_eng and vt, where the comparison is correctly paired;
- made no detectable difference on LongBench.

A matched rerun of the two unpaired tasks is needed before the correction's pooled effect is quoted.

## Recommended actions

1. Rerun `kvzip_bern_nobias_s1` at 8x for prefix_suffix and summary on L40S. Verify `c` and kept_frac equal E3's,
   then recompute D2.
2. Cluster LongBench by context; re-run `summarize.py`.
3. Rewrite CONCLUSION Finding 4, the GPU paragraph, the C2/verdict wording, the vt degeneracy paragraph and the
   item 7 slips. Retitle `3_nobias.png`.
4. Move the `cells.csv` write out of import; seed each Monte Carlo test separately.
5. Optional: a T7-style mask vs prune check on one 110k context.
