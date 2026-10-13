# Experiment 4: does Bernoulli sampling help both scorers?

**Status:** 2026-09-17. Complete: all 272 runs finished; results in `CONCLUSION.md` and `out/REPORT.md`. The criteria below were fixed before any Stage 1 result.

## Why

The professor asked for a small, fast experiment that gives an early, consistent signal: LongBench and
SCBench, 3-4 subtasks each, 50 samples per subtask, compared with Compactor, in about 10 hours.

E3 showed that Bernoulli sampling (`r = min(1, c*s)`, bias `log(1/r)`) on KVzip's scores beats KVzip's
own top-k at ~110k tokens. E4 asks whether that is a property of **sampling**, not of KVzip:
- does it help a second, differently built scorer (Compactor)?
- does it hold across 8 tasks and two length bands?

## Arms

| arm | score | selection | budget split | memory |
|---|---|---|---|---|
| `no_press` | | | | full |
| `kvzip` | KVzip (cached, E3) | top-k | global | masked |
| `kvzip_bern_s{1,2,3}` | same bytes | Bernoulli + `log(1/r)` | global `c` | masked |
| `kvzip_bern_nobias_s1` (8x only) | same bytes | same draws as s1, no bias | global `c` | masked |
| `compactor` | Compactor (kvpress defaults) | top-k | per head | pruned |
| `compactor_bern_s{1,2,3}` | same score | Bernoulli + `log(1/r)` | per head `c` | masked |

**`compactor_bern`** (`compactor_bern.py`):
- **Score map (user):** Compactor's scores are z-scores and can be negative, so each head's scores are
  shifted by that head's minimum.
- **Budget (user):** per KV head, like Compactor. Each head keeps `int(n*(1-ratio))` tokens in
  expectation.
- **Forced keeps:** Compactor's protected tokens (first 8, last 4) get `r = 1`, paid out of the same
  budget.
- **Selection is the only difference:** `compactor` vs `compactor_bern` is the same selection-only
  control as `kvzip` vs `kvzip_bern`.
- **Caveat:** "keep probability proportional to the shifted z-score" is our choice, not the Compactor
  paper's. A negative result could come from that map.

## Tasks (50 samples each)

| benchmark | task | family | n | GPU | source of reused arms |
|---|---|---|---|---|---|
| LongBench | qasper | single-doc QA | 50 (fraction 0.25, seed 42, E8's subset) | A40 | none, all new |
| LongBench | hotpotqa | multi-doc QA | 50 | A40 | none |
| LongBench | passage_retrieval_en | retrieval | 50 | A40 | none |
| LongBench | gov_report | summary | 50 | A40 | none |
| SCBench | prefix_suffix | retrieval | 10 ctx x 5 turns | A100 new arms (E3 arms: L40S) | E3 `S1n10` |
| SCBench | qa_eng | QA | 10 ctx x 5 (53 rows) | A100 | E3 `S1n10` |
| SCBench | summary | summary | 10 ctx x 5 | A100 new arms (E3 arms: L40S) | E3 `S1n10` |
| SCBench | vt | multi-hop tracking | 10 ctx x 5 | A100 | E3 `S1n10` (job 312108) |

**Ratios** are 8x, 4x, 16x, 2x, run in that order.

**GPU change after Stage 0c.** At 124k, `compactor_bern` peaks at 51.0 GB, against 47.3 GB for E3's
`kvzip_bern`. E3's L40S runs of prefix_suffix and summary already peaked at 44.2-44.4 GB, and an L40S
has about 45 GiB, so the new arms for those two tasks run on A100s. Each matched pair still shares one
GPU type: both Compactor arms on the A100, and E3's KVzip pair on the L40S. Only the cross-pair
comparisons (D3, D4) mix GPU types there.

**Compactor, two fixes to the stock kvpress press (both found by the smoke tests):**
- **Seeded sketch** (`SeededCompactorPress`). The leverage sketch is seeded from the context and the
  layer, so both Compactor arms read identical scores.
- **Cholesky fallback** (`FallbackLeverageScorePress`). Stock Compactor crashes at 124k tokens. The
  fallback redoes only the failing layer in float64 with the same sketch, and leaves stock output
  unchanged wherever stock works (T10). Every rescue is counted in `cost.json`.

LongBench runs on the A40, as E8 did.

## Measurement breakdown: exactly what is measured

### Vocabulary

- **Context:** the long document that gets compressed (prefilled once, then the KV cache is pruned or
  masked).
- **Question:** one query answered from that compressed cache, by greedy decoding. Nothing is sampled
  at decoding time.
- **Run:** one (task, arm, compression ratio) combination, evaluated on every question of that task.
  It produces one `predictions.csv` (one row per question) and one `metrics.json` (the task score).
- **Arm:** a method. The Bernoulli arms have 3 seeds, and each seed is a separate run.
- **Compression ratio:** the fraction of the context's KV pairs removed. 2x = 0.5, 4x = 0.75, 8x = 0.875,
  16x = 0.9375. So 8x keeps 12.5% of the context's KV pairs, counted over all 32 layers x 8 KV heads.
  - The question and the generated tokens are never compressed.
  - Protected tokens count toward the budget: KVzip's 4 sink tokens, and Compactor's first 8 and last 4.
  - Top-k arms keep exactly that many. Bernoulli arms keep that many **in expectation**; the realised
    count is logged per context in `memory.json` (measured 0.124-0.126 at 8x).
  - KVzip arms split the budget globally across layers and heads. Compactor arms split it evenly, the
    same count for every KV head.

### Tasks, contexts, questions

| benchmark | task | contexts | questions per context | questions | context length | max answer tokens | metric (per question, 0-100) |
|---|---|---|---|---|---|---|---|
| LongBench | qasper | 50 | 1 | 50 | median 3,564 words (1,464-13,401) | 148 | token F1 (max over gold answers) |
| LongBench | hotpotqa | 50 | 1 | 50 | median 10,261 words (2,412-12,372) | 52 | token F1 (max over gold answers) |
| LongBench | passage_retrieval_en | 50 | 1 | 50 | median 9,241 words (7,534-10,972) | 52 | retrieval score |
| LongBench | gov_report | 50 | 1 | 50 | median 7,061 words (2,096-40,508) | 532 | ROUGE-L F |
| SCBench | prefix_suffix | 10 | 5 | 50 | median 112,089 tokens (111,839-112,389) | 64 | substring match |
| SCBench | qa_eng | 10 | 5, 7, 6, then 5 x 7 | 53 | median 91,991 tokens (69,644-130,810) | 64 | token F1 (max over gold answers) |
| SCBench | summary | 10 | 5 | 50 | median 103,124 tokens (96,618-113,164) | 256 | ROUGE-L F1 |
| SCBench | vt | 10 | 5 | 50 | median 124,502 tokens (124,492-124,515) | 128 | fraction of gold variables found |
| **total** | **8** | **240** | | **403** | | | |

**Sample selection.**
- **LongBench:** each task has 200 rows. kvpress's `fraction=0.25, seed=42` picks 50 of them, the same
  50 as E8. Each row has its own context and one question.
- **SCBench:** the first 10 samples (contexts) of each task that fit under the model's 131,072-token
  limit, with all of each context's turns. That is E3's set, untruncated. qa_eng has only 23 fitting
  contexts out of 69, and three of the first 10 have more than 5 turns, hence 53 questions.
- **Length units:** LongBench lengths are the dataset's word counts (about 1.3 Llama tokens per word).
  SCBench lengths are measured Llama-3.1 tokens.
- **Shared contexts:** in SCBench all questions of a context are answered from **one** prefill and
  **one** compression, and so from one Bernoulli draw. The questions are not asked in a chat thread;
  each question sees only the compressed context.

### Runs and answers

**Per task:**

| arm | ratios | seeds | runs | LongBench source | SCBench source |
|---|---|---|---|---|---|
| no_press | full cache only | 1 | 1 | E4 | E3 (reused) |
| kvzip | 2x, 4x, 8x, 16x | 1 | 4 | E4 | E3 (reused) |
| kvzip_bern | 2x, 4x, 8x, 16x | 3 | 12 | E4 | E3 (reused) |
| compactor | 2x, 4x, 8x, 16x | 1 | 4 | E4 | E4 |
| compactor_bern | 2x, 4x, 8x, 16x | 3 | 12 | E4 | E4 |
| kvzip_bern_nobias | 8x only | 1 (seed 1) | 1 | E4 | E4 |
| **per task** | | | **34** | 34 new | 17 new + 17 reused |

**Totals:**
- **Runs:** 8 tasks x 34 = **272**. Of these, 136 are LongBench runs (all new) and 136 are SCBench runs
  (68 new, 68 reused from E3).
- **Generated answers:**
  - LongBench: 136 runs x 50 = 6,800;
  - SCBench: 102 runs x 50 (prefix_suffix, summary, vt) + 34 runs x 53 (qa_eng) = 6,902;
  - **total 13,702**.
- **Compressions (prefill, score, select):** LongBench 136 x 50 = 6,800; SCBench 136 x 10 = 1,360.
  no_press is a prefill with no compression.
- **E3 also has** expected_attention and random runs on the SCBench tasks. They are not part of the E4
  criteria.

**Independent units** (what the error bars and tests count):
- LongBench: the question, 50 per task, 200 in total;
- SCBench: the **context**, 10 per task, 40 in total, because a context's questions share one
  compression.

So SCBench carries 200 questions but only 40 independent units.

### What a score means

**Per question.** Every question gets a score in [0, 100]:

| metric | used for | what the score is |
|---|---|---|
| token F1 | qasper, hotpotqa, qa_eng | Overlap of words between answer and gold, after lowercasing and removing punctuation and articles: the harmonic mean of precision and recall, times 100. With several gold answers, the best match counts. Partial credit is normal; a verbose correct answer scores well below 100. |
| retrieval score | passage_retrieval_en | The fraction of numbers in the answer that equal the correct paragraph number, times 100. "Paragraph 12" scores 100 if 12 is correct; "Paragraph 12 or 13" scores 50. |
| ROUGE-L F | gov_report | F-score of the longest common word subsequence between the summary and the reference, times 100 (the `rouge` package). |
| ROUGE-L F1 | summary | The same idea on normalised tokens (`scbench_metrics.py`). |
| substring match | prefix_suffix | 100 if the gold string appears anywhere in the normalised answer, else 0. |
| fraction found | vt | The fraction of the gold variable names that appear in the answer, times 100. With 5 gold names, scores come in steps of 20. |

**Per run.** A run's **task score** (`metrics.json`) is the plain mean of its per-question scores over
all 50 (or 53) questions. For qa_eng this weights the 7-turn context more than a 5-turn one.

**Across tasks.** Metrics differ between tasks, so a number from one task is never compared with a
number from another. Only paired differences in points, and averages of those, are combined across
tasks.

### What is averaged, and what is not

**Averaged:**

| what | averaged over | where |
|---|---|---|
| a run's task score | the questions of that task | every table |
| a Bernoulli arm's score for a task | its 3 seeds, which equals the mean of the per-question seed means | score tables and plots |
| a Bernoulli arm's per-question score, in paired tests | its 3 seeds | D_KV, D_CP, D3 |
| paired difference per cluster | the questions in the cluster: 5 turns (7 or 6 for two qa_eng contexts) in SCBench; one question per cluster in LongBench | every test |
| a task's effect (D_KV, D_CP, ...) | the clusters, each weighted equally. For SCBench this can differ slightly from "Bernoulli task score minus top-k task score", because clusters, not questions, get equal weight. | per-task tables, forest plot |
| a task's SE | the spread of cluster differences: sd / sqrt(number of clusters) | per-task tables |
| pooled effect (C1, C2, C4, D2, D3) | the task effects, **unweighted**, over all 8 tasks. It is an average gain in points across different metrics; SCBench and LongBench tasks count equally despite different cluster counts. | criteria |
| benchmark-level effect (C3, D1) | the same, over one benchmark's 4 tasks | criteria, descriptive |
| headline plot | per benchmark, the unweighted mean over its 4 tasks of the task scores, only over tasks with every arm at every ratio | `0_headline.png` |
| realised kept fraction | the layers (and heads) within a context, then reported per run over contexts | `memory.json`, C0 |

**Not averaged:**
- **Compression ratios:** never. Every number belongs to one ratio.
- **The two benchmarks:** never combined in tables or plots. Only the pooled C1/C2/C4 statistic spans
  all 8 tasks; C3 and D1 keep them separate.
- **no_press, kvzip and compactor:** one run each and no seeds. Decoding is greedy, KVzip's scores are
  deterministic, and Compactor's random sketch is seeded per context, so a rerun gives the same answers
  (E8 reproduction: 98-100% identical).
- **The seed band is not an error bar.** It is mean ± 1 standard deviation (n - 1) across the 3 seed-level
  task scores (or benchmark means). It shows draw-to-draw variation. The score tables print the same
  quantity as `mean ± SD`. The statistical uncertainty is the clustered SE.
- **nobias (D2):** compared with **seed 1 only**, not the seed mean, because only seed 1 has the
  same draws with the correction removed.
- **SCBench turns** are not averaged into one score per context for the task score (that is a
  per-question mean). They are averaged only when forming clusters for tests.
- **Sign-flip p-values** are one-sided, "sampling helps". Per-task tests flip whole clusters. The pooled
  test flips clusters within each task at the same time. Exact for 10 clusters; 200,000 random flips
  for 50.

## Pre-registered analysis

**Primary effects.** Both are measured per task and ratio, paired per question:
- **D_KV** = kvzip_bern - kvzip;
- **D_CP** = compactor_bern - compactor.

**Statistics.**
- Each Bernoulli score is the per-question mean over its 3 seeds.
- **Clusters:** for SCBench, the context (its 5 turns share one compression draw); for LongBench, the
  sample.
- **Per-task test:** exact (or 10^5 Monte Carlo) sign-flip over clusters, with a clustered SE.
- **Pooled test:** stratified by task. Signs are flipped within each task, and the statistic is the
  unweighted mean over tasks of the task-mean differences.
- **Multiple testing:** Holm within each family. No |z|>2 rules (E3 audit).

| | criterion | passes if |
|---|---|---|
| C0 | harness trustworthy | T1-T10 pass in every job; realised kept fraction within 1% of target for both Bernoulli arms; `no_press` plausible per task; LongBench `no_press`/`kvzip` predictions compared with E8 on the 3 shared tasks (match rate reported, investigated if below 90%) |
| **C1** | sampling helps KVzip | at 8x, pooled D_KV > 0, one-sided p < 0.05, over all 8 tasks |
| **C2** | sampling helps Compactor | at 8x, pooled D_CP > 0, one-sided p < 0.05, over all 8 tasks |
| **C3** | consistent | at 8x, D_KV >= 0 and D_CP >= 0 on at least 3 of 4 tasks in **each** benchmark separately |
| C4 | other ratios | 4x: C1/C2 tests, Holm over {4x, 8x} x {KV, CP}. 2x: pooled 95% CI of D_KV and D_CP above -2 points. 16x: reported only |

**Verdict:**
- **"Sampling helps both scorers":** C1, C2 and C3 all pass.
- **"Helps one scorer":** exactly one of C1 and C2 passes, and we name which.
- **"No consistent help":** otherwise.

**Descriptive (no pass/fail):**
- **D1:** mean D_KV and D_CP, LongBench against SCBench.
- **D2:** kvzip_bern_s1 - kvzip_bern_nobias_s1 at 8x (same draws). Does the correction matter?
- **D3:** kvzip_bern against compactor_bern.
- **D4:** every arm against no_press.

## Self-tests (abort the job on failure)

**T1-T6** are E3's, unchanged (`experiment3/selftests.py`).

**T7-T10** are new (`selftests4.py`):
- **T7, masking matches pruning:** `CompactorBernPress(force_topk=True)` keeps Compactor's top-k through
  the -inf bias. Its logits must match plain `compactor`, which prunes:
  - relative max difference below 5e-2, the tolerance E3's T3 uses for the same bias-vs-masking
    question under bf16;
  - identical argmax at every position.

  The plan had 1e-3. That was tighter than bf16 kernels on different sequence lengths can promise, so
  the argmax check was added instead.
- **T8, budget:**
  - the expected count per head equals the budget to within 1e-6, at 4 ratios and 3 lengths;
  - forced tokens have r = 1;
  - bisection time is measured at 124k.
- **T9, end to end:**
  - bias present during generation and absent during prefill;
  - realised kept fraction within 1% of target;
  - ratio 0 gives the same answer as no_press;
  - the same seed gives the same draws, and different seeds give different draws.
- **T10, Cholesky fallback:**
  - the wrapper returns stock output bit for bit when stock succeeds;
  - the float64 path agrees with stock to 1e-2 on the same sketch (measured 3e-6);
  - a badly scaled key set that breaks stock is rescued with finite scores.

## Limits

- **Memory is counted, not freed,** for every masked arm. "8x" is behavioural.
- **One model:** Llama-3.1-8B-Instruct.
- **Small n.** 50 per task is a development signal. SCBench has 10 clusters per task, so per-task tests
  are weak by design, and the pooled tests carry the claim.
- **Two runtime environments.** The LongBench no_press/kvzip rows are new runs on A40. SCBench's are E3's
  runs on the matching GPU type.

## Files

| file | role |
|---|---|
| `compactor_bern.py` | the new press |
| `selftests4.py` | T7-T9 |
| `data.py` | offline loaders (E8 LongBench parquet, E3 SCBench), imported not copied |
| `run_sweep.py` | resumable sweep, ratio-first order, cost/memory logging |
| `sweeps/smoke.yaml`, `smoke_vt.yaml` | Stage 0 |
| `sweeps/lb_A.yaml`, `lb_B.yaml`, `sc_ps.yaml`, `sc_sum.yaml`, `sc_qa.yaml`, `sc_vt.yaml` | Stage 1 |
| `job_sweep.sbatch` | `SWEEP=... sbatch -p <part> --qos=<part> job_sweep.sbatch` |
| `summarize.py`, `plots.py` | analysis (after Stage 1) |
