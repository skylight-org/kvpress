# Phase 2, Experiment 3 — Long-context testing on SCBench (~110k, A100)

**Date:** 2026-09-16. **Depends on:** `../experiment1/CONCLUSION.md`, `../experiment2/CONCLUSION.md`,
`../DERIVATION_denominator_bernoulli.pdf`. **Plan of record:** approved plan in `.claude/plans/`.

---

## 1. Why

E1 and E2 both ran at ≤ 11k tokens, so every conclusion so far is untested where KV compression actually
matters. The professor's framing removes the theory question from the critical path: **the exact value
of ε does not matter now** — what matters is the *empirical* behaviour of the bound and how it feeds
dynamic (query-aware) fetching.

Carried forward:

- **E1:** the bound is correct and transfers (98.89% of real queries), but 10–14× loose. Certified
  memory is essentially a function of `T/n` where `T = Σᵢ maxq ωᵢ` (Spearman 0.957) — the looseness
  **is** the max over queries.
- **E2:** sampling ties top-k overall (+0.79 ± 0.51 at 8×, not significant) and both crush the cheap
  baselines. The one real effect was structural: sampling wins on **aggregation** (`cwe` +13 at 8×,
  +72 at 16×) and gains nothing on needle retrieval.

## 2. Questions

| # | Question | Answered by |
|---|---|---|
| **Q1** | Does the method still hold up against KVzip and the field at ~110k? | Stage 1 |
| **Q2** | Does sampling's **aggregation** advantage survive or grow at long context? | Stage 1, task-family split |
| **Q3** | Does compressibility improve with length — is `T/n` smaller at 110k than at 4–11k? | Stage 2 |
| **Q4** | How big is the gap between the query-agnostic score and a per-query oracle? | Stage 2 |

Q4 is the one that decides whether **dynamic fetching** is worth building: that gap is what a
query-aware method could recover.

## 3. Data — SCBench, unmodified

KVzip's own benchmark, and *shared-context multi-query*: one long context with 5 queries. Prefill and
calibration are paid per **context**; each extra query is a short decode. That is both what makes 110k
affordable and the method's real deployment setting — compress once, before any question, serve many.

**Nothing is synthesized and nothing is truncated.** `max_context_length` is `None` throughout. Samples
whose longest turn would exceed the model's 131,072-token limit are **dropped and counted**.

Measured by `measure_lengths.py` (`out/lengths.json`):

| task | samples | ctx tokens (median) | fits | dropped |
|---|---|---|---|---|
| `kv` | 100 | 124,002 | 100 | 0 |
| `prefix_suffix` | 100 | 112,089 | 100 | 0 |
| `vt` | 90 | 124,502 | 90 | 0 |
| `qa_eng` | 69 | 163,591 | **23** | **46** |
| `summary` | 70 | 103,124 | 70 | 0 |

**383 of 429 contexts → 1,915 queries, all at ~100–124k.** `qa_eng` loses 67% of its samples to the
model's context limit; that is a stated limitation, not a silent filter.

Lengths are tightly clustered (`vt` spans 124,492–124,515), so **there is no natural length variation to
bin on**. This is a ~110k benchmark, not a length sweep, and no claim will pretend otherwise.

**Task families:** retrieval/tracking = `kv`, `prefix_suffix`, `vt`; aggregation/free-text =
`summary`, `qa_eng`. Q2 rests on that split.

## 4. Arms and metrics

| arm | selection | role |
|---|---|---|
| `no_press` | — | ceiling |
| `kvzip` | global top-k on KVzip's score | **the control** |
| `kvzip_bern` (seeds 1–3) | Bernoulli `min(1, c·s)` + `log(1/r)` bias | **the method** |
| `expected_attention` | AdaKV top-k | best cheap baseline in E2 |
| `random` | uniform | sanity floor |

- **No no-bias ablation:** `kvzip_bern` carries the bias because that is the mathematically correct
  (Horvitz–Thompson, unbiased) estimator.
- **Chunked KVzip calibration everywhere**, exactly as kvpress implements it — one calibration scheme
  for every arm, so the only difference in B1 is top-k versus sampling.
- **Ratios:** 2×, 4×, 8×, 16×. E2 found 16× is where the accuracy ceiling breaks and arms separate.

**Metrics** (`scbench_metrics.py`): substring match for `kv` and `prefix_suffix`; `string_match_all`
for `vt` (fraction of the 5 gold variables found); token F1 for `qa_eng`; ROUGE-L for `summary`.

## 5. Pre-registered criteria

| | criterion | passes if |
|---|---|---|
| **C0** | harness trustworthy | `no_press` plausible per task; `random` collapses; every real arm beats it; T1–T6 pass at full length |
| **C1** | sampling ≥ top-k at 110k | `kvzip_bern` not significantly below `kvzip` at any ratio (paired per-query) |
| **C2** | competitive with the field | best Bernoulli arm ≥ `expected_attention` − 1 point at 2× and 4× |
| **C3** | **aggregation effect** | on `summary`/`qa_eng`, `kvzip_bern` − `kvzip` ≥ 0, and larger than on the retrieval tasks |
| **C4** | memory honest | realised kept fraction within ±1% of target and inside the §11 bound |
| **C5** | compressibility scales | report `T/n` at 110k against E1/E2's 4–11k values — **descriptive, no pass/fail** |

C1 and C3 are the experiment. C5 is a measurement, deliberately without a threshold.

## 6. Method notes that matter

- **Score cache** (`e3_press.py`): KVzip's score is deterministic given the context, but kvpress
  recomputes it per run. The probe measured 55 s per context with no press and 259 s with KVzip, so
  calibration is ~200 s and dominates; the matrix has 16 calibrated runs per context. Caching
  `score_val` per context cuts ~78 min per context to ~28 min **and** makes `kvzip` and `kvzip_bern`
  provably read the same bytes. Verified by running both paths and requiring identical predictions
  (`E3_NO_SCORE_CACHE=1` bypasses it).
- **Context count is limited by `E3_N_CTX`, never by kvpress's `fraction`**, which samples rows and
  would split a context's 5 turns across the subset, breaking the multi-query grouping.
- **Hardware:** dgx **A100-SXM4-80GB**. Peak measured 47.1 GB at 124k — this would not fit a 44 GB A40.

## 7. Honest limits

- **Memory is counted, not freed** (kvpress masks dropped tokens; a ragged-cache kernel is out of
  scope). "8×" means the model behaves as if 7/8 of the cache were gone.
- **One model, one length band.** Llama-3.1-8B-Instruct at ~110k. 1M is deferred: it needs a different
  model (its limit is 131,072), 137 GB of cache, and flash-attention.
- **`qa_eng` is 23 of 69 samples**, so its numbers are noisy and reported with that caveat.
- **Small context counts.** Paired per-query statistics are reported with the count stated plainly; no
  significance will be claimed that the data cannot carry.
- **No certificate here.** ε is not the object of study (professor's framing); the empirical error is.

## 8. Files

    measure_lengths.py   Stage 0: token-length census -> out/lengths.json
    scbench_data.py      loader: no truncation, drop-and-count, multi-query grouping
    scbench_metrics.py   per-task scorers
    e3_press.py          score cache + Bernoulli press + Stage 2 statistics
    selftests.py         T1-T6, copied verbatim from experiment2
    run_sweep.py         resumable sweep (dgx/A100), records peak memory and cost
    (Stage 2 T/n + heavy hitters ended up in summarize.py C5 and plots.py 6-7; the
     query-agnostic gap needs an instrumented pass and is NOT built -- see CONCLUSION.md)
    summarize.py         paired statistics, per-task-family tables -> out/REPORT.md
    plots.py             eight figures -> out/figures/
