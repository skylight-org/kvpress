# Phase 2, Experiment 2 — Benchmarking Bernoulli sampling against the published methods

**Date:** 2026-09-15. **Status:** design.
**Depends on:** Experiment 1 (`../experiment1/CONCLUSION.md`), `../DERIVATION_denominator_bernoulli.pdf`.

---

## 1. What Experiment 1 established, and what this one asks

Experiment 1 answered a question about the *certificate*: is the denominator bound tight enough to
set budgets? Answer: no — 13.7× loose at 2×, median certified memory 0.683n at ε = 0.1. But it also
produced a second result that was not the headline and is more actionable:

- the **sampling scheme itself is sound**. The estimator is unbiased, heavy hitters fall out of
  `rᵢ = min(1, cτᵢ)` by themselves, the `log(1/rᵢ)` bias is exact, and the transfer assumption holds
  on 98.89% of real queries;
- and in E9, Bernoulli sampling needed **0.3n where top-k needed 0.5n** to reach the same attention
  error — a 1.7× memory advantage at the attention level.

So the certificate is not yet usable, but the compressor might already be good. **Experiment 2 drops
the certificate entirely and asks a pure benchmark question: as a KV cache compression method, does
Bernoulli sampling beat the published query-agnostic methods on task accuracy?**

This is deliberately a different kind of experiment from everything in Phase 1 and from Experiment 1.
No attention errors, no η, no δ. Only task scores against published baselines.

## 2. Why RULER is the benchmark

The user asked for the dataset KVzip uses, so the numbers can be checked against their paper.
KVzip (arXiv 2505.23416) evaluates on **RULER**, plus SQuAD, LongBench and NIAH. Of these:

| Benchmark | Verdict | Why |
|---|---|---|
| **RULER** | **primary** | KVzip's headline benchmark; kvpress has the scorer built in (`ruler_scorer`); synthetic retrieval tasks where compression differences are large and unambiguous |
| **LongBench** | **secondary** | E8/E9 already ran it; gives continuity with our own prior numbers; but scores saturate and 50-sample noise is ±2–5 points |
| SQuAD | dropped | not in kvpress's registry; would need a custom loader for little gain |
| NIAH | dropped | RULER's `niah_*` subtasks already cover needle retrieval, with more variants |

**RULER is the right primary choice** and it is the one where Experiment 1's finding should bite.
LongBench scores are dominated by tasks a model can answer from a fraction of the context — E9 saw
KVzip *beat* the full cache at 2×, which means the benchmark is not measuring compression damage.
RULER's synthetic tasks have a precise answer located at a precise place, so dropping the wrong
tokens is visible immediately.

**Availability, checked:** `simonjegou/ruler` configs 4096 and 8192 are downloaded to `~/hf`
(24 MB + 68 MB). 16384 exists (189 MB) and can be added later. Compute nodes have no internet, so
everything must be cached on the login node first — this is done.

**Scope is capped at ~5 h wall** (user's constraint). Only RULER **4096** is run; 8192 is cached and
ready but deferred to a follow-up. See §5 for the cost model this is based on.

**Verifiability caveat, stated up front.** Our numbers will not equal KVzip's paper exactly, and we
should not pretend otherwise: kvpress's `KVzipPress` is a reimplementation ("minimal changes to this
repository", not the optimised original), it computes calibration in 2048-token chunks, and the model
and prompt formatting may differ from theirs. What *is* verifiable is the **ordering and the shape of
the degradation curve** — if kvpress's `kvzip` reproduces the paper's qualitative behaviour (near-
lossless at 2×, degrading at 8×), then a fair comparison inside the same harness is meaningful. E8
already confirmed `senszip_tau` reproduces `kvzip` predictions *exactly*, which validates the harness.

**And the chunking need not be a caveat at all.** `KVzipPress` separates scoring from selection:
`score_kvzip` builds a `score_val` tensor from the chunked repeat pass, and `compress_post` then does a
global top-k over it. Subclassing it and overriding **only `compress_post`** gives an arm with
byte-identical scoring — same chunk boundaries, same sink handling, same budget — that differs from
`kvzip` in exactly one respect: Bernoulli sampling instead of top-k. That is the controlled comparison,
and §4 makes it the primary arm.

## 3. The method under test

Straight from the PDF, unchanged from Experiment 1:

    τᵢ = max over calibration queries of ωᵢ(q)        (repeat-context calibration, per KV head,
                                                      pooled over the 4 query heads that share it)
    rᵢ = min(1, c·τᵢ)                                  one global c, solved from the memory target
    keep i with probability rᵢ, independently
    decode with logit bias log(1/rᵢ) on kept tokens

**Note the score is τ, not E9's π = max(σ, τ).** The PDF's derivation is denominator-only, and τ is
its native score. This also makes the comparison against KVzip clean: **KVzip's score *is* τ** (E8
verified this — `senszip_tau` reproduced `kvzip` predictions exactly). So `bern_tau` vs `kvzip` is a
controlled experiment isolating exactly one variable: **sampling + bias vs deterministic top-k**, at
identical scores and identical memory. That is the cleanest comparison available, and it is the
central claim of this experiment.

E9 already found `bern_tau` was its best LongBench variant (+3.72 ± 1.81 over KVzip at 4×), on 50
samples per task. This experiment is partly about whether that survives a proper sample size.

## 4. Arms

| Arm | Score | Selection | Bias | Role |
|---|---|---|---|---|
| `no_press` | — | — | — | the ceiling |
| `kvzip` | KVzip `score_val` (chunked) | global top-k | — | **the control** |
| **`kvzip_bern`** | **the same `score_val`** | **Bernoulli `min(1,c·s)`** | log(1/r) | **the method — differs from `kvzip` in selection ONLY** |
| `kvzip_bern_nobias` | the same `score_val` | Bernoulli | none | is the HT correction doing the work? |
| `bern_tau` | τ, full-context calibration | Bernoulli | log(1/r) | isolates calibration scope |
| `snapkv` | window attention | top-k | — | standard baseline |
| `expected_attention` | analytic | AdaKV top-k | — | query-agnostic, no extra pass |
| `streaming_llm` | position | sink + recent | — | the trivial floor |
| `random` | — | uniform | — | **sanity floor**: any method must beat this |

**The design decomposes the total difference into two measurable halves.**

| Comparison | Holds constant | Isolates |
|---|---|---|
| `kvzip` → `kvzip_bern` | score, chunking, budget, sinks | **top-k vs Bernoulli sampling** |
| `kvzip_bern` → `bern_tau` | selection rule (Bernoulli) | **chunked vs full-context calibration** |

Previously the chunking was an unmeasurable confound sitting inside a single comparison; now it is a
quantity this experiment reports. `kvzip` → `kvzip_bern` is the claim; the second row says whether our
calibration change is worth its extra cost.

Cut for time: `keydiff` (another cheap query-agnostic top-k — `expected_attention` covers that niche).

`random` matters more than it looks. E6 found uniform sampling is a strawman that σ-sampling beats
easily; including it here proves the harness discriminates at all, and if a "real" method ever ties
`random`, that method is broken at that ratio.

**Ablation `bern_tau_nobias` is the one to watch.** Experiment 1 showed the bias is what makes the
estimator unbiased. E9's LongBench run found it barely mattered end-to-end (63.8 vs 63.0 at 4×). If
that repeats on RULER, it means task accuracy is insensitive to the denominator correction and the
theory's main practical contribution is the *selection rule*, not the estimator — worth knowing.

## 5. Design

**Model:** Llama-3.1-8B-Instruct, bf16, kvpress v0.5.4 in the `kvpress` conda env
(`transformers<5.3`; never install into `llm`).

**Compression ratios:** 2×, 4×, 8× (ratios 0.5, 0.75, 0.875). E8 found π top-k lost to KVzip at 8× on
retrieval; 8× is where the arms should separate.

**Samples:** RULER **4096 only**, all 13 subtasks, **60 samples per subtask** = **780 samples per arm
per ratio**. That is 15× E9's per-task sample size, so a ~2-point difference is resolvable, which is
what B1 needs.

**Seeds:** Bernoulli arms are random, so **3 seeds** at every ratio, reported as mean ± SE over seeds.
Deterministic arms run once. This is the lesson from E9, where seed spread was 2–5 points and swamped
most reported differences — cutting seeds would be the wrong economy, so sample count was cut instead.

**Realised memory is recorded per sample.** Bernoulli's memory is random (`N = ΣZᵢ`), so equal-budget
claims must be *verified*, not assumed. The PDF's §11 high-probability bound
`N ≤ M + √(2M log(1/δ_N)) + (2/3)log(1/δ_N)` is checked against the realised counts — a cheap
empirical test of one more piece of the derivation.

### Cost model (from measured timings, not guessed)

E9's Part C logs give **~1.8 s/sample** for a calibrated arm once the repeat pass is cached, and E8
gives **~8 s/sample** for a context's first touch (prefill + repeat pass). RULER 4k contexts are
shorter than E9's LongBench ones, so 1.5 s and 6 s respectively, rounded up. Calibration is cached per
context and shared by every calibrated arm (`kvzip`, `bern_*`), so the repeat pass is paid once.

| Scope | GPU-h | wall on 3 A40 jobs |
|---|---|---|
| 4k+8k, 100/subtask, 9 arms | 47.7 | 15.9 h |
| 4k only, 100/subtask, 9 arms | 23.8 | 7.9 h |
| **4k only, 60/subtask, 7 arms (chosen)** | **12.3** | **≈ 4.1 h** |

Three A40 jobs is the account cap (`a40`: 3 running, 6 queued). ≈ 4.1 h leaves headroom inside the 5–6 h
target for queue wait and the ~4-minute cold-import tax per job.

### Ordering, so a partial run is still an answer

| Priority | Content | Answers |
|---|---|---|
| **P1** | `no_press`, `kvzip`, `kvzip_bern` (3 seeds), `random` at 2×/4×/8× | **the central claim (B1)** |
| **P2** | `snapkv`, `expected_attention`, `streaming_llm` at 2×/4×/8× | where we sit in the field (B2) |
| **P3** | `bern_tau` (3 seeds) at 2×/4×/8× | calibration scope (B6) |
| **P4** | `kvzip_bern_nobias` at 4× and 8× only | does the bias matter? (B5) |

All on RULER 4096. Every run writes its own directory and the sweep is resumable, so a job that
overruns loses only its current run. **Deferred, cached and ready if these results warrant it:**
RULER 8192, RULER 16384, and a LongBench arm for continuity with E8/E9.

P4 is trimmed to 4× and 8× because the bias can only matter where sampling is aggressive; at 2× nearly
everything is kept and E9 already saw no difference there.

If only P1 finishes, the experiment has still answered its main question.

## 6. Pre-registered criteria

| | Criterion | Passes if |
|---|---|---|
| **B0** | harness is trustworthy | `no_press` RULER scores plausible for Llama-3.1-8B; every real arm beats `random` at 2×; `kvzip_bern` at `c → ∞` reproduces `kvzip` (self-test T5) |
| **B1** | **sampling beats top-k, everything else held fixed** | `kvzip_bern` mean ≥ `kvzip` at 2×, 4× **and** 8×; **wins** if > +2 SE at any ratio |
| **B2** | competitive with the field | best Bernoulli arm within 1 point of the best non-KVzip baseline at 2× and 4× |
| **B3** | degrades gracefully | `kvzip_bern`'s drop from `no_press` at 8× no worse than `kvzip`'s |
| **B4** | memory is honest | realised kept fraction within ±1% of target; §11's high-probability bound holds on ≥ 95% of samples |
| **B5** | the bias earns its place | `kvzip_bern` ≥ `kvzip_bern_nobias` at 4× and 8×; if not, report that selection alone explains the result |
| **B6** | calibration scope | report `bern_tau` − `kvzip_bern` with SE; full-context calibration justifies its cost only if > +1 point |

**B1 is the experiment.** Identical scores, identical chunking, identical budget — the only difference
is Bernoulli sampling with a log(1/r) bias versus deterministic top-k. B6 is the second half of the
decomposition and is reported whichever way it falls.

## 7. What would make this a negative result

Worth writing down now, so it is not rationalised later:

- If `kvzip_bern` ties `kvzip` within noise at every ratio, then Bernoulli sampling's attention-level
  advantage (0.3n vs 0.5n in E9) **does not convert into task accuracy**, and the honest conclusion is
  that the theory's practical payoff is a certificate we cannot yet make tight. With scoring held
  byte-identical there is nowhere else to attribute the null result, which is precisely what makes
  this design worth running.
- If `kvzip_bern_nobias` ties `kvzip_bern`, the Horvitz–Thompson correction is not what is helping.
- If `bern_tau` ties `kvzip_bern`, full-context calibration does not repay its extra forward passes,
  and the chunked approximation should simply be adopted.
- If `random` is close to the real methods at 2×, the benchmark is too easy at that ratio and only
  8× (or 16k) carries information.

Any of these is a publishable finding about the method's limits; none of them is a reason to rerun
with different settings until the numbers improve.

## 8. Files

    phase2/experiment2/
      DESIGN.md         this document
      bern_tau_press.py kvpress press: tau scoring, Bernoulli keep, log(1/r) attention patch
                        (adapted from e9/bern_press.py, which is already verified: T1-T4 self-tests)
      run_sweep.py      resumable multi-run driver, one model load per job
      sweeps/*.yaml     P1-P5 run lists
      summarize.py      results -> tables, paired differences, figures
      job_sweep.sbatch  SLURM (a40)
      out/, results/    per-run predictions.csv + metrics.json, then REPORT.md

**Self-tests before any real run** (the job aborts if one fails). T1–T4 are inherited from E9, where
they passed; T5 is new and is what makes the controlled comparison trustworthy:

1. **T1** all `rᵢ = 1` (compression 0) → logits identical to `no_press`;
2. **T2** bias 0 with dropped tokens masked → identical to top-k's fake-key masking;
3. **T3** one layer offline: masked-softmax output == `Â/B̂`;
4. **T4** realised kept fraction matches the target within ±1%, **and the first `n_sink` tokens are
   kept in every layer and head** (see AUDIT.md #3 — Bernoulli drops sinks unless forced);
5. **T5 (new, decisive)** `kvzip_bern` and `kvzip` produce **identical `score_val`** on the same
   context. This proves the two arms share their scoring exactly, so any score difference between them
   is attributable to selection alone. If T5 fails, B1 is not a controlled comparison and the run is
   pointless — hence the abort.

## 9. Honest limits

- **Memory is counted, not freed.** Dropped tokens are masked, as kvpress does for every head-wise
  press. Real memory savings need a ragged-cache kernel, which is out of scope.
- **One model, one context length.** Llama-3.1-8B-Instruct at RULER 4096 only. Whether the ranking
  holds at 8k/16k is genuinely open — compression methods can reorder with context length, and this
  run cannot speak to that. The data is cached for the follow-up.
- **60 samples per subtask**, so a difference under ~2 points on a single subtask is noise; the
  13-subtask mean is the number to read.
- **Reimplemented baselines.** Comparisons are fair *within* kvpress; absolute numbers may differ from
  each method's paper.
- **No certificate.** This experiment says nothing about (ε, δ). Experiment 1 covers that, and the
  next step on the theory side is fixing the score, not the inequality.


---

## 10. As run (2026-09-16) — deviations from the plan above

- **Hardware.** P1 (the B1 pair), P3 and P4 ran on A40, matching the valid `kvzip`/`no_press`/`random`
  controls. The P2 baselines ran on L40S to fit the night, together with a `no_press` rerun there that
  measures the hardware difference directly rather than assuming it away.
- **`bern_tau` calibrates on every repeat position, not 512** (AUDIT #11). With 512 positions it missed
  the needles and scored 73; with all positions it scores in line with `kvzip_bern`. Without this
  change, B6 would not measure calibration scope. Remaining differences from `kvzip_bern`: E9's score
  is normalised over the context only, and it takes the max over the 4 query heads that share a KV head.
- **Exploratory 16× point (not pre-registered).** `kvzip` and `kvzip_bern` (3 seeds) at 16×, both on
  L40S. RULER 4k sits near the ceiling at 2×/4×, so this tests B1 where the arms can separate. It is
  reported separately and does not enter any verdict.
- **Statistics.** Arms are compared paired per sample (all runs score the same 780 samples), with a
  stratified standard error, in addition to the seed spread. Per-sample scores are recomputed and
  checked against kvpress's own metrics.
- **First P1 Bernoulli run discarded** (AUDIT #8). `keydiff` was already cut.
