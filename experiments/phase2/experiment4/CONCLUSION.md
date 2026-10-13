# Experiment 4: conclusion

**Question.** Does Bernoulli sampling (keep token i with probability `r_i = min(1, c*s_i)`, then add a
`log(1/r_i)` bias to kept tokens) improve on top-k selection for **two different scorers**, KVzip and
Compactor, across 4 LongBench and 4 SCBench tasks?

**Short answer.**
- **For KVzip, yes, and strongly at long context.**
  - Pooled over all 8 tasks at 8x, sampling beats top-k by **+4.9 ± 1.4 points** (p = 0.0005, Holm
    0.002).
  - The gain is concentrated in SCBench (~110k tokens: +8.2 at 8x, +7.9 at 4x).
  - On LongBench (~5-15k tokens) it is small at 4x and 8x and grows at 16x.
- **For Compactor, weakly.**
  - Pooled at 8x: **+2.5 ± 1.3** (p = 0.033), but this does not survive Holm correction (0.066).
  - It comes mostly from passage_retrieval_en and vt.
  - It turns negative at 16x on SCBench.
- **Formal verdict: the pre-registered label is "no consistent help".** C1 and C2 pass, but C3 (a
  non-negative effect on at least 3 of 4 tasks in each benchmark) fails. On LongBench only 2 of 4 tasks
  are non-negative for either scorer, because qasper and hotpotqa are ties (KVzip: -0.43 and -0.03
  points, with SEs of 3-4).
- **The log(1/r) correction matters.** At 8x, the same draws without it lose 2.8 points on average
  (p = 0.027), and 16 points on prefix_suffix.
- **The scorer matters more than the selection rule.** KVzip scores beat Compactor scores under sampling
  by 4-10 points at every ratio.

Full numbers are in `out/REPORT.md`, figures in `out/figures/`. Definitions of every count, metric and
average are in `DESIGN.md`, section "Measurement breakdown".

![headline](out/figures/0_headline.png)

## What was run

- **Model and data:**
  - Llama-3.1-8B-Instruct, kvpress v0.5.4;
  - 8 tasks and 403 questions: LongBench qasper, hotpotqa, passage_retrieval_en and gov_report (50 each,
    E8's subset); SCBench prefix_suffix, qa_eng, summary and vt (10 untruncated 70k-131k-token contexts
    x 5 turns each, E3's set).
- **Arms per task:**
  - no_press;
  - at 2x, 4x, 8x and 16x: kvzip, kvzip_bern (3 seeds), compactor, compactor_bern (3 seeds);
  - kvzip_bern without the correction, at 8x.
- **Scale:** 272 runs and 13,702 generated answers, all present. 68 SCBench runs (no_press, kvzip,
  kvzip_bern) are reused from E3; everything else is new.
- **Pairing:** each selection-only pair reads byte-identical scores:
  - kvzip vs kvzip_bern: E3's score cache;
  - compactor vs compactor_bern: the sketch is seeded per context.

  So each pair differs only in which tokens are kept.

## Results

### Benchmark averages (score, 0-100; mean of the 4 task scores)

| | 2x | 4x | 8x | 16x |
|---|---|---|---|---|
| **LongBench**, full cache 58.6 | | | | |
| kvzip | 58.7 | 56.0 | 50.1 | 31.8 |
| kvzip_bern | 58.5 | **57.3** | **51.7** | **35.7** |
| compactor | 57.6 | 47.1 | 35.7 | 26.4 |
| compactor_bern | 56.2 | 47.7 | 38.8 | 28.8 |
| **SCBench**, full cache 29.0 | | | | |
| kvzip | 28.8 | 20.6 | 18.1 | 17.8 |
| kvzip_bern | 28.5 | **28.4** | **26.3** | **19.7** |
| compactor | 22.6 | 20.0 | 17.9 | 19.7 |
| compactor_bern | 23.2 | 22.1 | 19.6 | 16.3 |

KVzip + sampling on SCBench stays within 0.6 points of the full cache at 4x and within 2.7 at 8x. With
top-k selection, KVzip loses 8.4 and 10.9 points there.

### Pre-registered criteria

| | criterion | result | outcome |
|---|---|---|---|
| C0 | harness trustworthy | all 272 runs present; self-tests pass in every job; kept fraction within 1% in all 200 Bernoulli runs; GPU types as planned; LongBench no_press and kvzip reproduce E8 (98-100% identical answers); E3's vt 8x runs reproduce across two jobs (100%) | **PASS** |
| C1 | sampling helps KVzip (8x, pooled) | +4.90 ± 1.38, p = 0.0005 | **PASS** |
| C2 | sampling helps Compactor (8x, pooled) | +2.49 ± 1.32, p = 0.033 | **PASS** (not after Holm: 0.066) |
| C3 | consistent: >= 0 on 3 of 4 tasks per benchmark (8x) | KVzip: LongBench 2/4, SCBench 4/4. Compactor: LongBench 2/4, SCBench 3/4 | **FAIL** |
| C4 | 4x, Holm over {4x, 8x} x {KV, CP} | KVzip +4.61 ± 1.21 (Holm 0.0007). Compactor +1.36 ± 1.24 (Holm 0.15) | KVzip passes, Compactor does not |
| C4 | 2x no harm (95% CI above -2) | KVzip CI [-1.81, +1.42]. Compactor CI [-2.32, +1.53] | KVzip PASS, Compactor FAIL (marginally) |

### Findings

**1. For KVzip, sampling is a long-context win and never a loss.**

| | 2x | 4x | 8x | 16x |
|---|---|---|---|---|
| LongBench | -0.2 | +1.3 | +1.6 | +3.9 (p = 0.036) |
| SCBench | -0.2 | **+7.9** (p = 0.0002) | **+8.2** (p < 0.0001) | +1.8 |

- **Per task at 8x:** all 4 SCBench tasks are positive (prefix_suffix +18.0, summary +6.0, vt +6.0,
  qa_eng +2.8).
- **On LongBench:** passage_retrieval_en and gov_report are positive (+6.0 and +0.9); qasper and hotpotqa
  are exact ties within noise.
- **At 16x** LongBench starts to benefit too (gov_report +4.3, p < 0.001; passage_retrieval_en +7.8).
- **This repeats E2 and E3.** Ties at ~4-15k tokens, a clear gain at ~110k. It is the consistency clause
  over LongBench's short-context ties that fails C3, not the direction of the effect.

**2. For Compactor, sampling helps less and less reliably.**

| | 2x | 4x | 8x | 16x |
|---|---|---|---|---|
| LongBench | -1.4 | +0.6 | +3.2 | +2.4 |
| SCBench | +0.6 | +2.1 | +1.8 | **-3.5** |

- **Where it helps:** passage_retrieval_en at 8x (+18.0, p = 0.006: 52 vs 34) and vt at 4x (+10.5).
- **Where it hurts:** hotpotqa at 4x and 8x (-5.2, -4.4, both not significant), summary at 16x (-2.7),
  and vt at 16x (-10.9).
  - The vt 16x number is partly an artefact of the baseline. Compactor top-k scores 33.2 there, above
    its own 2x-8x scores and close to the full cache, on a task whose answers are often degenerate.
- **prefix_suffix scores 0 for Compactor with either selection rule,** at every ratio including 2x.
  Sampling cannot recover a span that the scorer ranks low everywhere.

**3. The score matters more than the selection rule.**
- kvzip_bern beats compactor_bern by +3.8 (2x), +8.0 (4x), +9.8 (8x) and +5.1 (16x) points, pooled
  (all p < 0.001).
- Against the full cache at 8x:
  - LongBench: Compactor loses 22.9 (top-k) and 19.7 (sampled); KVzip loses 8.5 and 6.8.
- **Fairness caveat:** KVzip pays for its scores with an extra "repeat the context" pass (2-3x the
  prefill), while Compactor's scoring is cheap. This comparison is between scores at equal memory, not at
  equal compute.

**4. The correction is doing real work at long context.** Same draws with and without `log(1/r)`, 8x,
seed 1:

| | effect of the correction |
|---|---|
| pooled over 8 tasks | +2.80 ± 1.31 (p = 0.027) |
| prefix_suffix | **+16.0**. Without the correction the sampled cache scores 0, exactly like top-k. |
| vt | +6.4 |
| qa_eng | +5.2 |
| gov_report | +1.3 |
| qasper, hotpotqa, passage_retrieval_en | -0.1, -2.7, -4.0 (not significant) |

This answers part of the question the E3 audit left open (what causes the win at 110k): on
prefix_suffix, the correction and not just the random selection makes the difference.

**5. Interpretation (hypotheses, not tested here).**
- **Where sampling helps.** Sampling helps most on the long-context tasks where top-k falls furthest:
  prefix_suffix, vt, and summary at 110k. One candidate explanation is that top-k removes connective
  low-score tokens the answer needs.
  - The E3 audit found the "deleted span interior" version of this unsupported on prefix_suffix
    (top-k keeps word interiors at the base rate).
  - So the mechanism remains open. Finding 4 shows only that the correction is needed there.
- **Where it doesn't.** At LongBench length, top-k on a good scorer already keeps enough, and sampling
  mostly adds variance (seed-to-seed standard deviation of 0.6-7.0 points for kvzip_bern on qasper and hotpotqa, largest at 16x).
- **Why Compactor might benefit less.** Its per-head min-shifted z-scores may give a flatter keep
  distribution than KVzip's max-attention scores. More low-probability tokens would then be kept, with
  large corrections.
  - Consistent with this: a larger seed-to-seed standard deviation for compactor_bern on hotpotqa (3.4-6.7 points against 0.6-7.0 for kvzip_bern, and 3.4-5.4 against 0.6-2.0 at 2x-8x), and slightly
    more repetitive answers than plain Compactor on qasper (3% against 2%).
  - Not measured: the distribution of keep probabilities was not logged per arm. Whether a different
    score map (for example `exp(s)`, or only Compactor's attention component) does better is untested.

**6. Top-k makes vt answers degenerate; sampling does not.** On vt, an answer is counted as degenerate
if it repeats a 3-word phrase 3+ times, typically a variable assignment followed by the context's filler
sentences:

| arm | share degenerate |
|---|---|
| no_press | 34% |
| kvzip_bern | 34% |
| compactor_bern | 35% |
| kvzip (top-k) | 60% |
| compactor (top-k) | 82% |

So both sampled caches keep the model's output as well-formed as the full cache does. Elsewhere the rates
are the same across arms: 0-3% on the QA and retrieval tasks, and 82-93% on gov_report, whose long
summaries repeat phrases naturally.

## Trust and limits

**Harness.**
- Every job passed T1-T10.
- LongBench reproduces E8: 10 of the 12 compared runs give identical answers; qasper kvzip at 4x and 8x
  is 98% identical, differing in one answer each.
- Kept fractions are on target.
- Per-question scores are recomputed and match every `metrics.json`.

**Two fixes to stock kvpress Compactor were needed. Both are recorded and tested.**
- **Seeded sketch.** Stock draws its random sketch unseeded, so its scores change on every call.
- **Cholesky fallback.** At ~124k tokens the stock code crashes ("Cholesky failed"). A float64 fallback
  is used only on failure: 160 layer calls on vt, never elsewhere. Stock output is unchanged wherever
  stock works (T10).

**Small n.**
- SCBench has 10 independent contexts per task. Per-task tests there are weak, and the claims rest on
  the pooled tests.
- C2 does not survive multiple-testing correction, so the Compactor result is suggestive, not
  established.

**GPU types.**
- The new SCBench arms ran on A100s. compactor_bern needs 51 GB at 124k, more than an L40S has.
- E3's reused prefix_suffix and summary arms ran on L40S.
- Each selection pair is on one GPU type. Only the KVzip-vs-Compactor and vs-no_press comparisons mix
  types on those two tasks.

**Memory.** Memory is masked, not freed: the Bernoulli arms keep the full cache in GPU memory, so "8x"
describes behaviour only. The current SDPA bias path also slows decoding (Bernoulli runs took about
1.5-2x as long as top-k).

**Other limits.**
- One model.
- The Compactor score map (per-head minimum shift) is our choice.
- vt answers are often degenerate (a variable assignment followed by repeated filler), which makes vt
  noisy.

## For the professor, in one paragraph

On 8 tasks (4 LongBench, 4 SCBench, 50 questions each), Bernoulli sampling with the Horvitz-Thompson
correction beats top-k on KVzip's scores by about 5 points at 8x overall. The gain is about 8 points at
~110k-token SCBench contexts, where sampling stays near the full cache at 4x and 8x while top-k loses
8-11 points. At LongBench lengths the two tie, and at 2x sampling never hurts. On Compactor's scores
sampling helps less (+2.5 at 8x, not significant after correction) and is inconsistent across tasks.
KVzip's scores are much stronger than Compactor's under either selection rule. Removing the correction
costs 2.8 points on average and 16 on prefix_suffix, so the reweighting matters. By the pre-registered
rule the result is "not consistent across both benchmarks": the LongBench QA tasks are ties.

## Next steps

1. **Real memory savings and flash attention.** Carry the bias as an extra key coordinate (`k' = [k,
   b*sqrt(d)]`, `q' = [q, 1]`, scale 1/sqrt(128)), which is mathematically identical to the current bias.
   With flash-attn's variable-length mode per KV head, this would also drop tokens for real and remove
   the slow full-size bias mask.
2. **Compactor score map.** Try `exp(s)` and the attention-only component. Check whether the 16x SCBench
   loss and the 2x slight harm go away.
3. **More SCBench contexts** (for example 30 per task) to give per-task tests real power at ~110k.
4. **Report the Compactor Cholesky bug upstream** (it is also in skylight-org `main`), with the fallback
   as a proposed fix.
