# Phase 2, Experiment 3 — Result Audit

**Date:** 2026-09-16 · **Audited:** `CONCLUSION.md`, `out/REPORT.md`, `AUDIT.md`, `STATUS.md`, `DESIGN.md`,
and every script that produced them, plus the kvpress v0.5.4 code they depend on.
**Method:** I read the code line by line. I then recomputed every headline number from the raw
`results/S1n10/*/predictions.csv` with an independent script that does not import `summarize.py`. Finally
I tested the mechanistic claims against the saved score tensors in `out/score_cache/`. Scripts and outputs are in
`audit_results/` (see §7).

This is an independent check. It is **not** the running bug log that `AUDIT.md` is.

---

## 1. Bottom line

**The central result is real. Several claims around it are not supported as written.**

| Claim in `CONCLUSION.md` | Audit verdict |
|---|---|
| Bernoulli sampling beats KVzip top-k at 4× and 8× on ~110k SCBench | **Supported.** Pooled across tasks, exact sign-flip test: **p = 0.0002 at 4×, p = 0.00003 at 8×** |
| At 8× Bernoulli loses 0.4–5.6 pts vs uncompressed; top-k loses 6.4–20.0 | **Numbers reproduce exactly** |
| At 4× Bernoulli is lossless within noise | **Supported, but on 3 tasks, not 4** (`vt` has no 4× run). The 95% CIs are as wide as ±6 pts on `prefix_suffix` |
| `prefix_suffix`: top-k scores exactly 0 at ≥4×, and it is real | **Supported.** 0/50 at three ratios, and `expected_attention` also scores 0 |
| …*because* top-k deletes the low-scoring interior of the answer word | **Not supported.** Score data contradicts it (F4) |
| C1 PASS "with caveat" | **The reasoning is invalid, but the verdict is recoverable** with a proper small-sample test (F2) |
| C3 PASS | **Mis-graded. By the pre-registered wording, C3 FAILS** (F1) |
| C5: `T/n` median falls 2.4–8× with length, tail rises from 0.15 to ~1 | **Not a valid comparison.** E1 and E3 compute different statistics (F3) |
| Score cache verified: "identical predictions on 10/10 contexts" | **Not what the logs show.** The cache itself is fine; the stated evidence is wrong (F5) |
| "T1–T6 pass at full 110k length" | **False.** The self-tests run on ~1.4k- and ~6k-token synthetic contexts (F6) |
| E2's "aggregation only" story was a 4k artefact | **Confounded.** Benchmark and length both changed, and "retrieval" is one task at 3 of 4 ratios (F8) |

Nothing found here invalidates the controlled comparison. Both arms read the same bytes, are budget-matched, and run
on the same GPU within a task. The problems are in interpretation, grading and documentation. One gap is
experimental: nothing isolates *why* sampling wins (F7).

---

## 2. What was checked and found correct

These are closed. Each one was verified, not just read.

| Item | How verified |
|---|---|
| **Bias kernel is live.** kvpress falls back to SDPA because `flash_attn` is not installed (FlashAttention would silently ignore an additive mask) | log line "Flash Attention 2 not available"; `find_spec('flash_attn') is None` in env `kvpress` |
| Boolean-mask → additive conversion, and the causal mask rebuilt when the mask is `None` | `bern_press.py:_bias_patch`, and T1 = 0.0 exactly |
| Bias is attached during question prefill and decode, and absent during context prefill | T6: 256 generation calls with bias, 0 prefill calls; blocked fraction 0.8754 at 8× |
| GQA expansion `repeat_interleave(G, dim=1)` matches Llama's `repeat_kv` ordering | code reading |
| **Equal memory.** Bernoulli expected keeps = `(1-ratio)·numel` = top-k's keep count; forced sinks are paid for out of the same budget | `compress_post` re-solve loop; T4b budget error 2e-16; C4 realised within 0.012% |
| Sinks kept by both arms (top-k because score = 1.0 is the maximum; Bernoulli because it is forced) | code reading, T4b |
| **Same score for both arms.** Cache key = SHA-1 of the context token ids | I re-tokenised all 10 `prefix_suffix` contexts through the chat template and **all 10 hashes matched cache filenames**, so keying is correct |
| Cached vs freshly computed score is byte-identical | uncached `vcache` run and cached `S1n10` run give identical `c` (4.446896264068596 vs …595) and identical `T/n` on the shared contexts |
| Bernoulli draws seeded by context content, so contexts are independent | `zlib.crc32(context ids) ^ seed·7919` |
| No truncation; over-length samples dropped and counted | `scbench_data.py`, `lengths.json`. The longest kept context (130,839 tok + 29 query + 64 gen) fits 131,072 |
| Paired join on `(context_id, turn)`; SE clustered on context | `summarize.py:paired`, and my independent recomputation matches every diff and SE in `REPORT.md` |
| Per-query scorer equals `metrics.json` | asserted for all 82 runs in `summarize.py`, reproduced here |
| No cross-hardware confound inside a task | `cost.json`: `prefix_suffix` 25/25 L40S, `summary` 25/25 L40S, `qa_eng` 25/25 A100, `vt` A100 |
| `vt` top-k 8× result is in the numbered subdir `…kvzip__0.88/1/` (created on requeue) | `summarize.py` uses `rglob`, so it finds it. My first loader did not; fixed |
| Metric choices (substring / string_match_all / token F1 / ROUGE-L) are sensible and normalised identically across arms | `scbench_metrics.py` |

---

## 3. Findings

Ranked by how much they change what can be claimed.

### F1 — C3 is mis-graded: by its own pre-registered wording it FAILS · *severity: high*

`DESIGN.md` §5 says C3 passes if on `summary`/`qa_eng` the sampling-minus-top-k difference is
"≥ 0, **and larger than on the retrieval tasks**". `summarize.py` tests only the first clause.

| family | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| retrieval | −4.67 | **+20.67** | **+12.00** | **+3.33** |
| aggregation | **+2.02** | +5.75 | +4.39 | +0.06 |

Aggregation is larger at only 1 of 4 ratios. `CONCLUSION.md`'s own narrative says exactly this ("retrieval is now
*larger*, not zero") and then labels it PASS anyway.
**Fix:** C3 = **FAIL: the aggregation-specific hypothesis is refuted.** That is an interesting result,
and the prose already makes it. Add the second clause to `summarize.py`.

### F2 — The significance rule is too lenient for 10 clusters; C1's override uses the wrong argument · *severity: high*

`summarize.py` flags a cell as significant when |diff/SE| > 2. With 10 contexts, the right reference is a
t-distribution with 9 degrees of freedom (critical value 2.26) or an exact permutation test. (A sign-flip test
randomly flips the sign of each context's difference to build the null distribution; with 10 contexts all
2¹⁰ = 1024 flips can be enumerated.)

Sampling minus top-k, recomputed (`audit_results/audit_stats.csv`):

| cell | diff ± SE | t-test p | exact sign-flip p | contexts +/− | Holm-adjusted p (13 cells) |
|---|---|---|---|---|---|
| `prefix_suffix` 2× | −4.67 ± 2.23 | 0.066 | 0.125 | 0 / 4 | 1.00 |
| `prefix_suffix` 4× | +20.67 ± 6.55 | 0.012 | 0.016 | 7 / 0 | 0.17 |
| `prefix_suffix` 8× | +18.00 ± 5.07 | 0.006 | 0.008 | 8 / 0 | 0.09 |
| `qa_eng` 4× | +8.77 ± 4.37 | 0.076 | 0.068 | 7 / 3 | 0.68 |
| `summary` 8× | +5.97 ± 1.94 | 0.013 | **0.002** | **10 / 0** | **0.025** |
| `vt` 8× | +6.00 ± 4.67 | 0.23 | 0.25 | 5 / 4 | 1.00 |

Consequences:

1. **C1's failing cell is not significant** under either small-sample test (p = 0.066 / 0.125). So C1 can
   PASS legitimately. The argument `CONCLUSION.md` uses instead ("top-k beat the ceiling there, so the cell is
   noise") is not valid grounds for overriding a pre-registered rule. The test was not specified in
   `DESIGN.md`, so report **both**: FAIL under |z|>2, PASS under t₉ / sign-flip.
2. **"4 cells >2 SE" becomes 3 at p < 0.05**, and after correcting for 13 comparisons only `summary` 8×
   survives on its own. `qa_eng` 4× should not be called significant.
3. **The defensible evidence is pooled.** Each task's per-context differences were standardised and then pooled
   (`audit_results/pooled.py`, 200k sign flips):

   | ratio | tasks | contexts | pooled p |
   |---|---|---|---|
   | 2× | 3 | 30 | 0.26 |
   | **4×** | 3 | 30 | **0.0002** |
   | **8×** | 4 | 40 | **0.00003** |
   | 16× | 3 | 30 | 0.34 |

   **"Sampling beats top-k at 4× and 8× across tasks" is strongly supported.** No individual task-ratio cell
   except `summary` 8× is, after correction.

Minor counting errors in `CONCLUSION.md`: "positive at 11 of 13 cells" is actually **10 of 13**.
"Beats `expected_attention` significantly in 9 of 13" is **10 of 13** under the report's own rule (and
10 under sign-flip p<0.05 as well).

### F3 — C5 compares two different statistics; the "tail rises with length" story is an artefact · *severity: high*

`T/n` in E1 and `T/n` in E3 share a name but not a definition:

| | E1 (`run_exp1.py`) | E3 (KVzip `score_val` in kvpress) |
|---|---|---|
| calibration queries | **512** evenly spaced repeat positions | **every** repeat token (~110k) |
| softmax normalised over | the **full context** (n keys) | **one 2,048-token chunk** + sinks + the repeat chunk itself |
| sinks | ordinary tokens | score forced to 1.0 |

Three consequences, all measured:

1. **E1's max was capped by its query count.** With 512 queries × 4 pooled heads, T ≤ 2,048, so
   T/n ≤ 0.41 at E1's shortest context. E1's top head has T = 746 (layer 15, KV 7). Any head that puts
   ~1 unit of attention on a different token per query reads T/n ≈ 512/n ≈ 0.05–0.1 in E1, and ≈ 1 in E3
   where every token is a query. **The move from "max 0.15" to "max 0.98" is the change in query count, not
   the change in length.**
2. **The tail is a fixed set of heads, not a length effect.** Heads with T/n > 0.1 make up 16–18% of heads on
   `prefix_suffix`, `qa_eng` and `summary`. **The same ~40 (layer, KV-head) pairs** (e.g. (2,5), (4,4), (5,2),
   (8,0), (8,2), (9,6), (13,x), (14,x)) are in the tail in ≥80% of contexts **on all three tasks**. T/n
   correlates with "fraction of tokens scoring > 0.5" at r = 0.97–0.98. That is the signature of
   copy/induction heads under KVzip's "repeat the context" calibration: when reconstructing text verbatim,
   such a head puts nearly all its weight on the next original token, so every token scores ≈ 1.
3. **`vt` confirms the mechanism, and the interpretation of `vt` in `CONCLUSION.md` is backwards.** `vt`'s context is
   five filler sentences repeated **5,165 times each**. A copy head reconstructing it has thousands of
   identical candidates and spreads its weight across them, so no token scores high (max T/n 0.018, zero
   heads above 0.1). That is *diluted* attention, not "attention genuinely concentrated everywhere".

In addition, because KVzip scores each chunk against ~2k–4k keys, **the E3 score does not "see" 110k at
all**. A length effect on compressibility cannot be read off it.

**Fix:** withdraw "median `T/n` falls 2.4–8× with length" and "max rises from 0.15 to ~1 with length",
and Figure 6's E1-vs-E3 overlay. Keep, and relabel as properties of KVzip's score, (a) the bimodal head
distribution and (b) the budget finding below, which is new and useful. To answer Q3 properly,
compute E1's τ (full-context softmax, same number of queries) at 110k. That is the instrumented pass
already listed as "next" in `CONCLUSION.md`.

**What the score cache does show (new, measured).** Under the single global budget, the copy-head tail
absorbs most of the memory:

| task | budget share to T/n>0.1 heads, 4× (top-k / Bern) | 8× (top-k / Bern) | heads keeping <1% at 8× (top-k / Bern) |
|---|---|---|---|
| `prefix_suffix` | 59% / 58% | 88% / 84% | 71% / 65% |
| `qa_eng` | 62% / 61% | 81% / 77% | 60% / 56% |
| `summary` | 58% / 56% | 76% / 72% | 58% / 54% |

At 8×, ~16% of heads take ~80% of the cache and most other heads are essentially emptied, **under both
selectors**. This strengthens the case for per-head budgets (`CONCLUSION.md` "next" item 2) on
measured grounds. It also means the certified-memory argument in C5 ("tail heads set certified memory")
is still an inference: no certificate was computed in E3.

The layer-13 claim ("median `T/n` jumps 11× at layer 13") reproduces on `summary` (11.3×) and `qa_eng`
(11.0×), but it is **3.7× on `prefix_suffix` and 1.5× on `vt`**. It is task-dependent, and the text should say so.

### F4 — The `prefix_suffix` mechanism is not supported by the data · *severity: medium-high*

`AUDIT.md` A2 calls the prefix+suffix concatenations "decisive", and `CONCLUSION.md` says top-k deletes the
low-scoring interior of the answer word. Two checks contradict this:

**(a) Gluing is the model's normal failure mode, not a top-k symptom.** Predictions containing exactly
`prefix+suffix` (regex on the prompt's `prefix = "…"` / `suffix = "…"`):

| arm | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| **`no_press`** (full cache) | **19 / 50** | | | |
| `kvzip` | 19 | 28 | 16 | 6 |
| `kvzip_bern` s1/s2/s3 | 19/20/22 | 21/23/20 | 21/24/21 | 30/29/15 |
| `expected_attention` | 27 | 29 | 33 | 32 |

The uncompressed model glues on 19 of 50 queries. Seeing glued answers from top-k therefore says nothing about
*what* top-k removed.

**(b) The answer word is not preferentially dropped.** I located every gold word's tokens (5 × 10
contexts, 16.7 tokens each, depths 3%–99%) and read their scores from the cache:

| at 4× | global top-k | Bernoulli (expected) | context average |
|---|---|---|---|
| word **interior** tokens kept | 25.3% | 25.2% | 25.0% |
| word **edge** tokens kept | 27.5% | — | 25.0% |
| heads holding the **whole** word | **15.7%** | 13.4% | — |
| heads holding **none** of the word | 63.9% | 55.5% | — |
| in heads where top-k holds none: P(Bernoulli keeps ≥1 word token) | — | 14.8% (0.21 tokens) | — |

The interior scores no lower than anything else, and top-k keeps the *whole* word in *more* heads than
Bernoulli does. What Bernoulli does differently is keep a thin, heavily upweighted (1/r) sample in heads
that top-k empties. Whether that coverage, the 1/r reweighting, or something else (e.g. a few specific
retrieval heads) produces 18–21/50 vs 0/50 is **untested**.

**Fix:** keep the zero as a result. Replace the mechanism paragraph with "mechanism not identified" plus
the table above, and run the ablation in F7.

Also note how thin this headline is. The uncompressed ceiling is **10 correct answers in 6 of the 10
contexts**. Bernoulli's hits overlap those 8–10 of 9–12 times, which is a good sign that it preserves what the model
can do. But the effective sample is ~10 answerable queries.

### F5 — The score-cache verification is misreported; Bernoulli seeds are hardware-dependent · *severity: medium*

`AUDIT.md` A4 and `STATUS.md` say the cached and uncached runs gave "identical predictions on 10 of 10
contexts", checked via `E3_NO_SCORE_CACHE=1`. What the files show:

- No job script or log sets `E3_NO_SCORE_CACHE`. The `vcache` sweep used `n_contexts: 2` (10 predictions,
  not 10 contexts). Its `kvzip` run computed the score fresh and its Bernoulli run was a cache hit.
- The comparison that is possible (`vcache` `kvzip` uncached on A100 vs `S1n10` `kvzip` cached on L40S)
  gives **7 of 10 identical**. The three divergences start mid-generation (chars 19, 172, 226), which is typical
  of cross-GPU floating-point differences in a 256-token greedy decode.
- The same Bernoulli seed produced **different draws** on A100 and L40S. The score and `c` are identical, but `kept_frac`
  is 0.1250125 vs 0.1250185. `torch.rand` on a CUDA generator is not reproducible across GPU types.

**Is the cache safe?** Yes. The score and `c` are bit-identical, and the key matched all 10 re-tokenised contexts
(§2). **Fix the documentation**, not the code. Also add to `bern_press.py`'s docstring that seed
reproducibility holds only on the same GPU model. Within each task all arms ran on one GPU model, so the
comparison is unaffected.

### F6 — Self-tests do not run "at full length" · *severity: medium*

`AUDIT.md` A5, `STATUS.md` and the C0 line in `REPORT.md` say T1–T6 pass "at full 110k length".
`selftests.py` uses `"The capital of France is Paris. " * 200` (~1.4k tokens) for T1–T4 and a
400-item synthetic list (~6k tokens) for T6. **Fix the wording.** A cheap real guard would be to log
`T6`-style bias-call counts and the blocked fraction during the first real 110k context of each job.

Related, minor: T3 shows the `−inf` bias and kvpress's fake-key masking differ by 0.6–1.1% in max
logit. The fake key drives `exp(<q,k>)` towards 0 but not exactly to 0, and the value vectors are not
replaced. So the two arms do not drop tokens by bit-identical mechanisms. Any leakage slightly favours top-k,
so this cannot explain sampling's win.

### F7 — Nothing isolates why sampling wins · *severity: medium (scientific gap)*

The two arms differ in three ways at once: (1) randomised selection, (2) the `log(1/r)` reweighting,
(3) a different spread of budget across heads (F3 table: at 8× Bernoulli leaves fewer heads empty).
E2 measured the bias as worth only +0.4 to +0.5 points at 4k. `DESIGN.md` dropped the no-bias arm on the
grounds that the biased estimator is "the mathematically correct one". That is a reason to prefer it, but
not evidence that it causes the 110k gains. CONCLUSION's "sampling preserves distributional information"
is therefore a hypothesis.

**Cheapest discriminating runs** (scores are cached, so no recalibration is needed): `kvzip_bern_nobias` (already in
`METHODS`) and a deterministic control that spreads budget across heads the way Bernoulli does (e.g.
per-head top-k with Bernoulli's per-head expected budget). Run both on `prefix_suffix` and `summary` at 4× and 8×.

### F8 — Over-reach in the E2-vs-E3 story · *severity: medium*

- "E2's aggregation clause was a 4k artefact" treats **length** as the cause. But E2 was RULER and E3 is
  SCBench, with different tasks, prompts and metrics. Length and benchmark are confounded. Say "did not
  replicate on SCBench at ~110k".
- The "retrieval" family is **`prefix_suffix` alone at 2×, 4× and 16×**. `vt` exists only at 8×, where
  sampling vs top-k is +6.0 ± 4.7 (p = 0.23, 5 contexts up / 4 down). "The advantage appears on every
  task family" therefore rests on one retrieval task, as `CONCLUSION.md` "next" item 4 half-admits.
- "At 4× Bernoulli is lossless on **all four** tasks" (body text) — `vt` has no 4× run. The three-line
  summary says "every completed task", which is correct.
- "Lossless within noise" should carry the interval. The 95% CIs for Bernoulli minus full cache at 4× are
  `prefix_suffix` [−5.5, +6.8], `qa_eng` [−4.5, +2.8] and `summary` [−1.3, +1.8]. A 5-point loss on the first two is not excluded.

### F9 — Smaller issues

| # | Issue | Where | Effect |
|---|---|---|---|
| a | C0 says "every real arm beats `random`", but the code checks only Bernoulli. On `prefix_suffix` ≥4×, `kvzip` and `expected_attention` **tie** `random` at 0 | `summarize.py` C0 | The literal clause fails. Reword C0 as harness sanity; the tie is the headline, not a harness fault |
| b | C4 "memory honest" is near-tautological: the realised count of 3–26 M independent Bernoullis always sits within 0.01% of its mean | C4 | Keep it, but don't present it as strong evidence. Memory is still counted, not freed |
| c | `MAX_NEW["summary"]` is 512 in `measure_lengths.py` and 256 in `scbench_data.py`, though the comment says they must match | loaders | Harmless (512 is stricter), but summaries are capped at 256 tokens |
| d | `template_overhead` is recorded as 2 tokens; the Llama-3.1 system header is ~25–30 | `measure_lengths.py` | No sample actually overflows (largest total ≈ 130,937), but the margin is thinner than recorded |
| e | `memory.json` stores stats in processing order with no `context_id` | `run_sweep.py` | Per-context stats can only be matched by `n`. Add `context_id` |
| f | `per_query()` silently averages however many seeds exist | `summarize.py` | When `vt` finishes, a ratio with only s1 done would be reported as a 3-seed mean. Assert 3 seeds |
| g | `REPORT.md` and `CONCLUSION.md` will go stale as `vt` job 311762 (running, 7h+ of 12h) lands more ratios; the 16× retrieval family will then change | — | Regenerate and re-read after it ends |
| h | `DESIGN.md` §3 still lists `qa_eng` median 163,591 (all samples); the fitting samples' median is 91,991 | `DESIGN.md` | Stale table (already fixed in `REPORT.md`) |
| i | "SCBench, unmodified": SCBench's multi-turn protocol feeds earlier turns as chat history, while kvpress answers each turn independently from the same compressed context | `DESIGN.md` §3 | Fine for this experiment, but ceilings are not comparable to published SCBench numbers. Say so |
| j | `scbench_kv` exclusion ("gold keys at depth 0.00–0.02 of every sampled context") is plausible but not re-verified in this audit | `AUDIT.md` A1 | Low. The exclusion is correct whatever the cause, because `no_press` = 0 |

---

## 4. Pre-registered criteria, re-graded

| | Criterion (DESIGN §5) | CONCLUSION says | Audit |
|---|---|---|---|
| C0 | harness trustworthy | PASS | **PASS as harness sanity**; literal "every arm beats random" fails on `prefix_suffix` (F9a); "full length" wording false (F6) |
| C1 | sampling not significantly below top-k | PASS with caveat | **PASS under t₉ / sign-flip (p = 0.066 / 0.125); FAIL under the report's \|z\|>2 rule.** Test was not pre-specified, so report both (F2) |
| C2 | Bernoulli ≥ `expected_attention` − 1 at 2×/4× | PASS | **PASS** (confirmed) |
| C3 | aggregation diff ≥ 0 **and larger than retrieval** | PASS | **FAIL** (larger at 1 of 4 ratios) (F1) |
| C4 | kept fraction within ±1%, inside §11 bound | PASS | **PASS** (weak test, F9b) |
| C5 | `T/n` at 110k vs E1 — descriptive | "median down, tail up" | **Comparison invalid** (F3); the within-E3 budget-concentration result stands |

---

## 5. What the evidence supports — suggested replacement for "The answer in three lines"

1. **At ~110k on SCBench, Bernoulli sampling beats KVzip top-k at equal (counted) memory at 4× and 8×**
   (pooled sign-flip p = 0.0002 and 0.00003, 30–40 contexts). At 8× it loses 0.4–5.6 points against the full
   cache where top-k loses 6.4–20.0. At 2× and 16× the two are indistinguishable.
2. **The advantage is not specific to aggregation (C3 fails).** Its largest instance is `prefix_suffix`,
   where top-k and `expected_attention` score 0/50 at ≥4× and Bernoulli matches the uncompressed model's
   ~10/50. *Why* this happens is not yet identified. The answer word is not preferentially dropped by top-k.
3. **Under a single global budget, ~16% of KV heads take 60–88% of the cache** under both selectors, and most
   other heads are left nearly empty. These heads are the same across tasks and look like copy heads under
   KVzip's repeat-the-context calibration. That is a concrete argument for per-head budgets. Whether
   compressibility improves with length is **not** answered by E3.

---

## 6. Recommended actions, cheapest first

1. **Edit the text** (no compute): re-grade C3; report C1 under both tests; fix counts (10/13, 10/13);
   "3 tasks" at 4×; withdraw the length claims in C5 and the prefix_suffix mechanism; fix the
   "full length" and "10/10 contexts" wording in `AUDIT.md`/`STATUS.md`; correct the `vt` interpretation.
2. **Harden `summarize.py`:** the second clause of C3; t₉ or exact sign-flip p-values plus a Holm column; the pooled
   per-ratio test; an assertion of 3 seeds per Bernoulli cell; C0 checking every arm.
3. **Ablation (F7):** `kvzip_bern_nobias` + a head-budget-matched deterministic control, on
   `prefix_suffix` and `summary` at 4×/8×. This reuses the score cache, so it costs ~1–2 GPU-hours per task.
4. **More contexts on `prefix_suffix`** (the headline rests on ~10 answerable queries). The score cache makes
   20–40 contexts affordable. A second span-retrieval task would help most.
5. **A like-for-like Q3 measurement:** E1's τ (full-context softmax, fixed query count) at 110k. This is the same
   instrumented pass the query-agnosticism gap needs.
6. After `vt` job 311762 finishes: regenerate `REPORT.md` and re-check F2/F8, since the retrieval family changes.

---

## 7. Reproducing this audit

All scripts are in `audit_results/` and run on the login node in env `kvpress`, with no GPU.

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate kvpress
cd ~/phase2/experiment3/audit_results
python audit_stats.py            # paired stats, t / sign-flip / Holm, cache check, prefix_suffix ceiling -> audit_stats.csv   (~40 s)
python pooled.py                 # per-ratio pooled sign-flip test                                                           (~40 s)
python audit_scores.py           # per-head T/n, copy-head tail, budget share (reads out/score_cache)                        (~3 min)
HF_HOME=$HOME/hf python audit_span2.py   # gold-word retention in prefix_suffix; verifies cache keys -> span2.json            (~2 min)
```

`audit_scores.py` writes `score_rows.npy`. The per-head tables in F3 were printed from it with a short
inline snippet (medians and T/n>0.1 shares per task). The glued-answer counts in F4 came from a
one-off regex over `results/S1n10/*prefix_suffix*/predictions.csv` (pattern `prefix = "(.*?)"` /
`suffix = "(.*?)"`, then `prefix+suffix in prediction`).
