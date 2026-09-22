# Phase 2, Experiment 2 — Conclusion

**Run:** overnight 2026-09-15/16, 49 evaluation runs, ~30 GPU-hours on A40 and L40S.
**Numbers:** `out/REPORT.md` (auto-generated) · **bugs and checks:** `AUDIT.md` · **timeline:** `STATUS.md`
**Question:** as a compression method, does Bernoulli sampling beat top-k and the published
query-agnostic baselines on task accuracy?

---

## The answer in three lines

1. **Sampling is not worse than top-k, and is slightly better where the cache gets tight** — +0.79 ± 0.51
   at 8×, replicated at +0.63 ± 0.55 on independent hardware. Nothing reaches 2 SE.
2. **The advantage is not uniform across tasks: it is concentrated in aggregation.** On `cwe` (count the
   10 most common words) sampling gains **+13 at 8× and +72 at 16×**. On needle retrieval it gains nothing.
3. **Both KVzip-scored arms crush the cheap query-agnostic baselines** — 93.8 vs 45.2 (ExpectedAttention),
   26.0 (SnapKV), 25.7 (StreamingLLM) at 8×. The score matters far more than the selection rule.

---

## Pre-registered criteria

| | criterion | verdict | evidence |
|---|---|---|---|
| **B0** | harness trustworthy | **PASS** | `random` scores 7.0 / 0.3 / 0.0; hardware difference −0.10 ± 0.16 |
| **B1** | sampling ≥ top-k, everything else fixed | **PASS** | +0.31 / −0.02 / +0.79 at 2× / 4× / 8×; never significantly worse |
| **B2** | competitive with the field | **PASS** | 95.5 vs 90.6 at 2×; 95.5 vs 76.8 at 4× |
| **B3** | degrades gracefully | **PASS** | drop from full cache at 8×: sampling 1.58, top-k 2.36 |
| **B4** | memory honest | **PASS** | within 0.01% of target; 100% of samples inside the §11 bound |
| **B5** | the log(1/r) bias earns its place | **PASS** (weak) | +0.39 ± 0.30 at 4×, +0.50 ± 0.40 at 8× — positive, ~1.3 SE |
| **B6** | calibration scope | **no effect** | full-context vs chunked: −0.26, −0.30, +0.07 |

**B1 is the experiment**, and it was a genuinely controlled comparison: `kvzip` and `kvzip_bern` share
their scoring code byte for byte (same chunked repeat pass, same sink handling, same budget). Only the
selection rule differs.

---

## What the numbers say

### The headline table (RULER 4096, 780 samples per run, paired per sample)

| | 2× | 4× | 8× | 16× (exploratory) |
|---|---|---|---|---|
| full cache | 95.42 | | | |
| `kvzip` (top-k) | 95.22 | 95.50 | 93.05 | 73.86 |
| `kvzip_bern` (sampling) | **95.53** | **95.48** | **93.84** | **74.83** |
| paired difference | +0.31 ± 0.31 | −0.02 ± 0.32 | +0.79 ± 0.51 | +0.97 ± 1.15 |
| ExpectedAttention | 90.60 | 76.82 | 45.18 | — |
| SnapKV | 68.72 | 40.21 | 25.99 | — |
| StreamingLLM | 60.32 | 36.73 | 25.66 | — |
| `random` | 7.01 | 0.27 | 0.00 | — |

### The finding worth keeping: sampling helps *aggregation*, not retrieval

Per-subtask differences (sampling − top-k) are not spread evenly. They sit almost entirely on one task:

| | `cwe` (count common words) | needle subtasks |
|---|---|---|
| 8× (A40) | **+13** | −1 to +2 |
| 8× (L40S, replication) | **+17** | −1 to +1 |
| 16× (L40S) | **+72** | mixed, −41 to +40 |

This is exactly what the estimator predicts. `cwe` needs the *aggregate* weight of many individually
unimportant tokens. Top-k sorts by score and throws the whole tail away, so that mass is simply gone.
Bernoulli keeps a random subset of the tail and reweights it by 1/r, so the aggregate survives in
expectation. Needle retrieval depends on one high-score token that both methods keep (`r = 1`), so
sampling has nothing to add — and at 16× the needle results swing wildly in both directions, which is
sampling variance, not skill.

**So the right claim is not "sampling beats top-k". It is "sampling preserves distributional
information that top-k discards, and that matters when the task aggregates over the context."** A
benchmark of needle tasks will always understate this; `cwe` is the only RULER subtask that really tests it.

### The other three results

- **The bias helps a little, consistently.** +0.39 and +0.50, both positive at every ratio and every seed,
  each about 1.3 SE. Consistent with the theory (the correction is what makes the estimator unbiased) but
  too small to prove from task scores alone. Experiment 1 showed its effect on attention error is large;
  most of that is evidently absorbed before it reaches the output token.
- **Full-context calibration buys nothing** (−0.26, −0.30, +0.07). At 3.8k tokens KVzip's 2048-token
  chunking is only two chunks, so there is little to gain — this result should not be extrapolated to
  long contexts, where chunking is a much cruder approximation.
- **Memory accounting is exact:** mean kept fraction within 0.01% of target at every ratio, and every
  single sample of 780 × 8 runs fell inside the derivation's §11 high-probability bound. That is a clean
  empirical confirmation of one piece of the maths.

---

## Honest limits

- **No memory is actually saved.** kvpress masks dropped tokens rather than freeing them, because
  per-head selection needs a ragged cache (`AUDIT.md`). "8×" means the model behaves as if 7/8 of the
  cache were gone. Every arm is handicapped identically, so the comparison is fair, but the wall-clock
  and memory benefits are unmeasured.
- **One model, one context length** (Llama-3.1-8B-Instruct, RULER 4096). RULER 8192 and 16384 are cached
  and ready. Whether the ranking holds at longer context is open, and is the single most important gap.
- **Near the ceiling at 2× and 4×.** Full cache scores 95.4 and top-k 95.2–95.5, so those ratios cannot
  discriminate. The informative points are 8× and 16×, which is why 16× was added mid-run.
- **Baselines are kvpress reimplementations**, so absolute numbers need not match each method's paper.
  Comparisons *within* this harness are fair.
- **Nothing here is about the certificate.** This experiment dropped it entirely; Experiment 1 covers it.

---

## How much of this run was spent on bugs

Thirteen defects were found and fixed (`AUDIT.md`), and the pattern is worth stating plainly: the ones
that mattered all produced **plausible numbers rather than errors**.

- A boolean attention mask silently inverted causal masking (a zero bias moved logits 19%).
- Clearing the bias at the wrong point in kvpress's lifecycle made all nine first-run Bernoulli results
  **exactly equal to `no_press`** — a complete null that looked like a real experiment. Those runs are
  quarantined in `results/P1_INVALID_*`.
- `bern_tau` calibrated on 512 of ~3800 repeat positions, missing the needles, which would have turned
  B6 into a measurement of subsampling rather than calibration scope.

Kernel-level self-tests (T1–T4) passed through two of those. What caught them was end-to-end checking:
exact equality with `no_press`, per-subtask collapse, and an instrumented count of bias applications
during generation. **T6 now checks the bias is live during decoding**, which is the test that would have
caught the worst of them on day one.

---

## What to do next

1. **RULER 8192 / 16384** — the cached follow-up. If sampling's advantage grows with context, that is the
   paper's headline; if it vanishes, the honest conclusion is that this is a 4k-only effect.
2. **A benchmark with more aggregation tasks.** RULER has exactly one (`cwe`) and the effect is large
   there. This is now a specific, testable hypothesis rather than a hope.
3. **Back to the certificate, with the query-aware idea.** Experiment 1 showed the query-agnostic bound is
   10–14× loose because τ is a max over all calibration queries. Experiment 2 shows the *compressor* is
   already competitive without any certificate. The gap between them is where the remaining value is.
