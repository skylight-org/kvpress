# Phase 2, Experiment 1 — Conclusion

**Run:** job 310981, 2026-09-15, 4.3 minutes on one A40. Auto-generated numbers: `out/REPORT.md`.
**Question asked:** is the denominator-only Bernoulli bound (`DERIVATION_denominator_bernoulli.pdf`)
tight enough to be useful, or is it vacuous?

**Answer in one line: the bound is correct and it transfers to real questions, but it is
10–14× too loose to set budgets, and the reason is not the inequality — it is the score.**

---

## 1. What was run

4 LongBench contexts (qasper 4964, multifieldqa_en 7663, hotpotqa 6395, passage_retrieval_en 10490
tokens), Llama-3.1-8B-Instruct. Calibration by the KVzip trick: the model re-reads the context after
"Repeat the previous context exactly", and τᵢ is the max softmax share token i takes over 512 repeat
queries, pooled over the 4 query heads sharing each KV head.

Certificates computed for all **1024** KV heads; Monte Carlo (300 Bernoulli draws) on **128** KV
heads × 4 query heads. Future queries — the real question plus answer steps 1, 2, 4, 8, 16 at their
true positions — are never seen by calibration. No token is ever deleted, so `B` and `B̂` are both
computed exactly for every draw.

## 2. The four verdicts

| | criterion | verdict | number |
|---|---|---|---|
| **P1** | sound on calibration queries (a theorem) | **PASS** | 0.00% of 12 800 exceed δ+3SE |
| **P2** | transfers to future queries (≥ 95%) | **PASS** | 98.89% hold |
| **P3** | not vacuous (looseness ≤ 5× at 2×) | **FAIL** | 13.7× |
| **P4** | certifies real compression (< 0.5n at ε = 0.1) | **FAIL** | 0.683n |

### P1 — the maths and the code agree

Zero failures out of 12 800 (head, budget, query) tests on calibration queries, at every budget. This
is what the derivation guarantees, so it is a check on the implementation rather than on the theory,
and it passes cleanly. Independently verified on synthetic data before the run: `M(c)` inverts to 4
decimals, `ε(c)` is monotone non-increasing (so §11's bisection is exact, as claimed), the estimator
is unbiased (max |z| = 3.4 over 40 queries), and the `log(1/rᵢ)` kernel reproduces `B̂` to machine
precision.

### P2 — the transfer assumption largely survives

This was the one genuinely uncertain piece of theory. τ is measured on repeat-context queries, but
inequality (2.1) — `wᵢ(q) ≤ τᵢ·B(q)` — is then *assumed* for a question nobody has seen. It holds for
**98.89%** of future-query tests.

The failures are diffuse rather than structural, which matters:

- 80.5% of KV heads never fail at all; **no head fails more than 16%** of its queries.
- Among failing cases the median exceedance is only **2.0×** the certificate.
- Failures drift up slightly with answer step (0.7% at step 1 → 1.5% at step 16), the same
  generation-time degradation E6 and E9 both saw.
- By task: qasper 0.4%, hotpotqa 1.0%, multifieldqa_en 1.4%, passage_retrieval_en 1.7%.

So there is no population of pathological heads to exclude; a small tail of individual questions
simply falls outside the calibration envelope. **Repeat-context calibration is a reasonable proxy for
future questions on the denominator.** That is a real, positive finding, and it is the part of Phase 1
that carries over intact.

### P3 — but the bound is far too loose

| budget | stored | ε certified | real p95 error | looseness |
|---|---|---|---|---|
| 2× | 0.500n | 0.191 | 0.0128 | **13.7×** |
| 4× | 0.250n | 0.326 | 0.0362 | **9.8×** |
| ε=0.1's own budget | 0.800n | 0.100 | 0.0039 | **25.3×** |

At 2× compression the certificate promises 19% denominator error; the true p95 is 1.3%. The bound is
*true* at every budget — that is P2 — but it is describing a much worse estimator than the one
actually running.

Note the direction of the trend: looseness **grows as the budget grows** (9.8× at 4×, 13.7× at 2×,
25.3× at 0.8n). Bernstein's concentration is at its weakest relative to reality when the tail is
small and heavy hitters carry most of the mass — exactly the regime compression wants to live in.

### P4 — so it cannot certify useful compression

Median certified memory at ε = 0.1 is **0.683n**, and only 35% of heads get below half their cache.
Relaxing ε helps a lot (0.355n at ε = 0.2, 0.206n at ε = 0.3), but at ε = 0.1 the certificate is
close to saying "keep almost everything".

## 3. The ε = 0.1 ladder: loose, not broken

The pre-registered two-yardstick test separates the two failure modes cleanly:

| budget | stored | ε re-certified | fails vs **own** cert | fails vs **ε = 0.1** |
|---|---|---|---|---|
| M\* | 0.800n | 0.100 | 0.20% | 0.20% |
| M\*/2 | 0.400n | 0.237 | 1.48% | 11.45% |
| M\*/4 | 0.200n | 0.200n → 0.406 | 1.21% | 24.92% |

Read the two right-hand columns together:

- **Against its own re-certified ε, the bound holds at every budget** (1.2–1.5%, well under δ). Cut
  the budget and the certificate honestly widens to match. The maths is not breaking.
- **Against the original ε = 0.1 promise, failure climbs to 11% then 25%.** Spending half or a
  quarter of what the certificate asked for does genuinely void the guarantee — the certificate is
  not being pessimistic about *that*.

So the bound is **loose but never dishonest**. Every claim it makes is one it keeps.

## 4. Why it is loose — the diagnosis

The looseness is not mainly in Bernstein. It is in **T = Σᵢ τᵢ**, and this is measurable:

Spearman correlation between `T/n` and certified memory is **0.957** at ε = 0.1 and **0.985** at
ε = 0.3. Certified memory is almost a deterministic function of T. The inequality chain barely
matters by comparison.

And T is large because τ is a **max over ~2000 calibration queries**. Each τᵢ is the largest share
token i ever claimed under *any* repeat query; summing those maxima gives a number far above what any
single query spends. Median T is 22.9 (T/n = 0.0033) — small in absolute terms, which is why some
compression certifies at all — but the distribution has a long tail: p90 of T/n is 0.0385, max 0.15.

Two structural facts:

- **Depth.** Median T/n is 0.002 in layers 0–12 and 0.006 in layers 13–31, and certified memory
  follows (0.606n vs 0.751n at ε = 0.1). This is the same layer-13 boundary E5a and the Sensitivity
  experiment found, arriving here through a completely different statistic.
- **Skew is what pays.** The 68% of heads with T/n < 0.01 certify at a median 0.480n — roughly 2×
  compression. The bound only ever works where τ is genuinely concentrated.

## 5. What this means for Phase 2

**Keep:** the scheme. Bernoulli sampling with `rᵢ = min(1, cτᵢ)` is clean — heavy hitters fall out of
the rule rather than being bolted on, the budget is exactly `Σrᵢ` with no `m`-versus-distinct-tokens
gap, the estimator is unbiased, and the `log(1/rᵢ)` logit bias implements it in one line. The
denominator-only simplification also removed every numerator complication (χ, value norms, the ratio
inversion) at no cost to validity.

**The problem is the score, not the inequality.** Tightening Bernstein would buy a factor of maybe
2–3; the gap is 10–14×, and T explains almost all of it. Three directions, in the order I would try
them:

1. **Stop taking a raw max over calibration queries.** A high quantile of ωᵢ with a measured
   inflation factor (conformal-style) would cut T substantially and cost only a small, *quantified*
   increase in δ. P2's 98.89% suggests there is real headroom here — the max is buying more safety
   than the transfer assumption actually needs.
2. **Use the calibrated second moments.** The variance term `ν_L = Σ(1−cτᵢ)τᵢ` bounds `ω²` by `τω`
   (§7's "degrade the quadratic to a linear" step). E9's B2 rung showed measuring `Σ f ω²` directly
   instead is worth a lot, and it is a small change to `denom.py`.
3. **Empirical Bernstein on the kept sample.** Uses the realised variance rather than the worst case.

**What this experiment does not say.** It bounds the denominator only. A correct `B` does not imply a
correct attention output `A/B` — the numerator can be wrong in ways this never measures — and nothing
here touches task accuracy. Those remain separate questions.

## 6. Reproducing

    cd ~/phase2/experiment1 && sbatch job_exp1.sbatch          # ~4.5 min on one A40
    SMOKE=1 sbatch job_exp1.sbatch                             # 1 context, ~2 min

    denom.py      the bound, the scheme, both inversions — the ONLY implementation
    run_exp1.py   GPU: extract → calibrate → Monte Carlo
    report.py     CPU: tables + figures (rerunnable without the GPU)
    out/          head_cert.csv (1024 heads), mc.csv (12 800 rows), REPORT.md, figures/summary.png

Note `MAX_CTX = 11000`: passage_retrieval_en's shortest context is 10 408 tokens, so the initial
8000-token cap silently dropped that task entirely. Worth watching whenever a task disappears from
the context list.
