# Phase 2, Experiment 1 — How vacuous is the denominator-only Bernoulli bound?

**Date:** 2026-09-15. **Maths:** `../DERIVATION_denominator_bernoulli.pdf` (the attached derivation).
**Runtime target:** about half an hour on one A40.

---

## 1. Why this experiment exists

Phase 1 (E5a → E9) built a certificate for the **whole attention output** `A/B`. It needed a
numerator bound, a denominator bound, a χ factor to combine them, and a ratio inversion. Each piece
added looseness, and the final answer on LongBench was disappointing: at η = 0.2 the median head
needed 0.84n of its cache certified, which is barely compression at all.

Phase 2 throws most of that away and keeps **only the denominator**. The object is

    B(q) = Σ_i w_i(q),    w_i(q) = exp(⟨q, k_i⟩/√d)

and the claim is a bound on the *relative* error `|B̂ − B| / B`. No numerator, no χ, no ratio
inversion, no value vectors at all. This is a smaller claim, but it is a clean one, and the question
of this experiment is simply: **is it tight enough to be worth anything?**

A bound can fail in two opposite ways, and we check for both:

- **Unsound** — real errors exceed it more than δ of the time. Then it is wrong.
- **Vacuous** — it is true, but so loose it only ever permits budgets near the full cache. Then it is
  useless.

Phase 1's experience says the second failure mode is the one to watch.

---

## 2. The method being tested

Straight from the derivation, per KV head:

1. **Score.** `τ_i = max over calibration queries of ω_i(q)`, where `ω_i = w_i / B` is token i's
   softmax weight. So τ_i is the largest share of the denominator that token i ever claims.
   `T = Σ_i τ_i`.
2. **Keep probability.** For a budget constant `c > 0`, `r_i = min(1, c·τ_i)`, and keep token i with
   probability r_i, independently.
   - Tokens with `c·τ_i ≥ 1` are kept outright. **These are the heavy hitters, and they arise on
     their own** — they are not a separate mechanism bolted on. They contribute exactly zero error.
   - All the randomness lives in the tail `L = {i : c·τ_i < 1}`.
3. **Estimator.** Horvitz–Thompson: `B̂ = Σ_i (Z_i / r_i)·w_i`. Unbiased by construction.
   Equivalently: add a bias `log(1/r_i)` to each kept token's logit and take the softmax
   denominator over the kept set.
4. **The bound** (§10 of the derivation), for `ℓ = log(2/δ)`:

       |B̂ − B| / B  ≤  ℓ/(3c) + √( ℓ²/(9c²) + 2·ν_L(c)·ℓ/c )

   where `ν_L(c) = min(1, Σ_{i∈L} (1 − c·τ_i)·τ_i)` is the tail's variance mass.

5. **Budget inversion** (§11). Given a target ε, the smallest admissible c solves

       c > (2/3)·log(2/δ)·( 3·ν_L(c)/ε² + 1/ε )

   Both sides move the right way with c — the left side grows, the right side does not — so
   bisection finds the exact crossing. The memory that buys is `M(c) = Σ_i min(1, c·τ_i)`.

### One honest deviation from the PDF

The PDF defines τ_i as a supremum over the norm ball `𝒬 = {‖q‖ ≤ ρ}`. Phase 1 measured what that
gives on real caches: the ball is far too generous, τ_i saturates, `T ≈ n`, and every bound built on
it is trivially vacuous (the Sensitivity experiment found `S = 0.980n`).

So, as confirmed, **τ_i is the max over an empirical calibration set**, not over a ball: we ask the
model to repeat the context back to itself (KVzip's trick) and take the max ω_i over those repeat-pass
queries. The consequence has to be stated plainly:

> For queries in the calibration set this is a theorem. For a *future* query it is a theorem only if
> that query obeys `w_i ≤ τ_i·B` — inequality (2.1) — which is now an **assumption, measured rather
> than proved**. Checking how often that assumption survives contact with real questions is exactly
> what Part 2 of this experiment does.

This is the same transfer assumption E9 tested, and E9's answer was encouraging (B3 failed on 0.43%
of claims). The denominator-only version should do at least as well, since it drops the numerator's
requirements.

---

## 3. What gets measured

**Contexts:** 4 LongBench documents, one each from qasper, multifieldqa_en, hotpotqa and
passage_retrieval_en, 1.5k–8k tokens, chosen deterministically (seed 9, the same rule as E9).

**Calibration:** the repeat pass, 512 evenly spaced query positions. τ is maxed over those *and* over
the 4 query heads that share each KV head (GQA — the kept set is per KV head, so it must satisfy all
of them).

**Future queries — the actual test set:** the real question token plus answer steps 1, 2, 4, 8, 16,
at their true positions, generated greedily with the full cache. These are never seen by calibration.

**Crucially: nothing is deleted.** Every token's K stays in memory, so for any draw Z we compute the
true `B` and the estimate `B̂` exactly, and the error is measured rather than inferred.

**Heads:** all 1024 (32 layers × 32 KV heads... actually 32 × 8 = 256 KV heads) get a τ scan and a
certificate. Monte Carlo goes on 8 layers (0, 4, …, 28) × 4 KV heads = 32 KV heads × 4 query heads,
with 300 Bernoulli draws each. That is the same subsampling E9 used, and it is what keeps this to
half an hour.

---

## 4. The two parts

### Part 1 — fixed compression (2× and 4×)

Solve `M(c) = n/2` and `M(c) = n/4` for c by bisection. At each:

- report the bound's value ε(c) — what the certificate promises;
- draw 300 times, measure `|B̂ − B|/B` for every future query;
- report **the failure rate** `Pr[error > ε]`, which should sit at or below δ = 0.05;
- report the **looseness ratio** ε(c) ÷ (p95 of the real error), the headline vacuity number.

A looseness of 1 would be perfectly tight. Phase 1's full-attention bound ran 7.6× at its best rung
and 29× at its base rung.

### Part 2 — the budget ladder at ε = 0.1

1. Invert the bound at ε = 0.1, δ = 0.05 to get `c*`, and its memory `M* = M(c*)`.
2. Then test at **M\***, **M\*/2** and **M\*/4**, each with its own c by re-solving `M(c) = target`.

At each of the three budgets, report failure rates against **two different yardsticks**:

- **against the original ε = 0.1** — this shows how fast the promise breaks when you spend less than
  the certificate asked for. At M\* it should hold; at M\*/2 and M\*/4 it is expected to break, and
  the interesting number is *how badly*.
- **against that budget's own re-certified ε(c)** — this checks the bound stays internally honest at
  every budget, not just the one it was inverted at. This should hold everywhere. If it does not,
  something is wrong with the maths or the code, not with the method.

Reading the two together separates "the bound is loose" from "the bound is broken".

---

## 5. Success criteria, written before the run

| | Criterion | Passes if |
|---|---|---|
| **P1** | The bound is sound on calibration queries | failure rate ≤ δ + 3SE at every budget (a theorem — a breach here means a code bug) |
| **P2** | It transfers to future queries | ≥ 95% of (head, budget) pairs have failure rate ≤ δ + 3SE |
| **P3** | It is not vacuous | median looseness ≤ 5× at 2× compression |
| **P4** | It permits real compression | median certified memory at ε = 0.1 is below 0.5n |

P3 and P4 are the ones in genuine doubt. P4 is the sharper test: E9's full bound certified only 10.6%
of heads below half their cache, and if dropping the numerator does not fix that, the simplification
has not bought anything.

SE at δ = 0.05 with 300 draws is 1.3%, so δ + 3SE ≈ 0.089.

---

## 6. Files

    phase2/experiment1/
      DESIGN.md        this document
      denom.py         the bound, the scheme, the budget inversion — the ONLY implementation
      run_exp1.py      GPU: extract, calibrate, Monte Carlo, write results
      report.py        CPU: tables + figures from the CSVs
      job_exp1.sbatch  SLURM
      out/             head_cert.csv, mc.csv, REPORT.md, figures/

`denom.py` is deliberately separate from E9's `cert.py`. That file implements the full-attention
ladder B0–B4; this is a different, smaller claim, and mixing them would invite the copy-paste bug
Phase 1 already hit once (the m-bound was inlined in three files and all three had to be fixed).

---

## 7. What this can and cannot conclude

**Can:** whether the denominator bound holds on real queries, how loose it is in error units and in
memory units, and whether it certifies useful compression.

**Cannot:** anything about the attention *output*. A correct denominator does not imply a correct
`A/B` — the numerator can be wrong in ways this never sees. And it says nothing about task accuracy.
Those are later experiments; this one establishes whether the foundation is worth building on.
