# Phase 2, Experiment 1 — How vacuous is the denominator-only Bernoulli bound?

Bound: `|B̂ − B|/B ≤ u + √(u² + γu)` (derivation §10), δ = 0.05, 300 Bernoulli draws per (head, budget). SE of a failure rate at δ is 0.013, so the pass threshold is δ + 3SE = **0.088**.

Contexts: **4** (qasper n=4964, multifieldqa_en n=7663, hotpotqa n=6395, passage_retrieval_en n=10490). Certificates on **1024** KV heads; Monte Carlo on **128** KV heads.

## 0. What calibration produced

`T = Σᵢ τᵢ` is the derivation's total score: T ≈ 1 means the head is compressible, T ≈ n means it is not.

| | median | p10 | p90 | max |
|---|---|---|---|---|
| T | 22.9 | 3.98 | 284 | 746 |
| T/n | 0.00332 | 0.00055 | 0.0385 | 0.15 |

By depth (median T/n): layers 0–12 **0.002**, layers 13–31 **0.006**.

## 1. Certified memory (P4)

Fraction of the head's cache that must be kept to certify ε, before any question is seen. 1.00 = no compression certifiable. `M` is expected memory, `M⁺` the high-probability bound (§11).

| ε | M median | M p10 | M p90 | M⁺ median | heads < 0.5n | heads < 0.25n |
|---|---|---|---|---|---|---|
| 0.1 | 0.683 | 0.200 | 0.943 | 0.708 | 35.4% | 15.5% |
| 0.2 | 0.355 | 0.067 | 0.857 | 0.375 | 61.7% | 39.8% |
| 0.3 | 0.206 | 0.033 | 0.704 | 0.220 | 75.4% | 55.4% |

**P4 (FAIL):** median certified memory at ε = 0.1 is **0.683n** (passes if < 0.5n).

By depth, median certified memory at ε = 0.1: layers 0–12 **0.606n**, layers 13–31 **0.751n**.

## 2. Does the bound hold? (P1, P2)

`fail_own` = Pr over draws that the real error exceeds **that budget's own certificate**. On calibration queries this is a theorem (§10); on future queries it is the transfer assumption (2.1).

| budget | M/n | ε certified | calib: share over δ+3SE | calib: max | future: share over δ+3SE | future: max |
|---|---|---|---|---|---|---|
| `comp2x` | 0.500 | 0.1906 | 0.00% | 0.000 | 1.41% | 1.000 |
| `comp4x` | 0.250 | 0.3263 | 0.00% | 0.000 | 1.25% | 1.000 |
| `eps0.1_M` | 0.800 | 0.1 | 0.00% | 0.000 | 0.20% | 1.000 |
| `eps0.1_M/2` | 0.400 | 0.2374 | 0.00% | 0.000 | 1.48% | 1.000 |
| `eps0.1_M/4` | 0.200 | 0.406 | 0.00% | 0.000 | 1.21% | 1.000 |

**P1 (PASS):** on calibration queries 0.00% of (head, budget, query) exceed δ+3SE. This is a theorem — anything above ~0 is a code bug.
**P2 (PASS):** on future queries **98.89%** of pairs hold (passes if ≥ 95%).

Future-query failures by answer step (share over δ+3SE): step 1: 0.7%, step 2: 1.1%, step 4: 1.2%, step 8: 1.0%, step 16: 1.5%.

By task: hotpotqa: 1.0%, multifieldqa_en: 1.4%, passage_retrieval_en: 1.7%, qasper: 0.4%.

## 3. How loose? (P3)

Looseness = certified ε ÷ the p95 of the error actually measured. 1.0 would be perfectly tight.

| budget | M/n | ε certified | p95 error (median) | looseness (median) | p90 |
|---|---|---|---|---|---|
| `comp2x` | 0.500 | 0.1906 | 0.01278 | **13.7×** | 54.6× |
| `comp4x` | 0.250 | 0.3263 | 0.03624 | **9.8×** | 41.0× |
| `eps0.1_M` | 0.800 | 0.1 | 0.003945 | **25.3×** | 97.5× |
| `eps0.1_M/2` | 0.400 | 0.2374 | 0.02331 | **11.3×** | 42.0× |
| `eps0.1_M/4` | 0.200 | 0.406 | 0.05281 | **8.8×** | 34.9× |

**P3 (FAIL):** median looseness at 2× compression is **13.7×** (passes if ≤ 5×).

## 4. The ε = 0.1 ladder — spending less than the certificate asks for

Two yardsticks at each budget, as pre-registered. **Own certificate**: does the bound stay internally honest when re-certified at that budget (it should, everywhere). **ε = 0.1**: does the original promise survive (it should hold at M\*, and break below it).

| budget | M/n | ε re-certified | future fail vs own cert | future fail vs ε=0.1 | median error p95 |
|---|---|---|---|---|---|
| `eps0.1_M` | 0.800 | 0.1 | 0.20% | 0.20% | 0.003945 |
| `eps0.1_M/2` | 0.400 | 0.2374 | 1.48% | 11.45% | 0.02331 |
| `eps0.1_M/4` | 0.200 | 0.406 | 1.21% | 24.92% | 0.05281 |

The gap between the last two columns is the whole point: the bound stays true about *its own* claim at every budget, while the ε = 0.1 promise degrades as the budget is cut.

## 5. Verdict

| | criterion | verdict |
|---|---|---|
| P1 | sound on calibration queries (theorem) | PASS |
| P2 | transfers to future queries (≥95%) | PASS |
| P3 | not vacuous (looseness ≤ 5× at 2×) | FAIL |
| P4 | certifies real compression (median < 0.5n at ε=0.1) | FAIL |

Figures: `out/figures/summary.png` (validity, certified memory, looseness).
