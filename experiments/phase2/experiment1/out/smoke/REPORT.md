# Phase 2, Experiment 1 — How vacuous is the denominator-only Bernoulli bound?

Bound: `|B̂ − B|/B ≤ u + √(u² + γu)` (derivation §10), δ = 0.05, 300 Bernoulli draws per (head, budget). SE of a failure rate at δ is 0.013, so the pass threshold is δ + 3SE = **0.088**.

Contexts: **1** (qasper n=4964). Certificates on **256** KV heads; Monte Carlo on **2** KV heads.

## 0. What calibration produced

`T = Σᵢ τᵢ` is the derivation's total score: T ≈ 1 means the head is compressible, T ≈ n means it is not.

| | median | p10 | p90 | max |
|---|---|---|---|---|
| T | 16.8 | 3.22 | 159 | 391 |
| T/n | 0.00338 | 0.000648 | 0.032 | 0.0788 |

By depth (median T/n): layers 0–12 **0.003**, layers 13–31 **0.006**.

## 1. Certified memory (P4)

Fraction of the head's cache that must be kept to certify ε, before any question is seen. 1.00 = no compression certifiable. `M` is expected memory, `M⁺` the high-probability bound (§11).

| ε | M median | M p10 | M p90 | M⁺ median | heads < 0.5n | heads < 0.25n |
|---|---|---|---|---|---|---|
| 0.1 | 0.686 | 0.200 | 0.921 | 0.715 | 33.2% | 14.1% |
| 0.2 | 0.363 | 0.078 | 0.830 | 0.384 | 60.9% | 36.7% |
| 0.3 | 0.220 | 0.037 | 0.656 | 0.237 | 77.0% | 54.7% |

**P4 (FAIL):** median certified memory at ε = 0.1 is **0.686n** (passes if < 0.5n).

By depth, median certified memory at ε = 0.1: layers 0–12 **0.645n**, layers 13–31 **0.732n**.

## 2. Does the bound hold? (P1, P2)

`fail_own` = Pr over draws that the real error exceeds **that budget's own certificate**. On calibration queries this is a theorem (§10); on future queries it is the transfer assumption (2.1).

| budget | M/n | ε certified | calib: share over δ+3SE | calib: max | future: share over δ+3SE | future: max |
|---|---|---|---|---|---|---|
| `comp2x` | 0.500 | 0.3945 | 0.00% | 0.000 | 2.50% | 0.300 |
| `comp4x` | 0.250 | 0.7096 | 0.00% | 0.000 | 7.50% | 0.350 |
| `eps0.1_M` | 0.945 | 0.1 | 0.00% | 0.000 | 0.00% | 0.000 |
| `eps0.1_M/2` | 0.473 | 0.4157 | 0.00% | 0.000 | 5.00% | 0.190 |
| `eps0.1_M/4` | 0.236 | 0.7439 | 0.00% | 0.000 | 7.50% | 0.280 |

**P1 (PASS):** on calibration queries 0.00% of (head, budget, query) exceed δ+3SE. This is a theorem — anything above ~0 is a code bug.
**P2 (PASS):** on future queries **95.50%** of pairs hold (passes if ≥ 95%).

Future-query failures by answer step (share over δ+3SE): step 1: 0.0%, step 2: 10.0%, step 4: 5.0%, step 8: 7.5%, step 16: 0.0%.

By task: qasper: 4.5%.

## 3. How loose? (P3)

Looseness = certified ε ÷ the p95 of the error actually measured. 1.0 would be perfectly tight.

| budget | M/n | ε certified | p95 error (median) | looseness (median) | p90 |
|---|---|---|---|---|---|
| `comp2x` | 0.500 | 0.3945 | 0.04663 | **8.3×** | 15.6× |
| `comp4x` | 0.250 | 0.7096 | 0.1098 | **5.6×** | 10.9× |
| `eps0.1_M` | 0.945 | 0.1 | 0.003605 | **27.8×** | 88.8× |
| `eps0.1_M/2` | 0.473 | 0.4157 | 0.05319 | **7.4×** | 14.8× |
| `eps0.1_M/4` | 0.236 | 0.7439 | 0.1111 | **5.9×** | 12.1× |

**P3 (FAIL):** median looseness at 2× compression is **8.3×** (passes if ≤ 5×).

## 4. The ε = 0.1 ladder — spending less than the certificate asks for

Two yardsticks at each budget, as pre-registered. **Own certificate**: does the bound stay internally honest when re-certified at that budget (it should, everywhere). **ε = 0.1**: does the original promise survive (it should hold at M\*, and break below it).

| budget | M/n | ε re-certified | future fail vs own cert | future fail vs ε=0.1 | median error p95 |
|---|---|---|---|---|---|
| `eps0.1_M` | 0.945 | 0.1 | 0.00% | 0.00% | 0.003605 |
| `eps0.1_M/2` | 0.473 | 0.4157 | 5.00% | 22.50% | 0.05319 |
| `eps0.1_M/4` | 0.236 | 0.7439 | 7.50% | 47.50% | 0.1111 |

The gap between the last two columns is the whole point: the bound stays true about *its own* claim at every budget, while the ε = 0.1 promise degrades as the budget is cut.

## 5. Verdict

| | criterion | verdict |
|---|---|---|
| P1 | sound on calibration queries (theorem) | PASS |
| P2 | transfers to future queries (≥95%) | PASS |
| P3 | not vacuous (looseness ≤ 5× at 2×) | FAIL |
| P4 | certifies real compression (median < 0.5n at ε=0.1) | FAIL |

(figures skipped: Axes.boxplot() got an unexpected keyword argument 'labels')
