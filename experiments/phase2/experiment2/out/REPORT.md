# Phase 2, Experiment 2 — Bernoulli sampling vs the published methods (RULER 4096)

Llama-3.1-8B-Instruct, kvpress v0.5.4, RULER 4096, **780 samples per run** (13 subtasks). Scores are RULER's metric (0–100), averaged over subtasks. All arms score the same samples, so comparisons are paired per sample (± = stratified SE). Bernoulli arms average 3 seeds.

*Memory is counted, not freed:* dropped tokens are masked, so "8×" means the model behaves as if 7/8 of the cache were gone. GPU memory is not reduced.

## Scores

| arm | GPU | full | 2x | 4x | 8x | 16x |
|---|---|---|---|---|---|---|
| `no_press` | A40 | 95.42 | — | — | — | — |
| `no_press` | L40S | 95.32 | — | — | — | — |
| `kvzip` | A40 | — | 95.22 | 95.50 | 93.05 | — |
| `kvzip` | L40S | — | — | — | 92.87 | 73.86 |
| `kvzip_bern` | A40 | — | 95.53 (seed SE 0.05) | 95.48 (seed SE 0.38) | 93.84 (seed SE 0.28) | — |
| `kvzip_bern` | L40S | — | — | — | 93.51 (seed SE 0.33) | 74.83 (seed SE 0.54) |
| `kvzip_bern_nobias` | A40 | — | — | 95.09 (seed SE 0.03) | 93.33 (seed SE 0.08) | — |
| `bern_tau` | A40 | — | 95.27 (seed SE 0.14) | 95.18 (seed SE 0.11) | 93.90 (seed SE 0.17) | — |
| `snapkv` | L40S | — | 68.72 | 40.21 | 25.99 | — |
| `expected_attention` | L40S | — | 90.60 | 76.82 | 45.18 | — |
| `streaming_llm` | L40S | — | 60.32 | 36.73 | 25.66 | — |
| `random` | A40 | — | 7.01 | 0.27 | 0.00 | — |

## B0 — is the harness trustworthy?

- **Hardware check.** `no_press` on L40S vs A40: -0.10 ± 0.16 points. The P2 baselines ran on L40S; the hardware difference is negligible, so they can be compared with the A40 arms.
- **Random floor:** 2x 7.0, 4x 0.3, 8x 0.0. The harness punishes bad token selection.
- **B0: PASS** — the Bernoulli arm beats `random` by >10 points at 2x, 4x, 8x.

## B1 — sampling vs top-k, everything else identical (the central claim)

`kvzip` and `kvzip_bern` share their scores exactly (same chunked repeat pass, same sinks, same budget). The only difference is how tokens are selected: deterministic top-k, or Bernoulli sampling with a log(1/r) bias.

| compression | kvzip (top-k) | kvzip_bern (sampling, 3 seeds) | paired difference | per-seed differences |
|---|---|---|---|---|
| 2x | 95.22 | 95.53 | +0.31 ± 0.31 | +0.36, +0.35, +0.21 |
| 4x | 95.50 | 95.48 | -0.02 ± 0.32 | -0.13, +0.68, -0.62 |
| 8x | 93.05 | 93.84 | +0.79 ± 0.51 | +0.67, +1.32, +0.37 |
| 16x *(exploratory, L40S)* | 73.86 | 74.83 | +0.97 ± 1.15 | +0.20, +2.02, +0.69 |

**Independent-hardware replication of the 8× pair (L40S):** `kvzip` 92.87, `kvzip_bern` 93.51, difference +0.63 ± 0.55 — against +0.79 ± 0.51 on A40.

*16× was added overnight and is not part of the pre-registered B1 verdict.* Both arms ran on L40S, so the pair is within one hardware type. Difference: +0.97 ± 1.15.

**B1: PASS.** Sampling is not significantly better than top-k at any ratio.

## B2 / B3 — against the field

| compression | `kvzip` | `kvzip_bern` | `kvzip_bern_nobias` | `bern_tau` | `snapkv` | `expected_attention` | `streaming_llm` |
|---|---|---|---|---|---|---|---|
| 2x | 95.22 | 95.53 | — | 95.27 | 68.72 | 90.60 | 60.32 |
| 4x | 95.50 | 95.48 | 95.09 | 95.18 | 40.21 | 76.82 | 36.73 |
| 8x | 93.05 | 93.84 | 93.33 | 93.90 | 25.99 | 45.18 | 25.66 |

**B2: PASS** — best Bernoulli arm vs best non-KVzip baseline: 2x: 95.53 vs `expected_attention` 90.60; 4x: 95.48 vs `expected_attention` 76.82.
**B3: PASS** — drop from the full cache at 8×: `kvzip` 2.36, `kvzip_bern` 1.58.

## B5 — does the log(1/r) bias matter?

Same seeds, so the same tokens are kept. The only difference is the Horvitz–Thompson correction.

| compression | with bias | without bias | paired difference (bias − no bias) |
|---|---|---|---|
| 4x | 95.48 | 95.09 | +0.39 ± 0.30 |
| 8x | 93.84 | 93.33 | +0.50 ± 0.40 |

**B5: PASS** — no significant effect of the bias on task score.

## B6 — chunked vs full-context calibration

Both arms sample by Bernoulli with a log(1/r) bias. `kvzip_bern` uses KVzip's chunked (2048-token) repeat pass; `bern_tau` (E9's press) re-reads the whole context.

| compression | kvzip_bern (chunked) | bern_tau (full context) | paired difference (full − chunked) |
|---|---|---|---|
| 2x | 95.53 | 95.27 | -0.26 ± 0.23 |
| 4x | 95.48 | 95.18 | -0.30 ± 0.25 |
| 8x | 93.84 | 93.90 | +0.07 ± 0.31 |

**B6:** full-context calibration is not worth a significant +1 point at any ratio.

## B4 — is the memory accounting honest?

| arm | compression | target kept | realised kept (mean) | worst sample | samples within §11 bound M⁺ |
|---|---|---|---|---|---|
| `kvzip_bern` | 2x | 0.5000 | 0.5000 (-0.00%) | 0.5010 | 100.00% |
| `kvzip_bern` | 4x | 0.2500 | 0.2500 (+0.00%) | 0.2509 | 100.00% |
| `kvzip_bern` | 8x | 0.1250 | 0.1250 (+0.00%) | 0.1259 | 100.00% |
| `kvzip_bern_nobias` | 4x | 0.2500 | 0.2500 (+0.00%) | 0.2509 | 100.00% |
| `kvzip_bern_nobias` | 8x | 0.1250 | 0.1250 (+0.00%) | 0.1259 | 100.00% |
| `bern_tau` | 2x | 0.5000 | 0.5000 (+0.00%) | 0.5012 | n/a (E9 press) |
| `bern_tau` | 4x | 0.2500 | 0.2500 (+0.00%) | 0.2509 | n/a (E9 press) |
| `bern_tau` | 8x | 0.1250 | 0.1250 (+0.01%) | 0.1258 | n/a (E9 press) |

**B4: PASS** — mean kept fraction within 0.01% of target; §11's high-probability bound holds for 100.00% of samples.

Note: the target is set on the whole cache, so expected memory is exact by construction. The realised count per sample fluctuates, and the §11 bound says by how much.

## Per-subtask scores at 8×

| arm | cwe | fwe | niah_multikey_1 | niah_multikey_2 | niah_multikey_3 | niah_multiquery | niah_multivalue | niah_single_1 | niah_single_2 | niah_single_3 | qa_1 | qa_2 | vt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `kvzip` | 84 | 93 | 100 | 100 | 98 | 100 | 98 | 100 | 100 | 100 | 83 | 54 | 99 |
| `kvzip_bern` | 97 | 93 | 99 | 100 | 99 | 100 | 100 | 100 | 99 | 100 | 79 | 53 | 100 |
| `kvzip_bern_nobias` | 91 | 94 | 99 | 100 | 100 | 100 | 97 | 100 | 100 | 99 | 81 | 53 | 99 |
| `bern_tau` | 97 | 94 | 100 | 100 | 99 | 100 | 99 | 100 | 100 | 99 | 79 | 54 | 100 |
| `snapkv` | 35 | 74 | 13 | 6 | 0 | 15 | 13 | 71 | 16 | 0 | 35 | 28 | 33 |
| `expected_attention` | 45 | 66 | 59 | 15 | 2 | 46 | 39 | 95 | 73 | 0 | 40 | 34 | 74 |
| `streaming_llm` | 2 | 86 | 22 | 8 | 5 | 15 | 13 | 16 | 13 | 17 | 82 | 32 | 23 |
| `random` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Figure: `out/figures/scores.png` (score vs compression per arm; B1 paired difference with ±2 SE).
