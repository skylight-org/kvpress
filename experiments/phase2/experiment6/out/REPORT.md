# Experiment 6: downstream accuracy on E5's settings

180 runs, 30 samples each, every per-sample score recomputed and matched to metrics.json (180/180).

## Accuracy (string_match, %) at the budgeted ratios

Bernoulli = mean of seeds s1 and s2. no_press is the uncompressed ceiling.


### 4x (kept 0.25)

| model | task | no_press | KVzip | Bernoulli (global) | Bernoulli (per-head) |
|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 83 | 77 | 80 | 60 |
| Llama-3.1-8B | niah_single_3 | 100 | 100 | 100 | 7 |
| Llama-3.1-8B | niah_multikey_2 | 100 | 100 | 100 | 83 |
| Llama-3.1-8B | vt | 99 | 100 | 99 | 99 |
| Llama-3.1-8B | cwe | 43 | 52 | 53 | 21 |
| Qwen3-4B | qa_1 | 70 | 63 | 70 | 50 |
| Qwen3-4B | niah_single_3 | 100 | 100 | 100 | 0 |
| Qwen3-4B | niah_multikey_2 | 100 | 100 | 100 | 47 |
| Qwen3-4B | vt | 100 | 100 | 100 | 99 |
| Qwen3-4B | cwe | 78 | 80 | 78 | 71 |

### 8x (kept 0.125)

| model | task | no_press | KVzip | Bernoulli (global) | Bernoulli (per-head) |
|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 83 | 60 | 82 | 47 |
| Llama-3.1-8B | niah_single_3 | 100 | 100 | 100 | 0 |
| Llama-3.1-8B | niah_multikey_2 | 100 | 100 | 100 | 23 |
| Llama-3.1-8B | vt | 99 | 97 | 100 | 99 |
| Llama-3.1-8B | cwe | 43 | 13 | 47 | 13 |
| Qwen3-4B | qa_1 | 70 | 63 | 70 | 33 |
| Qwen3-4B | niah_single_3 | 100 | 100 | 100 | 0 |
| Qwen3-4B | niah_multikey_2 | 100 | 73 | 100 | 0 |
| Qwen3-4B | vt | 100 | 100 | 100 | 100 |
| Qwen3-4B | cwe | 78 | 62 | 66 | 53 |

### 16x (kept 0.0625)

| model | task | no_press | KVzip | Bernoulli (global) | Bernoulli (per-head) |
|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 83 | 60 | 52 | 33 |
| Llama-3.1-8B | niah_single_3 | 100 | 100 | 80 | 0 |
| Llama-3.1-8B | niah_multikey_2 | 100 | 70 | 98 | 0 |
| Llama-3.1-8B | vt | 99 | 97 | 99 | 99 |
| Llama-3.1-8B | cwe | 43 | 0 | 21 | 5 |
| Qwen3-4B | qa_1 | 70 | 33 | 55 | 27 |
| Qwen3-4B | niah_single_3 | 100 | 100 | 13 | 0 |
| Qwen3-4B | niah_multikey_2 | 100 | 23 | 97 | 3 |
| Qwen3-4B | vt | 100 | 100 | 99 | 100 |
| Qwen3-4B | cwe | 78 | 1 | 47 | 41 |

## VKvWR ladder (placed at its realised kept fraction)

| model | task | cr 0.875 | cr 0.9375 | cr 0.96875 | cr 0.984 | cr 0.995 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 77 @ 0.199 | 57 @ 0.152 | 60 @ 0.125 | 57 @ 0.105 | 60 @ 0.071 |
| Llama-3.1-8B | niah_single_3 | 70 @ 0.246 | 57 @ 0.196 | 17 @ 0.160 | 7 @ 0.132 | 0 @ 0.086 |
| Llama-3.1-8B | niah_multikey_2 | 93 @ 0.186 | 90 @ 0.142 | 87 @ 0.117 | 67 @ 0.100 | 27 @ 0.069 |
| Llama-3.1-8B | vt | 99 @ 0.174 | 99 @ 0.129 | 99 @ 0.106 | 100 @ 0.091 | 99 @ 0.061 |
| Llama-3.1-8B | cwe | 40 @ 0.192 | 37 @ 0.142 | 34 @ 0.116 | 27 @ 0.096 | 17 @ 0.067 |
| Qwen3-4B | qa_1 | 57 @ 0.235 | 77 @ 0.197 | 70 @ 0.173 | 57 @ 0.148 | 40 @ 0.100 |
| Qwen3-4B | niah_single_3 | 77 @ 0.291 | 53 @ 0.253 | 20 @ 0.219 | 7 @ 0.184 | 0 @ 0.119 |
| Qwen3-4B | niah_multikey_2 | 100 @ 0.211 | 97 @ 0.174 | 90 @ 0.153 | 67 @ 0.134 | 13 @ 0.095 |
| Qwen3-4B | vt | 100 @ 0.203 | 100 @ 0.164 | 100 @ 0.145 | 100 @ 0.127 | 100 @ 0.088 |
| Qwen3-4B | cwe | 72 @ 0.237 | 71 @ 0.197 | 66 @ 0.175 | 63 @ 0.150 | 50 @ 0.099 |

(qa_1 has no E5 memory measurement; its VKvWR kept fraction is the mean of the other four tasks.)

## Paired against KVzip at the same budget (per sample, pooled over tasks and models)

Difference in per-sample score (percentage points); positive = the arm beats KVzip. Tests: paired t over samples and an exact sign test on the non-tied samples; Holm over the whole family.

| arm | ratio | scope | n | mean diff | 95% CI | wins/losses | p (t) | Holm | p (sign) |
|---|---|---|---|---|---|---|---|---|---|
| bernoulli | 0.75 | all | 300 | +0.8 | ±2.1 | 31/29 | 0.463 | 0.463 | 0.897 |
| bernoulli | 0.75 | Llama-3.1-8B | 150 | +0.7 | ±3.4 | 20/12 | 0.67 |  | 0.215 |
| bernoulli | 0.75 | Qwen3-4B | 150 | +0.9 | ±2.6 | 11/17 | 0.519 |  | 0.345 |
| bernoulli | 0.875 | all | 300 | +9.5 | ±3.2 | 73/20 | 2.24e-08 | 6.71e-08 | 2.92e-08 |
| bernoulli | 0.875 | Llama-3.1-8B | 150 | +11.8 | ±4.4 | 43/4 | 6.09e-07 |  | 2.78e-09 |
| bernoulli | 0.875 | Qwen3-4B | 150 | +7.3 | ±4.7 | 30/16 | 0.0031 |  | 0.0541 |
| bernoulli | 0.9375 | all | 300 | +7.6 | ±6.0 | 110/54 | 0.0137 | 0.0275 | 1.46e-05 |
| bernoulli | 0.9375 | Llama-3.1-8B | 150 | +4.6 | ±6.6 | 45/21 | 0.169 |  | 0.00427 |
| bernoulli | 0.9375 | Qwen3-4B | 150 | +10.6 | ±10.1 | 65/33 | 0.0413 |  | 0.0016 |
| perhead_bern | 0.75 | all | 300 | -33.4 | ±5.6 | 15/131 | 1.36e-26 | 6.82e-26 | 2.68e-24 |
| perhead_bern | 0.75 | Llama-3.1-8B | 150 | -31.6 | ±7.5 | 8/59 | 7.78e-14 |  | 1.02e-10 |
| perhead_bern | 0.75 | Qwen3-4B | 150 | -35.2 | ±8.2 | 7/72 | 3.03e-14 |  | 1.06e-14 |
| perhead_bern | 0.875 | all | 300 | -40.0 | ±6.2 | 28/155 | 6.23e-30 | 3.74e-29 | 1.65e-22 |
| perhead_bern | 0.875 | Llama-3.1-8B | 150 | -37.5 | ±9.2 | 17/76 | 2.99e-13 |  | 4.45e-10 |
| perhead_bern | 0.875 | Qwen3-4B | 150 | -42.5 | ±8.2 | 11/79 | 1.3e-18 |  | 7.77e-14 |
| perhead_bern | 0.9375 | all | 300 | -27.7 | ±6.2 | 41/101 | 1.13e-16 | 4.52e-16 | 5.06e-07 |
| perhead_bern | 0.9375 | Llama-3.1-8B | 150 | -37.9 | ±8.7 | 13/62 | 1.74e-14 |  | 8.4e-09 |
| perhead_bern | 0.9375 | Qwen3-4B | 150 | -17.5 | ±8.4 | 28/39 | 7.81e-05 |  | 0.222 |

### Per task at 8x: Bernoulli (global) minus KVzip

| model | task | KVzip | Bernoulli | diff | wins/losses |
|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 60 | 82 | +21.7 | 11/4 |
| Llama-3.1-8B | niah_single_3 | 100 | 100 | +0.0 | 0/0 |
| Llama-3.1-8B | niah_multikey_2 | 100 | 100 | +0.0 | 0/0 |
| Llama-3.1-8B | vt | 97 | 100 | +3.0 | 4/0 |
| Llama-3.1-8B | cwe | 13 | 47 | +34.2 | 28/0 |
| Qwen3-4B | qa_1 | 63 | 70 | +6.7 | 6/3 |
| Qwen3-4B | niah_single_3 | 100 | 100 | +0.0 | 0/0 |
| Qwen3-4B | niah_multikey_2 | 73 | 100 | +26.7 | 8/0 |
| Qwen3-4B | vt | 100 | 100 | -0.3 | 0/1 |
| Qwen3-4B | cwe | 62 | 66 | +3.3 | 16/12 |

## VKvWR at matched realised memory

VKvWR's accuracy is interpolated (linear in log kept) onto the budgeted kept fractions. Out of range = its memory floor is above that budget.

| model | task | budget | KVzip | Bernoulli | VKvWR (interp.) |
|---|---|---|---|---|---|
| Llama-3.1-8B | qa_1 | 0.25 | 77 | 80 | out of range |
| Llama-3.1-8B | qa_1 | 0.125 | 60 | 82 | 60 |
| Llama-3.1-8B | qa_1 | 0.0625 | 60 | 52 | out of range |
| Llama-3.1-8B | niah_single_3 | 0.25 | 100 | 100 | out of range |
| Llama-3.1-8B | niah_single_3 | 0.125 | 100 | 100 | 6 |
| Llama-3.1-8B | niah_single_3 | 0.0625 | 100 | 80 | out of range |
| Llama-3.1-8B | niah_multikey_2 | 0.25 | 100 | 100 | out of range |
| Llama-3.1-8B | niah_multikey_2 | 0.125 | 100 | 100 | 88 |
| Llama-3.1-8B | niah_multikey_2 | 0.0625 | 70 | 98 | out of range |
| Llama-3.1-8B | vt | 0.25 | 100 | 99 | out of range |
| Llama-3.1-8B | vt | 0.125 | 97 | 100 | 99 |
| Llama-3.1-8B | vt | 0.0625 | 97 | 99 | 99 |
| Llama-3.1-8B | cwe | 0.25 | 52 | 53 | out of range |
| Llama-3.1-8B | cwe | 0.125 | 13 | 47 | 35 |
| Llama-3.1-8B | cwe | 0.0625 | 0 | 21 | out of range |
| Qwen3-4B | qa_1 | 0.25 | 63 | 70 | out of range |
| Qwen3-4B | qa_1 | 0.125 | 63 | 70 | 49 |
| Qwen3-4B | qa_1 | 0.0625 | 33 | 55 | out of range |
| Qwen3-4B | niah_single_3 | 0.25 | 100 | 100 | 50 |
| Qwen3-4B | niah_single_3 | 0.125 | 100 | 100 | 1 |
| Qwen3-4B | niah_single_3 | 0.0625 | 100 | 13 | out of range |
| Qwen3-4B | niah_multikey_2 | 0.25 | 100 | 100 | out of range |
| Qwen3-4B | niah_multikey_2 | 0.125 | 73 | 100 | 56 |
| Qwen3-4B | niah_multikey_2 | 0.0625 | 23 | 97 | out of range |
| Qwen3-4B | vt | 0.25 | 100 | 100 | out of range |
| Qwen3-4B | vt | 0.125 | 100 | 100 | 100 |
| Qwen3-4B | vt | 0.0625 | 100 | 99 | out of range |
| Qwen3-4B | cwe | 0.25 | 80 | 78 | out of range |
| Qwen3-4B | cwe | 0.125 | 62 | 66 | 58 |
| Qwen3-4B | cwe | 0.0625 | 1 | 47 | out of range |

In range: 12 cells. VKvWR >= KVzip in 4, >= Bernoulli in 2. Mean VKvWR - KVzip -22.0, VKvWR - Bernoulli -30.1 points.

## Does E5's upstream error predict downstream accuracy?

Cells = (model, task, arm, ratio) measured in both experiments (qa_1 has no E5 errors). Spearman rank correlation between E5's median attention error and the accuracy drop from the uncompressed ceiling.

| subset | cells | Spearman rho (attention err vs drop) | p |
|---|---|---|---|
| all arms | 112 | +0.10 | 0.274 |
| without per-head | 88 | +0.20 | 0.0583 |
| within kvzip | 24 | +0.53 | 0.00794 |
| within bernoulli | 24 | +0.56 | 0.00446 |
| within perhead_bern | 24 | +0.63 | 0.000859 |
| within vkvwr | 40 | +0.74 | 5.38e-08 |

| arm | median E5 attention error | mean accuracy drop (points) |
|---|---|---|
| KVzip (top-k) | 0.226 | 12.2 |
| Bernoulli (global c) | 0.067 | 6.9 |
| Bernoulli (per-head) | 0.024 | 49.9 |
| VKvWR (prof.) | 0.037 | 27.3 |

## SCBench prefix_suffix (8x budget, 25 questions = 5 shared contexts x 5 turns)

14 runs; every per-question score reproduces metrics.json (14/14).

| model | no_press | KVzip | Bernoulli | per-head | VKvWR ladder (acc @ realised kept) |
|---|---|---|---|---|---|
| Llama-3.1-8B | 12 | 0 | 8 | 0 | 0 @ 0.123, 0 @ 0.165, 0 @ 0.221 |
| Qwen3-4B | 0 | 0 | 0 | 0 | 0 @ 0.122, 0 @ 0.156, 0 @ 0.207 |

Paired per question against KVzip (VKvWR = its cr 0.96875 run, realised kept ~0.123):

| model | arm | mean diff (points) | wins/losses | exact sign p |
|---|---|---|---|---|
| Llama-3.1-8B | bernoulli | +8.0 | 2/0 | 0.5 |
| Llama-3.1-8B | perhead_bern | +0.0 | 0/0 | 1 |
| Llama-3.1-8B | vkvwr | +0.0 | 0/0 | 1 |
| Llama-3.1-8B | no_press | +12.0 | 3/0 | 0.25 |
| Qwen3-4B | bernoulli | +0.0 | 0/0 | 1 |
| Qwen3-4B | perhead_bern | +0.0 | 0/0 | 1 |
| Qwen3-4B | vkvwr | +0.0 | 0/0 | 1 |
| Qwen3-4B | no_press | +0.0 | 0/0 | 1 |
