# Experiment 6: Conclusion

**Question.** At the same memory, do our importance-sampling methods beat KVzip on *downstream accuracy*,
under exactly the settings where E5 measured their upstream attention error? RULER `qa_1` is added.

**Answer, in one line.** **Global Bernoulli** is the only method that beats KVzip on both counts: E5's upstream
error and E6's downstream accuracy, **+9.5 points at 8x** (73 samples won, 20 lost). The two methods that
**equalise the budget across heads**, per-head Bernoulli and VKvWR, had the lowest upstream error in E5 but
**lose badly downstream**. The cause is verified directly: retrieval depends on ~80 "copy heads" that an equal
per-head budget starves. **E5's error metric does not rank methods by downstream accuracy.**

---

## 1. Provenance

| | |
|---|---|
| code | `e5-upstream-error` @ `163ab1b`, the same as E5; error logging off, which never changes generation |
| grid | RULER 32k (`qa_1`, `niah_single_3`, `niah_multikey_2`, `vt`, `cwe`) x Llama-3.1-8B / Qwen3-4B |
| arms | `no_press` (ceiling), KVzip, Bernoulli s1/s2, per-head Bernoulli at 4x/8x/16x, and the VKvWR ladder (cr 0.875-0.995) |
| runs | 180, all SLURM `COMPLETED` (jobs 315628-315639, 315758) |
| samples | 30 per task, the same 30 for every arm (`df.sample(30, random_state=42)`); E5's 10 are a subset |
| answers | full length per task (32-128 tokens); E5 used 4 |
| verification | every per-sample score was recomputed with RULER's rule and reproduces each run's `metrics.json`, **180/180** |

Regenerate with `python analyze.py` (CPU only), which writes `out/REPORT.md`, `out/*.csv` and `out/figures/`.

## 2. Headline: paired against KVzip at the same budget

Per-sample difference in score (points), pooled over 5 tasks x 2 models (n = 300). Bernoulli = mean of its 2 seeds.

| | 4x (kept 0.25) | 8x (kept 0.125) | 16x (kept 0.0625) |
|---|---|---|---|
| **Bernoulli (global)** | +0.8 ± 2.1, n.s. | **+9.5 ± 3.2** (73/20, Holm p = 7e-8) | **+7.6 ± 6.0** (110/54, Holm p = 0.03) |
| **Bernoulli (per-head)** | **-33.4** ± 5.6 | **-40.0** ± 6.2 | **-27.7** ± 6.2 |

The 8x win holds for each model separately: Llama +11.8 (43/4), Qwen +7.3 (30/16).

**8x per task** (KVzip -> Bernoulli): Llama `qa_1` 60->82, `cwe` 13->47; Qwen `niah_multikey_2` 73->100, `qa_1`
63->70. Ties elsewhere (`niah_single_3` 100/100; `vt` saturated at ~100 for every arm).

**16x is mixed, not a clean win.** Bernoulli wins big on `niah_multikey_2` (Llama 98 vs 70, Qwen 97 vs 23) and `cwe`
(21 vs 0, 47 vs 1), but **loses on `niah_single_3`** (Llama 80 vs 100, Qwen 13 vs 100, consistent across both
seeds). Those losses are near-miss copies: the right UUID with 2-3 characters wrong. It's the same copy-head
token-dropping as in section 3, appearing once the budget is tight enough that needle tokens get r < 1.
KVzip keeps a single top-scored needle deterministically.

## 3. Per-head Bernoulli collapses on retrieval. Verified: not a code error

`niah_single_3`: **0-7%** at every ratio for both models, where KVzip and global Bernoulli score 100 at 4x and 8x.
It finds the needle, copies the first few characters, then makes up the rest.

Checked in detail (`diag/`):
1. **Plumbing is correct.** Forced to the shipped global `c` and seed, the per-head press gives a
   **bit-identical** attention bias to the shipped `BernoulliPress`, and identical, correct generations on the real
   model (5/5).
2. **Reproducible.** Re-running regenerates E6's saved answers exactly (5/5 for both arms).
3. **Mechanism measured.** ~80 of 1,024 query heads put >=20% of their attention on the needle while the answer is
   written ("copy heads"). Needle tokens kept **in those heads** at 4x:

   | | needle kept, averaged over all heads | needle kept in the copy heads | correct |
   |---|---|---|---|
   | Bernoulli, global `c` | 29-32% | **94-96%** | 5/5 |
   | Bernoulli, per-head | 27-38% | **30-38%** | 0/5 |
   | top-k, per-head (no sampling) | 26-39% | 31-40% | 1/5 |

   Averaged over all heads, both keep about the same share of the needle. The global budget puts it in the copy
   heads; an equal per-head cap can't. Deterministic per-head top-k fails too, so the cause is the
   **allocation**, not the sampling.

## 4. VKvWR at matched memory is worse than both

VKvWR placed at its realised kept fraction (from E5) and interpolated onto the budgets, 12 in-range cells:
**-22 points vs KVzip, -30 vs Bernoulli** on average. Retrieval collapses as its memory shrinks: Llama
`niah_single_3` 70% at kept 0.25 -> 6% at 0.125 -> 0% at 0.09, where KVzip and Bernoulli hold 100. Its
memory floor (0.06-0.12) again prevents it from reaching 16x.

## 5. E5's upstream error does not predict downstream accuracy across methods

| method | median E5 attention error | mean accuracy drop vs uncompressed |
|---|---|---|
| Bernoulli (per-head) | **0.024** (best) | **49.9** points (worst) |
| VKvWR | 0.037 | 27.3 |
| Bernoulli (global) | 0.067 | **6.9** (best) |
| KVzip | 0.226 (worst) | 12.2 |

Across all 112 cells measured in both experiments, Spearman rho = **+0.10 (p = 0.27)**, i.e. none. *Within* each
method, more compression does mean more error and more accuracy loss (rho +0.53 to +0.74). So the metric tracks
compression level, but it can't compare methods. It's a median over all 1,024 heads, and retrieval is decided by the ~8%
of heads that copy. **Allocation across heads** is what separates the methods downstream. The methods whose budget
can concentrate in those heads retrieve: KVzip (spread 1,500-37,000x across tasks) and global Bernoulli (150-11,700x). The ones
that equalise can't: VKvWR (9-33x) and per-head (1x). That ordering is an interpretation of four arms, backed
by the direct measurement in section 3.

## 6. Limits

- **30 samples per cell.** A single cell's accuracy is about +-9-18 points. The pooled, per-sample-paired tests are
  the reliable numbers; single-cell differences under ~15 points are not.
- **Saturated or weak tasks.** `vt` is ~100 for every arm and can't discriminate. Uncompressed Llama gets `cwe`
  wrong (it lists English stopwords, a ceiling of 43), so Llama `cwe` is a weak signal.
- **VKvWR's memory comes from E5's measurement**, not from these runs. For `qa_1`, which has no E5 errors, it's
  the mean over the other four tasks.
- **SCBench `prefix_suffix` cannot discriminate** (14 runs, 25 questions, 8x budget; see §8). The uncompressed
  ceiling is 12 for Llama and **0 for Qwen**, so there is almost nothing left for compression to lose. The one
  signal, Llama keeping 8 of that 12 under Bernoulli while every other arm reads 0, rests on 2 questions and is
  not significant. Pick a task with headroom next time.
- As in E5, **no memory is actually freed**. Every arm masks rather than evicts, so accuracy is compared at a
  *modelled* memory level.

## 7. What this changes

1. **Withdraws E5's recommendation** to give `BernoulliPress` a per-head budget. It lowers the median head's
   error and destroys retrieval.
2. **For the dynamic-Bernoulli work (P0 items 5 and 6):** keep the *global* allocation. The one place global
   Bernoulli loses (`niah_single_3` at 16x) is a top-scored needle getting r < 1 and being sampled away. A
   deterministic top slice with a sampled tail, i.e. item 6's "oracle top-k when we exceed epsilon", targets
   exactly that failure.
3. **The upstream metric needs to be head-aware** to predict retrieval: weight heads by their attention to the
   answer-relevant tokens, or report the copy heads separately. That can be tested on E5's existing data plus
   `diag/`'s copy-head identification.
4. **For the professor:** VKvWR's per-head adaptive sample size gave it low median error in E5, but here it
   equalises the budget across heads, and that costs it retrieval.

## 8. SCBench (added after the RULER runs)

`prefix_suffix`, E5's 25-question slice (5 shared contexts x 5 turns), 8x budget, 14 runs, every per-question
score reproduced from `metrics.json` (14/14).

| model | no_press | KVzip | Bernoulli | per-head | VKvWR (realised kept 0.12-0.22) |
|---|---|---|---|---|---|
| Llama-3.1-8B | 12 | 0 | **8** | 0 | 0 at all three ladder points |
| Qwen3-4B | **0** | 0 | 0 | 0 | 0 at all three ladder points |

The scorer is correct: uncompressed Llama answers by gluing the prefix and suffix together
("KEcFQ" + "ccLxD" -> "KEcFQccLxD") instead of retrieving the dictionary word, and Qwen never gets one right.
So the benchmark has no headroom here and the arms cannot be separated: Bernoulli's +8 over KVzip is 2 questions
out of 25 (sign test p = 0.5). It is consistent with the RULER direction but is not evidence on its own.

Tables: `out/REPORT.md` (appended by `analyze_scb.py`), `out/scbench_runs.csv`.

## 9. Files

`analyze.py` (all tables and figures) · `out/REPORT.md` · `out/runs.csv` · `out/paired_vs_kvzip.csv` ·
`out/vkvwr_matched.csv` · `out/upstream_vs_downstream.csv` · `out/figures/fig1_accuracy_vs_memory.png`,
`fig2_upstream_vs_downstream.png` · `diag/equivalence_test.py`, `diag/needle_diag.py`, `diag/needle_diag.json` ·
`STATUS.md`, `STATUS_jobs.txt`.
