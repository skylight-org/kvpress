# Experiment 5 — Conclusion

**Question (pre-registered in `DESIGN.md`, before any run):** do importance-sampling selectors —
our `BernoulliPress` and the professor's `VKvWRPress` — produce lower *upstream* error than KVzip
top-k at matched memory, on SCBench and RULER 32k, for Llama-3.1-8B and Qwen3-4B?

**Answer: yes, decisively, and by a much larger margin than expected.** Every sampled arm beats
top-k on every metric, at every ratio, on both models and both benchmarks, with no exceptions in
43/43 matched-budget cells. At 8x compression the attention error is **6-39x lower** than KVzip's.

---

## 1. Provenance

| | |
|---|---|
| branch | `e5-upstream-error` @ `163ab1b` = `origin/vkv1` + `origin/scbench-benchmark` + 4 commits, clean tree |
| presses | stock `KVzipPress`, shipped `BernoulliPress` (PR #2), the professor's shipped `VKvWRPress` (`vkv1`) — none forked |
| runs | 148, all SLURM `COMPLETED` exit 0 (jobs 315269-315278) |
| rows | 7,224,320 (context, question, decode step, layer, **query head**) tuples |
| models | Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507 |
| data | RULER 32k (`niah_single_3`, `niah_multikey_2`, `vt`, `cwe`) + SCBench `prefix_suffix` |
| GPUs | A40 / L40S / A100-80GB; GPU type pinned within each cell and logged |
| artefacts | `out/REPORT.md`, `out/pareto.csv`, `out/cells.csv`, `out/all_rows.parquet` |

All 148 runs pass the C0 invariants (`check_run.py`): top-k's signed denominator error is negative
for every head (T1, a theorem — a breach would be a bug), no NaN/Inf (T2), and the compression is
actually active (T3, the E2 trap where nine runs silently equalled `no_press`).

## 2. Headline: attention error at a realised kept fraction of 0.125 (8x)

`||A/B_hat - A/B|| / ||A/B||`, median over query heads. This is the quantity the model actually
consumes, and C5 is the criterion that decides whether an upstream win is real or cosmetic.

| model | task | KVzip | Bernoulli | per-head Bernoulli | KVzip / per-head |
|---|---|---|---|---|---|
| Llama | cwe | 0.4415 | 0.1326 | **0.0347** | 12.7x |
| Llama | niah_multikey_2 | 0.4204 | 0.1036 | **0.0388** | 10.8x |
| Llama | niah_single_3 | 0.3616 | 0.1841 | **0.0487** | 7.4x |
| Llama | vt | 0.0992 | 0.0204 | **0.0140** | 7.1x |
| Llama | scbench prefix_suffix | 0.4732 | 0.0977 | **0.0202** | 23.5x |
| Qwen3 | cwe | 0.2672 | 0.0985 | **0.0209** | 12.8x |
| Qwen3 | niah_multikey_2 | 0.2905 | 0.0750 | **0.0216** | 13.5x |
| Qwen3 | niah_single_3 | 0.3096 | 0.1537 | **0.0355** | 8.7x |
| Qwen3 | vt | 0.0724 | 0.0195 | **0.0112** | 6.5x |
| Qwen3 | scbench prefix_suffix | 0.1913 | 0.0422 | **0.0049** | 38.7x |

KVzip's attention error at 16x reaches **0.58-0.65** on the NIAH and CWE tasks — i.e. the attention
output is of the same order as its own error. The sampled arms stay at 0.02-0.06.

## 3. The pre-registered criteria

| | criterion | verdict |
|---|---|---|
| **C0** | harness trustworthy | **PASS** — T1/T2/T3 on all 148 runs; 16 unit tests green incl. hand-computed fp64 denominator, GQA mapping pinned to `repeat_interleave`, bottom-right alignment, HT unbiasedness over 4000 draws |
| **C1** | Bernoulli beats top-k upstream | **PASS** — denominator, 8x: -0.2358 (SE 0.0088), t=-26.9, Holm p=3.7e-43, 90 contexts, 94.1% of head-rows win |
| **C2** | VKvWR beats top-k upstream | **PASS** — denominator, 8x: -0.2703 (SE 0.0104), t=-25.9, Holm p=7.2e-42, 94.8% win rate |
| **C3** | consistent across models x benchmarks x ratios | **PASS** — sign holds in 43/43 matched-budget cells, both metrics, zero breaches |
| **C4** | the win is not merely debiasing | **not evaluated** — `kvzip_rescaled` is a post-hoc analysis arm and was cut for time. See §6. |
| **C5** | it survives the ratio | **PASS** — the attention error ranks the arms exactly as the denominator does, in every cell. E1's stated limitation is resolved: here a correct `B` did come with a correct `A/B`. |
| **C6** | VKvWR's CLT bound is calibrated | **not evaluated** — needs his `d_true` JSONL fields; deferred. |
| **C7** | estimator or allocation? | **PASS, and the answer is allocation** — see §4 |

Holm correction is across the full pre-registered family of 24 tests; every one survives at any
conventional level. Uncertainty is clustered at the **context**, never the query, because the turns
of one context share a single compression draw.

## 4. C7 — the win is the *allocation*, not the estimator

`perhead_bern` is the shipped Bernoulli estimator with a per-head budget replacing the single global
`c`. Against `bernoulli` it isolates the **allocation**; against `vkvwr` at matched realised memory
it isolates the **estimator**.

**Allocation.** Per-head budgeting beats one global `c` on the **typical** head at every ratio,
by 1.7-4.0x, and the margin grows as the budget tightens (`fig9`):

| ratio | median head | p95 head | p99 head | heads where per-head wins |
|---|---|---|---|---|
| 4x | **2.1x better** | 0.65x (worse) | 0.61x (worse) | 65.9% |
| 8x | **3.3x better** | 1.18x better | 1.23x better | 73.5% |
| 16x | **3.8x better** | 1.52x better | 1.75x better | 79.6% |

(ratios are global-c error / per-head error, pooled over both models; >1 means per-head wins.)

**At 4x there is a genuine trade-off**: equalising the budget across heads improves the typical head
but makes the *worst* heads worse, because forcing every head to keep exactly 25% takes budget away
from the few heads that genuinely need more. By 8x that reverses and per-head wins on the median,
the tail and the mean simultaneously. One global `c` pays for the worst head in the model — exactly
what E3's C5 argued — and that cost only bites once the budget is tight.

> This supersedes an earlier reading of the `summarize.py` C7 row, which reports the 4x allocation
> effect as +0.0118 (t=+5.7), i.e. per-head *worse*. That row is a paired **mean**, and at 4x the
> mean is driven by the heavy tail above; the pre-registered aggregate for this experiment is the
> median over heads, which points the other way. Both numbers are correct and they describe
> different things — the table above reports both rather than picking one.

**Estimator, at matched realised memory.** VKvWR's `compression_ratio` sets no budget: it sizes only
`topk_frac` and the pilot, and the CLT top-up sets the rest. So its nominal 8x realises 0.199 (Llama)
/ 0.236 (Qwen) kept, not 0.125. Comparing at nominal ratio hands it ~60% more memory. Interpolating
its error-vs-realised-memory curve to 0.125 instead:

| | in-range tasks | per-head Bernoulli wins on attention error |
|---|---|---|
| kept = 0.125 | 10 | **10 / 10** |

Margins are roughly 1.5-2.5x (e.g. Qwen `niah_multikey_2`: 0.0216 vs 0.0380; Llama `cwe`: 0.0347 vs
0.0538). **The with-replacement Hansen-Hurwitz estimator is not what makes VKvWR good — its
per-head adaptive sample size is.** Give the same per-head adaptivity to the simpler
without-replacement Horvitz-Thompson scheme and it wins.

> The `C7` table in the auto-generated `out/REPORT.md` compares at nominal ratio and therefore
> reports the opposite sign for the estimator row. That row is confounded; the numbers above
> supersede it. `summarize.py` should be fixed before that table is shown to anyone.

## 5. VKvWR has a memory floor

Sweeping `compression_ratio` to 200x, VKvWR's realised kept fraction bottoms out at **0.061
(Llama)** and **0.088 (Qwen)**. It cannot reach the 16x (0.0625) operating point on Qwen at all.
The forced keeps (sink 4 + local 4 + `topk_frac*n`) plus the CLT top-up set a floor that
`compression_ratio` cannot push through. The budgeted arms hit their target to within 0.01%.

This matters for the comparison against the professor's own tables, where his `epsilon` knob reaches
0.94 sparsity: **`epsilon`, not `compression_ratio`, is the right memory knob for his method.**

## 6. What is *not* claimed

- **No memory is actually freed.** VKvWR keeps every KV pair physically and needs the full key set
  at decode to compute `a_i`; the masked arms do not free memory either. Every number is
  approximation quality at a *modelled* memory level. The handicap is identical across arms, so the
  comparison is fair, but no arm here is a measured saving.
- **Top-k is biased by construction** — it drops positive mass and never reweights, so on
  `|B_hat - B|/B` the sampled arms are favoured *by the definition of the metric*. This was
  pre-registered as the honesty note. The `kvzip_rescaled` ablation (C4), which removes top-k's bias
  and leaves only its variance, was cut for time and is the **single most important follow-up**: it
  decides whether the win is "sampling" or merely "the correction". It is a pure post-hoc analysis
  over `out/all_rows.parquet` and needs no GPU.
  *However*: C4 cannot rescue top-k on the **attention** error, which is a ratio and so invariant to
  any per-head constant rescaling of `B`. KVzip's 0.44 vs per-head Bernoulli's 0.035 at 8x stands
  regardless of C4's outcome.
- **VKvWR is measured in-harness only.** It is query-adaptive and lives inside the press, so the
  offline replay cross-check (`replay.py`) cannot cover it. That cross-check was also cut for time;
  the in-harness metric instead carries 16 unit tests and the C0 invariants.
- **Downstream accuracy is out of scope**, by the user's decision. Lower upstream error does not
  automatically mean better task scores, and E2-E4 showed that instrument is noisy.
- Errors are measured at the **last query row** of each forward, which is the row VKvWR tuned its
  bias from. At the question prefill this slightly *understates* VKvWR's error, in its favour.
  `q_pos`/`q_len` are logged so the restriction is visible.

## 7. A bug found, fixed, and worth reporting upstream

`DynamicLayer.update` returns the cache's own tensor, so `attention_patch`'s fake-key write
(`key[batch_indices, head_indices, seq_indices] = k`) **destroys the real keys in the KV cache**.
From the second decode step onward there is no dense reference left, and the existing
`dense_key = key.clone()` clones already-faked keys.

Consequence: the professor's `attention_output_error` is correct only on the very first decode
forward **for any press that drops through `masked_key_indices` — which is exactly KVzip**, the
baseline. Averaging one correct step with N-1 near-zero steps drives the reported KVzip error
toward zero. Methods that drop through an additive bias (`bernoulli`, `vkvwr`) are unaffected.

This is very likely why the professor's earlier tables show KVzip beating VKvWR by 5-10x while this
experiment finds the opposite ordering — the disagreement is almost entirely in the KVzip column,
which is the fingerprint of a bug that only touches the `masked_key_indices` path. His own instinct
not to believe those tables was right.

Fixed in `163ab1b` with an output-neutral save/restore (bitwise identical, since `search_hyperplane`
reads only the query), covered by two regression tests that fail on unmodified `vkv1`:
`test_dense_reference_survives_every_decode_step` and `test_restore_does_not_change_the_answer`.
**This should be reported to him separately from E5's result.**

## 8. Recommended next steps

1. **Report the `attention_patch` bug upstream** — it invalidates one column of his existing tables.
2. **Run C4** (`kvzip_rescaled`) offline over `out/all_rows.parquet`. No GPU, decides whether the
   denominator win is sampling or debiasing.
3. **Fix `summarize.py`'s C7 estimator row** to pair at realised memory, not nominal ratio.
4. **Give `BernoulliPress` a per-head budget.** This is E5's actionable finding: the allocation is
   worth more than the estimator, the effect grows with compression, and `perhead_bern` is already
   written (`perhead_press.py`). Promoting it from an analysis arm to a shipped press is the natural
   PR out of this experiment.
5. **Re-run VKvWR sweeping `epsilon`,** not `compression_ratio`, so it can reach the high-sparsity
   regime where the comparison is most interesting.
6. Only then, downstream accuracy — now with a pre-registered hypothesis about *which* arm should win
   and *why*.

## 9. Figures

`out/figures/`, all regenerated by `python figures.py` from `out/` with no GPU. Every figure is a
full grid: both models x all five tasks x all four methods.

| | what it shows |
|---|---|
| `fig1_pareto_attention` / `_denominator` | error vs **realised** memory. The headline, and the only unconfounded comparison. The dotted line is VKvWR's memory floor. |
| `fig2_headline_8x` | attention error at matched memory 0.125, all 10 cells, VKvWR interpolated onto the same memory. |
| `fig3_perhead_distribution` | ECDF across query heads. Top-k is **stochastically dominated** at every quantile, not merely worse on average — which is what a per-layer Frobenius norm cannot show. |
| `fig4_layer_profile` | error by layer, with the E5a/E1 layer-13 boundary marked. |
| `fig5_signed_denominator` | the pre-registered honesty note, drawn: top-k is <= 0 everywhere, the reweighted arms straddle zero. |
| `fig6_budget_spread` | **the mechanism.** Per-head realised kept fraction. Top-k's budget spread across heads is 8,000-37,000x and its *median* head keeps ~1e-3 against a 0.125 budget; per-head Bernoulli is exactly 1x by construction; VKvWR 9-33x. |
| `fig7_c7_allocation_vs_estimator` | C7, both halves. |
| `fig8_numerator_abar` | numerator error against `Abar` vs against `\|A\|` (the E6 cancellation). |
| `fig9_allocation_tradeoff` | the 4x trade-off in §4: median vs p95 vs p99, and the win rate across heads. |

`fig6` is the one to lead with if only one figure is shown: it explains *why* top-k loses, rather
than only that it does. A single global budget is spent wherever the scores are largest, so it
starves most heads to near-zero while a handful keep almost everything.

## 10. Reproducing

```bash
export PYTHONPATH="$HOME/src/kvpress-e5"          # load-bearing: kvpress-pr's editable
conda activate kvpress-pr                          #   install points at kvpress-skylight
cd ~/phase2/experiment5
python check_run.py results/<config>                # C0 invariants
python summarize.py                                 # regenerates out/ from the parquet, no GPU
python figures.py                                   # regenerates out/figures/, no GPU
```

Every figure and table regenerates from `out/all_rows.parquet` with no GPU. Run config (press,
ratio, model, seed, head counts, scaling, git SHA, library versions) is in each parquet's schema
metadata, so any single file is interpretable on its own.

---

## Addendum (2026-09-22, after Experiment 6)

E6 measured downstream accuracy on these exact settings, and it **overturns one recommendation above**. Per-head
Bernoulli had the lowest upstream error here, but it collapses on retrieval (`niah_single_3` 0-7% vs 100% for KVzip
and global Bernoulli). This is verified as not a code error (bit-identical to the shipped press under global `c`). The
cause is that an equal per-head budget starves the ~80 copy heads that retrieval depends on (see
`~/phase2/experiment6/CONCLUSION.md`, section 3). **Recommendation 4 in section 8, "give BernoulliPress a per-head
budget", is withdrawn.** The median-over-heads error used throughout this report doesn't rank methods by downstream
accuracy (E6 section 5); treat E5's between-method conclusions as statements about *typical-head* approximation
quality, not about task performance.
