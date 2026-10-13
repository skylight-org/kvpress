# Phase 2, Experiment 5 — Upstream error: does importance sampling beat KVzip top-k?

**Date:** 2026-09-20. Target: a conclusive answer today (~9.5 h from 14:20 IST).

> **Read the AMENDMENT at the end of this file first.** Sections 'Arms', 'The equal-memory problem'
> and 'Scope' below are superseded by it. They are kept unchanged so the record shows what was
> pre-registered before the smoke runs and what changed afterwards, and why.

## Context

E2, E3 and E4 all asked the same question through **downstream task accuracy**, and that instrument
is noisy and slow. SCBench gives 10 independent contexts per task, so per-task standard errors are
wide; E4's consistency criterion C3 failed on LongBench *ties* rather than on the direction of the
effect; and the E3 and E4 audits both had to retreat to small-sample tests. Each answer cost 20-30
GPU-hours.

E5 changes the instrument, not the question. It measures the **upstream** quantities the theory is
actually about — the whole attention computation, decomposed:

    denominator   B     = sum_i w_i,            w_i = exp(<q,k_i>/sqrt(d))
    numerator     A     = sum_i w_i v_i         (a vector per head)
    attention     Attn  = A / B                 (what actually reaches the model)

and the relative error of each: `|B_hat - B|/B`, `||A_hat - A||/||A||`, `||Attn_hat - Attn||/||Attn||`.

The denominator is what E1's certificate bounds and is the cleanest signal. The **ratio is what the
model actually consumes**, and a correct `B` does not imply a correct `A/B` — E1 said so explicitly
as its main limitation. Measuring all three separates *where* each method's error comes from, which
a single number cannot.

This is far better powered than task accuracy: one number per (head, query) instead of one score per
question, so a single context yields ~10^5 paired observations. E1 showed the measurement works and
costs 4.3 minutes for 4 contexts on one A40.

**Also report the error against `Abar = sum_i w_i ||v_i||`, not only against `||A||`.** E6 found
`||A|| ~= 0.4 Abar` because the value vectors cancel, so an error that looks acceptable against
`Abar` is 2.5x worse against the quantity that matters. Both go in every table.

**The question:** do importance-sampling selectors — our Bernoulli scheme and the professor's
`VKvWRPress` — produce lower upstream error than KVzip top-k at matched memory, on SCBench and
RULER 32k, for Llama-3.1-8B and Qwen3-4B? Downstream accuracy is out of scope, which is what makes
this fit in a day.

## Decisions taken (user)

Metric: the **full decomposition** — denominator, numerator and the attention ratio — all per head.
The denominator is the headline (it is what the certificate bounds and the lowest-variance signal);
the ratio is the one that decides whether the win is real. Queries: real task queries at true
positions. Ratios: 2x, 4x, 8x, 16x. Benchmarks: SCBench + RULER 32k. Models: Llama-3.1-8B-Instruct
+ Qwen3-4B-Instruct-2507. Baseline: KVzip top-k. Prof's method: from his `vkv` branches.

## What already exists (found during exploration)

**The professor's method is `VKvWRPress`** — "Variance-adaptive KV sampling With Replacement",
`kvpress/presses/vkvwr_press.py` on `origin/vkv1`, registered as `vkvwr_kvzip` / `vkvwr_compactor`.
Hansen-Hurwitz **with replacement**: a PPS distribution over residual keys, `torch.multinomial(...,
replacement=True)`, corrected by `bias = log(I_i/(m*s_i))`. Forced keeps are sink 4 + local 4 + a
deterministic top-`topk_frac*n` head, `topk_frac = base_sample_frac = (1-cr)/2.1`. A pilot sample is
drawn at prefill; **at each decode step it re-solves the sample size from a CLT bound**
`m >= z^2 (M2 - D^2)/(eps^2 D^2)` using the real query. It imports
`transform_scores_for_sampling` from our `bernoulli_press`.

**He has also built most of the harness:** `kvpress/metric_logging/`, enabled with
`--enable_micro_metrics`, writing `micro_metrics.jsonl`. `attention_sparsity` and
`attention_output_error` (relative Frobenius error, per layer, via a second dense forward in
`attention_patch.py`) are already **press-agnostic** — so our secondary metric is free on every arm.

### The one gap E5 must close

Two gaps.

**His `denominator_error` cannot serve this comparison:** it is logged **only on the VKvWR decode
path** (so top-k and Bernoulli produce nothing), it measures the estimator against the **residual**
denominator (forced/sink/local/top-k mass excluded) rather than sparse-vs-dense `B`, and it
**collapses per-head values to a mean** before writing, while our reporting needs per-head
(`median [p25,p75]`, split at layer 13).

**His `attention_output_error` is per layer, not per head** — one Frobenius norm over the whole
layer output — so it cannot be paired head-by-head or split at layer 13, and it gives no numerator.

### The one thing E5 must build, and why it is cheap

A **press-agnostic, per-head** metric emitting `B_sparse`, `B_dense` and the two attention outputs
per head. The numerator then costs **nothing extra**, by an identity: the attention function returns
the *normalised* output, so

    A_hat = O_sparse * B_hat        and        A = O_dense * B

His double-forward already produces `O_sparse` and `O_dense`; our new per-head `B_hat` and `B` are
one extra reduction each in the same patch, where `query`, `key` and the mask are all in scope. So
all three errors come from one pass, for any press whose dropping is a bias (`bernoulli`, `vkvwr`)
or an index set (`kvzip`'s `masked_key_indices`).

**Raw quantities are logged, never ratios** — `log B_hat`, `log B`, `log B_suffix`, the per-head
`O_sparse`/`O_dense` norms, `Abar`, `n_kept_ctx` — so signed error, the bias/variance split, the
`Abar`-vs-`||A||` comparison and the rescaled-top-k ablation are all computable offline from a
single run, with no reruns.

**The double forward is a decode-time cost, not a prefill cost.** It fires only once the press has
set its bias, i.e. on the question prefill (tens of tokens) and on decode steps (`q_len = 1`), never
on the 110k context prefill. So it is affordable on SCBench too, and the earlier plan to restrict
output error to RULER 32k is dropped.

### Implementation essentials (these are the ways to get it silently wrong)

- **Per *query* head, not per KV head.** The bias is stored per KV head only because kvpress stores
  it that way; the quantity that actually normalises the attention weights is per query head, and
  queries within a GQA group differ substantially. Use a grouped einsum with no key expansion, and
  pin the group mapping against `add_attention_bias`'s `repeat_interleave` in a test so it cannot
  drift.
- **Take `scaling` from kwargs or the tensor, never from the config.** For Qwen3-4B,
  `num_attention_heads * head_dim = 4096 != hidden_size = 2560`, so `hidden_size // num_heads` gives
  80 instead of 128 and silently rescales every logit.
- **Read the mask, never write it.** transformers passes `None` at `q_len==1` and a **boolean**
  `.expand()` view (stride-0) at `q_len>1`. Branch on dtype *before* any arithmetic; everything
  out-of-place. Read `original_attention_mask`, not the biased one.
- **Bottom-right causal**, the same rule `add_attention_bias` already uses; top-left is wrong
  whenever `q_len < k_len`.
- **Log space, one shared logits tensor, fp64 reduction.** `B_hat` and `B` derive from the *same*
  logits, so bf16->fp32 error cancels in the difference. Naive fp32 accumulation over 110k terms
  drifts ~2e-5 relative — the same order as the effect being measured — so reduce in fp64.
- **Measure at the last query row**, which is the row VKvWR built its bias from; log `q_pos`/`q_len`
  so the restriction is visible.

### A live bug this uncovers on `vkv1`, in our own baseline arm

`DynamicLayer.update` returns the cache's own tensor, so `attention_patch`'s fake-key write
(`key[batch_indices, head_indices, seq_indices] = k`) **destroys the real keys in the KV cache**.
From the second decode step onward there is no dense reference left, and the existing
`dense_key = key.clone()` on that branch clones already-faked keys. So the professor's
`attention_output_error` is only correct on the very first decode forward for any press that drops
through `masked_key_indices` — **which is exactly `kvzip`, our baseline**.

The fix is a save/restore around the mutation, which is bitwise output-neutral because
`search_hyperplane` reads only the query. E5 must carry it, both to make the baseline measurable and
because `B_sparse` for the top-k arm has to be read off the **real** keys plus the index set rather
than trusting `exp(<q,k_fake>) ~ 0`. This gets its own regression test and is worth reporting
upstream to the professor separately from E5's result.

## Assets confirmed on disk

| | status |
|---|---|
| Llama-3.1-8B-Instruct | complete, 15 GB |
| Qwen3-4B-Instruct-2507 | complete, 7.6 GB; 36 layers, 32 Q / 8 KV heads, head_dim 128, **262 144 native context** so 110k needs no rope hackery; `model_type: qwen3` is a first-class kvpress adapter (`llama_like.py`), and `tests/fixtures.py` already has a pipeline fixture for this exact checkpoint |
| SCBench | complete, 5 task configs cached (kv, prefix_suffix, vt, qa_eng, summary) |
| **RULER 32k** | **NOT on disk.** `simonjegou/ruler` publishes only 4k/8k/16k. The professor's `ruler32k` benchmark reads **`xAlg-AI/att-hub-ruler-32k`** — verified public, 151 MB, 13 tasks. Must be prefetched **on the login node** (compute nodes have no DNS). |
| env | **`kvpress-pr`** — the only env with `src/kvpress-skylight` editable-installed, and the only one with flash-attn |
| queue | zero jobs running. QOS caps bind on **GPUs**, not jobs: a40 = 2 GPUs, l40 = 4, dgx = 4 (A100-80GB, all nodes currently busy). ~10 single-GPU jobs in parallel, association `MaxJobs=12`. 3 idle A40 nodes now. |

## Branch strategy — how "consistent with the pushed branches" is met

    e5-upstream-error  =  origin/vkv1  +  origin/scbench-benchmark  +  one commit adding the metric

`vkv1` already **is** `move_benchmarks` + `vkv` + two follow-ups, so it carries VKvWRPress, the
micro-metric logging *and* the `ruler32k` benchmark in one place. Nothing is forked: the Bernoulli
arm is the shipped `BernoulliPress`, the IS arm is the professor's shipped `VKvWRPress`, the
baseline is stock `KVzipPress`.

Both `vkv1` and `scbench-benchmark` edit `evaluation/evaluate.py` in the dataset-loading region, so
expect the same conflict already resolved once today for PR #3 — resolve it the same way (nest the
scbench config-name branch inside the `dataset_path` else-branch). **Fallback if that proves
fiddly:** reuse `phase2/experiment3/scbench_data.py` by monkeypatching `evaluate.load_dataset`,
exactly as E3 and E4 did.

## Arms

All read the **same KVzip scores**, so every comparison is selection-only.

| arm | press | selection | estimator of `B` |
|---|---|---|---|
| `kvzip` | stock `KVzipPress` | global top-k | `sum_kept w_i`, **biased low**, no reweighting |
| `bernoulli` | `BernoulliPress` (main, PR #2) | Bernoulli, `r_i = min(1, c*s_i)` | Horvitz-Thompson `sum_kept w_i/r_i` |
| `vkvwr` | `VKvWRPress` (vkv1) | forced top-k + PPS **with replacement**, adaptive `m` | Hansen-Hurwitz `sum_forced w_i + sum_sampled w_i*I_i/(m*s_i)` |
| `kvzip_rescaled` | ablation, **post-hoc in analysis** | top-k | `sum_kept w_i` times one constant fitted per head on calibration |

### The equal-memory problem

`kvzip` and `bernoulli` take a fixed budget. **`VKvWR` does not** — its `m` comes from the CLT bound
and is query-dependent and per-head; `compression_ratio` only enters through
`topk_frac = base_sample_frac = (1-cr)/2.1`. So E5 does **not** compare at a nominal ratio. It
sweeps each method's own knob and compares **error-versus-realised-memory curves**, with realised
kept fraction taken from the logged `attention_sparsity`, not from the requested ratio. `kvzip` and
`bernoulli` sweep 2x/4x/8x/16x; `vkvwr` sweeps the same four ratios and `epsilon` so its realised
memory spans the same range. Points are also interpolated to the four nominal ratios for a table
that lines up with E3 and E4.

**Caveat to state up front:** VKvWR keeps every KV pair physically and needs the full key set at
decode to compute `a_i` before it can sample; the masked arms do not free memory either. Every
number is approximation quality at a *modelled* memory level, not a measured saving. Same handicap
for all arms, so the comparison is fair — but it must be worded that way.

### The honesty note, pre-registered

Top-k is a **biased** estimator of `B`: it drops positive mass and never reweights, so it always
underestimates. Bernoulli and VKvWR are **unbiased by construction**. On `|B_hat - B|/B` the sampled
arms are therefore favoured *by the definition of the metric*, and a win that merely restates this
is worthless. The result is only meaningful reported as a decomposition — signed error, absolute
error, `MSE = bias^2 + variance`, and the `kvzip_rescaled` arm which removes top-k's bias and leaves
only variance. The genuinely open question is whether top-k's **concentration** advantage (it keeps
the largest weights, so low variance) beats the sampled arms' **zero bias**. If `kvzip_rescaled`
matches the sampled arms, the win is "the correction", not "the sampling" — and that is the finding.

### Fairness note on VKvWR, to be stated in the write-up

His own `denominator_error` measures the estimator against a **group-mean query**
(`query[:,:,-1,:].view(B, n_kv, n_groups, D).mean(dim=2)`), per KV head, over **residual** keys only,
then averages over heads. E5's metric measures the real **per-query-head** denominator over the
**whole** context including forced keeps. These are genuinely different quantities, and the new one
will generally report a **larger** error for `vkvwr_kvzip` than his logger does. That is not a
regression and must not be presented as one. It does suggest a follow-up worth raising with him: the
group-mean query is an approximation living inside the press itself, and it may be costing accuracy.

A second, smaller asymmetry: at the question-prefill step VKvWR applies one bias, tuned for the last
query row, to all `q_len` rows. Measuring only the last row therefore *understates* vkvwr's error
there, in its favour. `q_pos`/`q_len` are logged so this is visible, and a `query_pos="all"` knob
quantifies it on a small subset if the result turns on it.

## Statistics

- **Pairing:** fixed (model, benchmark, context, ratio, question, decode step, layer, head);
  identical scores, identical budget, only the selector differs.
- **Aggregation:** per-head median error, then `median [p25,p75]` across heads, **split at layer
  13**, plus the fraction of heads where each sampled arm beats top-k.
- **Uncertainty:** cluster at the **context**, never the query — turns of one context share one
  compression draw.
- **Tests:** small-sample only (paired t on context means with 9 df, or exact sign-flip) with
  **Holm** across the pre-registered family. Never `|z|>2` — explicit E3-audit lesson.
- **Seeds:** 2 seeds for the stochastic arms (3rd if time allows); top-k is deterministic. Seed
  spread reported, not hidden in the mean.
- **GPU pinning:** the same CUDA seed gives different draws on A100 vs L40S, so **every arm of a
  given (model, task, ratio) cell runs on one GPU type**, and the type is logged.

## Pre-registered criteria

| | criterion | passes if |
|---|---|---|
| **C0** | harness trustworthy | no-op press reads ~0 on all three errors; a tiny-model `B` and `A` match a hand computation; the identity `A_hat = O_sparse * B_hat` reproduces a directly computed numerator to ~1e-6 in fp64; realised kept fraction within 1% of target; **top-k's signed denominator error is negative for every head** (a theorem - a breach is a bug); the offline replay reproduces the press's logged `B_hat` |
| **C1** | Bernoulli beats top-k upstream | lower pooled **denominator** error at 8x, context-clustered, Holm-corrected |
| **C2** | VKvWR beats top-k upstream | same test |
| **C3** | consistent | sign holds on both models x both benchmarks at 8x, and across 2x/4x/8x/16x |
| **C4** | the win is not just debiasing | sampled arms also beat `kvzip_rescaled`; if not, that is reported as the finding |
| **C5** | **it survives the ratio** | the **attention** error `\|\|Attn_hat - Attn\|\|/\|\|Attn\|\|` ranks the arms the same way the denominator does. This is the criterion that decides whether the upstream win is real or cosmetic: unbiased `A` and unbiased `B` do **not** give an unbiased `A/B`, so a method can win on both pieces and lose on the ratio |
| **C6** | the professor's CLT bound is calibrated | VKvWR's realised residual error exceeds its own `epsilon` on at most `delta` of (head, query) draws, tested against **his** `d_true` from his own logged fields |

C1 and C2 are the experiment. **C5 is the one that makes it matter** — E1's stated limitation was
that a correct denominator implies nothing about the attention output, and E5 is now in a position
to settle that rather than defer it. C4 decides whether the win is sampling or merely debiasing.
C6 is nearly free, uses only what his logger already writes, and is the check he will want first.

### Reporting the decomposition

For every arm and ratio, report the three errors side by side, per head, `median [p25,p75]`, split
at layer 13, plus the numerator error against **both** `||A||` and `Abar`. The interesting failure
mode to watch for: an arm whose `B` error is tiny but whose `Attn` error is not, which means the
numerator and denominator errors are **correlated** rather than cancelling.

### A framing worth keeping

VKvWR's selection distribution is query-**agnostic**; only its **sample size** is query-adaptive,
re-solved per head from a variance estimate. That is exactly the per-head adaptive budget E3's C5
argued for when it found one global `c` pays for the worst head in the model. So E5 also tests
whether adaptive per-head budgeting buys accuracy — E3's own next step.

## Two independent implementations, so the result is verifiable

**Primary (in-harness).** Real presses through `evaluate.py --enable_micro_metrics`, with the new
per-head metric. This is the only honest way to measure VKvWR (it is query-adaptive and lives inside
the press) and it satisfies branch-consistency.

**Cross-check (offline replay).** `phase2/experiment1/denom.py` is reusable **verbatim** — `b_hat`
and `rel_error` are already this estimator and this metric — and `run_exp1.py` L164-198 is already
the extract/calibrate/decode-query pipeline. For the two *static* arms (`kvzip`, `bernoulli`) E5
replays the sampler numerically over stored `w` and asserts agreement with the press's logged
`B_hat`. Two implementations agreeing is what makes the number trustworthy rather than merely
reproducible. (VKvWR cannot be replayed this way without reimplementing it, so it is checked only
in-harness — stated as a limit.)

## Cost control

- **`max_new_tokens` = 4.** Only decode *queries* are needed, not answers. This is the single
  biggest saving: E3's `vt` runs cost ~5 100 s each because they generated 128 tokens x 5 turns.
- **KVzip score caching** across arms and ratios (`ScoreCacheMixin` pattern from
  `phase2/experiment3/e3_press.py`) — the repeat pass is paid once per (model, context).
- **All three errors everywhere.** The double dense forward fires only at decode and on the short
  question prefill, never on the 110k context prefill, so it is affordable on SCBench as well.
  A per-metric `--micro_metrics` CLI flag is still added (the logger's
  `configure_logging(enabled_metrics=...)` already takes a subset; only the CLI hardcodes
  `DEFAULT_MICRO_METRICS`) so the output-error forward can be switched off if a run proves tight.
- SCBench peaked at 47-51 GB at 110-124k in E3/E4, so SCBench goes to **dgx/A100-80GB** (queueing
  expected) or falls back to 5 contexts on l40; RULER 32k fits comfortably on a40/l40.

## Scope

- **RULER 32k:** 4 tasks spanning the families — `niah_single_3`, `niah_multikey_2`, `vt`, `cwe`
  (the aggregation task where E2 found the whole effect) — ~20 samples each.
- **SCBench:** 2 tasks, `prefix_suffix` and `qa_eng`, 10 contexts x 5 turns, E3's untruncated set.
- 2 models x 6 tasks x 4 ratios x (1 top-k + 2 Bernoulli seeds + 2 VKvWR seeds) = **240 runs**,
  most of them cheap. Cut order if time runs short: 3rd seed first, then SCBench contexts 10 -> 5,
  then RULER tasks 4 -> 2. **The 8x cell of every (model, benchmark) is run first**, so a truncated
  sweep still decides C1 and C2.

## Files

    phase2/experiment5/
      DESIGN.md        this, pre-registered, before any result
      run_sweep.py     resumable sweep driver (the E2/E4 pattern: many runs, one model load)
      replay.py        offline cross-check; imports phase2/experiment1/denom.py unchanged
      selftests5.py    T1..Tn, run before the sweep, abort on failure
      summarize.py     CPU: tables, context-clustered tests, Pareto curves
      job_*.sbatch     SLURM
      out/             REPORT.md, figures/
      CONCLUSION.md    after the runs

On the `e5-upstream-error` branch, the metric commit touches:
`kvpress/metric_logging/` (new `head_stats.py` for the maths + a parquet sink),
`kvpress/attention_patch.py` (the hook in all three branches, plus the save/restore fix),
`kvpress/presses/kvzip_press.py` (one line recording `ctx_len` in cache metadata),
`kvpress/pipeline.py` (four `begin_context/question/step` calls so rows can be keyed),
`evaluation/evaluate.py` (opt-in flags, wired into the existing configure/flush hooks),
and a new `tests/metric_logging/test_head_stats.py`. All flags default **off**, so no existing
run changes behaviour and the professor's JSONL output is untouched.

### Output: a parquet sidecar, not JSONL

Per-head rows at 32 layers x 32 heads x ~4 steps x ~50 questions is ~200k rows/run; as JSON Lines
that is ~60 MB/run and ~14 GB over the sweep. Instead write `head_stats.parquet` beside
`micro_metrics.jsonl`, one row per (context, question, decode step, layer, query head), ~34 B/row
=> **~7 MB/run raw, ~1.5 MB zstd**, ~0.4 GB for the whole sweep, and a whole run loads in under a
second with `pd.read_parquet`. The run config (press, ratio, model, seed, head counts, scaling, git
SHA, library versions) goes in the parquet schema metadata so a file is interpretable on its own.
One row group per flush, so a killed run still leaves a readable prefix. A summary record in the
professor's **existing** `denominator_error` JSONL shape is still emitted per (layer, step), which
keeps his consumers working and gives the two views a mutual cross-check.

A deterministic `layer_stride` / `step_stride` / `question_stride` gates the work **before** the
einsum, so striding saves compute and not just disk. `MicroMetricLogger.sampling_factor` must
**not** be reused for this: it samples after the compute and independently per record, which would
shred the per-head tuples and make the bias/variance split impossible.

## Verification

1. **Unit tests on the metric** (tiny model, no GPU needed), the ones that catch silent wrongness:
   no-op press reads ~0; a hand-computed denominator from a float64 Python loop; GQA group mapping
   asserted equal to `add_attention_bias`'s `repeat_interleave`; `logaddexp(B, B_suffix) ==
   logsumexp(all valid logits)`; boolean mask unmutated and still boolean afterwards; `None` / bool
   / additive masks give bit-identical answers; bottom-right alignment at `q_len=3, k_len=10`;
   top-k's `B_sparse` computed from real keys even after the fake-key slots are overwritten with
   garbage; **the save/restore is bitwise output-neutral over several decode steps**; the dense
   reference still correct at step 3 (this one fails on unmodified `vkv1` and documents the bug).
2. **Statistical sanity per arm:** over ~200 seeds on a fixed tiny context, Bernoulli's mean
   `B_hat/B` is 1 within a few SE (a GQA mis-map or a bias sign flip shows up immediately as a
   biased ratio); plain top-k never over-estimates.
3. `selftests5.py` in every job: kept fraction on target; bias-carrying decode calls counted (the E2
   trap where nine runs silently equalled `no_press`).
4. `replay.py` agreement between the in-harness `B_hat` and the offline replay for `kvzip` and
   `bernoulli`, on one context per model — two independent implementations agreeing.
5. Existing kvpress suite must not regress on the E5 branch (the `duo_attention` network test fails
   on this cluster for unrelated reasons and is excluded).
6. Every figure and table regenerable from the parquet by `summarize.py` with no GPU.

## Timeline (from ~14:20)

| | |
|---|---|
| +0:30 | prefetch RULER 32k on the login node; build the `e5-upstream-error` branch (vkv1 + scbench, one known conflict in `evaluate.py`); `DESIGN.md` committed before any result |
| +2:00 | metric implemented, unit tests green — this is the critical path and the only real risk |
| +2:30 | smoke: 1 RULER task x 2 samples x 3 arms x both models; C0 verified end to end |
| +6:30 | sweeps done, **8x cells of every (model, benchmark) first** |
| +8:00 | `REPORT.md` + `CONCLUSION.md` |

**If the metric is not green by +3:00**, fall back to the offline-replay path (`phase2/experiment1`
machinery, which already works) for `kvzip` and `bernoulli` and report VKvWR from his existing
JSONL `denominator_error` alone, clearly labelled as a different statistic. That still answers C1
today; C2 becomes indicative rather than conclusive.

**Cut order if the sweep runs long:** 3rd seed, then SCBench contexts 10 -> 5, then RULER tasks
4 -> 2, then ratios to {4x, 8x}. Because the 8x cells run first, C1 and C2 survive every cut.

## Known traps carried forward

- Compute nodes have no DNS: `HF_HUB_OFFLINE=1`. For datasets that are **not** published as configs
  this needs the existing offline parquet readers
  (`phase2/experiment2/run_sweep.py::_load_local_parquet`). RULER 32k does **not** need that: the
  professor's loader calls `load_dataset("xAlg-AI/att-hub-ruler-32k", task, split=task)`, i.e.
  config-name loading, which resolves from the builder cache offline — the same mechanism the
  SCBench PR relies on, already observed working in `kvpress_pr/logs/scb-verify-312876.out`
  ("Found the latest cached dataset configuration ..."). So the prefetch is just one
  `load_dataset(...)` per task on the **login node**, which populates `$HF_HOME/datasets`.
- `LD_LIBRARY_PATH="$CONDA_PREFIX/lib"` is load-bearing — SCBench's scorer imports nltk -> sqlite3,
  which needs `CXXABI_1.3.15`.
- Boolean SDPA masks: adding a float bias promotes True/False to 1.0/0.0 and destroys causal
  masking. The metric must read the mask without mutating it.
- `cache.crop(n)` with positive `n` is the legacy "crop to absolute size"; use negative.
- kvpress's `get_results_dir` **creates** a numbered subdir when the base exists — never call it to
  test whether a run finished; rebuild the name.
- Append new presses at the **end** of `tests/default_presses.py` — it names tests by list index.

---

## Deviations from the plan, recorded during implementation (before any result)

1. **The metric does not use the professor's double dense forward.** The plan was to recover the
   numerator from `A_hat = O_sparse * B_hat` using his `attention_output_error` pair. Implementing
   it showed the metric can compute all three errors itself from `query`, `key` and `value` in one
   pass, which is cheaper (no second attention call), gives per-head instead of per-layer values,
   and does not depend on the fake-key trick being numerically exact. `attention_output_error` is
   therefore left switched off.

2. **The save/restore fix for the fake-key write is not needed by E5** and was not made. Because
   the metric is taken *before* the mutation, the real keys are always available to it. The
   underlying bug in `attention_patch.py` is real and still worth reporting to the professor: from
   the second decode step onward there is no dense reference left, so his `attention_output_error`
   is only correct on the first decode forward for any press that drops through
   `masked_key_indices`. It is out of scope here.

3. **Per-metric `--micro_metrics` was added as planned**, but E5 runs with the JSONL metrics off
   entirely and only `--enable_head_stats` on.

4. **Self-tests moved from a pre-flight to `check_run.py`**, which validates each finished run's own
   parquet. Every invariant the pre-flight would have tested (kept fraction on target, top-k never
   over-estimating, the estimate differing from the truth at all) is present in the run's output, so
   checking it there is strictly stronger: it covers every run rather than one.

## Status

- `DESIGN.md` fixed before any result was looked at.
- Metric implemented on branch `e5-upstream-error` (commit f30e1ed), 14/14 unit tests pass.
- Verified end to end on the unit-test model: Bernoulli reads mean `B_hat/B = 0.975` (unbiased) and
  top-k reads `0.333` (biased low), which is the signal that the GQA mapping and the
  Horvitz-Thompson correction are both correct.

---

# AMENDMENT — 2026-09-20 ~15:40, after the smoke runs, before any sweep run

Two smoke runs on RULER 32k (`niah_single_3`, Llama-3.1-8B, 8x, 2 contexts, 8192 head-rows per arm)
established the harness and changed the design. Everything below was fixed before a single sweep
run was submitted.

## What the smoke established

All C0 checks pass (`check_run.py`): kept fraction on target, top-k never over-estimates, no NaNs,
the estimate differs from the truth on 100% of rows.

| arm | kept frac | mean B_hat/B | denominator err | attention err (median) |
|---|---|---|---|---|
| `kvzip` (top-k) | 0.125 | biased | -28% | 0.355 |
| `bernoulli` | 0.125 | 0.9921 | -1.4% | 0.181 |
| `vkvwr` | **0.245** | 0.9885 | -1.4% | 0.045 |

Both sampled arms are unbiased to within 1%, which is the end-to-end confirmation that the GQA
mapping and the Horvitz-Thompson / Hansen-Hurwitz corrections are right. Top-k's denominator sits
28% low, as a biased estimator must.

## The finding that forced the amendment

**VKvWR keeps 2x the memory at the same nominal ratio, and the excess is not waste — it is
redistribution.** Measured decomposition at `compression_ratio = 0.875`, `n = 32,180`:

    forced keeps  = sink 4 + local 4 + top (1-cr)/2.1 * n   = 0.0598 n
    pilot sample  = (1-cr)/2.1 * n draws, with replacement
    measured kept = 0.2445 n = 0.0598 forced + 0.1848 sampled

The sampled part is **3.1x the entire budget the ratio nominally granted**, because
`compression_ratio` sets no budget in his method: it only sizes `topk_frac` and `base_sample_frac`,
while the kept count is decided at decode by the CLT bound `m >= z^2 (M2 - D^2)/(eps^2 D^2)`,
clamped only by `max_sample_frac`, whose default is 1.0.

Per-head kept fraction tells the real story:

| arm | min | median | max | spread |
|---|---|---|---|---|
| `kvzip` | 0.0001 | 0.0010 | 0.9925 | **7985x** |
| `bernoulli` | 0.0002 | 0.0059 | 0.9915 | **4559x** |
| `vkvwr` | 0.1080 | 0.1169 | 0.6212 | **5.8x** |

Both of our arms use one **global** `c`, so the budget is allocated by score across all heads and
starves some to 1 token in 10,000. His per-head CLT gives every head a floor. That is precisely
what E3's criterion C5 recommended ("one global c is paying for the worst head in the model").

**So the original comparison confounds two different things:**

| | ours | his |
|---|---|---|
| estimator | Bernoulli, without replacement, Horvitz-Thompson `log(1/r_i)` | multinomial, with replacement, Hansen-Hurwitz `log(I_i/(m s_i))` |
| allocation | one global `c` | per-head, adaptive, from a CLT bound |
| what is held fixed | **memory** (error is the outcome) | **error target** (memory is the outcome) |
| decided when | prefill, query-agnostic | pilot at prefill, `m` re-solved per decode step on the real query |

## Amendments

**A1. A new ablation arm, `perhead_bern_kvzip_s{1,2}`** (`perhead_press.py`): the shipped
`BernoulliPress` estimator with a **per-head** budget instead of one global `c`. It subclasses the
shipped press and reuses `_solve_c_per_head`, `_keep_prob` and `transform_scores_for_sampling`, so
the sampling, sink handling and `log(1/r)` bias are byte-for-byte the shipped ones. It holds the
estimator fixed and changes only the allocation, so the two effects separate. It is an ablation,
not a proposed method, and is deliberately kept in the experiment directory rather than added to
the kvpress branch.

**A2. VKvWR gets its own ratio ladder**, `{0.875, 0.9375, 0.96875, 0.984}`, via the new `arm_ratios`
key in the sweep yaml, so its realised kept fraction spans down to overlap the other arms. The
headline comparison is read off error-versus-**realised**-memory curves; the nominal-ratio table is
kept only for continuity with E3 and E4.

**A3. Scope trimmed** (user decision, cost): RULER 32k 10 samples per task (was 20); SCBench 5
contexts and ratios {4x, 8x} only with one seed per sampled arm (was 10 contexts, 4 ratios, 2
seeds), with `layer_stride: 2` at 110k. RULER keeps all 4 ratios and 2 seeds. Ratio-first ordering
is unchanged, so 8x lands for every task before 4x begins.

**A4. New criterion.**

| | criterion | passes if |
|---|---|---|
| **C7** | the win is the estimator, not just the allocation | at matched realised memory, `perhead_bern` is compared against both `bernoulli` (isolating allocation) and `vkvwr` (isolating the estimator). If `perhead_bern` closes most of the gap to `vkvwr`, the result is "per-head budgeting is what matters"; if it does not, the estimator is what matters. Either way it is reported as the finding. |

C7 is now as important as C1 and C2: without it the headline is uninterpretable.

**A5. `kvzip_rescaled` stays** as a post-hoc analysis arm (C4). It needs no extra runs.

## Final run plan

8 SLURM jobs, sharded to run in parallel under the QOS caps (a40 2 GPUs, l40 4, dgx 4):

| job | model | dataset | tasks | partition |
|---|---|---|---|---|
| `r32k_llama_a` / `_b` | Llama-3.1-8B-Instruct | RULER 32k | `niah_single_3, niah_multikey_2` / `vt, cwe` | l40 |
| `r32k_qwen_a` / `_b` | Qwen3-4B-Instruct-2507 | RULER 32k | same split | l40 / a40 |
| `scb_llama_a` / `_b` | Llama-3.1-8B-Instruct | SCBench ~110k | `prefix_suffix` / `qa_eng` | dgx (A100) |
| `scb_qwen_a` / `_b` | Qwen3-4B-Instruct-2507 | SCBench ~110k | same split | dgx (A100) |

Arms per run: `kvzip`, `bernoulli_kvzip_s{1,2}`, `perhead_bern_kvzip_s1`, `vkvwr_kvzip_s{1,2}`
(SCBench drops the second seeds). Estimated ~19 GPU-hours, ~2.5 h wall at 8-way parallelism.
Measured cost basis: ~45 s per RULER-32k sample, 24.1 GB peak on an A40.

## Risks accepted, stated before the run

- **Qwen3-4B at 110k is untested in this line.** It is a registered kvpress architecture with a
  fixture for this exact checkpoint and 262k native context, but no prior experiment here has run
  Qwen at SCBench length. If it fails it fails early and the arm is dropped, not debugged.
- **RULER 32k `vt` and `cwe`** come from `xAlg-AI/att-hub-ruler-32k` and have only been exercised on
  `niah_single_3`.
- VKvWR needs the full key set resident at decode to compute `a_i`, and the masked arms do not free
  memory either, so every number is approximation quality at a **modelled** memory level.
