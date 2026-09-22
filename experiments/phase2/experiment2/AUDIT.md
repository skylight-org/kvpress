# Phase 2, Experiment 2 — Pre-launch audit

**Date:** 2026-09-15, before any GPU job was submitted. Every claim in `DESIGN.md` was checked against
the actual kvpress v0.5.4 source in `~/src/kvpress`, and the numerical claims were run.

Three bugs were found by reading, all three confirmed by measurement, all three fixed. Each would have
produced **plausible wrong numbers rather than an error** — the failure mode this project has been
bitten by before.

---

## Verified correct

| Item | How checked | Result |
|---|---|---|
| `KVzipPress` separates scoring from selection | read `kvzip_press.py`: `score_kvzip` fills `self.score_val` (L352); `compress_post` (L358) does global top-k over it | confirmed — overriding `compress_post` alone inherits scoring byte for byte, which is what makes B1 controlled |
| Budget matching vs top-k | `solve_c_for_memory` against KVzip's `numel - int(numel*ratio)` at 0.5/0.75/0.875 | exact, rel. err **0.00e+00** at every ratio |
| §11 high-probability memory bound | `memory_highprob` on 1M-element score tensor | M⁺ = M + 0.48%, tight and sane |
| RULER availability | `simonjegou/ruler` 4096 + 8192 downloaded to `~/hf` | 13 subtasks × 500 samples, columns match LongBench's loader |
| Cost model | measured from E9 Part C logs (~1.8 s/sample cached) and E8 (~8 s/sample first touch) | ≈ 4.1 h wall on 3 A40s for the chosen scope |

---

## Bugs found and fixed

### 1. Attention-patch signature was wrong (would silently drop the mask)

kvpress calls attention as `(module, query, key, value, attention_mask, dropout, **kwargs)` with
**`dropout` positional** (`attention_patch.py` L61). My wrapper declared
`(module, q, k, v, attention_mask=None, *args, **kwargs)`, which misbinds `dropout` into `*args` and,
on paths where the mask is passed positionally, corrupts the argument order.

**Fixed:** the wrapper now mirrors kvpress's own signature exactly.

### 2. The logit bias was applied during the prefill and calibration passes

*Terminology, because "mask" means two different things here:*
- *the **attention mask** — an additive bias on the logits before the softmax. That is what `e2_bias`
  is, and what this bug is about.*
- *"masking" a dropped token — see the note on deletion below.*

A context passes through three phases:

| Phase | What happens | Should the bias apply? |
|---|---|---|
| 1. Context prefill | read the context, build the cache | **No** — nothing is compressed yet |
| 2. Calibration repeat passes | re-read the context to compute `score_val` | **No** — scoring must see the true cache |
| 3. Question prefill + decoding | consume the compressed cache | **Yes** |

`e2_bias` lives on the attention module, so once set it stays set. With no phase guard it would still
be attached during phases 1 and 2 of the **next** sample, and scoring would read a cache already
corrupted by the previous context's dropped tokens.

kvpress distinguishes the phases by shape (`attention_patch.py` L62–64): during a prefill every query
is new, so `query_len == key_len`; during decoding it is 1 query against thousands of keys.

**Fixed:** the bias now applies only when `query_len != key_len`, the same test kvpress uses.

Like the other two, this produces no error — only subtly wrong scores, a degraded Bernoulli arm, and
the plausible-but-false conclusion "sampling loses to top-k".

---

### Note: why dropped tokens are masked rather than deleted

This is kvpress's design, inherited by every head-wise press including `kvzip`, not a choice of ours.

Selection is **per KV head**: head 3 may keep {5, 9, 200} while head 4 keeps {1, 9, 512}. But the cache
is one dense tensor, `[batch, n_kv_heads, seq_len, head_dim]`, with a single `seq_len` axis shared by
every head. Truly deleting different tokens per head needs a **ragged cache** and a custom kernel with
per-head offsets; PyTorch's attention kernels do not accept one.

So kvpress simulates it. Its own docstring is explicit:

> "The keys are updated with a fake key k such that exp(⟨q,k⟩) = 0 to fake head-wise compression. This
> solution is not optimal as it does not reduce peak memory and slightly increases runtime."

`search_hyperplane` (`attention_patch.py` L8) constructs a key whose attention weight is exactly zero.
Our press achieves the same thing more directly by adding `−inf` to the logit.

**Consequence for this experiment:** the model's output is exactly what it would be if those tokens
were gone, so accuracy comparisons are faithful and every arm is handicapped identically. But **no
memory is physically saved** — "2× compression" describes the algorithm, not GPU usage. Real savings
need a ragged-cache kernel, which is out of scope (DESIGN §9).

### 3. Bernoulli drops the attention sink (the serious one)

KVzip marks sinks by setting `score_val[..., :n_sink] = 1.0` (L246). But **1.0 is not distinguished** —
ordinary tokens reach it too — so `r = min(1, c·1.0) = c` whenever `c < 1`. Measured on a realistic
score distribution:

| compression | c | Pr[sink kept] |
|---|---|---|
| 2× | 1.11 | 1.00 |
| **4×** | 0.56 | **0.56** |
| **8×** | 0.28 | **0.28** |

Top-k keeps sinks because they tie at the top of the ranking; Bernoulli does not. E5a established that
layer 0 holds essentially *all* of its attention mass on the sink, so dropping it is catastrophic.

This would have crippled the Bernoulli arm at exactly the ratios where B1 is decided, and the result
would have read as "sampling loses to top-k" — a wrong conclusion with a plausible shape.

**Fixed:** force `r = 1` on the first `n_sink` tokens, then **re-solve `c` by bisection so the forced
keeps are paid out of the same budget**. Verified: sinks kept in every layer and head, expected memory
still exactly on budget (**+0.0000%** at 2×, 4× and 8×), so the equal-memory comparison against top-k
is preserved.

The derivation permits this explicitly (§3, "forced keeps are allowed"): any `r'ᵢ ≥ min(1, cπᵢ)` leaves
every bound valid, because each tail term `(1−r)/r` is non-increasing in `r` and forcing `r = 1` only
removes a token from the tail.

**Test T4 was extended** to assert sink preservation, so this cannot regress silently.

---

## Still to verify on GPU (the self-tests, before the real sweep)

T1–T3 and T5 require a loaded model and are run by the smoke job, which aborts on failure:

- **T1** compression 0 → logits identical to `no_press`
- **T2** bias 0 with dropped tokens masked → identical to top-k's fake-key masking
- **T3** one layer offline: masked-softmax output == `Â/B̂`
- **T5** `kvzip_bern` and `kvzip` produce **identical `score_val`** — the check that makes B1 a
  controlled comparison rather than a confounded one

T5 is the one that matters most: if it fails, the central claim of the experiment is not being tested
and the run should not proceed.

---

## Found after launch (2026-09-15/16), in order

Every item below produced plausible numbers or a quiet failure rather than a clear error, and each one
was caught by a test or a sanity check before it contaminated a result that is reported.

| # | Bug | How it surfaced | Fix |
|---|---|---|---|
| 4 | Offline RULER load failed: `datasets` 4.x cannot resolve a cached repo by name, and a partial `allow_patterns` download leaves the snapshot "incomplete" | pre-launch load check | read the snapshot parquet directly (E8's trap); completed the snapshot |
| 5 | `libstdc++` lacks `CXXABI_1.3.15`: RULER's registry imports loogle → nltk → sqlite3 | first smoke crashed at import | `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` |
| 6 | **Boolean attention mask.** transformers hands SDPA a *bool* mask (True = attend). Adding a float bias converts it to 1.0/0.0, destroying causal masking | self-test T1: an all-zero bias moved the logits by 19% | convert to the additive −inf convention before adding |
| 7 | Stale bias across samples: a 3039-token bias was applied to a 2193-token context | smoke crashed with a shape error (a same-length context would have run silently) | clear the bias per sample; added a hard guard against a stale bias |
| 8 | **The fix for #7 cleared the bias before generation.** kvpress wraps only the prefill in the press context, so clearing on exit discarded the compression | full P1 run: all 9 Bernoulli runs scored exactly 95.4154, identical to `no_press` | clear on entry instead. The 9 runs are quarantined in `results/P1_INVALID_*`; the controls stayed valid |
| 9 | T6 checked "does the answer change at 99% compression?", but KVzip's score keeps the needle even at 1% of the cache (KVzip itself gets 3/3) | T6 failed on a correct press; an instrumented diagnostic showed the bias live, blocking 0.990 | T6 now counts kernel calls carrying the bias and checks the blocked fraction |
| 10 | Draws were seeded from the tensor shape, so contexts of equal length got identical keep patterns | code review | seed from the context tokens |
| 11 | **`bern_tau` calibrated on 512 of ~3800 repeat positions**, missing the position that re-reads each needle. B6 would have measured subsampling, not calibration scope | all-arms smoke: `bern_tau` 73 vs 95, `niah_single_3` = 0 | `N_CAL` = all positions → 93.6 / 96.0, needles back to 100 |
| 12 | Summarizer's answer parser left escaped quotes in RULER answers | per-sample metric cross-checked against metrics.json: `qa_2` off by one sample | `ast.literal_eval` per token |
| 13 | Summarizer printed "B0 PASS" with no Bernoulli data | review of the test report | withhold a verdict without data |

**The lesson, again:** kernel-level self-tests (T1–T4) passed through bugs #8 and #11. Those bugs lived
in the *lifecycle* (when the bias exists) and the *configuration* (which queries calibrate). Only
end-to-end checks through the real pipeline caught them: exact equality with `no_press`, per-subtask
collapse. Every arm now runs through such a check before any long job.
