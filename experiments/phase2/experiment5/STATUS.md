# Experiment 5 — status log

Newest entry at the bottom. Final write-up: `CONCLUSION.md`. Design: `DESIGN.md` (read the
AMENDMENT at the end first).

---

**2026-09-20 16:20 — sweeps submitted.** 148 runs across 10 jobs.

| job | sweep | partition |
|---|---|---|
| 315269 | r32k_llama_niah_single_3 | l40 |
| 315270 | r32k_llama_niah_multikey_2 | l40 |
| 315271 | r32k_llama_vt | l40 (queued) |
| 315272 | r32k_llama_cwe | l40 (queued) |
| 315273 | r32k_qwen_niah_single_3 | a40 |
| 315274 | r32k_qwen_niah_multikey_2 | a40 |
| 315275 | r32k_qwen_vt | dgx |
| 315276 | r32k_qwen_cwe | dgx |
| 315277 | scb_llama (prefix_suffix, 110k) | dgx |
| 315278 | scb_qwen (prefix_suffix, 110k) | dgx |

## Before the sweeps

- **Branch** `e5-upstream-error` = `origin/vkv1` + `origin/scbench-benchmark` + 3 commits
  (the per-head metric, the key restore, the attn_implementation fix). 16/16 metric unit tests pass.
- **Smoke 1** (315245): 3 arms, RULER 32k, all C0 checks pass.
- **Smoke 2** (315253): the per-head ablation and the VKvWR ratio ladder; all C0 checks pass.
- **Smoke 3** (315260 Qwen on vt+cwe, 315261 SCBench 110k on A100): the two untested corners;
  all 10 runs pass.

## Bugs found by the pre-flight check, all fixed before submitting

1. `summarize.py` classified `perhead_bern_kvzip_s1` as the top-k baseline (it starts with neither
   "vkvwr" nor "bernoulli", so it fell through to the else branch) and **pooled the ablation arm
   with the arm it exists to be compared against**. Would have corrupted the headline.
2. `summarize.py` compared only at nominal ratio, which is exactly the confound the amendment
   exists to remove. Added an error-versus-realised-memory table and a C7 section.
3. `check_run.py` used the same brittle prefix match, and `all(...)` short-circuited so it skipped
   every run after the first failure.
4. SCBench `layer_stride: 2` distorted the realised kept fraction for the globally-budgeted arms
   only (0.1399 against a target of 0.1250) because KVzip's global budget is not uniform across
   layers, while a per-head budget hits the target in every layer. Set to 1.

## Earlier, during implementation

5. `evaluate.py` overwrote an explicitly requested `attn_implementation`, so the `kvzip` baseline
   would have run under flash-attention-2 while the sampled arms ran under sdpa.
6. The fake-key write destroys the real keys in the KV cache for every LATER forward, not just the
   current one, so from decode step 2 onward there was no dense reference: the kvzip arm showed a
   difference on 1 of 4 steps and an attention error of exactly 0. Fixed with an output-neutral
   save/restore.
