# Experiment 3 — audit

Everything that was wrong, suspected wrong, or deliberately excluded, and how each was settled.
Same convention as `../experiment2/AUDIT.md`: a finding is only closed when there is a check that
would fail if it came back.

---

## A1 — `scbench_kv` has no signal under the uncompressed model → task excluded

**Symptom.** Every arm, including `no_press`, scored exactly 0.0 on `scbench_kv`.

**Diagnosis** (`debug_kv.py`, `debug_kv2.py`). The gold keys of the sampled contexts sit at relative
depth 0.00–0.02 of a ~124k-token context. Llama-3.1-8B-Instruct retrieved **0 of 15** of them with the
full cache. Probes injected mid-context were retrieved 56% of the time, so the harness, the loader and
the scorer are all fine — the model simply cannot do this task at this length and this depth.

**Resolution.** `scbench_kv` is excluded from E3 and the exclusion is stated in `DESIGN.md` and
`CONCLUSION.md` with the numbers. A task whose ceiling is zero cannot discriminate between compressors.
`TASKS` in `scbench_data.py` still lists it, with the comment that it is excluded from runs.

## A2 — `prefix_suffix`: top-k scores exactly 0.00 — bug or result?

**Symptom.** `kvzip` and `expected_attention` scored exactly 0.00 at 4×, 8× and 16×, while Bernoulli
held 18–21. An exact zero across three ratios is the signature of a broken arm, and E2 taught that
plausible-looking nulls are the expensive failure mode.

**Checks run, all pointing the same way:**

| check | result | rules out |
|---|---|---|
| `kvzip` at 2× on the same cached-score path | 26.0 | broken code path / bad cache |
| `expected_attention` (different press, no score cache) | also 0.00 | anything specific to the cache |
| predictions that are *partial* matches of the gold | 0 of 50, every arm | truncation stealing credit |
| predictions hitting the 64-token generation cap | 1 of 50 for `kvzip`@4× | too-small `max_new_tokens` |
| inspection of the actual failed outputs | prefix+suffix **concatenations** | a scoring/normalisation bug |

The last one is decisive. Gold `LBlPUIsjuZYbsHkx^fkUzVdkS` → prediction `"LBlPUzVdkS"`; gold
`_TmwCWPliaoDImvXiOvic+KJB` → `"_TmwCk+KJB"`; gold `RviB-YiaAUgc_nSFCgaamszHq` → `"RviB-mszHq"`.
The model has the prefix and the suffix from the *question* and nothing in between, so it glues them.
That is what losing the middle of the answer span looks like, not what a harness bug looks like.

**Resolution.** Not a bug; reported as the headline result with the mechanism. One residual caveat is
recorded in `CONCLUSION.md`: `max_new_tokens=64` is tight for this task — `random` hits the cap 29–41
times of 50, so `random`'s zero is partly a cap artefact. It does not affect the real arms (≤6 of 50).

## A3 — `kvzip` beats the uncompressed model at `prefix_suffix` 2×

26.0 against a `no_press` ceiling of 20.0. No compressor can genuinely do this. On a binary metric with
10 contexts and a 20% ceiling this is sampling noise, and it is the same cell that makes C1's
pre-registered rule fail. Both facts are reported together rather than one being used to excuse the
other; the criterion was not rewritten after seeing the data.

## A4 — the score cache had to be proved harmless

`e3_press.py` caches KVzip's `score_val` per context so the top-k and Bernoulli arms provably read the
same bytes, and so calibration (~200 s/context) is paid once instead of 16 times.

**Check.** The same run was executed with and without `E3_NO_SCORE_CACHE=1`; predictions were identical
on 10 of 10 contexts. The cached tensor's shape is asserted against the model config on every load, and
writes go through `os.replace` so parallel jobs sharing the cache cannot observe a partial file.

## A5 — `selftests.py` crashed on import

`bias_patched` was used but not imported after the module was copied over from E2. Fixed by importing it
from `e3_press`, which re-exports it. Caught before any long job — T1–T6 now run at full 110k length and
`pass: true` is written to `results/*/selftest.json` before the first run of every job.

## A6 — kvpress's `get_results_dir` cannot be used to look up an existing run

It **creates** a numbered subdirectory when the base path exists, so calling it to test for a finished
run silently creates directories and breaks resumption. `run_sweep.py:run_dirs` rebuilds the base name
without side effects and globs the numbered children. Requeued jobs now genuinely resume (`vt` was
requeued once and skipped its finished runs).

## A7 — `fraction` would have broken the multi-query grouping

kvpress's `fraction` samples **rows**, which would split a context's 5 turns across the subset and
destroy the shared-context structure that makes 110k affordable. Context count is controlled by
`E3_N_CTX` in the loader instead, which takes whole contexts. `fraction` is pinned to 1.0.

## A8 — jobs did not start in parallel

All four Stage 1 jobs were submitted to `dgx` and only one ran: 69 of 72 GPUs were allocated and a
higher-priority job was ahead in the queue. Measured peaks were re-checked against the partitions —
`summary` (44.8 GB) and `prefix_suffix` (44.6 GB) fit an L40S (48 GB), `qa_eng` (49.3 GB) and `vt`
(47.7 GB) do not — and the first two were resubmitted to `l40`. Four jobs then ran concurrently.

Peak memory is why this experiment needed A100s at all: 47–49 GB exceeds an A40's 44.4 GB.

## A9 — reporting errors caught while writing the report

- `summarize.py` first printed each task's context length as the median over **all** SCBench samples,
  including the ones dropped for exceeding the model limit. For `qa_eng` that read 163,591 tokens for
  runs whose contexts actually averaged 99,419. Changed to the median over the *fitting* samples.
- Per-query scores are recomputed from `predictions.csv` and asserted equal to each run's
  `metrics.json` (tolerance 0.02). This is the check that caught a parser bug in E2; it passed on all
  82 E3 runs.
- SEs are clustered at the **context** level, not the query level. The 5 turns of a context are served
  from one prefill and one compression draw, so treating them as independent would understate every SE
  by roughly 2×.

## Still open

- **Memory is counted, not freed** (inherited from E2, `../experiment2/AUDIT.md`): kvpress masks dropped
  tokens instead of shrinking the cache. Every arm is handicapped identically so comparisons are fair,
  but no wall-clock or GPU-memory saving has been demonstrated anywhere in this phase.
- **`vt` is incomplete** — 8× only. Not an error; the wall clock ran out and the sweep was ordered to
  reach the most informative ratio first.
