# Experiment 3 — timeline

| when | what |
|---|---|
| Stage 0 | SCBench downloaded on the login node (~175 MB, 5 task files); `measure_lengths.py` tokenised every context → `out/lengths.json`: 383 of 429 contexts fit under 131,072 tokens, 46 dropped (all `qa_eng`). |
| Stage 0 | `scbench_data.py` (no truncation, drop-and-count, one row per context×turn), `scbench_metrics.py` (per-task scorers), `e3_press.py` (score cache + Stage 2 statistics). |
| Stage 0 | A100 probe: peak 47.1 GB at 124k context → A40 (44.4 GB) ruled out, `dgx` confirmed. Calibration measured at ~200 s/context, motivating the score cache. |
| Stage 0 | Score-cache equivalence verified (`E3_NO_SCORE_CACHE=1`): identical predictions on 10/10 contexts. |
| Stage 0 | `scbench_kv` scored 0.0 for **every** arm including `no_press`; diagnosed as model failure at depth 0.00–0.02, not a harness bug → task excluded (`AUDIT.md` A1). |
| Stage 1 submit | Four jobs (one per task), 25 runs each, ratios ordered 8×, 16×, 4×, 2× so a truncated job still yields complete ratios. `n_contexts: 10`. |
| Stage 1 | Jobs did not run concurrently — `dgx` was 69/72 GPUs allocated. `summary` and `prefix_suffix` moved to `l40` on measured peak memory; four jobs then ran in parallel (`AUDIT.md` A8). |
| Stage 1 | T1–T6 passed at full 110k length in every job (`results/S1n10/selftest.json`, `T6_blocked_frac` 0.875 at 8×). |
| Stage 1 mid-run | `prefix_suffix`: top-k collapsed to exactly 0.00 at ≥4×. Investigated as a suspected bug; five independent checks showed it is real (`AUDIT.md` A2). |
| Stage 1 done | `qa_eng`, `prefix_suffix`, `summary` complete at 25/25 runs. `vt` complete at 8× (7 runs) and still running. |
| Reporting | `summarize.py` → `out/REPORT.md` from 82 runs; per-query scores cross-checked against every `metrics.json`; `CONCLUSION.md`, `AUDIT.md` written. |

**Cost:** ~22 GPU-hours (prefix_suffix 5.6 on L40S, summary 6.8 on L40S, qa_eng 4.8 on A100, vt 5.2 on A100 so far).

**Not done:** `vt` at 2×/4×/16×; Stage 2's query-agnosticism gap (`Σᵢ maxq ωᵢ` vs per-query `Σᵢ ωᵢ(q)`),
which needs a separate instrumented pass storing per-query attention weights at 110k.
