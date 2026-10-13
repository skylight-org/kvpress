# Experiment 4: timeline

| when (2026-09-17) | what |
|---|---|
| Stage 0 | smoke 312112/312113 failed T9: kvpress Compactor's leverage sketch is unseeded, so scores change per call. Fixed with `SeededCompactorPress`. |
| Stage 0 | smoke 312114 (qasper, 5 samples, every arm at 8x) PASS: T1-T9, no_press 64.77 (E8: 64.8), kept fractions 0.1247-0.1251. |
| Stage 0 | smoke 312115 (vt, 124k) crashed in STOCK Compactor: "Cholesky failed after 5 tries". Fixed with `FallbackLeverageScorePress` (float64, same sketch, only on failure). CPU check: stock output unchanged when stock works; float64 path agrees to 2.6e-6. |
| Stage 0 | smoke 312128 (vt, 124k) PASS: T1-T10; compactor 59 s and 37.0 GB per context, compactor_bern 72 s and 51.0 GB; kept 0.1249; 1 fallback per run. |
| Stage 1 | LongBench on A40: 312116 `lb_A` (qasper, hotpotqa, passage_retrieval_en, 102 runs), 312117 `lb_B` (gov_report, 34 runs). |
| Stage 1 | gov_report 16x/2x split into a parallel A40 job `lb_B2` (lb_B skips them when it gets there); cuts the full finish from ~14 h to ~7-8 h. Job 312141. |
| Stage 1 | SCBench on A100 (51 GB peak rules out the L40S): 312131 `sc_qa`, 312132 `sc_vt`, 312133 `sc_ps`, 312134 `sc_sum` (queued behind the per-user dgx cap). |
| Stage 1 | E3 job 312108 (user's) fills vt's missing 16x/4x/2x reused arms in `experiment3/results/S1n10`. |

After all finish: `python summarize.py && python plots.py` (kvpress env), then `CONCLUSION.md`.
| Stage 1 done | All jobs COMPLETED: lb_A 4h48, lb_B 8h55, lb_B2 8h01, sc_qa 3h35, sc_ps 4h23, sc_vt 6h02, sc_sum 4h55; E3 vt job 312108 6h05 (25/25). 272/272 runs, no duplicates (lb_B skipped the 16 runs lb_B2 had done). |
| Reporting | `plots.py` -> `out/figures/` (4 figures) + `out/REPORT.md`; `CONCLUSION.md`. Report fixed: it had been overwritten mid-run by a verification import (summarize now writes only when asked). |
