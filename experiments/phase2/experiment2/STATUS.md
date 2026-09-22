# Experiment 2 — overnight status log

Newest entry at the bottom. Final write-up: `CONCLUSION.md` (written when all runs finish).

---

**2026-09-16, start of overnight run**

- **First P1 run invalid.** All 9 Bernoulli runs scored exactly 95.4154, the same as `no_press`.
  Cause: the bias was cleared when kvpress's press context exited, and that happens *before*
  generation. The model compressed nothing. Fixed by clearing on entry instead, and confirmed with an
  instrumented end-to-end diagnostic (`debug/dbg_e2e.py`): the bias is now present for every
  question-prefill and decode call, blocking 0.990 of the context at 99% compression and 0.875 at 8×.
- **The 9 invalid runs are kept** in `results/P1_INVALID_bias_cleared_before_generation/`. The P1
  control runs (`kvzip`, `no_press`, `random`) were unaffected and remain valid.
- **The old T6 test was wrong, not the press.** KVzip's score keeps the needle even with 1% of the
  cache (KVzip itself gets 3/3 there). T6 now checks that the bias is attached during generation and
  blocks the intended fraction.
- **Seeding fixed.** Draws were seeded from the tensor shape, so contexts of equal length got the same
  keep pattern. They are now seeded from the context tokens.
- **P3 (`bern_tau`) wired in.** It uses E9's verified press, imported under a distinct module name.
  Its attention patch is installed only during `bern_tau` runs, so the other arms run in the same
  environment as the valid controls.
- **Smoke tests running:** all 11 arm types on A40 (job 311522), and an environment check on L40S
  (job 311523).

**All-arms smoke passed (job 311522, 11 arm types, 52 samples each)**

- Every arm ran. The Bernoulli arm really compresses now: its predictions differ from `no_press` on
  8–11 of 52 samples, the two seeds differ from each other, the kept fraction is within 0.02% of
  target, and 100% of samples fall inside the §11 memory bound.
- L40S environment smoke also passed (job 311523), including the new T6 check: the bias is attached
  on 256 generation calls and 0 prefill calls, blocking 0.8751 of the context. L40S runs about 1.6×
  faster than A40.
- **Methodological problem found in `bern_tau`.** It scored 73 against 95 for `kvzip_bern`, and the
  whole loss sits in the needle subtasks (`niah_single_3` went to 0). Cause: E9's press uses only
  512 evenly spaced repeat positions as calibration queries (about 1 in 7 here), so it misses the
  query position that re-reads each needle. As configured, B6 would have measured "1-in-7 positions
  vs all positions" rather than "full context vs chunked". Fixed by calibrating on every repeat
  position. Verification running (job 311547).
- Summarizer rebuilt around **paired per-sample** comparisons, with a check against kvpress's own
  metrics.json. That check caught a parser bug (escaped quotes in `qa_2` answers), now fixed.

**Launched:** A40 shards A1 (P1 2× + P4 4×) and A2 (P1 4×, P1 8× seed 1, P4 8×), stored alongside the
valid controls in `results/P1`. L40S shards L1–L3 run the P2 baselines plus a `no_press` hardware check,
stored in `results/L40S`. A3 (P1 8× seeds 2–3 and all of P3) is held until the `bern_tau` fix verifies.

**`bern_tau` fix verified (job 311547).** Calibrating on every repeat position brings the needle subtasks
back to 100 (`niah_single_3`: 0 → 100). The mean on the same 52 samples rises from 72.4 (512 positions)
to 93.6 and 96.0 (seeds 1 and 2), level with `kvzip_bern`. Calibration costs 6.3 s/sample on first touch
and 1.5 s once cached. **A3 launched** (P1 8× seeds 2–3, then all nine P3 runs).

**Also launched: an exploratory 16× point on L40S (shard L4).** RULER 4k sits near the ceiling at 2× and
4×, so B1 will mostly be decided at 8×. The 16× runs (`kvzip` plus `kvzip_bern` with 3 seeds, all on
L40S so the pair shares hardware) test the claim where the arms should separate. It is reported
separately and is not part of the pre-registered verdict.

Jobs: A40 A1 311542, A2 311543, A3 (new); L40S L1 311544, L2 311545, L3 311546, L4 311554.

**03:10.** All 7 jobs healthy. The A40 and L4 logs are empty only because `grep` block-buffers when
writing to a file. `sstat` shows full CPU use and the run directories are filling. Future jobs use
`--line-buffered`.
- **Hardware check passed:** `no_press` scores 95.32 on L40S and 95.42 on A40, a 0.10-point
  difference, so the L40S baselines can be compared with the A40 arms.
- **Baselines so far** (2× / 4× / 8×):

  | method | 2× | 4× | 8× |
  |---|---|---|---|
  | ExpectedAttention | 90.6 | 76.8 | pending |
  | SnapKV | 68.7 | 40.2 | pending |
  | StreamingLLM | 60.3 | 36.7 | 25.7 |

  KVzip scores 95.2 / 95.5 / 93.1 on the same samples.
- StreamingLLM (L3) is complete.

**03:16. The first valid Bernoulli runs are in, and the fix holds.** `kvzip_bern` seed 1 scores 95.58 at 2×
and 95.38 at 4×. That is no longer the 95.4154 of the invalid runs. Its predictions differ from `no_press`
on 208 and 211 of 780 samples, so it is really compressing. It sits level with `kvzip` (95.22 and 95.51),
as expected near the ceiling. ExpectedAttention (L2) is complete.

ETAs: L1 ~03:30, L4 (16×) ~04:30, A1 ~07:05, A2 ~07:50, A3 ~08:10. The B1, B5 and B6 pairs stay on A40
even though L40S slots are free, because splitting a pair across hardware would add kernel noise to the
per-sample differences.

**04:15 interim (partial seeds).** All P2 baselines complete; 16× half done; A40 shards on their third seeds.

| | 2× | 4× | 8× | 16× (exploratory) |
|---|---|---|---|---|
| `kvzip` (top-k) | 95.22 | 95.50 | 93.05 | 73.86 |
| `kvzip_bern` (sampling) | 95.58 | 95.78 | 93.89 | 74.06 |
| paired difference | +0.36 ± 0.30 | +0.28 ± 0.34 | +0.84 ± 0.54 | +0.20 ± 1.54 |
| ExpectedAttention | 90.60 | 76.82 | 45.18 | — |
| SnapKV | 68.72 | 40.21 | 25.99 | — |
| StreamingLLM | 60.32 | 36.73 | 25.66 | — |

Sampling leads at every ratio but no gap yet clears 2 SE. 16× does break the ceiling (93 → 74), which is
what it was added for. Hardware check: −0.10 ± 0.16 points, negligible.

**05:10.** 16× shard complete (all seeds). A40 runs are taking ~49 min each, so A1 ≈ 08:20, A2 ≈ 09:10,
A3 ≈ 09:30. Idle L40S slots put to use replicating the **8× B1 pair on independent hardware** (job
311617): 8× is where sampling leads most on A40, and a single-GPU result is the weakest part of the
headline. Both arms run on L40S so the pair is internally consistent.

**06:10. B1 complete at every ratio with all 3 seeds.** Paired differences (`kvzip_bern` − `kvzip`):
2× +0.31 ± 0.31, 4× −0.02 ± 0.32, 8× +0.79 ± 0.51, 16× +0.97 ± 1.15. Sampling is never significantly
worse and never significantly better; the largest lead (8×) is about 1.5 SE. Verified that all three 8×
seeds are genuine runs and that the quarantined invalid runs are in a separate directory the summarizer
never reads. Remaining: B5 (bias ablation), B6 (`bern_tau`, 8 runs left), and the L40S 8× replication.

**07:10.** `bern_tau`'s cached runs take ~21 min, so everything converges ~08:30–08:45.
- **B5 (bias):** +0.36 ± 0.31 at 4×, +0.51 ± 0.55 at 8×. Consistently positive, about 1 SE.
- **B6 (calibration scope):** −0.30, −0.23, +0.33 — full-context calibration buys nothing over KVzip's
  chunked pass at this context length (3.8k ≈ two chunks).
- **B4: PASS** — memory within 0.01% of target, 100% of samples inside the §11 bound.

**07:35.** A1 and L5 complete. **The 8× effect replicates on independent hardware:** +0.63 ± 0.55 on L40S
against +0.79 ± 0.51 on A40 — same direction, similar size. The two are not pooled: they score the same
780 questions with the same seeds, so they are correlated; this is a replication, not extra power.
A2 has one run left, A3 three; both finish ~08:30.

**08:25 — COMPLETE.** All 49 runs finished, no failures. `out/REPORT.md` regenerated, `CONCLUSION.md`
written, session logged to `~/EXPLANATION2.md`.

Final: B0–B5 PASS, B6 no effect. Sampling never significantly worse than top-k, +0.79 ± 0.51 at 8×
(replicated +0.63 ± 0.55 on L40S). The gain is concentrated in `cwe` (+13 at 8×, +72 at 16×), the one
RULER subtask that aggregates over the context rather than retrieving a needle.
