# Experiment 6 - downstream accuracy on E5's settings (+ RULER qa_1)

Started 2026-09-21. Code: ~/src/kvpress-e5 @ 163ab1b (the E5 branch; the error logging is off and never
changes generation, so answers are identical to what that branch produces).

## What it is
The downstream accuracy counterpart to E5: same models, tasks, arms, ratios and VKvWR ladder, so every
score pairs with an E5 upstream-error number. Additions: RULER `qa_1` (P0) and a `no_press` ceiling arm.

## Changes from E5's driver (run_sweep.py)
- enable_head_stats False
- max_new_tokens = task default (qa_1 32, vt 30, cwe 120, niah 128) instead of 4
- a run is done when it has written metrics.json

## Grid
RULER 32k: qa_1, niah_single_3, niah_multikey_2, vt, cwe x Llama-3.1-8B / Qwen3-4B, 30 samples each.
Arms: no_press, kvzip, bernoulli s1/s2, perhead_bern s1 at 4x/8x/16x, and vkvwr at cr 0.875..0.995.
18 runs per (model, task). Qwen's niah x2 and cwe are split into _a (8x + no_press + vkvwr ladder) and
_b (4x, 16x). Both halves run on A100, so every cell stays on one GPU type.
SCBench prefix_suffix: E5's config (fraction 0.05, 8x only, vkvwr ladder), plus no_press.

## Jobs
See STATUS_jobs.txt (job id, partition, config). The A100 queue caps submissions at 5 per user, so
r32k_qwen_cwe_b, scb_llama and scb_qwen are submitted by a retry loop as slots free up.
Smoke tests: 315597 (Llama) and 315598 (Qwen), results/smoke_*. Scoring works on both models.
Timing: Llama/L40S ~20-25 s/sample; Qwen/A40 42 (short answers) to 90-120 (long answers) s/sample.

## Analysis (todo)
Match VKvWR at REALISED kept fraction, as in E5, never at nominal ratio. Pair arms per sample
(same samples across arms); pool over tasks and models; cluster uncertainty by sample.

## Progress 2026-09-21 16:10 IST
6/12 original jobs done, 6 running, 0 errors. The user approved the proposal: r32k_qwen_cwe_b (315758)
and an SCBench smoke (315759: Llama, 1 context x 5 questions, no_press + kvzip) were submitted to the A100s.
scb_llama / scb_qwen get submitted ONLY if the smoke scores are sensible (E5's SCBench scored 0.0,
but that was 4-token answers, so the scorer is not yet validated at full answer length).

## Interim finding: per-head Bernoulli collapses downstream (unconfirmed cause)
niah_single_3 at 4x: perhead_bern 7% (Llama) / 0% (Qwen), vs 100% for no_press, kvzip and global bernoulli.
It finds the needle and copies the first few UUID characters correctly, then invents the rest
("715629ee-e" + made-up digits). E5 measured only 4 decode steps, which are exactly the tokens it
still gets right, so E5 could not see this. No code bug found in perhead_press.py; the likely cause
is that an equal per-head cap randomly drops needle tokens in the heads that copy them (cf. E5 fig9:
the per-head tail is worse at 4x). To confirm: record per-head keep masks over the needle span,
perhead vs global, for one sample.
Also interim: at matched budget, downstream accuracy does NOT follow E5's upstream-error ranking on
niah (kvzip, which had the worst attention error, retrieves at 100%). Read nothing final until all jobs finish.

## Checked in detail (2026-09-21, ~16:50): the per-head collapse is NOT a code error
1. Plumbing: forced to the shipped global c and seed, perhead_press.py gives a BIT-IDENTICAL attention bias to the
   shipped BernoulliPress (tiny model, diag/equivalence_test.py) AND identical generations on the real model
   (diag/needle_diag.py, 5/5 samples). So the only difference between the arms is the per-head budget itself.
2. Reproducibility: the diagnostic re-generates E6's saved answers exactly (same GPU type, same seed).
3. Mechanism (diag/needle_diag.json, Llama niah_single_3 at 4x, 5 E6 samples): ~80 of 1024 query heads carry
   >=20% of their attention on the needle while the UUID is written ("copy heads"). In those heads:
     shipped (global c) Bernoulli keeps 94-96% of the needle tokens  -> 5/5 correct
     per-head Bernoulli keeps 30-38%                                -> 0/5 correct (invents the rest of the UUID)
     per-head TOP-K (same equal budget, no sampling) keeps 31-40%   -> 1/5 correct (near-miss copies)
   Averaged over ALL heads, both keep ~30-38% of the needle; the difference is WHERE. The global budget
   concentrates it in the copy heads; an equal per-head cap cannot. Per-head top-k fails too, so the cause is
   the allocation, not the sampling.
4. What this means for E5: its per-head error aggregates (median, and even p95/p99 over all 1024 heads)
   are dominated by the ~92% non-copy heads, where per-head budgets excel. The few heads that decide retrieval are
   invisible to them. An upstream metric that is meant to predict retrieval must weight or single out those heads.
5. Other checks: arms built as intended; seeds s1/s2 independent; the E6 per-head file is identical to E5's;
   Llama cwe no_press lists English stopwords (the ceiling of 43 is the model's), so Llama cwe is a weak signal.

## SCBench: NOT submitted
Smoke 315759: no_press 0/5 and kvzip 0/5 on prefix_suffix. The scorer is correct: uncompressed Llama concatenates
the prefix and suffix instead of retrieving the dictionary word. A zero ceiling cannot separate arms. Needs a decision.

## Done (2026-09-22)
All 180 RULER runs complete and verified (180/180 metrics.json reproduced). Conclusion: CONCLUSION.md.
E5's CONCLUSION.md has an addendum withdrawing its per-head-budget recommendation. SCBench still undecided.

## SCBench done (2026-09-22)
Jobs 316089 (Llama, 5h29m) / 316090 (Qwen, 9h53m), 14 runs, all scored. prefix_suffix has no headroom:
ceiling 12 (Llama) and 0 (Qwen). Bernoulli 8 vs KVzip/per-head/VKvWR 0 on Llama, but that is 2 of 25
questions (sign p=0.5). See CONCLUSION.md section 8. Analysis: analyze_scb.py.
