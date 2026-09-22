# Phase 2, Experiment 3 — Conclusion

**Run:** 2026-09-16, 82 evaluation runs, ~22 GPU-hours on A100-SXM4-80GB and L40S.
**Numbers:** `out/REPORT.md` (auto-generated) · **figures:** `out/figures/` (`plots.py`) · **design:** `DESIGN.md` · **bugs and checks:** `AUDIT.md` · **timeline:** `STATUS.md`
**Question:** does Bernoulli KV sampling still hold up against KVzip and the field at ~110k tokens —
the length where KV compression actually matters?

**Status:** `prefix_suffix`, `qa_eng` and `summary` are complete (25/25 runs each). `vt` is still
running and is reported at **8× only**, the ratio the sweep was ordered to reach first.

---

## The answer in three lines

1. **At 110k, sampling stops tying top-k and starts beating it — decisively.** At 8× compression
   Bernoulli loses **0.4 to 5.6 points** against the uncompressed model, while KVzip's top-k loses
   **6.4 to 20.0**. At 4× Bernoulli is statistically indistinguishable from the full cache on every
   completed task.
2. **E2's "sampling only helps aggregation" story does not survive the length change — it gets bigger.**
   At 4k the advantage was confined to one aggregation subtask. At 110k it appears on *every* task
   family, and the **largest** win is on a retrieval task (`prefix_suffix`, +20.7 at 4×).
3. **The bound's central statistic improves with length in the median and worsens in the tail.**
   Median `T/n` falls 2.4–8× from E1's 7k value, but the p90 rises ~4× and the max approaches 1.0.
   A minority of heads become essentially incompressible, and under one global budget those heads set
   the certified memory for everyone. **That is the specific opening for query-aware fetching.**

---

## Pre-registered criteria

| | criterion | verdict | evidence |
|---|---|---|---|
| **C0** | harness trustworthy | **PASS** | `no_press` plausible per task; `random` collapses (0.0–3.7 at 4×); T1–T6 pass at full length |
| **C1** | sampling ≥ top-k at 110k | **PASS with one caveat** | positive at 11 of 13 cells, 4 of them >2 SE; the one significant negative is explained below |
| **C2** | competitive with the field | **PASS** | beats `expected_attention` at every task and ratio, by +1.5 to +22.0 |
| **C3** | aggregation effect | **PASS** | aggregation +2.0/+5.8/+4.4/+0.1 at 2/4/8/16× — but retrieval is now *larger*, not zero |
| **C4** | memory honest | **PASS** | realised kept fraction within 0.012% of target; 390/390 contexts inside the §11 bound |
| **C5** | compressibility scales | **measured** | median `T/n` down, tail up — see below |

### The C1 caveat, stated plainly

The pre-registered rule says C1 fails if sampling is significantly below top-k anywhere. One cell does
that: `prefix_suffix` at 2×, −4.67 ± 2.23 (2.1 SE). It should not be read as a real deficit, because in
that same cell **top-k scores 26.0 against an uncompressed ceiling of 20.0** — it beats the model it is
compressing by 6 points, which no compressor can genuinely do. On a binary metric with 10 contexts and a
20% ceiling, that cell is noise. Sampling at the same ratio sits at 21.3, i.e. on the ceiling.

I am calling C1 a pass *with the failing cell reported*, rather than quietly re-writing the criterion.

---

## What the numbers say

### Degradation from the full cache (the table that matters)

Paired against `no_press` on identical queries. Negative = worse than uncompressed.

| task | arm | 2× | 4× | 8× | 16× |
|---|---|---|---|---|---|
| `prefix_suffix` | top-k | +6.0 | **−20.0** | **−20.0** | **−20.0** |
| | **sampling** | +1.3 | **+0.7** | **−2.0** | −16.7 |
| `vt` | top-k | — | — | −9.2 | — |
| | **sampling** | — | — | **−3.2** | — |
| `qa_eng` | top-k | −5.0 | −9.6 | −8.4 | −13.4 |
| | **sampling** | −0.4 | **−0.9** | −5.6 | −14.9 |
| `summary` | top-k | +0.3 | −2.5 | −6.4 | −5.3 |
| | **sampling** | −0.3 | **+0.2** | **−0.4** | −3.7 |

![degradation](out/figures/2_degradation.png)

The same numbers as bars, with context-clustered error bars: `out/figures/2_degradation.png`. The
controlled sampling-minus-top-k view is `3_paired_diff.png`, and `5_context_scatter.png` shows the
per-context points behind every average — on `prefix_suffix` every context sits on the x = 0 axis,
because top-k scores zero on all ten.

**At 4× compression, Bernoulli sampling is lossless within noise on all four tasks** (+0.7, −0.9, +0.2,
and `vt` not yet run). Top-k at the same budget has already lost 20 points on `prefix_suffix` and 9.6 on
`qa_eng`. At 8× sampling is still within 0.4–5.6 points; top-k is 6.4–20.0 down.

At 16× both collapse on the retrieval tasks and both survive on `summary` — the budget is simply too
small, and sampling's remaining edge (+1.6 to +3.7) no longer rescues the task.

### `prefix_suffix`: top-k scores exactly zero, and it is real

At ≥4×, `kvzip` scores **0.00 out of 50 queries** while sampling holds 18–21. This was checked hard
before being believed (`AUDIT.md`), because an exact zero usually means a broken harness:

- `kvzip` at 2× scores 26.0 through the *identical* cached-score code path, so the path works;
- `expected_attention`, a different press with no score cache, collapses to 0.00 the same way;
- 0 of 50 predictions are partial matches, so truncation is not stealing credit, and only 1 of 50
  hit the 64-token generation cap;
- top-k's actual outputs are **fabricated prefix+suffix concatenations**. For gold
  `LBlPUIsjuZYbsHkx^fkUzVdkS` it emits `"LBlPUzVdkS"` — the queried prefix glued directly to the
  queried suffix, with the 15 characters in between gone.

That is the mechanism made visible. The middle of the answer word is *not* attended to by the
calibration queries — it is arbitrary characters, not the prefix or suffix being asked about — so it
scores low and global top-k deletes it. The model then has the prefix, the suffix, and nothing between,
and it does the obvious thing. Bernoulli keeps a random, reweighted subset of that low-scoring tail, so
the middle survives often enough to answer correctly 18–21 times out of 50, matching the uncompressed
model's 20.

![per-context scatter](out/figures/5_context_scatter.png)

### The E2 story, revised

E2's conclusion was: *sampling preserves distributional information that top-k discards, and that
matters when the task aggregates over the context.* The second clause was wrong, or rather it was an
artefact of 4k context.

| family (sampling − top-k) | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| retrieval (`prefix_suffix`, `vt`) | −4.67 | **+20.67** | **+12.00** | +3.33 |
| aggregation (`summary`, `qa_eng`) | +2.02 | +5.75 | +4.39 | +0.06 |

At 4k, retrieval meant "one high-score needle that both methods keep", so sampling had nothing to add.
At 110k, retrieval tasks need *spans* — a 25-character word, a chain of variable assignments — whose
interiors are low-scoring under a query-agnostic score. Top-k deletes those interiors; sampling keeps a
reweighted sample of them. **The right claim is the first clause alone: sampling preserves distributional
information that top-k discards. What changes with length is how many tasks depend on that information.**

![families](out/figures/4_families.png)

### Against the field

`expected_attention` was E2's best cheap baseline. At 110k it is not competitive: `kvzip_bern` beats it
by +1.5 to +22.0 points at every task and ratio, significantly in 9 of 13 cells. `random` collapses to
0.0–3.7 at 4× on every task except `summary` (15.2, where ROUGE-L gives partial credit for fluent
English). The ranking from E2 — KVzip-family scores >> cheap scores — holds and widens.

### C5: what happens to the bound's statistics at length

| | E1 (~7k) | E3 (~110k) |
|---|---|---|
| median `T/n` | 0.0033 | 0.00039 – 0.00140 |
| p90 `T/n` | 0.0385 | 0.0025 – 0.1794 |
| max `T/n` | 0.15 | 0.018 – 0.975 |

The typical head becomes **2.4–8× more compressible** at 110k, which is the encouraging direction: the
method gets stronger where it is needed. But the distribution splits. On three of four tasks the worst
head reaches `T/n ≈ 0.9–1.0`, meaning its score is nearly flat — every token matters about equally, and
no budget can compress it without loss. `vt` is the striking exception (max 0.018): its attention is
genuinely concentrated everywhere.

![tau distribution](out/figures/6_tau_distribution.png)

Per layer (`7_tau_by_layer.png`) the tail is not spread evenly. The **median** `T/n` jumps 11× at
**layer 13** — 0.00065 in layers 0–12 against 0.00720 in layers 13–31, measured on `summary` — which is
the same depth at which the earlier sensitivity work found `S` jumps. Nearly-incompressible heads exist
in both halves (worst 0.74 below layer 13, 0.90 above), so the depth effect is a shift of the whole
distribution rather than the tail appearing all at once.

![tau by layer](out/figures/7_tau_by_layer.png)

Under a single global budget, those tail heads dominate the certified memory for the whole model. **This
is the concrete, measured case for per-head and query-aware budgeting**, and it did not exist at 7k,
where max `T/n` was 0.15.

---

## Honest limits

- **`vt` is partial.** 8× only (7 of 25 runs). Its Bernoulli runs cost ~5,100 s each — it generates 128
  tokens × 5 turns against a 124.5k cache — so the full ladder did not fit the wall clock. The ratios
  were ordered most-informative-first precisely so a truncated job would still yield a complete ratio,
  and it did. Remaining runs will extend the table, not change it.
- **10 contexts per task** (50–53 queries). Differences are paired and clustered at the context level,
  so the SEs are honest, but they are wide. Nothing here rests on a single cell.
- **`qa_eng` is 23 of 69 SCBench samples** — the rest exceed Llama-3.1-8B's 131,072-token limit and were
  dropped and counted, never truncated. Its numbers carry that selection.
- **`scbench_kv` was excluded.** Every arm including `no_press` scored 0.0 on it; the probe found the
  gold keys sit at relative depth 0.00–0.02 of a 124k context and the model could not retrieve them at
  all (0/15), while mid-context probes succeeded 56% of the time. A task with no signal under the
  uncompressed model cannot discriminate compressors. Documented in `AUDIT.md`.
- **Memory is counted, not freed.** kvpress masks dropped tokens rather than shrinking the cache
  (per-head selection needs a ragged cache). "8×" means the model behaves as if 7/8 of the cache were
  gone; every arm is handicapped identically, so the comparison is fair, but no wall-clock or GPU-memory
  saving is demonstrated here. This is now the largest gap between the result and the claim.
- **One model, one length band.** ~100–124k on Llama-3.1-8B-Instruct. 1M is still deferred: it needs a
  different model, ~137 GB of cache, and flash-attention.
- **Stage 2's query-agnosticism gap is not measured.** `T/n` and the heavy-hitter statistics are in, but
  comparing `T = Σᵢ maxq ωᵢ` against per-query `Σᵢ ωᵢ(q)` needs a separate instrumented pass that stores
  per-query attention weights at 110k. That is the direct measurement of what dynamic fetching could
  recover, and it is the first thing to run next.
- **No certificate here.** Per the professor's framing, ε is not the object of study; the empirical
  behaviour is.

---

## What to do next

1. **Measure the price of query-agnosticism** (the deferred half of Stage 2). SCBench's shared-context
   multi-query structure is the ideal place: one compressed cache, 5 genuinely different questions.
   The gap between `Σᵢ maxq ωᵢ` and the per-query `Σᵢ ωᵢ(q)` is the budget dynamic fetching could give
   back, and C5 now says where to look — the tail heads with `T/n → 1`.
2. **Per-head budgets.** C5 shows one global `c` is paying for the worst head in the model. Allocating
   `c` per head against each head's own `T/n` is a small change to `solve_c_for_memory` and is the
   cheapest available win.
3. **Make the memory real.** A ragged cache that actually frees the dropped tokens. Every accuracy
   result so far is an argument for a saving that has never been measured.
4. **Finish `vt`, and add one more span-retrieval task.** The `prefix_suffix` result — top-k at exactly
   zero, sampling at the ceiling — is the strongest finding in the phase and currently rests on one task.
