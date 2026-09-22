# KV cache compression experiments

Research code and results for a line of experiments on **KV cache compression by importance
sampling**: instead of keeping the top-k KV pairs, keep pair *i* with probability
`r_i = min(1, c * s_i)` and add a `log(1/r_i)` bias to the kept pairs, which makes the attention
numerator and denominator unbiased (Horvitz–Thompson) estimates of their full-cache values.

Six experiments (E1–E6) on **Llama-3.1-8B-Instruct** and **Qwen3-4B-Instruct-2507**, over RULER,
SCBench and LongBench. Everything here is committed: the drivers, the presses, the sweep configs,
the SLURM scripts, the pre-registered designs, the audits, the conclusions, and every run's
predictions and scores.

---

## 1. Where the work stands

| | question | answer |
|---|---|---|
| **E1** | Is the denominator-only Bernoulli bound tight enough to set budgets? | Correct and it transfers to real queries, but **10–14× too loose**. The looseness is the score, not the inequality. |
| **E2** | Does sampling beat top-k on task accuracy at 4k (RULER)? | **A tie**: +0.79 ± 0.51 at 8×, nothing reaching 2 SE. The gain is concentrated in aggregation (`cwe`: +13 at 8×, +72 at 16×). |
| **E3** | Does it hold at ~110k (SCBench)? | **Yes, decisively.** At 8× Bernoulli loses 0.4–5.6 points against the uncompressed model; KVzip top-k loses 6.4–20.0. |
| **E4** | Does it hold for two different scorers (KVzip and Compactor)? | **KVzip yes** (+4.9 ± 1.4 pooled at 8×, p = 0.0005). **Compactor weakly** (+2.5 ± 1.3, does not survive Holm). |
| **E5** | Do the sampled methods have lower *upstream* error (attention numerator/denominator) than top-k at matched memory? | **Yes, everywhere**: 43/43 cells, attention error 6–39× lower at 8×. Per-head budgeting looked best of all. |
| **E6** | Do those upstream wins show up as *downstream accuracy* on the same settings? | **Only for global Bernoulli** (+9.5 points vs KVzip at 8×). The two methods that equalise the budget across heads have the lowest error and the **worst** accuracy. |

### What to believe right now

1. **Global-`c` Bernoulli is the method that works.** It beats KVzip top-k both upstream (E5) and
   downstream (E6: +9.5 points at 8×, 73 samples won to 20 lost, pooled over 5 tasks × 2 models).
2. **The per-head budget recommendation from E5 is withdrawn.** E6 §3 shows it destroys retrieval
   (`niah_single_3`: 0–7% versus 100%), and proves it is not a bug: forced to a global `c` the same
   code is bit-identical to the shipped press. Retrieval depends on ~80 of 1,024 query heads that
   copy the answer; a global budget lets those heads keep 94–96% of the needle's tokens, an equal
   per-head cap only 30–38%.
3. **E5's upstream metric does not rank methods by accuracy** (E6 §5: Spearman +0.10 across 112
   cells). It is a median over all heads, and retrieval is decided by the ~8% that copy. Within a
   single method it does track compression level (+0.53 to +0.74). **Any upstream metric meant to
   predict retrieval has to be head-aware.**
4. **VKvWR** (the professor's method, `kvpress/presses/vkvwr_press.py`) has a **memory floor**: its
   `compression_ratio` sets no budget, so it cannot reach 16×, and at matched realised memory it
   loses to both KVzip and Bernoulli (E6 §4). Always place it at its *realised* kept fraction.
5. **SCBench `prefix_suffix` is nearly useless as a downstream task here**: the uncompressed
   ceiling is 12 (Llama) and 0 (Qwen). `vt` is saturated at ~100 for every method, and Llama's
   `cwe` ceiling is only 43. Pick tasks with headroom.

### Backlog (in priority order)

- **P0** Downstream runs mirroring the error settings (SCBench, RULER) — done for RULER in E6.
- **P0** Add `qa_1` to RULER — done in E6.
- **P0** Dynamic Bernoulli: re-solve the keep probabilities per decode step from the real query
  (as VKvWR re-solves its sample size). Add to both the micro-metrics and the downstream eval.
  **Keep the allocation global** — see §1.2 above.
- **P0** Dynamic Bernoulli + oracle top-k when the error exceeds ε. E6 gives this a concrete
  target: global Bernoulli's one failure is a top-scored needle token getting `r < 1` at 16× and
  being sampled away (E6 §2). A deterministic top slice with a sampled tail addresses exactly that.
  Note that forced keeps must get `r_i = 1` or the estimator stops being unbiased.
- **P1** One more kvpress baseline; **P1** LOFT-128k (musique, qampari); **P1/P2** kernel-level
  latency for KVzip and Bernoulli, then sparse attention.

---

## 2. Layout and setup

```
experiments/phase2/
  experiment1..6/    one directory per experiment (see §5)
  common/            vendored offline dataset loader used by experiment4
  environments/      conda environment exports
```

**The scripts use absolute paths** (`~/phase2/...`, `~/src/kvpress...`) exactly as they ran, so the
published results and the code that produced them match byte for byte. To run them, put the tree
where they expect it:

```bash
git clone -b compression-experiments https://github.com/skylight-org/kvpress.git ~/src/kvpress-e5
ln -s ~/src/kvpress-e5/experiments/phase2 ~/phase2        # the experiments expect ~/phase2
git clone https://github.com/NVIDIA/kvpress.git ~/src/kvpress && git -C ~/src/kvpress checkout 6d96555
```

### Environments

| env | used by | key versions | notes |
|---|---|---|---|
| `llm` | E1 | torch 2.11.0+cu128, transformers **5.16.1** | no kvpress; plain HF |
| `kvpress` | E2, E3, E4 | transformers 5.2.0, kvpress from `~/src/kvpress` @ **`6d96555`** (NVIDIA/kvpress) | each experiment adds its own press file |
| `kvpress-pr` | E5, E6 | transformers 5.2.0, **this branch** via `PYTHONPATH` | the only env with flash-attn |

```bash
conda env create -f environments/kvpress-pr.yml
export PYTHONPATH="$HOME/src/kvpress-e5"      # load-bearing for E5/E6: see §6
```

### Data

Compute nodes have **no DNS**, so every dataset must be fetched **on the login node** first, then
used with `HF_HUB_OFFLINE=1`:

```bash
python -c "from datasets import load_dataset as l; [l('xAlg-AI/att-hub-ruler-32k', t, split=t) for t in ['qa_1','niah_single_3','niah_multikey_2','vt','cwe']]"
python -c "from huggingface_hub import snapshot_download as s; s('microsoft/SCBench', repo_type='dataset'); s('Xnhyacinth/LongBench', repo_type='dataset')"
python -c "from datasets import load_dataset as l; l('simonjegou/ruler', '4096')"
```

Models: `meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`.

### Cluster (SLURM)

QOS caps bind on **GPUs**, not jobs: `a40` = 2, `l40` = 4, `dgx` = 4 (A100-80GB); association
`MaxJobs` = 12, `MaxSubmit` = 20, and **the dgx QOS accepts only 5 submitted jobs per user** — the
6th is rejected outright, so long sweeps need resubmission as slots free.

```bash
cd ~/phase2/experiment6
SWEEP=sweeps/r32k_llama_qa_1.yaml sbatch -p l40 --qos=l40 --export=ALL,SWEEP=sweeps/r32k_llama_qa_1.yaml job_sweep.sbatch
```

Rough costs: RULER 32k ≈ 20–25 s/sample (Llama, L40S) and 42–120 s/sample (Qwen, A40); SCBench at
110k needs an 80 GB card (44 GB peak) and ≈ 15–20 min per run.

---

## 3. Reproducing a result without a GPU

Every run's `metrics.json`, `config.yaml` and `predictions.csv` is committed, so all tables and
figures regenerate on CPU:

```bash
cd ~/phase2/experiment6 && python analyze.py       # E6 tables + figures, from results/
cd ~/phase2/experiment5 && python summarize.py     # E5 tables (needs the per-head parquet: see §4)
cd ~/phase2/experiment4 && python summarize.py
```

`analyze.py` recomputes every per-sample score with the benchmark's own scorer and asserts it
reproduces the committed `metrics.json` (180/180 for E6's RULER runs).

---

## 4. What is **not** committed

Large intermediates, all regenerable by re-running the sweeps:

- `experiment5/out/all_rows.parquet` (405 MB) and every run's `head_stats.parquet` — E5's per-head
  rows, 7.2M of them. **`experiment5/summarize.py` and `figures.py` need these**; re-run the E5
  sweeps to rebuild, or use the committed `out/*.csv` and `out/REPORT.md`, which already contain
  the aggregates.
- `experiment2/out/calib_cache_*` (5.7 GB) and E3's caches (3.2 GB).
- SLURM logs, `__pycache__`, model weights.

---

## 5. The experiments

Each directory holds `DESIGN.md` (pre-registered, written before any run), `CONCLUSION.md` (the
answer), usually `STATUS.md` (timeline and bugs) and `AUDIT.md` (post-hoc checks), plus `out/` with
the generated report and figures.

| dir | what it does | run | analyse |
|---|---|---|---|
| `experiment1` | Is the denominator bound vacuous? Certificates + Monte Carlo on 4 LongBench contexts. | `sbatch job_exp1.sbatch` | `python report.py` |
| `experiment2` | RULER 4k accuracy sweep, sampling vs top-k vs query-agnostic baselines. | `SWEEP=sweeps/N_L1.yaml sbatch job_sweep.sbatch` | `python summarize.py` |
| `experiment3` | SCBench at ~110k, untruncated, contexts capped by `E3_N_CTX`. | `SWEEP=sweeps/S1_prefix_suffix.yaml sbatch job_sweep.sbatch` | `python summarize.py`, `plots.py` |
| `experiment4` | LongBench + SCBench, two scorers (KVzip and Compactor). | `SWEEP=sweeps/sc_ps.yaml sbatch job_sweep.sbatch` | `python summarize.py`, `plots.py` |
| `experiment5` | **Upstream** error per query head (denominator, numerator, attention ratio) at matched memory. | `SWEEP=sweeps/r32k_llama_cwe.yaml sbatch job_sweep.sbatch` | `check_run.py`, `summarize.py`, `figures.py` |
| `experiment6` | **Downstream** accuracy on E5's exact settings, plus RULER `qa_1` and a `no_press` ceiling. | `SWEEP=sweeps/r32k_llama_qa_1.yaml sbatch job_sweep.sbatch` | `analyze.py`, `analyze_scb.py` |

`experiment6/diag/` holds the two diagnostics that decided the per-head question:
`equivalence_test.py` (the per-head press is bit-identical to the shipped one under a global `c`)
and `needle_diag.py` (which heads copy the needle, and which of its tokens each method keeps).

The experiments import each other: E2 imports E1, E3 imports E1 and E2, E4 imports E3. That is why
the `~/phase2` symlink matters.

---

## 6. Traps that cost time

- **`PYTHONPATH`.** The `kvpress-pr` env has a *different* kvpress editable-installed. Without
  `export PYTHONPATH="$HOME/src/kvpress-e5"`, E5/E6 silently import the wrong library.
- **`LD_LIBRARY_PATH="$CONDA_PREFIX/lib"`** is load-bearing: SCBench's scorer imports nltk →
  sqlite3, which needs `CXXABI_1.3.15`.
- **The fake-key measurement bug.** `attention_patch` fakes head-wise pruning by writing fake keys
  into `key`, which during generation *is* the KV cache's own storage, so the write persists. Any
  press that drops through `masked_key_indices` (i.e. **KVzip**) therefore had a corrupted dense
  reference from the second forward onward, and `attention_output_error` stopped measuring
  anything. Fixed in **PR #5** (`fix-attention-output-error`), with a test that fails without it.
  Earlier KVzip-vs-VKvWR tables produced before that fix understate KVzip's error.
- **VKvWR's `compression_ratio` sets no memory budget** — only `topk_frac` and the pilot. Its
  nominal 8× actually keeps ~0.20. Compare it at its *realised* kept fraction, never nominally.
- **GPU type changes the random draws.** Run every arm of one (model, task, ratio) cell on the same
  GPU type and log it.
- **`scaling` must come from kwargs or the tensor, never `hidden_size // num_heads`** — for Qwen3
  that gives 80 instead of 128 and silently rescales every logit.
- **Boolean SDPA masks**: adding a float bias promotes `True/False` to `1.0/0.0` and destroys causal
  masking. Read the mask, never write it.
- **`get_results_dir` creates a numbered subdirectory** when the base exists — never call it to
  test whether a run finished; rebuild the name.
- **New presses go at the end of `tests/default_presses.py`** — tests are named by list index.

---

## 7. Branches

| branch | what |
|---|---|
| `main` | the fork's baseline |
| `vkv`, `vkv1` | the professor's work: `VKvWRPress`, micro-metric logging, the `ruler32k` benchmark |
| `bernoulli-press` | `BernoulliPress` (merged as PR #2) |
| `scbench-benchmark` | the SCBench benchmark (**PR #3, open, has conflicts with `vkv1`**) |
| `fix-attention-output-error` | the measurement fix above (**PR #5, open**) |
| `compression-experiments` | **this branch**: `vkv1` + `scbench-benchmark` + the per-head upstream metric + all six experiments |
