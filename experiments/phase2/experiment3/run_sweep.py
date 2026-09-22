"""Phase 2, Experiment 3: resumable SCBench sweep at ~110k context, one model load per job.

    SWEEP=sweeps/probe.yaml sbatch job_sweep.sbatch

Differences from experiment2's runner:
  * dataset is SCBench, loaded from the local snapshot with NO truncation (scbench_data.py);
  * the number of contexts is limited by E3_N_CTX, not by kvpress's `fraction`, because `fraction`
    samples ROWS and would split a context's 5 turns across the subset, breaking the multi-query
    grouping that makes long contexts affordable;
  * peak GPU memory is recorded per run (A100 sizing);
  * score statistics for the Stage 2 scaling analysis are logged by E3BernPress.

Runs that already hold a metrics.json are skipped, so a requeued job resumes.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "experiment1"))
sys.path.insert(0, os.path.join(HERE, "..", "experiment2"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "src", "kvpress", "evaluation"))

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
ap.add_argument("--group", default=None, help="results subdir (default: sweep name before '_')")
args = ap.parse_args()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SWEEP = yaml.safe_load(open(os.path.join(HERE, args.sweep)))
NAME = Path(args.sweep).stem
GROUP = args.group or NAME.split("_")[0]
OUTDIR = os.path.join(HERE, "results", GROUP)
os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


# Contexts per task must be set BEFORE the dataset is loaded (the loader reads this).
N_CTX = int(SWEEP.get("n_contexts", 0))
if N_CTX:
    os.environ["E3_N_CTX"] = str(N_CTX)

import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402

import scbench_data  # noqa: E402
import scbench_metrics  # noqa: E402
from e3_press import CachedKVzipPress, E3BernPress, bias_patched  # noqa: E402
from selftests import self_tests  # noqa: E402

# ---- SCBench into kvpress's registries, and our loader over datasets.load_dataset (the E2 pattern:
# compute nodes have no internet and datasets 4.x cannot resolve a cached repo by name offline).
evaluate.load_dataset = scbench_data.load_scbench
evaluate_registry.DATASET_REGISTRY["scbench"] = "microsoft/SCBench"
evaluate_registry.SCORER_REGISTRY["scbench"] = scbench_metrics.calculate_metrics
evaluate.DATASET_REGISTRY["scbench"] = "microsoft/SCBench"
evaluate.SCORER_REGISTRY["scbench"] = scbench_metrics.calculate_metrics

# ---- arms. No no-bias ablation (the bias-corrected estimator is the mathematically correct one) and
# no full-context calibration arm: every arm uses kvpress's chunked KVzip calibration, so scoring is
# identical across the comparison and the only difference is top-k versus sampling.
for s in (1, 2, 3):
    evaluate_registry.PRESS_REGISTRY[f"kvzip_bern_s{s}"] = E3BernPress(method="kvzip_bern", seed=s)
# The top-k control reads the SAME cached score tensor as the sampling arm, so B1 differs in selection
# alone -- and calibration is paid once per context instead of once per run (see e3_press.py).
evaluate_registry.PRESS_REGISTRY["kvzip"] = CachedKVzipPress()

RUNS = [(r["press"], float(r["ratio"])) for r in SWEEP["runs"]]
TASKS = SWEEP["tasks"]
log(f"sweep {NAME}: {len(RUNS)} runs x {len(TASKS)} tasks {TASKS}, n_contexts={N_CTX or 'all'}, "
    f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, outdir {OUTDIR}")


def run_dirs(cfg):
    """kvpress's get_results_dir CREATES a numbered subfolder when the base exists, so it cannot be used
    to look up an existing run. Rebuild the base name without side effects."""
    comp = [cfg.dataset, str(cfg.data_dir), cfg.model.replace("/", "--"), cfg.press_name,
            f"{cfg.compression_ratio:.2f}"]
    if cfg.fraction < 1.0:
        comp.append(f"fraction{cfg.fraction:.3f}")
    if cfg.max_context_length is not None:
        comp.append(f"max_context{cfg.max_context_length}")
    base = Path(OUTDIR) / "__".join(comp)
    if not base.exists():
        return []
    return [base] + sorted((d for d in base.glob("[0-9]*") if d.is_dir()), key=lambda d: int(d.name))


pipe, tested = None, False
todo = [(t, p, r) for t in TASKS for p, r in RUNS]
for i, (task, press_name, ratio) in enumerate(todo):
    cfg = evaluate.EvaluationConfig(**{**asdict(evaluate.EvaluationConfig()),
                                       "dataset": "scbench", "data_dir": task, "model": MODEL,
                                       "press_name": press_name, "compression_ratio": ratio,
                                       "fraction": 1.0, "seed": 42, "output_dir": OUTDIR,
                                       "max_context_length": None})   # NEVER truncate
    evaluate.logger.handlers.clear()                       # each runner adds one; they accumulate
    runner = evaluate.EvaluationRunner(cfg)
    if any((d / "metrics.json").exists() for d in run_dirs(cfg)):
        log(f"{i + 1}/{len(todo)} {task} {press_name} cr={ratio:.3f}: done, skipping")
        continue
    if pipe is None:
        runner._setup_press()
        runner._setup_model_pipeline()
        pipe = runner.pipeline
        # one timestamped line per context, so a hung run shows up within minutes instead of hours
        _fwd = pipe._forward

        def _forward_logged(input_tensors, *a, **kw):
            t = time.time()
            out = _fwd(input_tensors, *a, **kw)
            log(f"   context n={input_tensors['context_ids'].shape[1]} done in {time.time() - t:.0f}s")
            return out
        pipe._forward = _forward_logged
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)
    if not tested:
        self_tests(pipe, log, OUTDIR)
        tested = True

    press = evaluate_registry.PRESS_REGISTRY[press_name]
    if isinstance(press, E3BernPress):
        press.stats_log.clear()
    log(f"{i + 1}/{len(todo)} {task} {press_name} cr={ratio:.3f}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with bias_patched(pipe.model):
        runner.run_evaluation()
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9

    if isinstance(press, E3BernPress) and press.stats_log:
        st = press.stats_log
        kf = sum(s["kept_frac"] for s in st) / len(st)
        log(f"   kept {kf:.4f} (target {1 - ratio:.4f}), T/n med "
            f"{sum(s.get('T_over_n_median', 0) for s in st) / len(st):.4f}, "
            f"heavy {sum(s.get('heavy_frac_median', 0) for s in st) / len(st):.4f}")
        for d in run_dirs(cfg):
            if (d / "metrics.json").exists():
                json.dump(st, open(d / "memory.json", "w"), indent=1)
    for d in run_dirs(cfg):
        if (d / "metrics.json").exists():
            json.dump({"seconds": dt, "peak_gpu_gb": peak, "n_contexts": N_CTX,
                       "gpu": torch.cuda.get_device_name(0)}, open(d / "cost.json", "w"), indent=1)
    log(f"   done in {dt:.0f}s, peak GPU {peak:.1f} GB")

log("all runs done")
