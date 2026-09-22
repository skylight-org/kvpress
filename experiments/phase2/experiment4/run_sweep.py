"""Experiment 4: resumable sweep over LongBench or SCBench, one model load per job.

    SWEEP=sweeps/smoke.yaml sbatch -p l40 --qos=l40 job_sweep.sbatch

Generalises experiment3/run_sweep.py (left untouched):
  * the yaml picks the dataset (longbench | scbench), the fraction or context count, and the arms;
  * run order is no_press per task first, then RATIO-FIRST across tasks (8x, 4x, 16x, 2x), so a job
    that runs long still has the headline ratios for every task;
  * arms: kvzip (cached scores), kvzip_bern_s1-3, kvzip_bern_nobias_s1, compactor, compactor_bern_s1-3;
  * self-tests: E3's T1-T6, then T7-T9 for CompactorBernPress (selftests4.py). Both abort on failure.

Yaml keys: dataset, group, tasks, fraction (longbench), n_contexts (scbench), no_press (bool),
ratios, arms, extra (list of {press, ratio}), and optional max_samples (smoke only: first N rows).
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
HOME = os.path.expanduser("~")
for p in (os.path.join(HOME, "src", "kvpress", "evaluation"), os.path.join(HOME, "phase2", "experiment1"),
          os.path.join(HOME, "phase2", "experiment2"), os.path.join(HOME, "phase2", "experiment3"), HERE):
    sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
args = ap.parse_args()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SWEEP = yaml.safe_load(open(os.path.join(HERE, args.sweep)))
NAME = Path(args.sweep).stem
DATASET = SWEEP["dataset"]
OUTDIR = os.path.join(HERE, "results", SWEEP["group"])
os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


N_CTX = int(SWEEP.get("n_contexts", 0))
if N_CTX:
    os.environ["E3_N_CTX"] = str(N_CTX)          # read by scbench_data at load time

import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402

import data  # noqa: E402
import scbench_metrics  # noqa: E402
from compactor_bern import CompactorBernPress, SeededCompactorPress  # noqa: E402
from e3_press import CachedKVzipPress, E3BernPress, bias_patched  # noqa: E402
from selftests import self_tests  # noqa: E402
from selftests4 import compactor_tests  # noqa: E402

evaluate.load_dataset = data.load
for mod in (evaluate, evaluate_registry):
    mod.DATASET_REGISTRY["scbench"] = "microsoft/SCBench"
    mod.SCORER_REGISTRY["scbench"] = scbench_metrics.calculate_metrics

# ---- arms. PRESS_REGISTRY is shared by name between the two modules (same dict object).
REG = evaluate_registry.PRESS_REGISTRY
for s in (1, 2, 3):
    REG[f"kvzip_bern_s{s}"] = E3BernPress(method="kvzip_bern", seed=s)
    REG[f"compactor_bern_s{s}"] = CompactorBernPress(press=SeededCompactorPress(), seed=s)
REG["kvzip_bern_nobias_s1"] = E3BernPress(method="kvzip_bern_nobias", seed=1)
REG["kvzip"] = CachedKVzipPress()
REG["compactor"] = SeededCompactorPress()   # kvpress Compactor, sketch seeded per context
assert evaluate.PRESS_REGISTRY is REG

TASKS = SWEEP["tasks"]
todo = []
if SWEEP.get("no_press", True):
    todo += [(t, "no_press", 0.0) for t in TASKS]
for ratio in SWEEP.get("ratios", []):
    for t in TASKS:
        todo += [(t, a, float(ratio)) for a in SWEEP["arms"]]
        todo += [(t, e["press"], float(e["ratio"])) for e in SWEEP.get("extra", []) if float(e["ratio"]) == ratio]
log(f"sweep {NAME}: {DATASET} {len(todo)} runs, tasks {TASKS}, n_contexts={N_CTX or '-'}, "
    f"fraction={SWEEP.get('fraction', 1.0)}, GPU {torch.cuda.get_device_name(0)}, outdir {OUTDIR}")


def run_dirs(cfg):
    """kvpress's get_results_dir CREATES a numbered subfolder when the base exists, so it cannot be used
    to look up an existing run. Rebuild the base name without side effects (as in E3)."""
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


def press_stats(press):
    """Per-context kept fraction, averaged over layers (compactor_bern) or as logged (kvzip_bern)."""
    if isinstance(press, CompactorBernPress):
        out = []
        for rec in press.stats_log:
            L = rec["layers"]
            if L:
                out.append({"n": rec["n"], "kept_frac": sum(x["kept_frac"] for x in L) / len(L),
                            "expected_frac": sum(x["expected_frac"] for x in L) / len(L),
                            "target_frac": L[0]["target_frac"],
                            "c_median_by_layer": [x["c_median"] for x in L]})
        return out
    if isinstance(press, E3BernPress):
        return list(press.stats_log)
    return None


pipe, tested = None, False
MAX_SAMPLES = SWEEP.get("max_samples")
for i, (task, press_name, ratio) in enumerate(todo):
    cfg = evaluate.EvaluationConfig(**{**asdict(evaluate.EvaluationConfig()),
                                       "dataset": DATASET, "data_dir": task, "model": MODEL,
                                       "press_name": press_name, "compression_ratio": ratio,
                                       "fraction": float(SWEEP.get("fraction", 1.0)), "seed": 42,
                                       "output_dir": OUTDIR, "max_context_length": None})
    evaluate.logger.handlers.clear()                       # each runner adds one; they accumulate
    runner = evaluate.EvaluationRunner(cfg)
    if any((d / "metrics.json").exists() for d in run_dirs(cfg)):
        log(f"{i + 1}/{len(todo)} {task} {press_name} cr={ratio:.4f}: done, skipping")
        continue
    if pipe is None:
        runner._setup_press()
        runner._setup_model_pipeline()
        pipe = runner.pipeline
        assert pipe.model.config._attn_implementation == "sdpa", pipe.model.config._attn_implementation
        _fwd = pipe._forward

        def _forward_logged(input_tensors, *a, **kw):
            t = time.time()
            out = _fwd(input_tensors, *a, **kw)
            log(f"   context n={input_tensors['context_ids'].shape[1]} done in {time.time() - t:.0f}s")
            return out
        pipe._forward = _forward_logged
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)
    if MAX_SAMPLES:                                        # smoke only: keep the first N rows
        _prep = runner._load_and_prepare_dataset

        def _prep_head(r=runner, _p=_prep):
            _p()
            r.df = r.df.iloc[:MAX_SAMPLES].copy()
        runner._load_and_prepare_dataset = _prep_head
    if not tested and not SWEEP.get("skip_selftests"):
        self_tests(pipe, log, OUTDIR)
        compactor_tests(pipe, log, OUTDIR)
        tested = True

    press = REG[press_name]
    if hasattr(press, "stats_log"):
        press.stats_log.clear()
    for obj in (press, getattr(press, "press", None)):
        if isinstance(obj, SeededCompactorPress):
            obj._leverage_press.n_fallback = 0
    log(f"{i + 1}/{len(todo)} {task} {press_name} cr={ratio:.4f}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with bias_patched(pipe.model):
        runner.run_evaluation()
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9

    st = press_stats(press)
    inner = press.press if isinstance(press, CompactorBernPress) else press
    fallbacks = getattr(inner, "n_fallback", None)
    if fallbacks:
        log(f"   Compactor Cholesky fallback used on {fallbacks} layer calls")
    done = [d for d in run_dirs(cfg) if (d / "metrics.json").exists()]
    if st:
        kf = sum(s["kept_frac"] for s in st) / len(st)
        log(f"   kept {kf:.4f} (target {1 - ratio:.4f}) over {len(st)} contexts")
    for d in done:
        if st:
            json.dump(st, open(d / "memory.json", "w"), indent=1)
        json.dump({"seconds": dt, "peak_gpu_gb": peak, "n_contexts": N_CTX,
                   "gpu": torch.cuda.get_device_name(0),
                   "cholesky_fallbacks": fallbacks}, open(d / "cost.json", "w"), indent=1)
    log(f"   {json.load(open(done[-1] / 'metrics.json')) if done else 'NO metrics.json'}")
    log(f"   done in {dt:.0f}s, peak GPU {peak:.1f} GB")

log("all runs done")
