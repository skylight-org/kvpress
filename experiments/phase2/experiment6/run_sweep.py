"""Experiment 5: resumable sweep measuring per-head UPSTREAM error, one model load per job.

    SWEEP=sweeps/smoke.yaml sbatch -p a40 --qos=a40 job_sweep.sbatch

E6: downstream accuracy on E5's exact settings (tasks, models, arms, ratios, VKvWR ladder), so each
score pairs with an E5 error measurement. Head stats are off. (E5's docstring follows.) One row per
(context, question, decode step, layer, query head) holding the softmax denominator with and
without compression, the numerator error and the attention-ratio error. See
kvpress/metric_logging/head_stats.py on the e5-upstream-error branch.

Everything runs through the SHIPPED presses, so the arms are exactly what is on the branches:
  kvzip          stock KVzipPress, global top-k              (the baseline)
  bernoulli_*    BernoulliPress from main / PR #2            (our scheme)
  vkvwr_*        VKvWRPress from the professor's vkv1 branch (his scheme)

Run order is RATIO-FIRST (8x, 4x, 16x, 2x) across tasks, so a job that runs long still has the
headline ratio for every task. Runs holding a metrics.json are skipped, so a requeued job
resumes.

Yaml keys: dataset, group, tasks, model, n_samples/fraction/n_contexts, ratios, arms, seeds,
max_new_tokens, strides, max_context_length.

Invariants are checked after the fact by check_run.py against the parquet, rather than by a
pre-flight: every quantity they test (kept fraction, top-k never over-estimating, the Bernoulli
estimator being unbiased) is present in every run's own output.
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
sys.path.insert(0, os.path.join(HOME, "src", "kvpress-e5", "evaluation"))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
args = ap.parse_args()

SWEEP = yaml.safe_load(open(os.path.join(HERE, args.sweep)))
NAME = Path(args.sweep).stem
DATASET = SWEEP["dataset"]
MODEL = SWEEP.get("model", "meta-llama/Llama-3.1-8B-Instruct")
OUTDIR = os.path.join(HERE, "results", SWEEP["group"])
os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402

from kvpress import BernoulliPress, KVzipPress  # noqa: E402
from kvpress.presses.vkvwr_press import VKvWRPress  # noqa: E402

import perhead_press  # noqa: E402

REG = evaluate_registry.PRESS_REGISTRY
assert evaluate.PRESS_REGISTRY is REG
for s in SWEEP.get("seeds", [1, 2]):
    REG[f"bernoulli_kvzip_s{s}"] = BernoulliPress(press=KVzipPress(), seed=s)
    REG[f"vkvwr_kvzip_s{s}"] = VKvWRPress(press=KVzipPress(), seed=s)
    REG[f"perhead_bern_kvzip_s{s}"] = perhead_press.make(seed=s)
REG["kvzip"] = KVzipPress()

TASKS = SWEEP["tasks"]
RATIOS = [float(r) for r in SWEEP.get("ratios", [])]
ARM_RATIOS = SWEEP.get("arm_ratios", {})   # arm-name prefix -> its own ratio list
todo = []
for ratio in RATIOS:                     # ratio-first: the headline budget lands for every task
    for t in TASKS:
        for a in SWEEP["arms"]:
            rs = next((v for k, v in ARM_RATIOS.items() if a.startswith(k)), None)
            if rs is None:
                todo += [(t, a, ratio)]
            elif ratio == RATIOS[0]:      # arm has its own ladder: queue it all on the first pass
                todo += [(t, a, float(r)) for r in rs]

log(f"sweep {NAME}: {DATASET} {MODEL} {len(todo)} runs, tasks {TASKS}, "
    f"GPU {torch.cuda.get_device_name(0)}, outdir {OUTDIR}")


def run_dirs(cfg):
    """kvpress's get_results_dir CREATES a numbered subdir when the base exists, so it cannot be
    used to test whether a run finished. Rebuild the base name without side effects."""
    comp = [cfg.dataset, str(cfg.data_dir), cfg.model.replace("/", "--"), cfg.press_name,
            f"{cfg.compression_ratio:.2f}"]
    if cfg.n_samples is not None:
        comp.append(f"nsamples{cfg.n_samples}")
    elif cfg.fraction < 1.0:
        comp.append(f"fraction{cfg.fraction:.3f}")
    if cfg.max_context_length is not None:
        comp.append(f"max_context{cfg.max_context_length}")
    base = Path(OUTDIR) / "__".join(comp)
    if not base.exists():
        return []
    return [base] + sorted((d for d in base.glob("[0-9]*") if d.is_dir()), key=lambda d: int(d.name))


BASE = asdict(evaluate.EvaluationConfig())
pipe = None
for i, (task, press_name, ratio) in enumerate(todo):
    over = {
        "dataset": DATASET, "data_dir": task, "model": MODEL,
        "press_name": press_name, "compression_ratio": ratio, "seed": 42,
        "output_dir": OUTDIR,
        "max_new_tokens": SWEEP.get("max_new_tokens"),        # None = the task's own answer length
        "max_context_length": SWEEP.get("max_context_length"),
        "enable_head_stats": False,                           # E6 is downstream only
        "head_stats_layer_stride": SWEEP.get("layer_stride", 1),
        "head_stats_step_stride": SWEEP.get("step_stride", 1),
        "head_stats_question_stride": SWEEP.get("question_stride", 1),
        "head_stats_max_rows": SWEEP.get("max_rows"),
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    if SWEEP.get("n_samples"):
        over["n_samples"] = int(SWEEP["n_samples"])
    if SWEEP.get("fraction"):
        over["fraction"] = float(SWEEP["fraction"])
    cfg = evaluate.EvaluationConfig(**{**BASE, **over})

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
        log(f"model loaded: {MODEL} ({pipe.model.config.num_hidden_layers} layers, "
            f"{pipe.model.config.num_attention_heads}q/{pipe.model.config.num_key_value_heads}kv heads)")
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)
    log(f"{i + 1}/{len(todo)} {task} {press_name} cr={ratio:.4f}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner.run_evaluation()
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    dirs = run_dirs(cfg)
    dest = dirs[-1] if dirs else None
    if dest is not None:
        json.dump({"seconds": dt, "peak_gb": peak, "gpu": torch.cuda.get_device_name(0),
                   "press": press_name, "ratio": ratio, "task": task, "model": MODEL},
                  open(dest / "cost.json", "w"), indent=2)
    log(f"   done in {dt:.0f}s, peak {peak:.1f} GB -> {dest}")

log("sweep complete")
