"""Phase 2, Experiment 2: resumable RULER sweep, one model load per job.

    TASKS=P1 python run_sweep.py --sweep sweeps/P1.yaml [--smoke]

Registers our presses into kvpress's PRESS_REGISTRY, then runs every (press, ratio) in the sweep file
against RULER. One model is loaded and reused for every run (the ~4 min cold-import + weight load is
paid once). Runs already holding a metrics.json are skipped, so a requeued job resumes.

Self-tests T1-T5 run once, before the first real run, and abort the job on failure (DESIGN.md 8).
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
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "src", "kvpress", "evaluation"))

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--group", default=None, help="override results subdir (e.g. separate hardware)")
args = ap.parse_args()

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SWEEP = yaml.safe_load(open(os.path.join(HERE, args.sweep)))
NAME = Path(args.sweep).stem
# Shards of one priority (P1_2x, P1_4x, P1_8x) share a results dir: run directories already encode
# press and ratio, so the shards cannot collide, and summarize.py sees them as one sweep. This also
# makes a requeued shard skip work another shard already finished.
GROUP = args.group or (NAME.split("_")[0] if not args.smoke else "smoke")
OUTDIR = os.path.join(HERE, "results", GROUP)
os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402

from bern_press import KVzipBernPress, bias_patched  # noqa: E402


def _load_local_parquet(path, data_dir=None, split="test", **kw):
    """Offline replacement for datasets.load_dataset (the E8 trap, hit again here).

    Compute nodes have no internet, and datasets 4.x cannot resolve a cached repo by name in offline
    mode -- load_dataset('simonjegou/ruler', '4096') raises OfflineModeIsEnabled even though the
    snapshot is on disk. So read the parquet straight out of the hub snapshot downloaded on the login
    node. For RULER the config ('4096') is the subdirectory."""
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    try:
        root = snapshot_download(path, repo_type="dataset", local_files_only=True)
    except Exception:
        # A partial download (allow_patterns) leaves the snapshot "incomplete" and snapshot_download
        # refuses it offline even when the parquet we need is present. Locate the snapshot directly.
        cache = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                             "hub", "datasets--" + path.replace("/", "--"), "snapshots")
        snaps = sorted(Path(cache).iterdir(), key=lambda d: d.stat().st_mtime)
        assert snaps, f"no cached snapshot for {path} under {cache}"
        root = str(snaps[-1])
    sub = os.path.join(root, str(data_dir)) if data_dir else root
    files = sorted(f for f in os.listdir(sub) if f.startswith(split) and f.endswith(".parquet"))
    assert files, f"no {split}*.parquet in {sub}"
    return Dataset.from_parquet([os.path.join(sub, f) for f in files])


evaluate.load_dataset = _load_local_parquet

# ---- register our arms. kvzip_bern* inherit KVzip's scoring and override selection only.
for m in ("kvzip_bern", "kvzip_bern_nobias"):
    for s in (1, 2, 3):
        evaluate_registry.PRESS_REGISTRY[f"{m}_s{s}"] = KVzipBernPress(method=m, seed=s)

# ---- P3: bern_tau = E9's verified BernPress(method="bern_tau"): full-context repeat calibration (no
# 2048-token chunks), tau score, Bernoulli r = min(1, c tau), log(1/r) bias, sinks forced kept.
# Loaded under a DISTINCT module name: e9/bern_press.py has the same filename as ours, so a plain
# import would silently return this experiment's module. Its calibration cache is redirected here so
# RULER contexts never land in E9's directory.
import importlib.util  # noqa: E402
from contextlib import contextmanager, nullcontext  # noqa: E402

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF  # noqa: E402

# E9's module patches EVERY attention implementation at import. The valid P1 controls (kvzip, no_press,
# random) ran without that patch, so it must not leak into other arms: snapshot the registry, import,
# capture E9's wrapped versions, restore the originals, and install E9's patch only around bern_tau runs.
_orig_attn = {k: _AAF[k] for k in list(_AAF.keys())}
_spec = importlib.util.spec_from_file_location(
    "e9_bern_press", os.path.join(os.path.expanduser("~"), "e9", "bern_press.py"))
e9bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e9bp)
_e9_attn = {k: _AAF[k] for k in list(_AAF.keys())}
for _k, _f in _orig_attn.items():
    _AAF[_k] = _f
e9bp.CACHE_DIR = os.path.join(HERE, "out", "calib_cache_bern_tau")
# Use EVERY repeat position as a calibration query, as KVzip does. E9 subsampled to 512 evenly spaced
# positions (~1 in 7 on RULER 4k), which misses the single query position that re-reads each needle:
# the all-arms smoke gave bern_tau 73 vs kvzip_bern 95, with the loss concentrated in niah subtasks
# (niah_single_3: 0 and 29) while qa/cwe were untouched. Left at 512, B6 would measure "all positions vs
# 1-in-7 positions" instead of "full-context vs chunked". E9's cache filename includes N_CAL, so the
# 512-position statistics can never be reused by mistake.
e9bp.N_CAL = 1_000_000
for s in (1, 2, 3):
    evaluate_registry.PRESS_REGISTRY[f"bern_tau_s{s}"] = e9bp.BernPress(method="bern_tau", seed=s)
BERN_TYPES = (KVzipBernPress, e9bp.BernPress)


@contextmanager
def e9_patched():
    """Install E9's bias kernel for the duration of a bern_tau run, then restore the originals."""
    saved = {k: _AAF[k] for k in list(_AAF.keys())}
    for k, f in _e9_attn.items():
        _AAF[k] = f
    try:
        yield
    finally:
        for k, f in saved.items():
            _AAF[k] = f
        for L in (getattr(e9_patched, "model", None).model.layers if getattr(e9_patched, "model", None) else []):
            L.self_attn.e9_bias = None

RUNS = [(r["press"], float(r["ratio"])) for r in SWEEP["runs"]]
CONFIG = SWEEP.get("config", "4096")
SAMPLES = 4 if args.smoke else SWEEP.get("samples_per_subtask", 60)
log(f"sweep {NAME}: {len(RUNS)} runs, RULER {CONFIG}, {SAMPLES} samples/subtask, GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, outdir {OUTDIR}")


# ---------------------------------------------------------------- self-tests (DESIGN 8)
def self_tests(pipe):
    """T1-T5. Aborts the job on failure: a broken bias kernel or a score mismatch makes B1 meaningless."""
    model = pipe.model
    dev = model.device
    from transformers import DynamicCache
    tok = pipe.tokenizer
    ids = tok("The capital of France is Paris. " * 200, return_tensors="pt").input_ids.to(dev)
    qids = tok(" What is the capital?", return_tensors="pt").input_ids.to(dev)
    n = ids.shape[1]
    nkv = model.config.num_key_value_heads

    def logits_with(set_fn):
        cache = DynamicCache()
        with torch.no_grad(), bias_patched(model):
            model.model(input_ids=ids, past_key_values=cache)
            set_fn(cache)
            out = model(input_ids=qids, past_key_values=cache,
                        position_ids=torch.arange(n, n + qids.shape[1], device=dev)[None]).logits[0].double()
        for L in model.model.layers:
            L.self_attn.e2_bias = None
            L.self_attn.masked_key_indices = None
        return out

    base = logits_with(lambda c: None)
    scale = base.abs().max()

    # T1: bias of all zeros (every r = 1) must reproduce no_press
    zeros = logits_with(lambda c: [setattr(L.self_attn, "e2_bias", torch.zeros(1, nkv, n, device=dev))
                                   for L in model.model.layers])
    t1 = ((zeros - base).abs().max() / scale).item()

    # T2/T3: -inf bias must equal kvpress's fake-key masking on the same dropped set
    g = torch.Generator().manual_seed(1)
    gmask = torch.rand(len(model.model.layers), nkv, n, generator=g) < 0.3
    gmask[..., :4] = False                                    # never drop sinks

    def set_bias(c):
        for l, L in enumerate(model.model.layers):
            L.self_attn.e2_bias = torch.where(gmask[l], -float("inf"), 0.0)[None].to(dev)

    def set_fake(c):
        for l, L in enumerate(model.model.layers):
            hi, si = gmask[l].nonzero(as_tuple=True)
            L.self_attn.masked_key_indices = (torch.zeros_like(hi), hi, si)

    t3 = ((logits_with(set_bias) - logits_with(set_fake)).abs().max() / scale).item()

    # T4a: the patch must actually be live -- heavy masking must move the logits
    heavy = logits_with(lambda c: [setattr(L.self_attn, "e2_bias",
                                           torch.where(torch.rand(1, nkv, n, device=dev) < 0.9, -float("inf"), 0.0))
                                   for L in model.model.layers])
    t4 = ((heavy - base).abs().max() / scale).item()

    # T4b: sinks kept and budget exact, at every ratio (AUDIT.md #3)
    import denom
    sink_ok, mem_err = True, 0.0
    sc = torch.rand(32, 1, 8, 1024, dtype=torch.float64) * 0.9
    sc[..., :4] = 1.0
    flat = sc.reshape(-1)
    for ratio in (0.5, 0.75, 0.875):
        budget = (1.0 - ratio) * flat.numel()
        c = denom.solve_c_for_memory(flat, budget)
        r = denom.keep_prob(flat, c).view(32, 1, 8, 1024)
        r[..., :4] = 1.0
        if r.sum().item() > budget:                            # re-solve, as the press does
            lo, hi = 0.0, c
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                rr = denom.keep_prob(flat, mid).view(32, 1, 8, 1024)
                rr[..., :4] = 1.0
                hi, lo = (mid, lo) if rr.sum().item() > budget else (hi, mid)
            r = denom.keep_prob(flat, hi).view(32, 1, 8, 1024)
            r[..., :4] = 1.0
        sink_ok &= bool((r[..., :4] == 1.0).all())
        mem_err = max(mem_err, abs(r.sum().item() - budget) / budget)

    # T6 (END-TO-END, through the real kvpress pipeline): is the bias ATTACHED during generation, and
    # does it block the intended fraction of the context?
    #   History: T1-T4 all passed while the bias was being cleared by the press context BEFORE
    #   generation (kvpress wraps only the prefill), so all nine P1 Bernoulli runs decoded with the full
    #   cache and scored exactly no_press. A first T6 asked "does the answer change at 99% compression?"
    #   -- but KVzip's score keeps the needle even at 1% of the cache (kvzip itself gets 3/3 there,
    #   debug/dbg_e2e.py), so that test fails on a CORRECT press. The right question is whether the
    #   kernel sees the bias during question prefill and decode, which is what this counts.
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    impl = model.config._attn_implementation
    seen = {"gen_calls_with_bias": 0, "prefill_calls_with_bias": 0, "blocked": []}

    def counter(func):
        def w(module, q, k, v, attention_mask, dropout=0.0, **kw):
            b = getattr(module, "e2_bias", None)
            if b is not None:
                if q.shape[2] == k.shape[2]:
                    seen["prefill_calls_with_bias"] += 1
                else:
                    seen["gen_calls_with_bias"] += 1
                    seen["blocked"].append(torch.isinf(b).float().mean().item())
            return func(module, q, k, v, attention_mask, dropout, **kw)
        return w

    try:
        ctx = " ".join(f"The code for item {i} is {(i * 7919) % 99991}." for i in range(400))
        p8 = KVzipBernPress(method="kvzip_bern", seed=1, compression_ratio=0.875)
        with bias_patched(model):
            inner = ALL_ATTENTION_FUNCTIONS[impl]
            ALL_ATTENTION_FUNCTIONS[impl] = counter(inner)
            try:
                ans = pipe(ctx, question="What is the code for item 57?", press=p8, max_new_tokens=8)["answer"]
            finally:
                ALL_ATTENTION_FUNCTIONS[impl] = inner
        blocked = sum(seen["blocked"]) / len(seen["blocked"]) if seen["blocked"] else 0.0
        t6_ok = (seen["gen_calls_with_bias"] > 0 and seen["prefill_calls_with_bias"] == 0
                 and abs(blocked - 0.875) < 0.02)
        res_t6 = {"T6_gen_calls_with_bias": seen["gen_calls_with_bias"],
                  "T6_prefill_calls_with_bias": seen["prefill_calls_with_bias"],
                  "T6_blocked_frac": blocked, "T6_answer": ans[:30]}
        for L in model.model.layers:
            L.self_attn.e2_bias = None
    except Exception as e:                                                 # noqa: BLE001
        t6_ok, res_t6 = False, {"T6_error": repr(e)[:300]}

    res = {"T1_keepall_rel": t1, "T3_bias_vs_fakekeys_rel": t3, "T4a_heavy_mask_changes_rel": t4,
           "T4b_sinks_kept": sink_ok, "T4b_budget_rel_err": mem_err, **res_t6}
    ok = (t1 < 2e-2 and t3 < 5e-2 and t4 > 1e-2 and sink_ok and mem_err < 1e-3 and t6_ok)
    log(f"self-tests {res} -> {'PASS' if ok else 'FAIL'}")
    json.dump({**res, "pass": ok}, open(os.path.join(OUTDIR, "selftest.json"), "w"), indent=1)
    if not ok:
        sys.exit("SELF-TESTS FAILED — aborting before any real run")


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


# RULER has 500 samples per subtask x 13 subtasks; `fraction` subsamples proportionally.
FRACTION = min(1.0, SAMPLES / 500.0)

pipe, tested = None, False
for i, (press_name, ratio) in enumerate(RUNS):
    cfg = evaluate.EvaluationConfig(**{**asdict(evaluate.EvaluationConfig()),
                                       "dataset": "ruler", "data_dir": str(CONFIG), "model": MODEL,
                                       "press_name": press_name, "compression_ratio": ratio,
                                       "fraction": FRACTION, "seed": 42, "output_dir": OUTDIR})
    evaluate.logger.handlers.clear()                       # each runner adds one; they accumulate
    runner = evaluate.EvaluationRunner(cfg)
    if any((d / "metrics.json").exists() for d in run_dirs(cfg)):
        log(f"{i + 1}/{len(RUNS)} {press_name} cr={ratio:.3f}: done, skipping")
        continue
    if pipe is None:
        runner._setup_press()
        runner._setup_model_pipeline()
        pipe = runner.pipeline
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)
    if not tested:
        self_tests(pipe)
        tested = True

    press = evaluate_registry.PRESS_REGISTRY[press_name]
    if isinstance(press, BERN_TYPES):
        press.stats_log.clear()
    log(f"{i + 1}/{len(RUNS)} {press_name} cr={ratio:.3f}")
    t0 = time.time()
    is_e9 = isinstance(press, e9bp.BernPress)
    e9_patched.model = pipe.model
    # Order matters: E9's patch OUTER, ours INNER. bias_patched wraps whatever is registered at entry, so
    # entering it first and E9's second would let E9 overwrite our wrapper.
    with (e9_patched() if is_e9 else nullcontext()), bias_patched(pipe.model):
        runner.run_evaluation()
    dt = time.time() - t0

    if isinstance(press, BERN_TYPES) and press.stats_log:
        kf = sum(s["kept_frac"] for s in press.stats_log) / len(press.stats_log)
        hps = [s["M_hp_frac"] for s in press.stats_log if "M_hp_frac" in s]   # E9's log has no M_hp
        log(f"   kept fraction {kf:.4f} (target {1 - ratio:.4f})"
            + (f", max M_hp {max(hps):.4f}" if hps else ""))
        for d in run_dirs(cfg):
            if (d / "metrics.json").exists():
                json.dump(press.stats_log, open(d / "memory.json", "w"), indent=1)
    log(f"   done in {dt:.0f}s")

log("all runs done")
