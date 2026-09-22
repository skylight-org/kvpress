"""T7-T9: self-tests for CompactorBernPress. Run after E3's T1-T6 (experiment3/selftests.py, unchanged).

T7  masking path == pruning path: CompactorBernPress(force_topk=True) keeps exactly Compactor's top-k
    through the -inf bias; its logits must match plain CompactorPress, which prunes the cache.
T8  budget: per head, expected kept count == int(n * (1 - ratio)); forced tokens have r = 1.
T10 Cholesky fallback (stock kvpress Compactor crashes at ~124k): exact stock output when stock works.
T9  end to end through the kvpress pipeline: the bias is live during generation and absent during
    prefill, the blocked fraction matches the target, ratio 0 == no_press, seeds reproduce and differ.
"""

import json
import os
import sys
import time

import torch
from transformers import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import math

from compactor_bern import (CompactorBernPress, FallbackLeverageScorePress, SeededCompactorPress as CompactorPress,
                            forced_mask, keep_probabilities, stable_leverage_scores)
from kvpress.presses.leverage_press import LeverageScorePress
from e3_press import bias_patched

CTX = " ".join(f"The code for item {i} is {(i * 7919) % 99991}." for i in range(160))   # ~1.9k tokens
QUESTION = "What is the code for item 57?"


def _clear(model):
    for L in model.model.layers:
        L.self_attn.e2_bias = None
        L.self_attn.masked_key_indices = None


def _logits(pipe, press):
    """Context prefill under `press`, then one question forward on the resulting cache."""
    model, tok, dev = pipe.model, pipe.tokenizer, pipe.model.device
    ids = tok(CTX, return_tensors="pt").input_ids.to(dev)
    qids = tok(" " + QUESTION, return_tensors="pt").input_ids.to(dev)
    n = ids.shape[1]
    cache = DynamicCache()
    with torch.no_grad(), bias_patched(model):
        with press(model):
            model.model(input_ids=ids, past_key_values=cache)
        biases = [None if L.self_attn.e2_bias is None else L.self_attn.e2_bias.clone() for L in model.model.layers]
        out = model(input_ids=qids, past_key_values=cache,
                    position_ids=torch.arange(n, n + qids.shape[1], device=dev)[None]).logits[0].double()
    _clear(model)
    return out, biases, n


def t8_budget():
    """Synthetic z-scores; expected count per head must hit the budget at every ratio."""
    g = torch.Generator().manual_seed(0)
    worst, forced_ok = 0.0, True
    for n in (37, 1024, 4096):
        s = torch.randn(1, 8, n, generator=g, dtype=torch.float64)
        s = (s - s.mean()) / s.std()
        for ratio in (0.5, 0.75, 0.875, 0.9375):
            r, c, k = keep_probabilities(s, ratio, 8, 4)
            f = forced_mask(n, 8, 4, s.device)
            forced_ok &= bool((r[..., f] == 1.0).all())
            nf = int(f.sum())
            n_pos_free = ((s - s.amin(-1, keepdim=True)).masked_fill(f, 0.0) > 0).sum(-1).double()
            expected = torch.where(torch.tensor(k > nf), torch.clamp(n_pos_free + nf, max=k),
                                   torch.full_like(n_pos_free, nf))
            err = ((r.sum(-1) - expected).abs() / expected.clamp_min(1)).max().item()
            worst = max(worst, err)
    # timing at SCBench length
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    big = torch.randn(1, 8, 124_000, device=dev)
    t0 = time.time()
    keep_probabilities(big, 0.875, 8, 4)
    if dev == "cuda":
        torch.cuda.synchronize()
    return {"T8_budget_rel_err": worst, "T8_forced_kept": forced_ok, "T8_seconds_per_layer_124k": time.time() - t0}


def t10_fallback(dev):
    """The Cholesky fallback: stock output untouched when stock works; float64 path agrees with stock on
    the same sketch; a badly scaled, near-rank-deficient key set that breaks stock is rescued."""
    g = torch.Generator(device=dev).manual_seed(0)
    K = torch.randn(1, 8, 3000, 128, device=dev, generator=g)
    torch.manual_seed(1)
    a = LeverageScorePress.compute_leverage_scores(K, 48)
    fb = FallbackLeverageScorePress()
    torch.manual_seed(1)
    b = fb.compute_leverage_scores(K, 48)
    torch.manual_seed(1)
    phi = torch.randn(1, 8, 128, 48, device=dev) / math.sqrt(48)
    c = stable_leverage_scores(K, phi)
    base = torch.randn(1, 8, 20000, 4, device=dev, generator=g) * 3000
    K2 = base @ torch.randn(4, 128, device=dev, generator=g) + 1e-3 * torch.randn(1, 8, 20000, 128, device=dev, generator=g)
    fb2 = FallbackLeverageScorePress()
    torch.manual_seed(1)
    s2 = fb2.compute_leverage_scores(K2, 48)
    return {"T10_wrapper_equals_stock": bool(torch.equal(a, b)) and fb.n_fallback == 0,
            "T10_stable_vs_stock_rel": ((c - a).abs().max() / a.abs().max()).item(),
            "T10_pathological_fallbacks": fb2.n_fallback, "T10_pathological_finite": bool(torch.isfinite(s2).all())}


def compactor_tests(pipe, log, OUTDIR):
    model = pipe.model
    impl = model.config._attn_implementation
    res = {}

    # T7: masking == pruning on the same kept set
    base, _, _ = _logits(pipe, CompactorPress(compression_ratio=0.0))
    pruned, _, _ = _logits(pipe, CompactorPress(compression_ratio=0.875))
    masked, _, _ = _logits(pipe, CompactorBernPress(press=CompactorPress(compression_ratio=0.875), force_topk=True))
    scale = base.abs().max()
    res["T7_mask_vs_prune_rel"] = ((masked - pruned).abs().max() / scale).item()
    res["T7_argmax_equal"] = bool((masked.argmax(-1) == pruned.argmax(-1)).all())
    res["T7_prune_moves_logits_rel"] = ((pruned - base).abs().max() / scale).item()

    res.update(t8_budget())
    res.update(t10_fallback(model.device))

    # T9a: reproducible and seed-dependent draws
    def draws(seed):
        _, b, _ = _logits(pipe, CompactorBernPress(press=CompactorPress(compression_ratio=0.875), seed=seed))
        return torch.stack([torch.isinf(x) for x in b])
    d1, d1b, d2 = draws(1), draws(1), draws(2)
    res["T9_same_seed_identical"] = bool(torch.equal(d1, d1b))
    res["T9_seeds_differ_frac"] = (d1 != d2).float().mean().item()

    # T9b: through the real pipeline, count bias-carrying kernel calls (as E3's T6 does)
    seen = {"gen": 0, "prefill": 0, "blocked": []}

    def counter(func):
        def w(module, q, k, v, attention_mask, dropout=0.0, **kw):
            b = getattr(module, "e2_bias", None)
            if b is not None:
                if q.shape[2] == k.shape[2]:
                    seen["prefill"] += 1
                else:
                    seen["gen"] += 1
                    seen["blocked"].append(torch.isinf(b).float().mean().item())
            return func(module, q, k, v, attention_mask, dropout, **kw)
        return w

    p8 = CompactorBernPress(press=CompactorPress(compression_ratio=0.875), seed=1)
    try:
        with bias_patched(model):
            inner = ALL_ATTENTION_FUNCTIONS[impl]
            ALL_ATTENTION_FUNCTIONS[impl] = counter(inner)
            try:
                ans = pipe(CTX, question=QUESTION, press=p8, max_new_tokens=8)["answer"]
            finally:
                ALL_ATTENTION_FUNCTIONS[impl] = inner
            p0 = CompactorBernPress(press=CompactorPress(compression_ratio=0.0), seed=1)
            ans0 = pipe(CTX, question=QUESTION, press=p0, max_new_tokens=8)["answer"]
            ans_np = pipe(CTX, question=QUESTION, press=None, max_new_tokens=8)["answer"]
    finally:
        _clear(model)
    layers = p8.stats_log[-1]["layers"]
    kept = sum(x["kept_frac"] for x in layers) / len(layers)
    target = layers[0]["target_frac"]
    blocked = sum(seen["blocked"]) / len(seen["blocked"]) if seen["blocked"] else 0.0
    res.update({"T9_gen_calls_with_bias": seen["gen"], "T9_prefill_calls_with_bias": seen["prefill"],
                "T9_blocked_frac": blocked, "T9_kept_frac": kept, "T9_target_frac": target,
                "T9_answer": ans[:30], "T9_ratio0_equals_no_press": ans0 == ans_np})

    ok = (res["T7_mask_vs_prune_rel"] < 5e-2 and res["T7_argmax_equal"] and res["T7_prune_moves_logits_rel"] > 1e-3
          and res["T8_budget_rel_err"] < 1e-6 and res["T8_forced_kept"]
          and res["T9_same_seed_identical"] and res["T9_seeds_differ_frac"] > 0.01
          and res["T9_gen_calls_with_bias"] > 0 and res["T9_prefill_calls_with_bias"] == 0
          and abs(kept - target) < 0.01 and abs(blocked - (1 - kept)) < 1e-4
          and res["T9_ratio0_equals_no_press"]
          and res["T10_wrapper_equals_stock"] and res["T10_stable_vs_stock_rel"] < 1e-2
          and res["T10_pathological_finite"])
    log(f"compactor self-tests {res} -> {'PASS' if ok else 'FAIL'}")
    json.dump({**res, "pass": ok}, open(os.path.join(OUTDIR, "selftest_compactor.json"), "w"), indent=1)
    if not ok:
        sys.exit("COMPACTOR SELF-TESTS FAILED, aborting before any real run")
    return res
