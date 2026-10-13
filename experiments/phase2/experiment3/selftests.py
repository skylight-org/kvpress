"""T1-T6 self-tests, copied VERBATIM from experiment2/run_sweep.py (which they were verified in),
with two changes: the press class is E3's subclass, and `log`/`OUTDIR` are parameters instead of module
globals. experiment2's runner is deliberately left untouched so its published runs stay reproducible.

T6 is the important one: it counts kernel calls carrying the bias during GENERATION. E2's worst bug
(the bias cleared before decoding, making every Bernoulli run score exactly no_press) passed T1-T4 and
was caught only by an end-to-end check like this.
"""

import json
import os
import sys

import torch

# In E2 these were module globals of run_sweep.py; here they are imported explicitly.
from e3_press import E3BernPress, bias_patched


def self_tests(pipe, log, OUTDIR):
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
        p8 = E3BernPress(method="kvzip_bern", seed=1, compression_ratio=0.875)
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


