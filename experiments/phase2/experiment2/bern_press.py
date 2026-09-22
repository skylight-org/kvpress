"""Phase 2, Experiment 2: Bernoulli selection on top of KVzip's own scoring.

The point of this file is the CONTROLLED COMPARISON (DESIGN.md §4):

    kvzip        KVzip score_val -> global top-k                       (kvpress, unmodified)
    kvzip_bern   THE SAME score_val -> Bernoulli min(1, c*s) + log(1/r) bias

`KVzipBernPress` subclasses kvpress's `KVzipPress` and overrides **only `compress_post`**, the
selection step. Scoring (`score_kvzip`, the chunked repeat pass, sink handling) is inherited byte for
byte, so any difference in task score between the two arms is attributable to selection alone.
Self-test T5 asserts the two produce identical `score_val`.

The bias is applied by an attention patch (`e2_bias`), reusing the mechanism verified in E9:
  log(1/r) on kept context tokens, -inf on dropped ones, 0 on question/generated tokens.
Memory is COUNTED, not freed -- dropped tokens are masked, as kvpress does for every head-wise press.

Maths: ../DERIVATION_denominator_bernoulli.pdf sections 3-4 (r_i = min(1, c s_i), Horvitz-Thompson).
Budget inversion and the memory bound live in ../experiment1/denom.py -- the ONLY implementation.
"""

import math
import os
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvpress.presses.kvzip_press import KVzipPress

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment1"))
import denom  # noqa: E402   the budget inversion, shared with Experiment 1

N_SINK = 4
METHODS = ("kvzip_bern", "kvzip_bern_nobias")


# ---------------------------------------------------------------- attention patch (bias kernel)
def _bias_patch(func):
    """Wrap an attention implementation so `module.e2_bias` is added to the attention mask.

    e2_bias is [1, NKV, n_ctx]: log(1/r) for kept context tokens, -inf for dropped. Expanded to the
    query heads that share each KV head. Question and generated tokens get 0 (they are never
    compressed), and causal masking among them is left to the existing mask.

    Signature is POSITIONAL and must match kvpress's own patch exactly
    (module, query, key, value, attention_mask, dropout, **kwargs) -- `dropout` is passed
    positionally by transformers, so a *args catch-all would misbind it.

    The bias is skipped when query_len == key_len (a prefill). That is the same test kvpress's
    attention_patch uses: during the CONTEXT prefill and the calibration repeat passes the cache must
    be read unmodified, and compression has not been decided yet. The bias applies only to the
    question prefill and decoding, which is where the compressed cache is actually consumed."""
    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        bias = getattr(module, "e2_bias", None)
        if bias is not None and bias.shape[2] > key.shape[2]:
            # Stale bias from a previous, longer context. Should be impossible now that
            # _reset_internal_parameters clears it per sample, but a silent mis-application would
            # compress this context with the previous one's decisions, so fail loudly instead.
            raise RuntimeError(f"stale e2_bias: n_ctx={bias.shape[2]} > n_keys={key.shape[2]}")
        if bias is not None and query.shape[2] != key.shape[2]:
            nq, nk = query.shape[2], key.shape[2]
            nkv, n_ctx = bias.shape[1], bias.shape[2]
            G = query.shape[1] // nkv
            add = bias.repeat_interleave(G, dim=1)[:, :, None, :]           # [1, NH, 1, n_ctx]
            full = torch.zeros((1, query.shape[1], nq, nk), dtype=query.dtype, device=query.device)
            full[..., :n_ctx] = add.to(query.dtype)
            if attention_mask is None:
                # transformers' sdpa_attention_forward computes
                #     is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
                # so handing it a mask where there was None disables causal masking. When we replace
                # a None mask we must therefore carry the causal structure ourselves. Keys are
                # [context | exact] and the queries are the last nq of them, so query i (absolute
                # nk-nq+i) may see key j <= nk-nq+i.
                if nq > 1:
                    idx = torch.arange(nk, device=query.device)
                    causal = idx[None, :] > (torch.arange(nq, device=query.device)[:, None] + (nk - nq))
                    full = full.masked_fill(causal[None, None], -float("inf"))
                attention_mask = full
            elif attention_mask.dtype == torch.bool:
                # CRITICAL: transformers hands SDPA a BOOLEAN mask (True = attend). Adding a float
                # tensor to it promotes bool -> float, turning True/False into 1.0/0.0, so blocked
                # positions become "allowed with zero bias" and the causal structure is destroyed.
                # Measured: an all-zero bias moved the logits by 19% (self-test T1) purely from this.
                # Convert to the additive convention (0 where allowed, -inf where blocked) first.
                attention_mask = torch.zeros_like(attention_mask, dtype=query.dtype).masked_fill_(
                    ~attention_mask, -float("inf")) + full
            else:
                attention_mask = attention_mask + full
        return func(module, query, key, value, attention_mask, dropout, **kwargs)
    return wrapper


@contextmanager
def bias_patched(model):
    """Install the patched attention implementation for the duration of a block."""
    name = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[name]
    ALL_ATTENTION_FUNCTIONS[name] = _bias_patch(orig)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS[name] = orig
        for layer in model.model.layers:
            layer.self_attn.e2_bias = None


# ---------------------------------------------------------------- the press
@dataclass
class KVzipBernPress(KVzipPress):
    """KVzip's scoring, Bernoulli selection.

    method: "kvzip_bern" (log(1/r) bias) or "kvzip_bern_nobias" (selection only, no correction).
    seed:   Bernoulli draws are seeded per (context, seed) so runs are reproducible and the
            no-bias ablation sees the SAME draws as its biased counterpart (paired comparison)."""

    method: str = "kvzip_bern"
    seed: int = 1
    stats_log: list = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert self.method in METHODS, self.method

    def __call__(self, model):
        """Clear e2_bias on ENTRY to each sample, never on exit.

        kvpress's pipeline (`pipeline.py` L215-232) wraps ONLY the prefill in the press context:

            with press(model):          # <- compress_post runs here, sets e2_bias
                model.model(context)
            ...                         # context EXITS
            for question in questions:  # <- generation happens OUT HERE
                generate_answer(...)

        So anything cleared in the context's exit (`_reset_internal_parameters`) is gone before a
        single token is generated. Clearing there made every Bernoulli run decode with the FULL
        cache: all nine P1 runs scored exactly 95.4154, identical to no_press, while memory.json
        correctly logged kept_frac 0.25 -- the compression was computed and then discarded.

        Clearing on entry instead gives the same protection against a stale bias from the previous
        sample (the crash this originally fixed) while leaving the bias live through generation.

        KVzipPress.__call__ is @contextmanager-decorated, so super().__call__(model) merely BUILDS
        the context manager; the clearing below therefore happens before the `with` body runs, which
        is exactly the entry point we want."""
        for layer in model.model.layers:
            layer.self_attn.e2_bias = None
        self._model_ref = model
        return super().__call__(model)

    def compress_post(self, model):
        """Override of KVzipPress.compress_post: Bernoulli instead of global top-k.

        self.score_val is [n_layer, bsz, n_kv_heads, ctx_len], built by the inherited scoring and
        already carrying score=1.0 on the first n_sink tokens. Budget matching is on EXPECTED memory:
        c solves sum_i min(1, c*s_i) = (1 - compression_ratio) * numel, the same total the top-k arm
        keeps, so the two arms are compared at equal memory."""
        self._model_ref = model              # so _reset_internal_parameters can clear e2_bias
        if self.compression_ratio <= 0:
            return
        s = self.score_val                                                  # [NL, bsz, NKV, n]
        assert s.shape[1] == 1, "batch size 1 only"
        NL, _, NKV, n = s.shape
        dev = s.device
        flat = s.reshape(-1).double().clamp_min(0.0)
        budget = (1.0 - self.compression_ratio) * flat.numel()

        # c from the shared implementation (bisection on a monotone M(c)); r = min(1, c*s)
        # NOTE the sink correction below is NOT cosmetic. KVzip marks the first n_sink tokens by
        # setting their score to 1.0, but 1.0 is not distinguished -- ordinary tokens reach it too --
        # so at c < 1 the sink would be kept with probability only c (measured: 0.56 at 4x, 0.28 at
        # 8x). Top-k keeps sinks because they tie at the top; Bernoulli would not. Dropping the sink
        # is catastrophic (E5a: layer 0 holds essentially all its mass there), so force r = 1 on the
        # sink, exactly as E9's press did. The derivation permits this: section 3, "forced keeps are
        # allowed" -- any r'_i >= min(1, c s_i) leaves every bound valid, since each tail term
        # (1-r)/r is non-increasing in r and forcing r = 1 only removes a token from the tail.
        c = denom.solve_c_for_memory(flat, budget)
        r = denom.keep_prob(flat, c)
        r = r.view(NL, 1, NKV, n)
        r[..., :self.n_sink] = 1.0
        r = r.reshape(-1)
        # re-solve so the FORCED keeps are paid for out of the same budget (equal memory vs top-k)
        if r.sum().item() > budget:
            lo, hi = 0.0, c
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                rr = denom.keep_prob(flat, mid).view(NL, 1, NKV, n)
                rr[..., :self.n_sink] = 1.0
                if rr.sum().item() > budget:
                    hi = mid
                else:
                    lo = mid
            c = hi
            r = denom.keep_prob(flat, c).view(NL, 1, NKV, n)
            r[..., :self.n_sink] = 1.0
            r = r.reshape(-1)

        g = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
        # Seed from the CONTEXT TOKENS, not the tensor shape: seeding from (NL, NKV, n) gave every
        # context of the same length the identical keep pattern, correlating samples across the
        # dataset. Content-seeding keeps draws independent across contexts, and still pairs
        # kvzip_bern with kvzip_bern_nobias on the same draws (same context, same seed).
        ctx_bytes = self._context_ids.detach().cpu().numpy().tobytes()
        g.manual_seed((zlib.crc32(ctx_bytes) ^ (self.seed * 7919)) & 0x7FFFFFFF)
        Z = torch.rand(r.shape, generator=g, dtype=torch.float64, device=r.device) < r

        keep = Z.view(NL, 1, NKV, n)
        r_v = r.view(NL, 1, NKV, n)
        if self.method == "kvzip_bern_nobias":
            bias = torch.where(keep, 0.0, -float("inf"))
        else:
            bias = torch.where(keep, -torch.log(r_v.clamp_min(1e-300)), torch.tensor(-float("inf"), device=r.device))

        for layer in model.model.layers:
            li = int(layer.self_attn.layer_idx)
            layer.self_attn.e2_bias = bias[li].float()                      # [1, NKV, n]
            layer.self_attn.masked_key_indices = None                       # bias carries the -inf

        self.stats_log.append({"n": n, "c": c, "kept_frac": keep.double().mean().item(),
                               "expected_frac": r.mean().item(),
                               "M_hp_frac": min(1.0, denom.memory_highprob(r, 0.05) / r.numel())})
