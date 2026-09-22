"""Experiment 5 ablation: BernoulliPress on KVzip scores with a PER-HEAD budget.

Why this arm exists
-------------------
The shipped `BernoulliPress` gives KVzip a single GLOBAL keep-probability constant `c`, exactly as
KVzip's own top-k does, so the budget is allocated across all layers and heads by score. On RULER
32k at 8x that starves some heads to 1 token in 10,000 while others keep 99%:

    arm         min kept   median    max kept   spread
    kvzip        0.0001    0.0010    0.9925     7985x
    bernoulli    0.0002    0.0059    0.9915     4559x
    vkvwr        0.1080    0.1169    0.6212       5.8x

The professor's VKvWRPress instead solves a sample size per (layer, head) from a CLT bound, which
gives every head a floor. So a straight comparison confounds two different things:

    the ESTIMATOR   Bernoulli + Horvitz-Thompson   vs   multinomial + Hansen-Hurwitz
    the ALLOCATION  one global c                   vs   per-head adaptive

This arm holds the estimator fixed and changes only the allocation, so the two can be separated.
It is an ablation, not a proposed method: it is deliberately kept here rather than added to the
kvpress branch, and it subclasses the shipped press rather than forking it, so the sampling, the
sink handling and the log(1/r) bias are byte-for-byte the shipped ones.
"""

import torch
from kvpress import BernoulliPress, KVzipPress
from kvpress.presses.bernoulli_press import (
    _keep_prob,
    _solve_c_per_head,
    transform_scores_for_sampling,
)
from kvpress.utils import get_cache_metadata
from transformers import PreTrainedModel


class PerHeadBernoulliPress(BernoulliPress):
    """BernoulliPress with one keep-probability constant per (layer, KV head) instead of one global."""

    def compress_post(self, model: PreTrainedModel):
        if self.compression_ratio <= 0:
            return
        self._check_sdpa(model.config._attn_implementation)

        scores = self.press.score_val                       # [n_rows, bsz, n_kv_heads, ctx_len]
        assert scores.shape[1] == 1, "PerHeadBernoulliPress only supports batch size 1"
        n_rows, _, n_kv_heads, ctx_len = scores.shape
        n_sink = self.press.n_sink
        device = scores.device

        # One row per (layer, KV head); every row keeps the same count in expectation.
        flat = transform_scores_for_sampling(scores.double()).reshape(-1, ctx_len)
        budget = (1.0 - self.compression_ratio) * ctx_len

        # Sinks are forced to r = 1 and paid for out of the same per-head budget, as the global
        # path does: solve for the free positions only, then put the sinks back at 1.
        free = flat.clone()
        free[:, :n_sink] = 0.0
        k_free = torch.full((free.shape[0],), max(budget - n_sink, 0.0), dtype=torch.float64, device=device)
        n_pos = (free > 0).sum(-1).double()
        k_free = torch.minimum(k_free, n_pos)
        c = _solve_c_per_head(free, k_free)
        r = _keep_prob(free, c[:, None])
        r[:, :n_sink] = 1.0

        context_crc = self._context_crc
        if context_crc is None:
            import zlib
            context_crc = zlib.crc32(self.press._context_ids.detach().cpu().numpy().tobytes())
        generator = self._generator(device, context_crc ^ (self.seed * 7919))
        keep = torch.rand(r.shape, generator=generator, dtype=torch.float64, device=device) < r

        bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)),
                           torch.tensor(-float("inf"), device=device))
        bias = bias.view(n_rows, 1, n_kv_heads, ctx_len)

        metadata = get_cache_metadata(self.press._cache)
        attention_bias = metadata.setdefault("attention_bias", {})
        rows = self.press._score_row_by_layer
        for module in self._attention_modules(model):
            attention_bias[int(module.layer_idx)] = bias[rows[int(module.layer_idx)]].float()
            module.masked_key_indices = None


def make(seed: int) -> PerHeadBernoulliPress:
    return PerHeadBernoulliPress(press=KVzipPress(), seed=seed)
