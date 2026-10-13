"""Diagnostic variants of the E5/E6 per-head press (used only by diag/ scripts)."""
import torch
import perhead_press as PH
from kvpress.presses.bernoulli_press import _keep_prob, _solve_c, transform_scores_for_sampling
from kvpress.utils import get_cache_metadata


def shipped_global_c(scores, cr, n_sink):
    """BernoulliPress.compress_post's c, copied line for line."""
    n_layers, _, n_kv, ctx = scores.shape
    flat = transform_scores_for_sampling(scores.reshape(-1).double())
    budget = (1.0 - cr) * flat.numel()
    c = _solve_c(flat, budget)
    r = _keep_prob(flat, c).view(n_layers, 1, n_kv, ctx); r[..., :n_sink] = 1.0
    if r.sum().item() > budget:
        lo, hi = 0.0, c
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rm = _keep_prob(flat, mid).view(n_layers, 1, n_kv, ctx); rm[..., :n_sink] = 1.0
            hi, lo = (mid, lo) if rm.sum().item() > budget else (hi, mid)
        c = hi
    return c


class PerHeadWithGlobalC(PH.PerHeadBernoulliPress):
    """The per-head press with every head's c set to the shipped global c: must equal BernoulliPress."""
    def compress_post(self, model):
        c = shipped_global_c(self.press.score_val, self.compression_ratio, self.press.n_sink)
        orig = PH._solve_c_per_head
        PH._solve_c_per_head = lambda free, k: torch.full((free.shape[0],), c, dtype=torch.float64, device=free.device)
        self._context_crc = None
        try:
            super().compress_post(model)
        finally:
            PH._solve_c_per_head = orig


class PerHeadTopK(PH.PerHeadBernoulliPress):
    """Same equal per-head budget, but deterministic top-k within each head and no reweighting."""
    def compress_post(self, model):
        scores = self.press.score_val
        n_rows, _, n_kv, ctx = scores.shape
        flat = scores.double().reshape(-1, ctx).clone()
        flat[:, : self.press.n_sink] = float("inf")                            # sinks always kept
        k = int(round((1.0 - self.compression_ratio) * ctx))
        idx = flat.topk(k, dim=-1).indices
        keep = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
        bias = torch.where(keep, 0.0, float("-inf")).view(n_rows, 1, n_kv, ctx)
        md = get_cache_metadata(self.press._cache).setdefault("attention_bias", {})
        rows = self.press._score_row_by_layer
        for module in self._attention_modules(model):
            md[int(module.layer_idx)] = bias[rows[int(module.layer_idx)]].float()
            module.masked_key_indices = None
