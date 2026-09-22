"""Experiment 4: Bernoulli selection on top of Compactor's own scoring.

The controlled comparison mirrors E2/E3's KVzip pair:

    compactor        Compactor score -> per-head top-k, pruned         (kvpress, unmodified)
    compactor_bern   THE SAME score  -> per-head Bernoulli min(1, c*s') + log(1/r) bias, masked

`CompactorBernPress` wraps a kvpress `CompactorPress` (composition, not a subclass) and calls its
`score` unchanged. Only the selection step differs:

  * Compactor's scores are signed z-scores, so each head's scores are shifted by the head minimum,
    s' = s - min_head(s). This keeps the order and the gaps (user's choice, DESIGN.md).
  * The budget is per KV head, exactly Compactor's: every head keeps int(n * (1 - ratio)) tokens in
    expectation. One c per head, found by bisection.
  * Compactor's protected tokens (first sink_size_start, last sink_size_end) get r = 1 and are paid for
    out of the same budget, as E2 does with KVzip's sinks.
  * Kept tokens get the Horvitz-Thompson bias log(1/r), dropped ones -inf, through E2's verified kernel
    (`module.e2_bias`, installed by `bias_patched`). Keys and values are not pruned: memory is counted,
    not freed, as for kvzip_bern.
"""

import math
import os
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.compactor_press import CompactorPress
from kvpress.presses.leverage_press import LeverageScorePress

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment3"))


def forced_mask(n: int, start: int, end: int, device) -> torch.Tensor:
    """Tokens Compactor always keeps: the first `start` and the last `end` (same clipping as its score)."""
    left = min(start, n)
    right = min(end, max(0, n - left))
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[:left] = True
    if right:
        m[n - right:] = True
    return m


def solve_c_per_head(t: torch.Tensor, k_free: torch.Tensor, iters: int = 100) -> torch.Tensor:
    """Smallest c per head with sum_i min(1, c * t_i) >= k_free.

    t: [H, n] non-negative float64 scores with forced tokens already zeroed.
    k_free: [H] float64 target expected count among the free tokens, already capped at the number of
    positive scores. Bisection runs in log c; at c = 1 / min(positive t) every positive token is kept,
    which is the upper bracket."""
    pos = t > 0
    big = torch.finfo(t.dtype).max
    t_min = torch.where(pos, t, torch.full_like(t, big)).amin(-1)
    t_max = t.amax(-1)
    lo = (1e-12 / t_max.clamp_min(1e-300)).log()
    hi = (1.0 / t_min.clamp_min(1e-300)).log()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = (mid.exp()[:, None] * t).clamp(max=1.0).sum(-1)
        ok = m >= k_free
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi.exp()


def keep_probabilities(scores: torch.Tensor, ratio: float, start: int, end: int):
    """scores [B, H, n] (Compactor z-scores) -> r [B, H, n] float64, c [B, H], per-head budget k.

    Every head's expected kept count is exactly k = int(n * (1 - ratio)), Compactor's own count."""
    B, H, n = scores.shape
    s = scores.double()
    s = s - s.amin(-1, keepdim=True)                                   # the user's score map
    forced = forced_mask(n, start, end, s.device)
    n_forced = int(forced.sum())
    k = int(n * (1 - ratio))
    t = s.masked_fill(forced, 0.0).reshape(B * H, n)
    n_pos = (t > 0).sum(-1).double()
    k_free = torch.full((B * H,), float(max(k - n_forced, 0)), dtype=torch.float64, device=s.device)
    k_free = torch.minimum(k_free, n_pos)
    c = solve_c_per_head(t, k_free)
    r = (c[:, None] * t).clamp(max=1.0)
    r = torch.where(k_free[:, None] > 0, r, torch.zeros_like(r))       # budget used up by forced keeps
    r = r.view(B, H, n).masked_fill(forced, 1.0)
    return r, c.view(B, H), k


def stable_leverage_scores(key_states: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    """LeverageScorePress.compute_leverage_scores with the SAME sketch, in float64 with scale-relative jitter.

    Leverage scores are invariant to rescaling X, so the jitter is taken relative to mean(diag(G)).
    Upstream uses an absolute jitter (at most 100) on a float32 G whose entries reach ~1e8 at 124k
    tokens, where rounding can leave G indefinite and every retry fails."""
    X = key_states.double() - key_states.double().mean(dim=-2, keepdim=True)
    X = X @ Phi.double()
    XT = X.transpose(-2, -1)
    G = XT @ X
    G = 0.5 * (G + G.transpose(-2, -1))
    scale = G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    for rel in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        L, info = torch.linalg.cholesky_ex(G + rel * scale * eye, upper=False)
        if bool((info == 0).all()):
            inv_Xt = torch.cholesky_solve(XT, L, upper=False)
            return (X * inv_Xt.transpose(-2, -1)).sum(dim=-1).clamp_min(0).to(torch.float32)
    raise RuntimeError("stable Cholesky failed too")


@dataclass
class FallbackLeverageScorePress(LeverageScorePress):
    """Stock leverage scores; only when the stock Cholesky fails, the float64 path on the same sketch.

    Wherever upstream works, the output is upstream's, bit for bit. `n_fallback` counts the layers that
    needed the fallback, so the report can say how often Compactor was rescued."""

    n_fallback: int = 0

    def compute_leverage_scores(self, key_states, sketch_dimension):
        dev = key_states.device
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
        try:
            return LeverageScorePress.compute_leverage_scores(key_states, sketch_dimension)
        except RuntimeError as e:
            if "Cholesky failed" not in str(e):
                raise
        torch.set_rng_state(cpu_state)                   # redraw the identical sketch
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, dev)
        d, k = key_states.shape[-1], sketch_dimension
        Phi = torch.randn(key_states.shape[0], key_states.shape[1], d, k, device=dev,
                          dtype=key_states.dtype) * (1 / math.sqrt(k))
        self.n_fallback += 1
        return stable_leverage_scores(key_states, Phi)


def context_crc(ids: torch.Tensor) -> int:
    return zlib.crc32(ids.detach().cpu().numpy().tobytes())


@dataclass
class SeededCompactorPress(CompactorPress):
    """kvpress's CompactorPress with its Gaussian leverage sketch seeded from (context, layer).

    CompactorPress draws the sketch with an unseeded torch.randn, so its scores change from call to call
    (found by self-test T9). Seeding makes Compactor reproducible and, more importantly, makes
    `compactor` and every `compactor_bern` seed read the SAME scores, so they differ in selection alone.
    The algorithm is unchanged; the global RNG state is restored after each call.

    It also uses FallbackLeverageScorePress: stock kvpress Compactor crashes at ~124k tokens
    ("Cholesky failed after 5 tries", found by smoke job 312115); the fallback recomputes only the
    failing layer in float64 with the same sketch."""

    def __post_init__(self):
        super().__post_init__()
        self._sketch_seed = None
        self._leverage_press = FallbackLeverageScorePress(
            compression_ratio=self.compression_ratio, sketch_dimension=self.sketch_dimension)

    @property
    def n_fallback(self):
        return self._leverage_press.n_fallback

    def _capture_context(self, module, args, kwargs):
        self._sketch_seed = context_crc(kwargs.get("input_ids", args[0] if args else None))

    @contextmanager
    def __call__(self, model):
        pre = model.model.register_forward_pre_hook(self._capture_context, with_kwargs=True)
        try:
            with super().__call__(model):
                yield
        finally:
            pre.remove()

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        assert self._sketch_seed is not None, "context ids were not captured"
        seed = (self._sketch_seed ^ (int(module.layer_idx) * 104729) ^ 0x5EED) & 0x7FFFFFFF
        devices = [hidden_states.device] if hidden_states.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            return super().score(module, hidden_states, keys, values, attentions, kwargs)


@dataclass
class CompactorBernPress(BasePress):
    """Compactor's scoring, Bernoulli selection with a log(1/r) correction. Batch size 1.

    seed: draws are seeded per (context, seed, layer), so runs are reproducible and every seed gives an
          independent draw.
    force_topk: TEST ONLY (self-test T7). Keep exactly Compactor's top-k with zero bias, which must
          reproduce plain `compactor` (which prunes) and so validates the masking path."""

    press: SeededCompactorPress = field(default_factory=SeededCompactorPress)
    seed: int = 1
    force_topk: bool = False
    stats_log: list = field(default_factory=list, repr=False)  # kvpress saves str(press) per run

    def __post_init__(self):
        assert isinstance(self.press, SeededCompactorPress), "CompactorBernPress wraps a SeededCompactorPress"
        self._ctx_seed = None

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    def _capture_context(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self._ctx_seed = context_crc(ids)
        self.press._sketch_seed = self._ctx_seed          # same sketch as plain `compactor` on this context
        self.stats_log.append({"n": int(ids.shape[1]), "layers": []})

    @contextmanager
    def __call__(self, model):
        # Clear on ENTRY, never on exit: kvpress generates answers after this context closes, and the
        # bias must stay live for that (E2's worst bug cleared it on exit).
        for layer in model.model.layers:
            layer.self_attn.e2_bias = None
        pre = model.model.register_forward_pre_hook(self._capture_context, with_kwargs=True)
        try:
            with super().__call__(model):
                yield
        finally:
            pre.remove()

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio <= 0:
            return keys, values
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)   # [1, H, n]
        assert scores.shape[0] == 1, "batch size 1 only"
        n = scores.shape[-1]
        dev = scores.device

        if self.force_topk:
            k = int(n * (1 - self.compression_ratio))
            idx = scores.topk(k, dim=-1).indices
            keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, idx, True)
            bias = torch.where(keep, 0.0, -float("inf"))
            r, c = keep.double(), None
        else:
            r, c, k = keep_probabilities(scores, self.compression_ratio,
                                         self.press.sink_size_start, self.press.sink_size_end)
            assert self._ctx_seed is not None, "context ids were not captured"
            g = torch.Generator(device=dev if dev.type == "cuda" else "cpu")
            li = int(module.layer_idx)
            g.manual_seed((self._ctx_seed ^ (self.seed * 7919) ^ (li * 104729)) & 0x7FFFFFFF)
            keep = torch.rand(r.shape, generator=g, dtype=torch.float64, device=dev) < r
            bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)), torch.tensor(-float("inf"), device=dev))

        module.e2_bias = bias.float()                     # [1, H, n], read by E2's kernel
        module.masked_key_indices = None
        if self.stats_log:
            self.stats_log[-1]["layers"].append({
                "layer": int(module.layer_idx), "kept_frac": keep.double().mean().item(),
                "expected_frac": r.mean().item(), "target_frac": int(n * (1 - self.compression_ratio)) / n,
                "c_median": float(c.median()) if c is not None else None})
        return keys, values                               # masked, not pruned
