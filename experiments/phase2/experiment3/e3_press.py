"""E3's presses: experiment2's verified press + a per-context score cache + Stage 2 statistics.

experiment2/bern_press.py is left untouched so E2 stays reproducible. Everything here is additive.

**Why the score cache matters.** KVzip's score is a deterministic function of the context, but kvpress
recomputes it on every run. At ~124k tokens the probe measured 55 s for a whole context with no press
and 259 s with KVzip, so calibration is ~200 s and dominates. The full matrix has 16 calibrated runs
per context (kvzip x 4 ratios + 3 seeds x 4 ratios), all recomputing the identical tensor. Caching it
turns ~78 min per context into ~28 min.

It also strengthens the comparison: `kvzip` (top-k) and `kvzip_bern` (sampling) then demonstrably read
the *same bytes*, so B1 differs in selection alone. Set E3_NO_SCORE_CACHE=1 to bypass, which is how the
cached and uncached paths are verified to agree.
"""

import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment2"))
from bern_press import KVzipBernPress, bias_patched  # noqa: E402,F401  (bias_patched re-exported)

from kvpress.presses.kvzip_press import KVzipPress  # noqa: E402

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "score_cache")


class ScoreCacheMixin:
    """Cache KVzip's `score_val` per context, skipping the repeat passes when it is already on disk."""

    def _score_cache_path(self, model):
        ids = self._context_ids.detach().cpu().numpy().tobytes()
        key = hashlib.sha1(ids).hexdigest()[:24]
        name = str(getattr(model.config, "name_or_path", "model")).replace("/", "--")
        return os.path.join(CACHE_DIR, f"{name}_{key}_sink{self.n_sink}_n{self._context_ids.shape[1]}.pt")

    def _perform_kvzip_compression(self, model, tokenizer):
        if os.environ.get("E3_NO_SCORE_CACHE"):
            return super()._perform_kvzip_compression(model, tokenizer)

        self.context_length = self._context_ids.shape[1]
        path = self._score_cache_path(model)
        if os.path.exists(path):
            sv = torch.load(path, map_location=model.device)
            expected = (model.config.num_hidden_layers, 1, model.config.num_key_value_heads, self.context_length)
            assert tuple(sv.shape) == expected, f"cached score shape {tuple(sv.shape)} != {expected}"
            self.score_val = sv.to(model.dtype)
            self.compress_post(model)          # the parent's last step; the repeat passes are skipped
            return

        super()._perform_kvzip_compression(model, tokenizer)     # computes score_val, then compress_post
        if self.score_val is not None:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"                    # atomic: parallel jobs share this cache
            torch.save(self.score_val.detach().cpu(), tmp)
            os.replace(tmp, path)


class CachedKVzipPress(ScoreCacheMixin, KVzipPress):
    """The top-k control arm, reading the same cached scores as the sampling arm."""


class E3BernPress(ScoreCacheMixin, KVzipBernPress):
    """Bernoulli sampling + cached scores + the score statistics the Stage 2 analysis needs.

    Recorded per context (aggregated over the 256 KV heads):
        T_h/n        the derivation's T per head, normalised. E1 found certified memory is essentially
                     a function of T/n, so whether T/n falls at ~110k is E3's central question.
        heavy_frac   |H(c)|/n, tokens kept outright because c*tau >= 1
        nu_L         tail mass, sum over the tail of (1 - c tau) tau
    """

    def compress_post(self, model):
        super().compress_post(model)
        if self.compression_ratio <= 0 or not self.stats_log or self.score_val is None:
            return
        s = self.score_val.double()                       # [NL, 1, NKV, n]
        NL, _, NKV, n = s.shape
        c = self.stats_log[-1].get("c")
        per_head_T = s.sum(dim=-1).reshape(-1) / n
        rec = {"n_ctx": int(n), "n_heads": int(NL * NKV),
               "T_over_n_median": float(per_head_T.median()),
               "T_over_n_p90": float(torch.quantile(per_head_T, 0.9)),
               "T_over_n_max": float(per_head_T.max())}
        if c:
            r = (c * s).clamp(max=1.0)
            heavy = (r >= 1.0).double().sum(dim=-1).reshape(-1) / n
            tail = ((1.0 - c * s).clamp_min(0.0) * s * (r < 1.0).double()).sum(dim=-1).reshape(-1)
            rec.update({"heavy_frac_median": float(heavy.median()),
                        "heavy_frac_p90": float(torch.quantile(heavy, 0.9)),
                        "nu_L_median": float(tail.median())})
        self.stats_log[-1].update(rec)
