"""Is perhead_press.py plumbing-identical to the shipped BernoulliPress?

Force PerHeadBernoulliPress to use the shipped global c and the shipped seed source. If its attention bias is
then bit-identical to BernoulliPress's on every layer, the per-head c is the ONLY difference between the arms.
Also checks whether the unpatched seed source (self._context_crc) differs across contexts.
"""
import sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perhead_press as PH
from kvpress import BernoulliPress, KVzipPress
from kvpress.presses.bernoulli_press import _keep_prob, _solve_c, transform_scores_for_sampling
from kvpress.utils import get_cache_metadata

CKPT, CR, SEED = "maxjeblick/llama2-0b-unit-test", 0.75, 1
tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, attn_implementation="sdpa", dtype=torch.float32).eval()


def run(press, text):
    ids = tok(text, return_tensors="pt").input_ids
    cache = DynamicCache()
    with torch.no_grad(), press(model):
        model(input_ids=ids, past_key_values=cache)
    return {k: v.clone() for k, v in get_cache_metadata(cache)["attention_bias"].items()}


def shipped_global_c(scores, cr, n_sink):
    """The shipped BernoulliPress.compress_post's c, copied line for line."""
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
    def compress_post(self, model):
        c = shipped_global_c(self.press.score_val, self.compression_ratio, self.press.n_sink)
        orig = PH._solve_c_per_head
        PH._solve_c_per_head = lambda free, k_free: torch.full((free.shape[0],), c, dtype=torch.float64, device=free.device)
        self._context_crc = None                       # -> falls back to crc32(_context_ids), the shipped source
        try:
            super().compress_post(model)
        finally:
            PH._solve_c_per_head = orig


texts = ["The quick brown fox jumps over the lazy dog. " * 30,
         "A completely different context about ships, harbours and tides. " * 25]
for i, text in enumerate(texts):
    a = run(BernoulliPress(press=KVzipPress(compression_ratio=CR), seed=SEED), text)
    b = run(PerHeadWithGlobalC(press=KVzipPress(compression_ratio=CR), seed=SEED), text)
    same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    kept = torch.cat([torch.isfinite(v).flatten() for v in a.values()]).float().mean().item()
    print(f"context {i}: layers={len(a)}  kept={kept:.3f}  per-head press with global c == BernoulliPress: {same}")
    assert same, "PLUMBING DIFFERS"

# Seed source of the UNPATCHED per-head press: is it the context, or the last forward pass?
crcs = []
for text in texts:
    p = PH.PerHeadBernoulliPress(press=KVzipPress(compression_ratio=CR), seed=SEED)
    captured = {}
    orig_cp = p.compress_post
    p.compress_post = lambda m, p=p: (captured.setdefault("crc", p._context_crc), orig_cp(m))
    run(p, text)
    crcs.append(captured["crc"])
print(f"unpatched per-head seed source across the two contexts: {crcs} -> "
      f"{'SAME seed for different contexts' if crcs[0] == crcs[1] else 'differs per context'}")
