"""End-to-end diagnostic: does the Bernoulli bias reach the attention kernel during GENERATION?

Uses the real kvpress pipeline on RULER niah samples. A counter sits OUTERMOST on the attention function
and records, per call, whether module.e2_bias is set and whether the call is a prefill (q==k) or not.
"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP); sys.path.insert(0, os.path.join(EXP, "..", "experiment1"))
sys.path.insert(0, os.path.expanduser("~/src/kvpress/evaluation"))
import pandas as pd
from pathlib import Path
from transformers import pipeline
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import kvpress  # registers the pipeline + kvpress's attention patch
from kvpress import KVzipPress, RandomPress
from bern_press import KVzipBernPress, bias_patched

MID = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline("kv-press-text-generation", model=MID, device_map="auto", trust_remote_code=True)
pipe.model.eval()
impl = pipe.model.config._attn_implementation
print("attn_implementation:", impl, "dtype:", pipe.model.dtype, flush=True)

snap = sorted(Path(os.environ["HF_HOME"], "hub", "datasets--simonjegou--ruler", "snapshots").iterdir())[-1]
df = pd.read_parquet(snap / "4096" / "test-00000-of-00001.parquet")
rows = df[df.task == "niah_single_1"].head(3)

C = {}
def counter(func):
    def w(module, q, k, v, attention_mask, dropout=0.0, **kw):
        b = getattr(module, "e2_bias", None)
        phase = "prefill" if q.shape[2] == k.shape[2] else ("qprefill" if q.shape[2] > 1 else "decode")
        key = (phase, b is not None)
        C[key] = C.get(key, 0) + 1
        if b is not None and phase != "prefill":
            C.setdefault("blocked_frac", []).append(torch.isinf(b).float().mean().item())
        return func(module, q, k, v, attention_mask, dropout, **kw)
    return w

def run(press, label):
    C.clear()
    out = []
    with bias_patched(pipe.model):
        inner = ALL_ATTENTION_FUNCTIONS[impl]
        ALL_ATTENTION_FUNCTIONS[impl] = counter(inner)          # outermost
        try:
            for _, r in rows.iterrows():
                a = pipe(r["context"], question=r["question"], answer_prefix=r["answer_prefix"],
                         press=press, max_new_tokens=int(r["max_new_tokens"]))["answer"]
                out.append((a.strip()[:50], list(r["answer"])))
        finally:
            ALL_ATTENTION_FUNCTIONS[impl] = inner
    bf = C.pop("blocked_frac", [])
    hits = sum(any(x.lower() in a.lower() for x in ref) for a, ref in out)
    print(f"\n== {label}: needle found {hits}/{len(out)}", flush=True)
    print("   kernel calls (phase, bias_set):", {f"{k[0]}|{k[1]}": v for k, v in C.items()})
    if bf:
        print(f"   bias blocked fraction during generation: mean {sum(bf)/len(bf):.3f}")
    for a, ref in out:
        print(f"   ans={a!r}  ref={ref}")
    if isinstance(press, KVzipBernPress) and press.stats_log:
        print("   stats kept_frac:", [round(s['kept_frac'], 4) for s in press.stats_log[-len(out):]])
    return hits

run(None, "no_press")
run(RandomPress(compression_ratio=0.99), "random 0.99 (positive control: must fail)")
run(KVzipPress(compression_ratio=0.99), "kvzip 0.99")
run(KVzipBernPress(method="kvzip_bern", seed=1, compression_ratio=0.99), "kvzip_bern 0.99 (must degrade)")
run(KVzipBernPress(method="kvzip_bern", seed=1, compression_ratio=0.875), "kvzip_bern 0.875")
print("\nDONE", flush=True)
