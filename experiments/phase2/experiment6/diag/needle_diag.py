"""Why does per-head Bernoulli fail niah_single_3 when global Bernoulli does not?

On E6's own failing samples (Llama-3.1-8B, cr=0.75, seed 1):
 1. uncompressed generation, recording each query head's attention mass on the needle (UUID) tokens while the
    answer is written -> the heads that actually copy the needle ("copy heads");
 2. which needle tokens each KV head keeps under global Bernoulli vs per-head Bernoulli (and their r);
 3. generations for: shipped bernoulli / E6 per-head / per-head forced to global c / per-head top-k.
    (per-head with global c must reproduce shipped bernoulli exactly; per-head top-k separates the equal
    allocation from the sampling within a head.)
"""
import ast, sys, json, torch, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import perhead_press as PH
from equivalence_helpers import PerHeadWithGlobalC, PerHeadTopK
from kvpress import BernoulliPress, KVzipPress
from kvpress.utils import get_cache_metadata

MODEL, CR, SEED, N = "meta-llama/Llama-3.1-8B-Instruct", 0.75, 1, 5
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa", dtype=torch.bfloat16, device_map="cuda").eval()
pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tok)
L, NKV, NQ = model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.num_attention_heads
G = NQ // NKV

# E6's failing samples: the per-head predictions file gives question+answer; match them back to the dataset.
res = Path.home() / "phase2/experiment6/results/r32k_llama_niah_single_3"
pred = pd.read_csv(next(res.glob("*perhead_bern_kvzip_s1__0.75*/predictions.csv")))
ds = load_dataset("xAlg-AI/att-hub-ruler-32k", "niah_single_3", split="niah_single_3").to_pandas()
ds["gold"] = ds.answer.map(lambda a: list(a)[0] if not isinstance(a, str) else ast.literal_eval(a)[0])
pred["gold"] = pred.answer.map(lambda a: ast.literal_eval(a.replace("' '", "', '"))[0])
rows = [ds[(ds.question == r.question) & (ds.gold == r.gold)].iloc[0] for r in pred.head(N).itertuples()]

# --- instrumentation -------------------------------------------------------------------------------
STATE = {"ctx_ids": None, "record": False, "span": None, "mass": []}
lm = model.model
def pre(mod, args, kwargs):
    if STATE["ctx_ids"] is None:
        STATE["ctx_ids"] = kwargs.get("input_ids", args[0] if args else None)[0].tolist()
lm.register_forward_pre_hook(pre, with_kwargs=True)

orig_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
def recording(module, query, key, value, attention_mask, dropout=0.0, **kw):
    if STATE["record"] and query.shape[2] == 1 and STATE["span"] is not None:
        a, b = STATE["span"]
        scale = kw.get("scaling") or module.scaling
        q = query[0, :, 0].float().view(NKV, G, -1)                           # [kv, g, d]
        logits = torch.einsum("hgd,hnd->hgn", q, key[0].float()) * scale       # [kv, g, k_len]
        p = logits.softmax(-1)
        STATE["mass"].append((int(module.layer_idx), p[..., a:b].sum(-1).reshape(NQ).cpu()))
    return orig_sdpa(module, query, key, value, attention_mask, dropout, **kw)
ALL_ATTENTION_FUNCTIONS["sdpa"] = recording

def stash(press):
    """Keep a copy of the attention bias the press writes (the pipeline's cache is not reachable afterwards)."""
    inner = press.compress_post
    def wrapped(m):
        inner(m)
        press._bias = {k: v.detach().clone() for k, v in get_cache_metadata(press.press._cache)["attention_bias"].items()}
    press.compress_post = wrapped
    return press

def needle_span(ctx_ids, context, gold):
    raw = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    cs = context.index(gold); ce = cs + len(gold)
    idx = [i for i, (s, e) in enumerate(raw.offset_mapping) if s < ce and e > cs]
    pat = raw.input_ids[idx[0]: idx[-1] + 1]
    for a in range(len(ctx_ids) - len(pat) + 1):
        if ctx_ids[a: a + len(pat)] == pat:
            return a, a + len(pat)
    raise RuntimeError("needle span not found in the context ids")

def gen(press, row):
    STATE["ctx_ids"] = None
    out = pipe(row.context, questions=[row.question], answer_prefix=row.answer_prefix, press=press,
               max_new_tokens=int(row.max_new_tokens))
    return out["answers"][0]

report = []
for si, row in enumerate(rows):
    # 1. uncompressed, recording needle attention mass during the answer
    STATE.update(record=False, span=None, mass=[])
    STATE["ctx_ids"] = None
    # need the span before recording: one cheap pass to get ctx_ids via a no-press short generation
    _ = pipe(row.context, questions=[row.question], answer_prefix=row.answer_prefix, press=None, max_new_tokens=1)
    span = needle_span(STATE["ctx_ids"], row.context, row.gold)
    STATE.update(record=True, span=span, mass=[])
    dense_answer = gen(None, row)
    STATE["record"] = False
    steps = min(24, len(STATE["mass"]) // L)                                   # the UUID is written in these
    mass = torch.stack([m for _, m in STATE["mass"][: steps * L]]).view(steps, L, NQ).mean(0)   # [L, NQ]
    copy = (mass >= 0.2).nonzero().tolist()                                    # (layer, q_head)

    # 2 + 3. compressed variants
    variants = {
        "bernoulli (shipped)": stash(BernoulliPress(press=KVzipPress(compression_ratio=CR), seed=SEED)),
        "perhead (E6)": stash(PH.PerHeadBernoulliPress(press=KVzipPress(compression_ratio=CR), seed=SEED)),
        "perhead forced to global c": stash(PerHeadWithGlobalC(press=KVzipPress(compression_ratio=CR), seed=SEED)),
        "perhead top-k (no sampling)": stash(PerHeadTopK(press=KVzipPress(compression_ratio=CR), seed=SEED)),
    }
    a, b = span
    entry = {"sample": si, "gold": row.gold, "span_tokens": b - a, "dense_answer": dense_answer[:60],
             "n_copy_heads": len(copy), "copy_heads": copy[:12], "variants": {}}
    for name, press in variants.items():
        ans = gen(press, row)
        kept = torch.stack([torch.isfinite(press._bias[l][0, :, a:b]) for l in range(L)]).float()   # [L, kv, span]
        allk = torch.stack([torch.isfinite(press._bias[l][0]) for l in range(L)]).float()
        ch = [kept[l, h // G].mean().item() for l, h in copy] or [float("nan")]
        entry["variants"][name] = {
            "answer": ans[:60], "correct": row.gold in ans,
            "overall_kept": round(allk.mean().item(), 4),
            "needle_kept_all_heads": round(kept.mean().item(), 4),
            "needle_kept_in_copy_heads": round(sum(ch) / len(ch), 4),
            "copy_heads_missing_any_needle_token": sum(x < 1.0 for x in ch),
        }
    report.append(entry)
    print(json.dumps(entry, indent=1), flush=True)

json.dump(report, open(HERE / "needle_diag.json", "w"), indent=1)
