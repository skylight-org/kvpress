"""Is scbench_kv's zero score a harness bug or a model limitation?

Same pipeline, same data, three context sizes. If short contexts answer correctly, the prompt handling
is right and the 8B model simply cannot retrieve a uuid from ~124k tokens of uuid pairs.
Truncation here is a DIAGNOSTIC only; the experiment itself never truncates.
"""
import json, os, sys, re
sys.path.insert(0, '.')
from transformers import pipeline, AutoTokenizer
import kvpress  # noqa: F401  (registers the pipeline)
from scbench_data import snapshot_dir

MID = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
row = json.loads(open(snapshot_dir() / "scbench_kv.jsonl").readline())
ctx, turns = row["context"], row["multi_turns"]
pipe = pipeline("kv-press-text-generation", model=MID, device_map="auto", trust_remote_code=True)
pipe.model.eval()

pairs = re.findall(r'"([0-9a-f-]{36})":\s*"([0-9a-f-]{36})"', ctx)
print(f"context has {len(pairs)} kv pairs, {len(tok(ctx).input_ids)} tokens", flush=True)

for n_tok in (4000, 16000, None):
    if n_tok is None:
        sub, label = ctx, "FULL"
    else:
        ids = tok(ctx, add_special_tokens=False).input_ids[:n_tok]
        sub, label = tok.decode(ids), f"{n_tok//1000}k"
    # a key that is definitely inside this prefix: take one from the first 40% of the retained text
    inside = [p for p in re.findall(r'"([0-9a-f-]{36})":\s*"([0-9a-f-]{36})"', sub)]
    k, v = inside[len(inside)//3]
    q = f'\nKey: "{k}"\nThe value associated with the specified key is: '
    ans = pipe(sub, question=q, press=None, max_new_tokens=48)["answer"]
    print(f"{label:5s} pairs_in_prefix={len(inside):5d}  correct={v.lower() in ans.lower()}  pred={ans.strip()[:80]!r}", flush=True)

# and the original question of the full-context sample, for reference
ans = pipe(ctx, question=turns[0]["input"], press=None, max_new_tokens=48)["answer"]
print(f"orig  gold={turns[0]['answer']}  correct={turns[0]['answer'].lower() in ans.lower()}  pred={ans.strip()[:80]!r}")
