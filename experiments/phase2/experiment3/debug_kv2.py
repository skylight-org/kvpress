"""What is scbench_kv's actual no_press accuracy? Decides whether the task carries signal.

One prefill, many queries (kvpress supports questions=[...]): the 5 real turns plus probe keys drawn at
known depths through the context.
"""
import json, re, sys
sys.path.insert(0, '.')
from transformers import pipeline
import kvpress  # noqa: F401
from scbench_data import snapshot_dir

MID = "meta-llama/Llama-3.1-8B-Instruct"
rows = [json.loads(l) for l in open(snapshot_dir() / "scbench_kv.jsonl")][:3]
pipe = pipeline("kv-press-text-generation", model=MID, device_map="auto", trust_remote_code=True)
pipe.model.eval()

real_hits = probe_hits = real_n = probe_n = 0
for ci, row in enumerate(rows):
    ctx, turns = row["context"], row["multi_turns"]
    pairs = re.findall(r'"([0-9a-f-]{36})":\s*"([0-9a-f-]{36})"', ctx)
    depths = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    probes = [pairs[int(d * (len(pairs) - 1))] for d in depths]
    qs = [t["input"] for t in turns] + [f'\nKey: "{k}"\nThe value associated with the specified key is: ' for k, _ in probes]
    golds = [t["answer"] for t in turns] + [v for _, v in probes]
    ans = pipe(ctx, questions=qs, press=None, max_new_tokens=48)["answers"]
    ok = [g.lower() in a.lower() for g, a in zip(golds, ans)]
    real_hits += sum(ok[:len(turns)]); real_n += len(turns)
    probe_hits += sum(ok[len(turns):]); probe_n += len(probes)
    print(f"ctx{ci}: real turns {sum(ok[:len(turns)])}/{len(turns)} | "
          f"probes by depth " + " ".join(f"{d:.2f}:{int(o)}" for d, o in zip(depths, ok[len(turns):])), flush=True)
    # where do the real keys sit?
    keys = [re.search(r'"([0-9a-f-]{36})"', t["input"]).group(1) for t in turns]
    pos = [next((i / len(pairs) for i, (k, _) in enumerate(pairs) if k == kk), None) for kk in keys]
    print(f"      real-key depths: {[round(p, 2) if p is not None else None for p in pos]}", flush=True)

print(f"\nTOTAL no_press: real turns {real_hits}/{real_n} = {100*real_hits/real_n:.0f}%   "
      f"depth probes {probe_hits}/{probe_n} = {100*probe_hits/probe_n:.0f}%")
