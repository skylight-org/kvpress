"""Stage 0: what does SCBench actually contain, in Llama-3.1 tokens?

No truncation is allowed in E3, so the run matrix is decided by how many samples fit under the model's
131,072-token limit. This script tokenizes every context and every query of the five chosen tasks and
reports the distribution, the per-task budget, and how many samples must be DROPPED (never cut).

The usable context budget is smaller than the model limit:

    n_ctx + n_query + max_new_tokens + chat-template overhead  <=  131072

    python measure_lengths.py
"""

import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MID = "meta-llama/Llama-3.1-8B-Instruct"
TASKS = ["kv", "prefix_suffix", "vt", "qa_eng", "summary"]
MODEL_MAX = 131072
# generation budget per task (SCBench answers: uuids and words are short, summaries are long)
MAX_NEW = {"kv": 64, "prefix_suffix": 64, "vt": 128, "qa_eng": 64, "summary": 512}
HERE = os.path.dirname(os.path.abspath(__file__))


def snapshot():
    """Locate the SCBench snapshot, tolerating the 'incomplete snapshot' state that allow_patterns
    leaves behind (AUDIT #4: snapshot_download refuses it offline even though the files are present)."""
    root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snaps = root / "hub" / "datasets--microsoft--SCBench" / "snapshots"
    return sorted(snaps.iterdir(), key=lambda d: d.stat().st_mtime)[-1] / "data"


def context_of(row):
    """kv/prefix_suffix/qa_eng/summary store the shared context in 'context'; vt stores it in 'input'."""
    return row["context"] if "context" in row else row["input"]


def main():
    tok = AutoTokenizer.from_pretrained(MID)
    # chat-template overhead, measured rather than guessed
    empty = tok.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    overhead = len(empty)
    d = snapshot()
    out, summary = {}, []

    for t in TASKS:
        rows = [json.loads(l) for l in open(d / f"scbench_{t}.jsonl")]
        ctx_tokens = [len(tok(context_of(r), add_special_tokens=False).input_ids) for r in rows]
        q_tokens = [[len(tok(x["input"], add_special_tokens=False).input_ids) for x in r["multi_turns"]] for r in rows]
        turns = [len(r["multi_turns"]) for r in rows]
        # a sample fits only if its longest turn fits
        need = [c + max(q) + MAX_NEW[t] + overhead for c, q in zip(ctx_tokens, q_tokens)]
        fits = [i for i, nd in enumerate(need) if nd <= MODEL_MAX]
        c = np.array(ctx_tokens)
        out[t] = {"n_samples": len(rows), "turns_per_sample": int(np.median(turns)),
                  "ctx_tokens": {"min": int(c.min()), "p25": int(np.percentile(c, 25)),
                                 "median": int(np.median(c)), "p75": int(np.percentile(c, 75)),
                                 "max": int(c.max())},
                  "max_query_tokens": int(max(max(q) for q in q_tokens)),
                  "max_new_tokens": MAX_NEW[t], "template_overhead": overhead,
                  "n_fits": len(fits), "n_dropped": len(rows) - len(fits),
                  "fits_idx": fits,
                  "fitting_ctx_tokens": {"min": int(c[fits].min()), "median": int(np.median(c[fits])),
                                         "max": int(c[fits].max())} if fits else None}
        r = out[t]
        summary.append(f"{t:14s} n={r['n_samples']:4d} turns={r['turns_per_sample']}  ctx tokens "
                       f"min {r['ctx_tokens']['min']:7d} med {r['ctx_tokens']['median']:7d} max {r['ctx_tokens']['max']:7d}"
                       f"  | FITS {r['n_fits']:4d}  DROP {r['n_dropped']:4d}"
                       + (f"  fitting ctx med {r['fitting_ctx_tokens']['median']:7d}" if fits else ""))
        print(summary[-1], flush=True)

    tot_fit = sum(v["n_fits"] for v in out.values())
    tot = sum(v["n_samples"] for v in out.values())
    print(f"\nTOTAL fitting samples {tot_fit}/{tot}; queries = {sum(v['n_fits'] * v['turns_per_sample'] for v in out.values())}")
    print(f"chat-template overhead {overhead} tokens; model max {MODEL_MAX}")
    json.dump(out, open(os.path.join(HERE, "out", "lengths.json"), "w"), indent=1)
    print("written out/lengths.json")


if __name__ == "__main__":
    main()
