"""SCBench loader for kvpress — natural contexts, NO truncation.

SCBench ships one long shared context per sample plus `multi_turns` (5 queries). kvpress's runner groups
rows by the context string and sends all of a context's questions through one prefill + one compression
(`evaluate.py` `_run_inference`), which is exactly the method's deployment setting and what makes ~110k
affordable.

Rules this loader enforces (E3 DESIGN):
  * `max_context_length` is never used; nothing is ever truncated.
  * A sample whose longest turn would exceed the model's 131,072-token limit is DROPPED and COUNTED,
    using the fits list precomputed by `measure_lengths.py` (out/lengths.json) so we do not re-tokenize
    ~170 MB of text on every job.
  * Every turn of a sample carries the identical context string (kvpress groups on it) and an identical
    answer_prefix (the runner asserts this).

Environment:
  E3_N_CTX  limit the number of contexts per task (deterministic: first N of the fitting ones)
"""

import json
import os
from pathlib import Path

from datasets import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ("kv", "prefix_suffix", "vt", "qa_eng", "summary")   # kv is EXCLUDED from E3 runs: see DESIGN
# generation budget per task; must match measure_lengths.py, which sized the drop list with it
MAX_NEW = {"kv": 64, "prefix_suffix": 64, "vt": 128, "qa_eng": 64, "summary": 256}


def snapshot_dir():
    """Locate the SCBench snapshot directly.

    `snapshot_download(..., local_files_only=True)` refuses a snapshot fetched with `allow_patterns`,
    calling it incomplete, even though the files we need are present (AUDIT #4 in experiment2)."""
    root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snaps = root / "hub" / "datasets--microsoft--SCBench" / "snapshots"
    assert snaps.exists(), f"SCBench not downloaded: {snaps}"
    return sorted(snaps.iterdir(), key=lambda d: d.stat().st_mtime)[-1] / "data"


def context_of(row):
    """kv / prefix_suffix / qa_eng / summary keep the context in 'context'; vt keeps it in 'input'."""
    return row["context"] if "context" in row else row["input"]


def load_scbench(path=None, data_dir=None, split="test", **kw):
    """Return a Dataset with kvpress's expected columns for one SCBench task.

    Signature matches `datasets.load_dataset` because it is monkeypatched over it in run_sweep.py."""
    task = str(data_dir)
    assert task in TASKS, f"unknown SCBench task {task!r}"
    rows = [json.loads(l) for l in open(snapshot_dir() / f"scbench_{task}.jsonl")]

    lengths = json.load(open(os.path.join(HERE, "out", "lengths.json")))
    fits = lengths[task]["fits_idx"]                      # indices that fit under the model limit
    limit = int(os.environ.get("E3_N_CTX", "0"))
    if limit > 0:
        fits = fits[:limit]

    recs = []
    for ci in fits:
        r = rows[ci]
        ctx = context_of(r)
        for ti, turn in enumerate(r["multi_turns"]):
            ans = turn["answer"]
            recs.append({
                "context": ctx,
                "question": turn["input"],
                "answer_prefix": "",                      # SCBench prompts already end with their cue
                # lists (vt) are JSON-encoded so the column stays a flat string through predictions.csv
                "answer": json.dumps(ans) if isinstance(ans, (list, tuple)) else str(ans),
                "task": task,
                "context_id": int(ci),
                "turn": ti,
                "max_new_tokens": MAX_NEW[task],
            })
    print(f"[scbench] {task}: {len(fits)} contexts x turns -> {len(recs)} queries "
          f"(dropped {lengths[task]['n_dropped']} over-length samples of {lengths[task]['n_samples']})",
          flush=True)
    return Dataset.from_list(recs)
