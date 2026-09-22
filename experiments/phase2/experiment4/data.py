"""One offline `datasets.load_dataset` replacement for both benchmarks (monkeypatched in run_sweep.py).

  LongBench -> E8's hub-parquet reader (~/e8/run_eval.py), the loader behind E8's runs
  SCBench   -> experiment3/scbench_data.load_scbench (untruncated, contexts limited by E3_N_CTX)

Both are imported, never copied, so the rows are exactly the ones E8 and E3 used.
"""

import os
import sys

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "phase2", "experiment3"))
sys.path.insert(0, os.path.join(HOME, "e8"))

import scbench_data  # noqa: E402

try:  # the machine the experiments ran on: E8's own loader, so the rows are byte-for-byte E8's
    from run_eval import _load_local_parquet  # noqa: E402
except ImportError:  # a fresh clone of this repo: the vendored copy of that same function
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
    from local_parquet import _load_local_parquet  # noqa: E402

# E8 has its own run_sweep.py / summarize.py: take its folder off the path again so later imports
# cannot pick those up.
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != os.path.join(HOME, "e8")]


def load(path, data_dir=None, split="test", **kw):
    if "SCBench" in str(path):
        return scbench_data.load_scbench(path, data_dir=data_dir, split=split, **kw)
    return _load_local_parquet(path, data_dir=data_dir, split=split)
