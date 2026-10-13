"""Offline replacement for ``datasets.load_dataset``, used by experiment4 (LongBench).

Copied verbatim from ``~/e8/run_eval.py`` on the machine the experiments were run on, so the rows
are exactly the ones experiment 4 evaluated. It lives here because that file also imports an
unrelated project (SensZip), which is not part of this work.

Compute nodes on the cluster have no internet, and datasets 4.x cannot find a cached ``data_dir``
subset in offline mode, so the parquet files are read straight out of the hub snapshot that was
downloaded on the login node.
"""

import os


def _load_local_parquet(path, data_dir=None, split="test"):
    """Offline replacement for datasets.load_dataset.

    Compute nodes have no internet, and datasets 4.x cannot find a cached `data_dir` subset in
    offline mode ("Couldn't find cache ... for config 'default-data_dir=qasper'"). So read the
    parquet file straight out of the hub snapshot that was downloaded on the login node.
    """
    from datasets import Dataset
    from huggingface_hub import snapshot_download

    root = snapshot_download(path, repo_type="dataset", local_files_only=True)
    sub = os.path.join(root, data_dir) if data_dir else root
    files = sorted(f for f in os.listdir(sub) if f.startswith(split) and f.endswith(".parquet"))
    assert files, f"no {split}*.parquet in {sub}"
    return Dataset.from_parquet([os.path.join(sub, f) for f in files])
