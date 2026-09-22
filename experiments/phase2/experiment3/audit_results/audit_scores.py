"""Score-cache analysis: what are the T/n~1 heads, and how do top-k vs Bernoulli spend a global budget?"""
import glob, json, re, torch, numpy as np
torch.set_num_threads(8)
E3 = "/home/rethinkingai-self/23b1835/phase2/experiment3/"
# n -> (task, {ratio: c}) from memory.json of seed 1
N2 = {}
for f in glob.glob(E3 + "results/S1n10/*kvzip_bern_s1__*/memory.json"):
    task = f.split("scbench__")[1].split("__")[0]; ratio = f.split("__")[-1].split("/")[0]
    for m in json.load(open(f)):
        N2.setdefault(m["n"], [task, {}])[1][ratio] = m["c"]
rows = []
for p in sorted(glob.glob(E3 + "out/score_cache/*.pt")):
    n = int(re.search(r"_n(\d+)\.pt", p).group(1))
    if n not in N2: continue
    task, cs = N2[n]
    s = torch.load(p).float()[:, 0]                       # [32, 8, n]
    T = s.sum(-1) / n                                      # [32, 8]
    hi = (s > 0.5).float().mean(-1)                        # fraction of tokens with score > 0.5
    flat = s.reshape(-1)
    rec = {"task": task, "n": n, "T": T.numpy(), "hi": hi.numpy(), "zero_frac": float((flat == 0).float().mean())}
    for ratio, key in (("0.75", "4x"), ("0.88", "8x")):
        keep = 1 - float(ratio.replace("0.88", "0.875"))
        k = int(round(keep * flat.numel()))
        thr = torch.topk(flat, k).values[-1]
        topk_head = (s >= thr).float().mean(-1)            # per-head kept fraction under global top-k
        c = cs.get(ratio)
        bern_head = (c * s).clamp(max=1).mean(-1) if c else None
        rec[key] = (topk_head.numpy(), bern_head.numpy() if c is not None else None)
    rows.append(rec); print(task, n, "done", flush=True)
np.save("score_rows.npy", rows, allow_pickle=True)
