"""prefix_suffix: are the gold word's tokens actually dropped by global top-k and kept by Bernoulli?"""
import glob, hashlib, json, os, sys
import numpy as np, torch
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
E3 = "/home/rethinkingai-self/23b1835/phase2/experiment3/"
sys.path.insert(0, E3)
import scbench_data
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
ds = scbench_data.load_scbench(data_dir="prefix_suffix") if False else None
os.environ["E3_N_CTX"] = "10"
ds = scbench_data.load_scbench(data_dir="prefix_suffix").to_pandas()
C = {}
for f in glob.glob(E3 + "results/S1n10/*prefix_suffix*kvzip_bern_s1__*/memory.json"):
    r = f.split("__")[-1].split("/")[0]
    for m in json.load(open(f)): C.setdefault(m["n"], {})[r] = m["c"]
sep = "\n" + "#" * len("dummy context")
out = []
for cid, g in ds.groupby("context_id"):
    context = g.context.iloc[0]
    templ = tok.apply_chat_template([{"role": "user", "content": context + sep}], add_generation_prompt=True, tokenize=False)
    ctx_text = templ.split(sep)[0]
    enc = tok(ctx_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc.input_ids])
    key = hashlib.sha1(ids.numpy().tobytes()).hexdigest()[:24]
    path = glob.glob(E3 + f"out/score_cache/*_{key}_sink4_n{ids.shape[1]}.pt")
    if not path:
        print("NO CACHE MATCH for context", cid, ids.shape); continue
    s = torch.load(path[0]).float()[:, 0]                       # [32, 8, n]
    flat = s.reshape(-1); n = s.shape[-1]
    thr = {r: torch.topk(flat, int(round((1 - k) * flat.numel()))).values[-1] for r, k in (("0.75", .75), ("0.88", .875))}
    offs = np.array(enc.offset_mapping)
    for word in g.answer:
        a = ctx_text.find(word); b = a + len(word)
        span = np.where((offs[:, 0] < b) & (offs[:, 1] > a))[0]
        ss = s[:, :, span]                                       # [32, 8, len]
        rec = {"cid": cid, "word": word, "ntok": len(span), "pos": a / len(ctx_text)}
        for r in thr:
            keep_tk = ss >= thr[r]
            rr = (C[n][r] * ss).clamp(max=1)
            rec[r] = {"topk_all_kept_heads": float(keep_tk.all(-1).float().mean()),        # heads holding the whole word
                      "topk_tok_kept": float(keep_tk.float().mean()),
                      "topk_interior_kept": float(keep_tk[..., 1:-1].float().mean()),
                      "topk_edge_kept": float(keep_tk[..., [0, -1]].float().mean()),
                      "bern_all_kept_heads": float(rr.prod(-1).mean()),                      # expected
                      "bern_tok_kept": float(rr.mean()),
                      "bern_interior_kept": float(rr[..., 1:-1].mean()),
                      "ctx_topk_kept": 1 - float(r.replace("0.88", "0.875")),
                      "topk_none_heads": float((~keep_tk.any(-1)).float().mean()),
                      "bern_anykept_in_topk_none": float((1 - (1 - rr).prod(-1))[~keep_tk.any(-1)].mean()) if (~keep_tk.any(-1)).any() else float("nan"),
                      "bern_ntok_in_topk_none": float(rr.sum(-1)[~keep_tk.any(-1)].mean()) if (~keep_tk.any(-1)).any() else float("nan"),
                      "bern_none_heads": float((1 - rr).prod(-1).mean())}
        out.append(rec)
    print("ctx", cid, "cache matched, n", n, flush=True)
json.dump(out, open("span2.json", "w"), indent=1)
for r in ("0.75", "0.88"):
    print(r, {k: round(float(np.mean([o[r][k] for o in out])), 4) for k in out[0][r]})
print("tokens per word", np.mean([o["ntok"] for o in out]), "depth range", min(o["pos"] for o in out), max(o["pos"] for o in out))
