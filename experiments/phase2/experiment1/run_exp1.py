"""Phase 2, Experiment 1 (GPU): how vacuous is the denominator-only Bernoulli bound?

Per context (DESIGN.md §3-4):
  1. prefill chat(context) -> the context tokens C, post-RoPE keys kept for every layer
  2. repeat pass ("Repeat the previous context exactly." + context) on the same cache; N_CAL evenly
     spaced repeat positions give the calibration queries; the cache is then cropped back to C
  3. question + greedy answer with the full cache -> the FUTURE queries (never seen by calibration),
     at answer steps STEPS, at their true positions
  4. all KV heads: tau (max over calibration queries AND the 4 query heads sharing the KV head),
     T, and certified memory at eps in EPSS
  5. MC_LAYERS x MC_KV heads: Bernoulli Monte Carlo at every budget of the two parts ->
     Part 1: 2x and 4x compression
     Part 2: the eps = 0.1 ladder, at M*, M*/2, M*/4
     failure rates measured on calibration queries (theorem) and future queries (transfer),
     against both the budget's own certificate and the original eps = 0.1

Nothing is deleted: every key stays, so B and B_hat are both computed exactly for each draw.

    python run_exp1.py [--smoke]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import denom  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

MID = "meta-llama/Llama-3.1-8B-Instruct"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "passage_retrieval_en"]
PER_TASK, MAX_CTX, MIN_CTX = 1, 11000, 1500          # 4 contexts; 11k admits passage_retrieval_en (shortest is 10.4k)
N_CAL, STEPS = 512, (1, 2, 4, 8, 16)
N_CALTEST = 32                                       # calibration queries held out as the theorem test
DELTA, DELTA_N = 0.05, 0.05
EPS_TARGET = 0.1                                     # Part 2's epsilon
EPSS = (0.1, 0.2, 0.3)                               # certified memory reported for these
COMP = (2.0, 4.0)                                    # Part 1 compressions
MC_LAYERS = list(range(0, 32, 4))                    # 8 layers
MC_KV = [0, 2, 4, 6]                                 # 4 KV heads each -> 32 KV heads per context
TRIALS = 300                                         # SE at delta=0.05 is 1.3%
if args.smoke:
    PER_TASK, MC_LAYERS, MC_KV, TRIALS, N_CAL = 1, [0, 16], [0], 100, 256
    TASKS = TASKS[:1]

OUT = os.path.join(HERE, "out", "smoke" if args.smoke else "")
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

tok = AutoTokenizer.from_pretrained(MID)
root = snapshot_download("Xnhyacinth/LongBench", repo_type="dataset", local_files_only=True)


def chat_split(context):
    """Split the chat template around the context, so the context tokens can be isolated."""
    sep = "#" * (len(context) + 10)
    s = tok.apply_chat_template([{"role": "user", "content": context + sep}], add_generation_prompt=True, tokenize=False)
    ctx, suffix = s.split(sep)
    prefix = tok.apply_chat_template([{"role": "user", "content": "dummy context" + "#"}], add_generation_prompt=True,
                                     tokenize=False).split("dummy context")[0]
    return ctx, suffix, prefix


# ---------------------------------------------------------------- context selection (deterministic, seed 9)
sel_path = os.path.join(OUT, "contexts.json")
if os.path.exists(sel_path):
    CTX = json.load(open(sel_path))
else:
    CTX, rng = [], np.random.default_rng(9)
    for task in TASKS:
        df = pd.read_parquet(os.path.join(root, task, "test-00000-of-00001.parquet"))
        got = 0
        for i in rng.permutation(len(df)):
            row = df.iloc[int(i)]
            ctx, _, _ = chat_split(row["context"])
            n = len(tok.encode(ctx, add_special_tokens=False))
            if MIN_CTX <= n <= MAX_CTX:
                CTX.append({"task": task, "row": int(i), "n_ctx": n})
                got += 1
                if got == PER_TASK:
                    break
    json.dump(CTX, open(sel_path, "w"), indent=1)
log(f"{len(CTX)} contexts: " + ", ".join(f"{c['task']}:{c['n_ctx']}" for c in CTX))

model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
cfg = model.config
NL, NH, NKV, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
GQ = NH // NKV                                        # query heads per KV head
SC = model.model.layers[0].self_attn.scaling
STOP = [i for i in set(tok.convert_tokens_to_ids(["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"])) if i is not None]

CAP = {"pos": None, "q": {}}


def q_hook(l):
    def h(mod, inp, out):
        if CAP["pos"] is not None:
            CAP["q"].setdefault(l, []).append(out[0, CAP["pos"]].detach().view(-1, NH, HD))
    return h


for l, layer in enumerate(model.model.layers):
    layer.self_attn.q_proj.register_forward_hook(q_hook(l))


@torch.no_grad()
def fwd(ids, cache, pos=None):
    CAP["pos"], CAP["q"] = pos, {}
    out = model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    CAP["pos"] = None
    return out


def rope(q_pre, positions):
    """q_pre [Q, NH, HD], positions [Q] -> post-RoPE, scaled by 1/sqrt(d), float64.

    Keeping the 1/sqrt(d) inside the query is what makes logits = q @ k.T directly."""
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(torch.zeros(1, 1, 1, device=dev), positions[None].to(dev))
    cos, sin = cos[0].double()[:, None, :], sin[0].double()[:, None, :]
    q = q_pre.double()
    return SC * (q * cos + torch.cat((-q[..., HD // 2:], q[..., :HD // 2]), -1) * sin)


# ---------------------------------------------------------------- per-context work
cert_rows, mc_rows, meta = [], [], []
gen_mc = torch.Generator(device=dev).manual_seed(2026)

for ci, cinfo in enumerate(CTX):
    tag = f"ctx{ci:02d}"
    df = pd.read_parquet(os.path.join(root, cinfo["task"], "test-00000-of-00001.parquet"))
    row = df.iloc[cinfo["row"]]
    ctx_text, suffix, prefix = chat_split(row["context"])
    ctx_ids = tok.encode(ctx_text, add_special_tokens=False)
    pre_len = len(tok.encode(prefix, add_special_tokens=False))
    q_ids = tok.encode(row["question"] + suffix + (row["answer_prefix"] or ""), add_special_tokens=False)
    rep_ids = tok.encode("\n\nRepeat the previous context exactly.", add_special_tokens=False) + \
        tok.encode(suffix, add_special_tokens=False) + ctx_ids[pre_len:]
    n = len(ctx_ids)
    log(f"{tag}: {cinfo['task']} n={n}, question {len(q_ids)} tok, repeat {len(rep_ids)} tok")

    # ---- 1. context prefill; keep post-RoPE keys
    cache = DynamicCache()
    with torch.no_grad():
        model(input_ids=torch.tensor([ctx_ids], device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    Kc = [cache.layers[l].keys[0, :, :n].detach() for l in range(NL)]        # [NKV, n, HD] bf16

    # ---- 2. repeat pass -> calibration queries
    cal_rel = torch.linspace(0, len(rep_ids) - 1, min(N_CAL, len(rep_ids))).round().long().unique()
    fwd(rep_ids, cache, pos=cal_rel.to(dev))
    Qcal_pre = {l: CAP["q"][l][0] for l in range(NL)}                        # [Qc, NH, HD]
    cal_pos = n + cal_rel
    cache.crop(-len(rep_ids))                                               # NEGATIVE: remove that many tokens
    assert cache.get_seq_length() == n, "crop must return the cache to the context"

    # ---- 3. question + greedy answer -> the FUTURE queries
    out = fwd(q_ids, cache, pos=slice(-1, None))
    real_pre = {l: [CAP["q"][l][0][-1]] for l in range(NL)}
    gen = []
    for step in range(1, max(STEPS) + 1):
        logits = out.logits[0, -1].float()
        logits[STOP] = -float("inf")
        nxt = int(logits.argmax())
        gen.append(nxt)
        if step < max(STEPS):
            out = fwd([nxt], cache, pos=slice(-1, None))
            if step + 1 in STEPS:
                for l in range(NL):
                    real_pre[l].append(CAP["q"][l][0][-1])
    n_q = len(q_ids)
    real_pos = torch.tensor([n + n_q - 1 + s - 1 for s in STEPS])
    del cache, out
    torch.cuda.empty_cache()
    log(f"{tag}: extracted; answer {tok.decode(gen)[:60]!r}")
    meta.append({"ctx": ci, "task": cinfo["task"], "n": n, "n_q": n_q, "n_cal": len(cal_rel),
                 "answer": tok.decode(gen)})

    # ---- 4/5. per KV head
    for l in range(NL):
        Qcal = rope(Qcal_pre[l], cal_pos)                                    # [Qc, NH, HD] float64
        Qreal = rope(torch.stack(real_pre[l]), real_pos)                     # [Qr, NH, HD]
        K = Kc[l].double()                                                   # [NKV, n, HD]
        for kv in range(NKV):
            qh = slice(kv * GQ, (kv + 1) * GQ)
            # omega over the context, pooled across the GQ query heads that share this KV head
            lg_cal = torch.einsum("qhd,nd->qhn", Qcal[:, qh], K[kv])         # [Qc, GQ, n]
            om_cal = torch.softmax(lg_cal, dim=-1).reshape(-1, n)            # [Qc*GQ, n]
            tau = denom.score_tau(om_cal)
            Tsum = tau.sum().item()

            cert = {}
            for eps in EPSS:
                c_e, M_e = denom.certified_memory(tau, eps, DELTA)
                cert[eps] = (c_e, M_e, denom.memory_highprob(denom.keep_prob(tau, c_e), DELTA_N))
            cert_rows.append({"ctx": ci, "task": cinfo["task"], "layer": l, "kv": kv, "n": n,
                              "T": Tsum, "T_over_n": Tsum / n, "tau_max": tau.max().item(),
                              **{f"M_eps{e}": cert[e][1] / n for e in EPSS},
                              **{f"Mhp_eps{e}": min(1.0, cert[e][2] / n) for e in EPSS},
                              **{f"c_eps{e}": cert[e][0] for e in EPSS}})

            if l not in MC_LAYERS or kv not in MC_KV:
                continue

            # ---------- budgets: Part 1 (2x, 4x) and Part 2 (eps=0.1 ladder)
            budgets = []
            for comp in COMP:
                budgets.append((f"comp{comp:g}x", denom.solve_c_for_memory(tau, n / comp)))
            c_star, M_star = cert[EPS_TARGET][0], cert[EPS_TARGET][1]
            for name, frac in (("eps0.1_M", 1.0), ("eps0.1_M/2", 0.5), ("eps0.1_M/4", 0.25)):
                tgt = max(1.0, M_star * frac)
                budgets.append((name, denom.solve_c_for_memory(tau, tgt)))

            # test queries: calibration held-out (theorem) and future (transfer)
            ct = torch.linspace(0, om_cal.shape[0] - 1, min(N_CALTEST, om_cal.shape[0])).round().long().unique()
            w_cal = torch.exp(lg_cal.reshape(-1, n)[ct] - lg_cal.reshape(-1, n)[ct].max(1, keepdim=True).values)
            lg_real = torch.einsum("qhd,nd->qhn", Qreal[:, qh], K[kv]).reshape(-1, n)
            w_real = torch.exp(lg_real - lg_real.max(1, keepdim=True).values)

            for bname, c in budgets:
                r = denom.keep_prob(tau, c)
                M = r.sum().item()
                eps_c = denom.eps_bound(tau, c, DELTA)
                Z = denom.sample_keep(r, gen_mc, trials=TRIALS)
                for qset, w in (("cal", w_cal), ("real", w_real)):
                    err = denom.rel_error(Z, r, w)                            # [TRIALS, Q]
                    p_own = (err > eps_c).double().mean(0) if math.isfinite(eps_c) else torch.ones(err.shape[1])
                    p_tgt = (err > EPS_TARGET).double().mean(0)
                    for qi in range(err.shape[1]):
                        mc_rows.append({
                            "ctx": ci, "task": cinfo["task"], "layer": l, "kv": kv, "budget": bname,
                            "qset": qset, "qhead": qi % GQ, "step": (STEPS[qi // GQ] if qset == "real" else -1),
                            "n": n, "c": c, "M_over_n": M / n, "eps_cert": eps_c, "T_over_n": Tsum / n,
                            "fail_own": p_own[qi].item(), "fail_eps0.1": p_tgt[qi].item(),
                            "err_p95": torch.quantile(err[:, qi], 0.95).item(),
                            "err_mean": err[:, qi].mean().item(), "err_max": err[:, qi].max().item()})
        if l % 8 == 0:
            log(f"{tag}: layer {l} done")

    pd.DataFrame(cert_rows).to_csv(os.path.join(OUT, "head_cert.csv"), index=False)
    pd.DataFrame(mc_rows).to_csv(os.path.join(OUT, "mc.csv"), index=False)
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    log(f"{tag}: written ({len(cert_rows)} cert rows, {len(mc_rows)} mc rows)")

log("done")
