import pickle, numpy as np, pandas as pd, re
D = pickle.load(open('runs.pkl','rb')); R = D['RUNS']
def g(t,p,c): return [r for r in R if r['task']==t and r['press']==p and r['comp']==c][0]
print("== context lengths per context: E3 kvzip_bern_s1 (n) vs E4 compactor_bern_s1 (n), 8x")
for t in ["prefix_suffix","qa_eng","summary","vt"]:
    a=[m['n'] for m in g(t,'kvzip_bern_s1','8x')['mem']]; b=[m['n'] for m in g(t,'compactor_bern_s1','8x')['mem']]
    print(t, 'E3', sorted(a)[:3], '...', 'E4', sorted(b)[:3], 'diff', sorted(set(np.array(b)-np.array(a))) if len(a)==len(b) else 'len differs', 'median', int(np.median(a)))
def rep(ans):
    w=str(ans).split(); gr=[" ".join(w[i:i+3]) for i in range(len(w)-2)]
    return bool(gr) and max(gr.count(x) for x in set(gr))>=3
print("== vt: degenerate share and mean score, by arm and ratio")
for arm in ['no_press','kvzip','kvzip_bern','compactor','compactor_bern','kvzip_bern_nobias']:
    row=[]
    for c in ['full','2x','4x','8x','16x']:
        rs=[r for r in R if r['task']=='vt' and r['arm']==arm and r['comp']==c]
        if not rs: continue
        p=pd.concat([r['pred'] for r in rs]); d=p.predicted_answer.map(rep)
        row.append(f"{c}: deg {100*d.mean():3.0f}% score(deg) {p.s[d].mean() if d.any() else float('nan'):5.1f} score(ok) {p.s[~d].mean() if (~d).any() else float('nan'):5.1f}")
    print(f"{arm:18s}", " | ".join(row))
p=g('vt','compactor','16x')['pred']; print("vt compactor 16x sample answers:"); [print('  ',repr(x[:150]), s) for x,s in zip(p.predicted_answer[:4], p.s[:4])]
p=g('vt','no_press','full')['pred']; print("vt no_press sample:"); [print('  ',repr(x[:150]), s, '| gold', a[:60]) for x,s,a in zip(p.predicted_answer[:3], p.s[:3], p.answer[:3])]
print("== prefix_suffix 8x answers vs gold (ctx 0-1)")
for pr in ['no_press','kvzip','kvzip_bern_s1','kvzip_bern_nobias_s1','compactor','compactor_bern_s1']:
    r=[x for x in R if x['task']=='prefix_suffix' and x['press']==pr and x['comp']==('full' if pr=='no_press' else '8x')][0]['pred']
    print(pr, 'score', r.s.mean(), [ (str(a)[:40], str(b)[:30]) for a,b in zip(r.predicted_answer[:2], r.answer[:2])])
print("== prefix_suffix per-context hits (of 5 turns) at 8x")
for pr in ['no_press','kvzip_bern_s1','kvzip_bern_s2','kvzip_bern_s3','kvzip_bern_nobias_s1']:
    r=[x for x in R if x['task']=='prefix_suffix' and x['press']==pr and x['comp']==('full' if pr=='no_press' else '8x')][0]['pred']
    print(f"{pr:22s}", (r.groupby('cluster').s.sum()/100).astype(int).tolist())
