import pickle, numpy as np, pandas as pd, glob, hashlib, math
from pathlib import Path
D = pickle.load(open('runs.pkl','rb')); R = D['RUNS']
rng = np.random.default_rng(1)
LB=["qasper","hotpotqa","passage_retrieval_en","gov_report"]; SC=["prefix_suffix","qa_eng","summary","vt"]; T=LB+SC
# qasper context mapping
f=glob.glob(str(Path.home()/"hf/hub/datasets--Xnhyacinth--LongBench/snapshots/*/qasper/test*.parquet"))
df=pd.concat([pd.read_parquet(x) for x in f]); ctx={r['_id']: hashlib.md5(r['context'].encode()).hexdigest() for _,r in df.iterrows()}
def runs(t,arm,comp,seed=None):
    return [r for r in R if r['task']==t and r['arm']==arm and r['comp']==comp and (seed is None or r['seed']==seed)]
def perq(t,arm,comp,seed=None,recluster=False):
    rs=runs(t,arm,comp if arm!='no_press' else 'full',seed)
    fr=[r['pred'].set_index('key') for r in rs]
    s=pd.concat([x.s for x in fr],axis=1).mean(axis=1)
    cl=fr[0].cluster.reindex(s.index)
    if recluster and t=='qasper': cl=pd.Series([ctx[i] for i in s.index], index=s.index)
    return s, cl
def diffs(t,a,b,comp,sa=None,sb=None,recluster=False):
    A,cl=perq(t,a,comp,sa,recluster); B,_=perq(t,b,comp,sb,recluster)
    return (A-B.reindex(A.index)).groupby(cl).mean().to_numpy()
def pooled(xs,draws=400_000):
    obs=np.mean([x.mean() for x in xs]); null=np.zeros(draws)
    for x in xs: null+=(rng.choice([-1,1],size=(draws,len(x)))@x)/len(x)
    null/=len(xs); se=math.sqrt(sum(x.var(ddof=1)/len(x) for x in xs))/len(xs)
    return obs,se,(null>=obs-1e-12).mean()
def show(lbl,xs): o,se,p=pooled(xs); print(f"  {lbl:45s} {o:+6.2f} ± {se:4.2f}  p={p:.4f}")
for comp in ['8x','4x']:
  for nm,(a,b) in {'D_KV':('kvzip_bern','kvzip'),'D_CP':('compactor_bern','compactor')}.items():
    print(f"== {nm} {comp}")
    X={t:diffs(t,a,b,comp) for t in T}
    show('as reported', list(X.values()))
    Xr=dict(X); Xr['qasper']=diffs('qasper',a,b,comp,recluster=True)
    print(f"    qasper clusters {len(X['qasper'])}->{len(Xr['qasper'])}: task {X['qasper'].mean():+.2f}±{X['qasper'].std(ddof=1)/len(X['qasper'])**.5:.2f} -> {Xr['qasper'].mean():+.2f}±{Xr['qasper'].std(ddof=1)/len(Xr['qasper'])**.5:.2f}")
    show('qasper clustered by context', list(Xr.values()))
    for t in T: show(f'leave out {t}', [X[u] for u in T if u!=t])
    for s in (1,2,3): show(f'single seed s{s} only', [diffs(t,a,b,comp,sa=s) for t in T])
    # sign test on task means
    k=sum(X[t].mean()>0 for t in T); print(f"  tasks with positive effect {k}/8")
print("== D2 (bias on - off, seed 1, 8x)")
X={t:diffs(t,'kvzip_bern','kvzip_bern_nobias','8x',sa=1) for t in T}
show('as reported (8 tasks)', list(X.values()))
ok=[t for t in T if t not in ('prefix_suffix','summary')]
show('6 tasks with truly identical draws', [X[t] for t in ok])
show('  ...of which LongBench', [X[t] for t in LB]); show('  ...of which qa_eng+vt', [X['qa_eng'],X['vt']])
Y={t:diffs(t,'kvzip_bern','kvzip_bern_nobias','8x') for t in T}
show('3-seed mean vs nobias (8 tasks)', list(Y.values()))
for t in T: print(f"    {t:22s} s1-nobias {X[t].mean():+6.2f}   seedmean-nobias {Y[t].mean():+6.2f}")
print("== D2 leave-one-out"); 
for t in T: show(f'leave out {t}', [X[u] for u in T if u!=t])
