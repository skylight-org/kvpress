import pickle, numpy as np, pandas as pd
D = pickle.load(open('runs.pkl','rb')); R = D['RUNS']
def g(task, press, comp): 
    x=[r for r in R if r['task']==task and r['press']==press and r['comp']==comp]; return x[0] if x else None
print("== D2 same-draw check: per-context kept_frac, kvzip_bern_s1 vs nobias_s1 at 8x")
for t in ["qasper","hotpotqa","passage_retrieval_en","gov_report","prefix_suffix","qa_eng","summary","vt"]:
    a,b = g(t,'kvzip_bern_s1','8x'), g(t,'kvzip_bern_nobias_s1','8x')
    ka=[m['kept_frac'] for m in a['mem']]; kb=[m['kept_frac'] for m in b['mem']]
    na=[m['n'] for m in a['mem']]; nb=[m['n'] for m in b['mem']]
    ca=[m.get('c') for m in a['mem']]; cb=[m.get('c') for m in b['mem']]
    same=sum(x==y for x,y in zip(ka,kb))
    print(f"{t:22s} src {a['source']}/{b['source']} gpu {a['cost']['gpu'][7:12]}/{b['cost']['gpu'][7:12]} n_equal={na==nb} c_equal={ca==cb} kept identical {same}/{len(ka)}  max|dk|={max(abs(x-y) for x,y in zip(ka,kb)):.2e}")
    # also: same answers where bias doesn't matter? fraction identical answers
    pa=a['pred'].set_index('key').predicted_answer.fillna(''); pb=b['pred'].set_index('key').predicted_answer.fillna('')
    print(f"{'':22s} identical answers {100*(pa==pb.reindex(pa.index)).mean():.0f}%")
