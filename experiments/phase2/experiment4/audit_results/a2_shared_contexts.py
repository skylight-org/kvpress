import pickle, numpy as np, pandas as pd, hashlib
D = pickle.load(open('runs.pkl','rb')); R = D['RUNS']
for t in ["qasper","hotpotqa","passage_retrieval_en","gov_report"]:
    r=[x for x in R if x['task']==t and x['press']=='compactor_bern_s1' and x['comp']=='8x'][0]
    p=r['pred']
    print(t, p.columns.tolist()[:12], len(p), 'unique _id', p._id.nunique(), 'mem contexts', len(r['mem']))
    if 'context' in p.columns:
        h=p.context.map(lambda s: hashlib.md5(str(s).encode()).hexdigest())
        print('   unique contexts', h.nunique(), 'group sizes', sorted(h.value_counts().tolist(), reverse=True)[:6])
