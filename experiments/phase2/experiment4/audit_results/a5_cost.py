import pickle, numpy as np
D = pickle.load(open('runs.pkl','rb')); R = D['RUNS']
T=["qasper","hotpotqa","passage_retrieval_en","gov_report","prefix_suffix","qa_eng","summary","vt"]
print("time ratio bern/top-k (mean seconds over ratios & seeds), and kvzip_bern/no_press")
for t in T:
    m=lambda a: np.mean([r['cost']['seconds'] for r in R if r['task']==t and r['arm']==a and r['cost']])
    print(f"{t:22s} compactor_bern/compactor {m('compactor_bern')/m('compactor'):.2f}  kvzip/no_press {m('kvzip')/m('no_press'):.2f} kvzip_bern/no_press {m('kvzip_bern')/m('no_press'):.2f} nobias/bern {m('kvzip_bern_nobias')/m('kvzip_bern'):.2f}")
