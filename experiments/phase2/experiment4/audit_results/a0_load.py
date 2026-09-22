"""Load every E4 run once (about 5 min) into runs.pkl for the a1-a5 audit scripts.

Imports a copy of summarize.py whose OUT points here, because summarize.py writes out/cells.csv
on import. Run from this folder:  python a0_load.py && python a1_d2_same_draws.py ...
"""
import pathlib, pickle, re, sys
here = pathlib.Path(__file__).resolve().parent
src = (here.parent / "summarize.py").read_text()
src = re.sub(r'^HERE = .*$', f'HERE = pathlib.Path("{here.parent}")', src, flags=re.M)
src = re.sub(r'^OUT = .*$', f'OUT = pathlib.Path("{here}")', src, flags=re.M)
(here / "_summ.py").write_text("import pathlib\n" + src)
sys.path.insert(0, str(here))
import _summ
pickle.dump({"RUNS": _summ.RUNS, "cells": _summ.cells}, open(here / "runs.pkl", "wb"))
