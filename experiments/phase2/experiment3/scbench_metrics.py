"""Per-task scorers for SCBench, in kvpress's scorer shape: calculate_metrics(df) -> {task: {metric: score}}.

Metric per task, chosen to match what the answer actually is:
  kv             single uuid          -> substring match
  prefix_suffix  single word          -> substring match
  vt             list of variables    -> fraction of the gold variables present (RULER's string_match_all)
  qa_eng         short free-text span -> SQuAD-style token F1, plus substring EM reported alongside
  summary        a sentence           -> ROUGE-L F1 (LCS based, implemented here: no extra dependency,
                                        and the `rouge` package's tokenizer drops the LaTeX and maths
                                        that fill these physics/CS abstracts)

Every scorer normalises away the stray quoting in SCBench's gold answers (e.g. '"Peyton"', '"Annalisa').
"""

import json
import re
import string

import numpy as np
import pandas as pd

CTRL = re.compile(r"[\x00-\x1f]")
ARTICLES = re.compile(r"\b(a|an|the)\b")


def norm(s):
    """Lowercase, strip control chars, quotes and punctuation, collapse whitespace."""
    s = CTRL.sub(" ", str(s)).lower()
    s = "".join(" " if c in string.punctuation else c for c in s)
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())


def gold_list(answer):
    """vt answers are JSON lists; everything else is a single string."""
    a = str(answer)
    if a.startswith("["):
        try:
            return [str(x) for x in json.loads(a)]
        except json.JSONDecodeError:
            pass
    return [a]


def substring_match(pred, golds):
    p = norm(pred)
    return 100.0 * float(any(norm(g) in p for g in golds if str(g).strip()))


def all_match(pred, golds):
    """Fraction of gold items present in the prediction (vt: 5 variable names)."""
    p = norm(pred)
    golds = [g for g in golds if str(g).strip()]
    return 100.0 * (sum(norm(g) in p for g in golds) / len(golds)) if golds else 0.0


def token_f1(pred, golds):
    """SQuAD-style max token F1 over the gold answers."""
    pt = norm(pred).split()
    best = 0.0
    for g in golds:
        gt = norm(g).split()
        if not gt or not pt:
            best = max(best, float(pt == gt))
            continue
        common = sum(min(pt.count(w), gt.count(w)) for w in set(gt))
        if common:
            prec, rec = common / len(pt), common / len(gt)
            best = max(best, 2 * prec * rec / (prec + rec))
    return 100.0 * best


def rouge_l(pred, golds):
    """ROUGE-L F1 via longest common subsequence on normalised tokens."""
    pt = norm(pred).split()
    best = 0.0
    for g in golds:
        gt = norm(g).split()
        if not pt or not gt:
            continue
        # LCS length, O(len(pt) * len(gt)) with a rolling row
        prev = [0] * (len(gt) + 1)
        for a in pt:
            cur = [0]
            for j, b in enumerate(gt):
                cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
            prev = cur
        lcs = prev[-1]
        if lcs:
            prec, rec = lcs / len(pt), lcs / len(gt)
            best = max(best, 2 * prec * rec / (prec + rec))
    return 100.0 * best


SCORERS = {
    "kv": {"substring_match": substring_match},
    "prefix_suffix": {"substring_match": substring_match},
    "vt": {"string_match_all": all_match},
    "qa_eng": {"f1": token_f1, "substring_match": substring_match},
    "summary": {"rouge_l": rouge_l},
}


def score_one(task, pred, answer):
    """Per-sample score used by both the harness and the paired analysis (first metric of the task)."""
    fn = next(iter(SCORERS[task].values()))
    return fn(pred, gold_list(answer))


def calculate_metrics(df: pd.DataFrame) -> dict:
    """kvpress scorer entry point."""
    out = {}
    for task, g in df.groupby("task"):
        preds = g["predicted_answer"].fillna("").tolist()
        golds = [gold_list(a) for a in g["answer"].tolist()]
        out[str(task)] = {name: round(float(np.mean([fn(p, gl) for p, gl in zip(preds, golds)])), 4)
                          for name, fn in SCORERS[str(task)].items()}
    return out
