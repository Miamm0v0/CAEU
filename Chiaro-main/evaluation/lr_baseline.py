"""Lexical-artifact baseline suite (Reviewer pZG2 C1), mirroring the paper's RoBERTa protocol.

Split: train on the 800 train scenes (1,600 pair-examples), select C on val-100,
report on test-100 (200 slots). Gold = release adjudicated human gold.
Baselines: pair-input TF-IDF+LR (headline), sentence-only, role-only,
majority-class, random. Scoring: polarity-restricted argmax (matches encoders/LLMs)
plus unrestricted 10-class.
"""
import io, json, os, collections, random, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

POS = ["joy", "pride", "relief", "gratitude", "excitement"]
NEG = ["anger", "sadness", "fear", "disgust", "embarrassment"]
ALL10 = POS + NEG
FULL = json.load(io.open("../data/chiaro_full.json", encoding="utf-8"))

def items(split):
    out = []
    for r in FULL:
        if r.get("split") != split: continue
        for x in ("a", "b"):
            out.append(dict(sent=r["sentence"], role=r["agent_" + x + "_role"],
                            gold=r["human_gold_" + x]))
    return out

TR, VA, TE = items("train"), items("val"), items("test")
print(f"train {len(TR)} | val {len(VA)} | test {len(TE)} pair-examples")

def score(preds_proba, classes, golds):
    raw = pol = 0
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    for i, g in enumerate(golds):
        pr = classes[int(np.argmax(preds_proba[i]))]
        raw += (pr == g)
        pool = POS if g in POS else NEG
        idx = [classes.index(e) for e in pool if e in classes]
        pp = classes[idx[int(np.argmax(preds_proba[i][idx]))]]
        pol += (pp == g)
        if pp == g: tp[g] += 1
        else: fp[pp] += 1; fn[g] += 1
    f1s = []
    for e in ALL10:
        p = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 0
        r = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
    return 100 * raw / len(golds), 100 * pol / len(golds), 100 * sum(f1s) / 10

def run_lr(text_fn, name):
    Xtr_t = [text_fn(d) for d in TR]; ytr = [d["gold"] for d in TR]
    best = None
    for ng in [(1, 1), (1, 2)]:
        for C in [0.25, 1.0, 4.0]:
            vec = TfidfVectorizer(ngram_range=ng, min_df=2, sublinear_tf=True)
            Xtr = vec.fit_transform(Xtr_t)
            clf = LogisticRegression(max_iter=3000, C=C).fit(Xtr, ytr)
            Xva = vec.transform([text_fn(d) for d in VA])
            _, pol_va, _ = score(clf.predict_proba(Xva), list(clf.classes_), [d["gold"] for d in VA])
            if best is None or pol_va > best[0]:
                best = (pol_va, ng, C, vec, clf)
    pol_va, ng, C, vec, clf = best
    Xte = vec.transform([text_fn(d) for d in TE])
    raw, pol, f1 = score(clf.predict_proba(Xte), list(clf.classes_), [d["gold"] for d in TE])
    # agent-invariance: same prediction for both agents of a scene?
    inv = None
    if name != "pair-input (sentence+role)":
        pass
    print(f"  {name:32s} raw10 {raw:5.1f} | polarity-restricted {pol:5.1f} | macro-F1 {f1:5.1f}   (val-selected ngram={ng} C={C}, val {pol_va:.1f})")
    return vec, clf

print("\n=== TF-IDF + Logistic Regression (val-tuned) ===")
run_lr(lambda d: d["sent"] + " || " + d["role"], "pair-input (sentence+role)")
run_lr(lambda d: d["sent"], "sentence-only (agent-invariant)")
run_lr(lambda d: d["role"], "role-only (artifact check)")

print("\n=== Floors ===")
maj = collections.Counter(d["gold"] for d in TR)
maj_pos = max(POS, key=lambda e: maj[e]); maj_neg = max(NEG, key=lambda e: maj[e])
c = sum(1 for d in TE if d["gold"] == (maj_pos if d["gold"] in POS else maj_neg))
print(f"  majority-class (per polarity: {maj_pos}/{maj_neg}): polarity-restricted {100*c/len(TE):.1f}")
random.seed(0)
c = sum(1 for d in TE if random.choice(POS if d["gold"] in POS else NEG) == d["gold"])
print(f"  random (polarity-restricted): {100*c/len(TE):.1f}  (expected 20.0)")
print("\nReference: RoBERTa-large (paper) 69.5 acc on same test split; frontier LLMs 59.9-67.3 macro-F1 on full release.")
