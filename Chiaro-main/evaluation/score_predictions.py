"""Score raw model predictions against the adjudicated human gold labels.

Reproduces the paper's main results from the files shipped in this repo:
  * Table 2  - macro-F1 per LLM (raw_predictions/llm/)
  * Table 3  - per-emotion precision/recall/F1/support for any one model
  * Table 5  - macro-F1 split by physical vs non-physical causal mode
  * Encoder table - macro-F1 for the four off-the-shelf classifiers
    (raw_predictions/encoders/), scored with polarity-restricted argmax

Every metric is computed against ``human_gold_a/b`` in data/chiaro_full.json
(the adjudicated two-annotator gold), joined to each prediction record by
sentence + agent role.

Usage:
  python score_predictions.py                 # summary for every file
  python score_predictions.py --per-emotion raw_predictions/llm/eval_openai_gpt_5_5.json
"""
import argparse
import collections
import glob
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "chiaro_full.json")

POS = ["joy", "pride", "relief", "gratitude", "excitement"]
NEG = ["anger", "sadness", "fear", "disgust", "embarrassment"]
ALL10 = POS + NEG

_norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())


def load_gold():
    full = json.load(io.open(DATA, encoding="utf-8"))
    return {_norm(r["sentence"]): r for r in full}


def gold_for(rec, slot, gold_map):
    """Adjudicated human gold for one agent slot of one prediction record."""
    fr = gold_map.get(_norm(rec["sentence"]))
    if fr is None:
        return None, None
    role = rec["agent_a_role"] if slot == "A" else rec["agent_b_role"]
    if fr["agent_a_role"] == role:
        s = "a"
    elif fr["agent_b_role"] == role:
        s = "b"
    else:  # fall back to valence matching
        s = "a" if ((fr["human_gold_a"] in POS) == (rec["gold_" + slot] in POS)) else "b"
    return fr["human_gold_" + s], fr.get("version")


def macro_f1(pairs):
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    n = c = 0
    for p, g in pairs:
        n += 1
        c += (p == g)
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    f1s = {}
    for e in ALL10:
        pr = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 0.0
        rc = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 0.0
        f1s[e] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = 100.0 * c / n if n else 0.0
    return acc, 100.0 * sum(f1s.values()) / len(ALL10), n


def score_file(path, gold_map):
    recs = json.load(io.open(path, encoding="utf-8"))
    pairs, by_mode = [], collections.defaultdict(list)
    parsed = total = 0
    for r in recs:
        for slot in ("A", "B"):
            g, mode = gold_for(r, slot, gold_map)
            if g is None:
                continue
            total += 1
            p = r.get("llm_emotion_" + slot)
            if p is None:
                continue
            parsed += 1
            pairs.append((p, g))
            by_mode[mode].append((p, g))
    acc, f1, n = macro_f1(pairs)
    out = {"file": os.path.basename(path), "n": n, "parsed": parsed, "total": total,
           "acc": acc, "macro_f1": f1}
    for mode, ps in sorted(by_mode.items()):
        out["f1_" + str(mode)] = macro_f1(ps)[1]
    return out


def per_emotion(path, gold_map):
    recs = json.load(io.open(path, encoding="utf-8"))
    tp = collections.Counter(); fp = collections.Counter(); fn = collections.Counter()
    support = collections.Counter()
    for r in recs:
        for slot in ("A", "B"):
            g, _ = gold_for(r, slot, gold_map)
            p = r.get("llm_emotion_" + slot)
            if g is None or p is None:
                continue
            support[g] += 1
            if p == g:
                tp[g] += 1
            else:
                fp[p] += 1
                fn[g] += 1
    print(f"\nPer-emotion breakdown for {os.path.basename(path)} (vs human gold):")
    print(f"  {'emotion':14s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'support':>8s}")
    for e in ALL10:
        pr = 100 * tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 0.0
        rc = 100 * tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        print(f"  {e:14s} {pr:6.1f} {rc:6.1f} {f1:6.1f} {support[e]:8d}")


def human_ceiling(gold_map):
    """Annotator-vs-gold macro-F1 (the paper's ~93 human ceiling)."""
    pairs = []
    for fr in gold_map.values():
        for s in ("a", "b"):
            for ann in ("annotator_1_", "annotator_2_"):
                a = fr.get(ann + s.upper())
                if a:
                    pairs.append((a, fr["human_gold_" + s]))
    return macro_f1(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-emotion", metavar="FILE", help="print Table-3-style breakdown for one predictions file")
    args = ap.parse_args()

    gold_map = load_gold()

    if args.per_emotion:
        per_emotion(args.per_emotion, gold_map)
        return

    print(f"{'file':44s} {'acc':>6s} {'macroF1':>8s} {'F1 phys':>8s} {'F1 nonph':>9s} {'parsed':>11s}")
    for sub in ("llm", "encoders"):
        for path in sorted(glob.glob(os.path.join(HERE, "raw_predictions", sub, "*.json"))):
            r = score_file(path, gold_map)
            phys = r.get("f1_physical", float("nan"))
            nonp = r.get("f1_non_physical", r.get("f1_nonphysical", float("nan")))
            print(f"{sub + '/' + r['file']:44s} {r['acc']:6.1f} {r['macro_f1']:8.1f} "
                  f"{phys:8.1f} {nonp:9.1f} {r['parsed']:>5d}/{r['total']}")
    acc, f1, n = human_ceiling(gold_map)
    print(f"\nHuman ceiling (annotators vs adjudicated gold): acc {acc:.1f}, macro-F1 {f1:.1f} over {n} judgments")


if __name__ == "__main__":
    main()
