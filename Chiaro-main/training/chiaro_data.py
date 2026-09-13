"""Load the released CHIARO data in the record shape the training scripts expect.

The release file (data/chiaro_full.json) carries every label source:
  * human_gold_a/b        - adjudicated two-annotator gold (canonical)
  * annotator_1/2_A/B     - each annotator's raw labels
  * generation_emotion_a/b- the label pair the scene was generated toward
plus the frozen train/val/test split (800/100/100 scenes).
"""
import io, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "chiaro_full.json")


def load_release():
    full = json.load(io.open(DATA, encoding="utf-8"))
    recs = []
    for r in full:
        recs.append({
            "mcq_id_a": r["id"],
            "sentence": r["sentence"],
            "agent_a_role": r["agent_a_role"],
            "agent_b_role": r["agent_b_role"],
            "gold_A": r["generation_emotion_a"], "gold_B": r["generation_emotion_b"],
            "human_gold_A": r["human_gold_a"],   "human_gold_B": r["human_gold_b"],
            "annotator_1_A": r.get("annotator_1_A"), "annotator_1_B": r.get("annotator_1_B"),
            "annotator_2_A": r.get("annotator_2_A"), "annotator_2_B": r.get("annotator_2_B"),
            "split": r["split"],
        })
    tr = [x for x in recs if x["split"] == "train"]
    va = [x for x in recs if x["split"] == "val"]
    te = [x for x in recs if x["split"] == "test"]
    assert (len(tr), len(va), len(te)) == (800, 100, 100), (len(tr), len(va), len(te))
    return tr, va, te
