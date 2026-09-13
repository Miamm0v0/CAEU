"""
Experiment B — combined CHIARO + GoEmotions training, matched to 1600
examples to make the 3-way comparison fair (CHIARO-only and GoEm-only
both at 1600).

Combined source mix:
  * 400 CHIARO sentences → 800 pair-input examples (agent_role as 2nd seq)
  * 800 GoEmotions items (stratified 80/class across CHIARO's 10 emotions,
                          pair-input with "the speaker" as 2nd seq)

Total: 1600 training examples. Same hyperparameters as the existing
CHIARO-only and GoEm-only training scripts. Pair-input throughout so the
model trains and evaluates under one consistent input contract — matching
the "the speaker" fill used by _expB_eval_external.py on external corpora.

Output dir: expb_roberta_large_combined_seed{SEED} / .
"""
import json, os, random, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
from chiaro_data import load_release

MODEL_ID = "roberta-large"
SEED = int(os.environ.get("EXPB_SEED", "42"))
LABEL_SOURCE = os.environ.get("EXPB_LABEL_SOURCE", "human_gold")  # human_gold | gold (=generation labels, the protocol used in the paper) | annotator_1 | annotator_2
assert LABEL_SOURCE in ("human_gold", "gold", "annotator_1", "annotator_2"), f"unknown EXPB_LABEL_SOURCE={LABEL_SOURCE}"
EPOCHS = 5
BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
# Combined corpus = full CHIARO + soft-stratified GoEm. Total 3200 examples
# (1600 CHIARO pair-examples + 1600 GoEm items via soft-stratified sample).
# Soft-stratification means rare GoEm classes (pride, relief, embarrassment)
# contribute what they have and the shortfall is filled from rich classes,
# so the total actually reaches 1600 instead of falling short.
CHIARO_TRAIN_SENTENCES = 800         # → 1600 pair-input examples
GOEMO_TRAIN_TARGET     = 1600        # soft-stratified, 160/class soft cap

CHIARO_EMOTIONS = ['joy','pride','relief','gratitude','excitement',
                   'anger','sadness','fear','disgust','embarrassment']
CHIARO_SET = set(CHIARO_EMOTIONS)
LABEL2ID = {e: i for i, e in enumerate(CHIARO_EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}

# Write training artifacts to local /tmp (NFS write failures during model
# shard save); the final best/ checkpoint and predictions are copied back
# to HERE at the end.
# Slug includes the data-size profile so the 1600-example and
# 3200-example Combined runs don't overwrite each other.
_label_slug = "" if LABEL_SOURCE == "gold" else f"_{LABEL_SOURCE}"
RUN_SLUG = f"combined_full{_label_slug}_seed{SEED}"   # 1600 CHIARO + 1600 GoEm
TMP_DIR = f"/tmp/expb_roberta_large_{RUN_SLUG}"
OUT_DIR = TMP_DIR
FINAL_DIR = os.path.join(HERE, f"expb_roberta_large_{RUN_SLUG}")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)


# ---------- Build CHIARO examples (pair-input with real agent role) ----

def build_chiaro_examples(sentences):
    """Each CHIARO sentence → 2 examples (one per agent slot).
    Training label is drawn from LABEL_SOURCE (gold | annotator_1 | annotator_2)."""
    rows = []
    for s in sentences:
        for ag, role_key in [("A", "agent_a_role"), ("B", "agent_b_role")]:
            train_label = s.get(f"{LABEL_SOURCE}_{ag}")
            if train_label not in LABEL2ID:
                continue
            rows.append({
                "source": "chiaro",
                "text": s["sentence"],
                "role": s[role_key],
                "label": LABEL2ID[train_label],
                "label_str": train_label,
            })
    return rows


# ---------- Build GoEmotions examples (pair-input with "the speaker") ---

def filter_goemo_to_chiaro(split, names):
    out = []
    for r in split:
        if len(r["labels"]) != 1: continue
        label_name = names[r["labels"][0]]
        if label_name in CHIARO_SET:
            out.append({"text": r["text"], "gold": label_name})
    return out


def stratified_sample_goemo(items, n_per_class):
    """Strict per-class sample. Used for the validation set."""
    by_class = {c: [] for c in CHIARO_EMOTIONS}
    for r in items:
        if r["gold"] in by_class:
            by_class[r["gold"]].append(r)
    rng = random.Random(SEED)
    out = []
    for c in CHIARO_EMOTIONS:
        pool = by_class[c]
        rng.shuffle(pool)
        out.extend(pool[:n_per_class])
    rng.shuffle(out)
    return out


def sample_up_to_goemo(items, target_total):
    """Soft-stratified: take min(target_total/10, available) per class to
    preserve coverage of CHIARO's ten emotions, then top up from leftover
    items in rich classes until target_total is hit. Used for training so
    rare-class shortfall doesn't make the total fall short of target."""
    by_class = {c: [] for c in CHIARO_EMOTIONS}
    for r in items:
        if r["gold"] in by_class:
            by_class[r["gold"]].append(r)
    rng = random.Random(SEED)
    for pool in by_class.values():
        rng.shuffle(pool)
    per_class_cap = target_total // len(CHIARO_EMOTIONS)
    out, leftover = [], []
    for c in CHIARO_EMOTIONS:
        out.extend(by_class[c][:per_class_cap])
        leftover.extend(by_class[c][per_class_cap:])
    rng.shuffle(leftover)
    out.extend(leftover[:target_total - len(out)])
    rng.shuffle(out)
    return out


def build_goemo_examples(rows):
    """Each GoEmotions item → 1 example with "the speaker" as role."""
    return [{
        "source": "goemo",
        "text": r["text"],
        "role": "the speaker",
        "label": LABEL2ID[r["gold"]],
        "label_str": r["gold"],
    } for r in rows]


# ---------- Combined PairDataset ---------------------------------------

class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], r["role"], truncation=True,
                       max_length=self.max_length, padding="max_length")
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(r["label"])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, axis=-1) == labels).mean()}


# ---------- Main ------------------------------------------------------

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # CHIARO splits ----------------------------------------------------
    chiaro_train_sentences, chiaro_val_sentences, test_sentences = load_release()
    chiaro_train_sentences = chiaro_train_sentences[:CHIARO_TRAIN_SENTENCES]
    assert len(chiaro_train_sentences) == CHIARO_TRAIN_SENTENCES

    chiaro_train_rows = build_chiaro_examples(chiaro_train_sentences)
    chiaro_val_rows   = build_chiaro_examples(chiaro_val_sentences)
    chiaro_test_rows  = build_chiaro_examples(test_sentences)
    print(f"CHIARO  train={len(chiaro_train_rows)}  val={len(chiaro_val_rows)}  test={len(chiaro_test_rows)}")

    # GoEmotions splits ------------------------------------------------
    print("\nLoading GoEmotions ...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names
    goemo_train_filtered = filter_goemo_to_chiaro(ds["train"], names)
    goemo_val_filtered   = filter_goemo_to_chiaro(ds["validation"], names)
    print(f"  GoEmotions train (single-label, CHIARO 10):      {len(goemo_train_filtered)}")
    print(f"  GoEmotions validation (single-label, CHIARO 10): {len(goemo_val_filtered)}")

    goemo_train_sample = sample_up_to_goemo(goemo_train_filtered, GOEMO_TRAIN_TARGET)
    goemo_val_sample   = stratified_sample_goemo(goemo_val_filtered, max(1, 100 // 10))
    print(f"  Sampled: train={len(goemo_train_sample)}  val={len(goemo_val_sample)}")
    print(f"  GoEmo train per-class: {dict(Counter(r['gold'] for r in goemo_train_sample).most_common())}")

    goemo_train_rows = build_goemo_examples(goemo_train_sample)
    goemo_val_rows   = build_goemo_examples(goemo_val_sample)

    # Combine ----------------------------------------------------------
    combined_train_rows = chiaro_train_rows + goemo_train_rows
    combined_val_rows   = chiaro_val_rows + goemo_val_rows
    rng.shuffle(combined_train_rows)
    rng.shuffle(combined_val_rows)
    print(f"\nCombined train: {len(combined_train_rows)}  "
          f"(chiaro {len(chiaro_train_rows)} + goemo {len(goemo_train_rows)})")
    print(f"Combined val:   {len(combined_val_rows)}")

    print(f"\nLoading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(CHIARO_EMOTIONS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH * 2,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        greater_is_better=True, logging_steps=20, report_to="none",
        seed=SEED, bf16=True,
    )

    trainer = Trainer(
        model=mdl, args=args,
        train_dataset=PairDataset(combined_train_rows, tok),
        eval_dataset=PairDataset(combined_val_rows, tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Evaluate on CHIARO held-out 100 -----------------------------------
    print(f"\nEvaluating on CHIARO held-out test (N={len(chiaro_test_rows)} pair examples) ...")
    test_metrics = trainer.evaluate(PairDataset(chiaro_test_rows, tok))
    print(f"  In-distribution metrics: {test_metrics}")

    preds_out = trainer.predict(PairDataset(chiaro_test_rows, tok))
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    rows_out = []
    for r, p in zip(chiaro_test_rows, pred_labels):
        rows_out.append({**r, "pred": ID2LABEL[int(p)]})
    correct = sum(1 for r in rows_out if r["pred"] == r["label_str"])
    print(f"  CHIARO test vs {LABEL_SOURCE} (training target): {correct}/{len(rows_out)} = {correct/len(rows_out)*100:.1f}%")

    # Save best model to /tmp (fast local disk), then copy to NFS final dir
    best_path_tmp = os.path.join(TMP_DIR, "best")
    trainer.save_model(best_path_tmp); tok.save_pretrained(best_path_tmp)

    # Copy best/ from /tmp to FINAL_DIR (NFS), then drop /tmp scratch
    import shutil
    best_path_final = os.path.join(FINAL_DIR, "best")
    if os.path.exists(best_path_final): shutil.rmtree(best_path_final)
    shutil.copytree(best_path_tmp, best_path_final)
    with open(os.path.join(FINAL_DIR, "chiaro_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nSaved best model + predictions under {FINAL_DIR}")


if __name__ == "__main__":
    main()
