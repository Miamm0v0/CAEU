"""
Experiment B — GoEmotions-only matched baseline at 1600 items.

Same as _expB_train_goemotions_matched.py except trained on 1600
GoEmotions items (160/class × 10) instead of 800. This is the matched
comparator for the combined CHIARO+GoEmotions run (which also uses 1600
training examples total). Putting all three comparators at 1600 makes
the 3-way comparison fair on training size.

Key difference from the original matched-size script: pair-input format
("the speaker" as 2nd seq) instead of single-input, so that the
checkpoint can be evaluated on external datasets via the same
"the speaker" fill used by _expB_eval_external.py.

Output dir: expb_roberta_large_goemo_1600_seed{SEED} / .
"""
import json, os, random, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "roberta-large"
SEED = int(os.environ.get("EXPB_SEED", "42"))
# Write training artifacts to local /tmp (NFS write failures during model
# shard save); the final best/ checkpoint and predictions are copied back
# to HERE at the end.
TMP_DIR = f"/tmp/expb_roberta_large_goemo_1600_seed{SEED}"
OUT_DIR = TMP_DIR
FINAL_DIR = os.path.join(HERE, f"expb_roberta_large_goemo_1600_seed{SEED}")
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

EPOCHS = 5
BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01
TARGET_TRAIN_SIZE = 1600
VAL_SIZE = 100

CHIARO_EMOTIONS = ['joy','pride','relief','gratitude','excitement',
                   'anger','sadness','fear','disgust','embarrassment']
CHIARO_SET = set(CHIARO_EMOTIONS)
LABEL2ID = {e: i for i, e in enumerate(CHIARO_EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}


def filter_to_chiaro(split, names):
    out = []
    for r in split:
        if len(r["labels"]) != 1: continue
        label_name = names[r["labels"][0]]
        if label_name in CHIARO_SET:
            out.append({"text": r["text"], "gold": label_name})
    return out


def stratified_sample(items, n_per_class):
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


def sample_up_to(items, target_total):
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


class PairDataset(Dataset):
    """Pair input: (text, "the speaker") — matches downstream eval."""
    def __init__(self, rows, tok, max_length=256):
        self.rows = rows; self.tok = tok; self.max_length = max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["text"], "the speaker", truncation=True,
                       max_length=self.max_length, padding="max_length")
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(LABEL2ID[r["gold"]])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (np.argmax(logits, axis=-1) == labels).mean()}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("Loading GoEmotions ...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    names = ds["train"].features["labels"].feature.names

    train_filtered = filter_to_chiaro(ds["train"], names)
    val_filtered   = filter_to_chiaro(ds["validation"], names)
    test_filtered  = filter_to_chiaro(ds["test"], names)
    print(f"  train (single-label, CHIARO 10):      {len(train_filtered)}")
    print(f"  validation:                            {len(val_filtered)}")
    print(f"  test:                                  {len(test_filtered)}")

    train_rows = sample_up_to(train_filtered, TARGET_TRAIN_SIZE)
    val_rows   = stratified_sample(val_filtered, max(1, VAL_SIZE // len(CHIARO_EMOTIONS)))
    test_rows  = test_filtered
    print(f"\nSampled: train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")
    print(f"  train per-class: {dict(Counter(r['gold'] for r in train_rows).most_common())}")

    print(f"\nLoading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(CHIARO_EMOTIONS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH*2,
        learning_rate=LR, weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        greater_is_better=True, logging_steps=20, report_to="none",
        seed=SEED, bf16=True,
    )

    trainer = Trainer(
        model=mdl, args=args,
        train_dataset=PairDataset(train_rows, tok),
        eval_dataset=PairDataset(val_rows, tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print(f"\nEvaluating on GoEmotions test (filtered to CHIARO 10, N={len(test_rows)}) ...")
    preds_out = trainer.predict(PairDataset(test_rows, tok))
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    rows_out = []
    correct = 0
    for r, p in zip(test_rows, pred_labels):
        pred_str = ID2LABEL[int(p)]
        rows_out.append({**r, "pred": pred_str})
        if pred_str == r["gold"]:
            correct += 1
    print(f"  GoEmotions test (pair-input, 'the speaker' fill): {correct}/{len(rows_out)} = {correct/len(rows_out)*100:.1f}%")

    best_path_tmp = os.path.join(TMP_DIR, "best")
    trainer.save_model(best_path_tmp); tok.save_pretrained(best_path_tmp)

    import shutil
    best_path_final = os.path.join(FINAL_DIR, "best")
    if os.path.exists(best_path_final): shutil.rmtree(best_path_final)
    shutil.copytree(best_path_tmp, best_path_final)
    with open(os.path.join(FINAL_DIR, "goemotions_test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(rows_out, f, ensure_ascii=False, indent=2)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nSaved best model + predictions under {FINAL_DIR}")


if __name__ == "__main__":
    main()
