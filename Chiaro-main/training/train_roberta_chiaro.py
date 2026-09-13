"""
Experiment B — Train RoBERTa-base on CHIARO from scratch fine-tuning.

Designed to be the source-side of a CHIARO → external-dataset transfer
test. Input format: "{sentence}</s></s>{agent_role}" (RoBERTa pair-
classification). Output: 10-class softmax over CHIARO's taxonomy.

Split design: re-uses the same 100-sentence pilot subset (seed=42) as
the held-out CHIARO test, so the in-distribution number is directly
comparable to the LLM pilot table and Experiment A. The remaining 900
sentences are split 800 train / 100 val.

GPU placement: CUDA_VISIBLE_DEVICES=7.
"""
import json, os, random, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

HERE = os.path.dirname(os.path.abspath(__file__))
from chiaro_data import load_release

MODEL_ID = os.environ.get("EXPB_MODEL_ID", "roberta-base")
LABEL_SOURCE = os.environ.get("EXPB_LABEL_SOURCE", "human_gold")  # human_gold | gold (=generation labels, the protocol used in the paper) | annotator_1 | annotator_2
SEED = int(os.environ.get("EXPB_SEED", "42"))
SEED_OVERRIDE = SEED  # keep both names for the slug
assert LABEL_SOURCE in ("human_gold", "gold", "annotator_1", "annotator_2"), f"unknown EXPB_LABEL_SOURCE={LABEL_SOURCE}"
_slug = MODEL_ID.split("/")[-1].replace("roberta-", "").replace("-", "_") or "model"
_label_slug = "" if LABEL_SOURCE == "gold" else f"_{LABEL_SOURCE}"
_seed_slug = "" if SEED_OVERRIDE == 42 else f"_seed{SEED_OVERRIDE}"
OUT_DIR = os.path.join(HERE, f"expb_roberta_{_slug}_chiaro{_label_slug}{_seed_slug}")
os.makedirs(OUT_DIR, exist_ok=True)
EPOCHS = 5
BATCH = 16
LR = 2e-5
WEIGHT_DECAY = 0.01

CHIARO_EMOTIONS = ['joy','pride','relief','gratitude','excitement',
                   'anger','sadness','fear','disgust','embarrassment']
LABEL2ID = {e: i for i, e in enumerate(CHIARO_EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}


def build_examples(sentences):
    """Each sentence yields TWO training examples (one per agent).
    The training LABEL comes from LABEL_SOURCE; rows where that source
    is missing or out-of-taxonomy are skipped."""
    rows = []
    for s in sentences:
        for ag, role_key in [("A", "agent_a_role"), ("B", "agent_b_role")]:
            train_label = s.get(f"{LABEL_SOURCE}_{ag}")
            if train_label not in LABEL2ID:
                continue
            rows.append({
                "mcq_id_a": s["mcq_id_a"],
                "agent_slot": ag,
                "sentence": s["sentence"],
                "agent_role": s[role_key],
                "label": LABEL2ID[train_label],
                "label_str": train_label,
                "annotator_1": s.get(f"annotator_1_{ag}"),
                "annotator_2": s.get(f"annotator_2_{ag}"),
                "gold": s.get(f"gold_{ag}"),
                "human_gold": s.get(f"human_gold_{ag}"),
            })
    return rows


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc = self.tok(r["sentence"], r["agent_role"], truncation=True,
                       max_length=self.max_length, padding="max_length")
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(r["label"])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    train_sentences, val_sentences, test_sentences = load_release()

    train_rows = build_examples(train_sentences)
    val_rows   = build_examples(val_sentences)
    test_rows  = build_examples(test_sentences)
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    # Label distribution sanity check
    from collections import Counter
    train_dist = Counter(r["label_str"] for r in train_rows)
    print(f"train label dist: {dict(train_dist.most_common())}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(CHIARO_EMOTIONS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_ds = PairDataset(train_rows, tok)
    val_ds   = PairDataset(val_rows, tok)
    test_ds  = PairDataset(test_rows, tok)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH * 2,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
        seed=SEED,
        bf16=True,
    )

    trainer = Trainer(
        model=mdl,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Eval on test (in-distribution CHIARO 100 pilot)
    test_metrics = trainer.evaluate(test_ds)
    print(f"\nIn-distribution CHIARO test (N=100 sentences, 200 judgments):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Save best model in standard HF format for downstream eval
    best_path = os.path.join(OUT_DIR, "best")
    trainer.save_model(best_path)
    tok.save_pretrained(best_path)
    print(f"\nSaved best model to {best_path}")
    print(f"Trained with label source: {LABEL_SOURCE}")

    # Also save the test predictions for direct comparison with Exp A panel
    preds_out = trainer.predict(test_ds)
    logits = preds_out.predictions
    pred_labels = np.argmax(logits, axis=-1)
    test_results = []
    for r, p_id in zip(test_rows, pred_labels):
        test_results.append({**r, "pred": ID2LABEL[int(p_id)]})
    with open(os.path.join(OUT_DIR, "chiaro_test_predictions.json"), "w",
              encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"Wrote test predictions to chiaro_test_predictions.json")

    # Accuracy vs annotator_1 / annotator_2 / gold on test
    for label, key in [("A1", "annotator_1"), ("A2", "annotator_2"), ("Generation", "gold"), ("HumanGold", "human_gold")]:
        tot, hit = 0, 0
        for r in test_results:
            g = r.get(key); p = r["pred"]
            if g and p:
                tot += 1
                if g == p: hit += 1
        print(f"  vs {label}: {hit}/{tot} = {hit/tot*100:.1f}%")


if __name__ == "__main__":
    main()
