# Fine-tuning (CHIARO as a training signal)

RoBERTa-large fine-tuning scripts for the paper's training study: three checkpoints
that differ only in their training corpus, evaluated in-distribution and on ten
external emotion benchmarks.

## Files

- **`chiaro_data.py`**: loads `../data/chiaro_full.json` into the frozen
  800/100/100 scene split, exposing every label source.
- **`train_roberta_chiaro.py`**: *CHIARO-only*: each scene becomes two pair-input
  examples (`sentence </s></s> agent_role`, 1,600 training examples), 10-class head.
- **`train_goemotions_matched_1600.py`**: *GoEm-only*: 1,600 GoEmotions items
  soft-stratified across CHIARO's ten emotions (160/class soft cap; rare classes
  contribute what they have, the shortfall is filled from frequent classes).
- **`train_combined.py`**: *Combined*: union of both corpora (3,200 examples),
  each source keeping its native input shape.
- **`eval_external.py`**: evaluates a checkpoint on the external benchmarks
  (loaded from the Hugging Face hub), filtered to items whose gold is one of
  CHIARO's ten emotions.

## Label source

`EXPB_LABEL_SOURCE` selects the training/eval labels:

- `human_gold` (default): the adjudicated gold, canonical for the release
- `gold`: the generation-label protocol used in the submitted paper's §5.2
- `annotator_1` / `annotator_2`: single-annotator robustness variants

`EXPB_SEED` sets the seed (paper default 42). Scripts expect a single CUDA GPU
(RoBERTa-large, batch 16, 5 epochs, lr 2e-5); each run finishes in minutes on an
A100-class card.
