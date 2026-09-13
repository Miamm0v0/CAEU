# CHIARO

**Chiaroscuro for Emotions: A Contrastive Emotion Benchmark Grounded in Appraisal Theory**
Divyesh Bommana, Mohammad Saim, Tianyu Jiang

CHIARO is a 1,000-sentence benchmark for *contrastive emotion inference*: each sentence
describes one shared event involving two agents, and the task is to predict one fine-grained
emotion per agent (one positive, one negative) from situational context alone. No explicit
affect words appear in any sentence. The benchmark is grounded in appraisal theory and fully
human-annotated (two independent annotators plus adjudication, κ̄ = 0.827).

This repository contains everything needed to **re-evaluate, re-train, or regenerate** the
benchmark:

```
data/         the released benchmark (all label sources + frozen splits) and datasheet
evaluation/   evaluation scripts, the exact prompt, and raw predictions for all 11 models
generation/   the full two-stage generation + validation pipeline and every prompt
training/     RoBERTa fine-tuning scripts (CHIARO-only / GoEm-only / Combined) + external eval
```

## Quickstart

```bash
pip install -r requirements.txt

# Reproduce the paper's main tables from the shipped predictions (no API keys needed):
cd evaluation
python score_predictions.py
python score_predictions.py --per-emotion raw_predictions/llm/eval_openai_gpt_5_5.json
```

`score_predictions.py` recomputes, against the adjudicated human gold:
macro-F1 for all seven LLMs (Table 2), the per-emotion breakdown (Table 3),
the physical vs non-physical split (Table 5), the four off-the-shelf encoder
results, and the ~93 macro-F1 human ceiling, from the raw prediction files
in `evaluation/raw_predictions/`.

## Evaluate your own model

```bash
cd evaluation
OPENAI_API_KEY=... python eval_llm_joint.py --model <model-name> --out my_preds.json
# any OpenAI-compatible endpoint works via --base-url / --key-env
python score_predictions.py
```

The runner uses the paper's exact joint two-agent prompt (reproduced verbatim in the
paper's Evaluation Prompt appendix and in `eval_llm_joint.py`).

## Data and label sources

`data/chiaro_full.json` (1,000 scenes; schema in `data/README.md`) carries **every** label
source so any protocol can be reproduced or re-audited:

| field                                | meaning                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `human_gold_a/b`                   | adjudicated two-annotator gold;**the canonical evaluation labels** |
| `annotator_1_*`, `annotator_2_*` | each annotator's independent raw labels                                  |
| `generation_emotion_a/b`           | the emotion pair the scene was generated toward                          |
| `split`                            | frozen 800/100/100 train/val/test scene split                            |
| `version`                          | causal mode of the released variant (`physical` / `non_physical`)    |

All benchmark numbers in the paper (LLM and encoder tables) are scored against
`human_gold_a/b`. The fine-tuning study's in-distribution accuracies in the submitted
version were computed under a generation-label protocol; `training/` supports both
(`EXPB_LABEL_SOURCE=human_gold` is the default, `EXPB_LABEL_SOURCE=gold` reproduces the
submitted protocol).

## Regenerate the corpus

`generation/` contains the complete pipeline: Stage-1 draft prompt, Stage-2 paired
physical/non-physical renders, lexical constraints, six validators with a bounded repair
loop, the balanced-sampling driver, and the MCQ builder. Source AITA posts are **not**
redistributed; supply your own via the Reddit API (format in `generation/README.md`).

## License

Code is released under the MIT License. The dataset (`data/`) is released under
CC BY 4.0. See `LICENSE`.

## Citation
