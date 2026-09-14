# crowd-enVent first-person Policy evaluation

These scripts evaluate the CAREBench-trained Policy on native crowd-enVent
ratings and emotions without training or tuning on crowd-enVent.

## 1. Prepare the primary external test set

The default selects the 653 descriptions that are both confirmed real events
and included in the five-annotator validation set.  It uses `hidden_emo_text`,
masks all 12 emotion class-name tokens and dataset ellipses uniformly, and adds
a non-leaking first-person prefix.

```bash
python crowd-envent_utils/prepare_crowd_envent_eval.py \
  --subset strict-validated \
  --output_file FirstPersonMethod/data/crowd_envent/test.json
```

Other supported subsets are `strict-all` (3,528 confirmed-real events),
`all-except-imagined`, and `all`.

`--max_samples N` performs a deterministic random pilot selection using
`--sample_seed 42` by default; it does not take the category-sorted first N
rows.

## 2. Generate local Base/SFT/GRPO Policy predictions

`--generation_schema` has two meaningful choices:

- `chain` (default) runs two appraisal-mediated branches: situation -> five
  appraisals -> CAREBench ratings, and situation -> five appraisals -> native
  crowd-enVent emotion. The converter maps the CAREBench ratings to the 21
  crowd-enVent fields.
- `official-direct` follows the official paper's separate T->A and T->E task
  decomposition: situation -> 21 native ratings, and situation -> one of the 13
  native emotion labels plus 1--5 intensity. It does not generate appraisal
  reasoning and does not project ratings from CAREBench.

```bash
CUDA_VISIBLE_DEVICES=0 python crowd-envent_utils/generate_policy_crowd_envent.py \
  --generation_schema chain \
  --emotion_output_mode single-label \
  --eval_file FirstPersonMethod/data/crowd_envent/test.json \
  --model /path/to/adapter-or-model \
  --model_is_adapter \
  --base_model /path/to/base-model \
  --tokenizer /path/to/base-model \
  --dtype bf16 \
  --device cuda \
  --do_sample false \
  --enable_thinking false \
  --predictions_file output/crowd-envent/model-name/predictions.jsonl
```

For the official-style direct path, change only the schema and output path:

```bash
CUDA_VISIBLE_DEVICES=0 python crowd-envent_utils/generate_policy_crowd_envent.py \
  --generation_schema official-direct \
  --emotion_output_mode single-label \
  --eval_file FirstPersonMethod/data/crowd_envent/test.json \
  --model /path/to/adapter-or-model \
  --model_is_adapter \
  --base_model /path/to/base-model \
  --tokenizer /path/to/base-model \
  --dtype bf16 \
  --device cuda \
  --do_sample false \
  --enable_thinking false \
  --predictions_file output/crowd-envent/model-name/official-direct/predictions.jsonl
```

Both schemas make exactly two model calls per sample. A sample is valid only
when both branches pass strict parsing.

Emotion output is controlled independently with
`--emotion_output_mode {single-label,multi-label}`:

| Generation schema | `single-label` (default) | `multi-label` |
| --- | --- | --- |
| `chain` | five appraisals -> one native label and intensity | five appraisals -> unchanged CAEU positive/negative output plus an evaluation-only ranking |
| `official-direct` | situation -> one of 13 labels and intensity | situation -> a ranked list chosen from the 13 labels and one intensity |

For `official-direct + multi-label`, labels are ordered from most to least
likely. The converter uses the first label for standard crowd-enVent
single-label metrics and preserves the ranking in `multilabel_emotion.labels`.
For `chain + multi-label`, the native `emotion` object is unchanged and the
model additionally returns `evaluation_only_emotion_ranking`, containing
exactly the native positive/negative candidates in global likelihood order.
For chain multi-label evaluation, its first label is used by accuracy/F1 and
observer agreement metrics. Emotion intensity is taken from that label's
native positive or negative valence score.

The shared Transformers backend includes the Qwen3.5 wrapped-language-model
LoRA compatibility remapping.  Omit `--model_is_adapter`, `--base_model`, and
`--tokenizer` for a full/base model when its own tokenizer should be used.

## 3. Generate through an OpenAI-compatible API

```bash
python crowd-envent_utils/generate_policy_crowd_envent.py \
  --backend api \
  --api_provider glm \
  --api_model glm-5.1 \
  --eval_file FirstPersonMethod/data/crowd_envent/test.json \
  --api_thinking disabled \
  --predictions_file output/crowd-envent/glm-5.1/predictions.jsonl
```

GLM reads `ZAI_API_KEY` or `ZHIPUAI_API_KEY`; OpenAI reads `OPENAI_API_KEY`.

## 4. Compute automatic metrics

```bash
python crowd-envent_utils/compute_crowd_envent_metrics.py \
  --eval_file FirstPersonMethod/data/crowd_envent/test.json \
  --predictions_file output/crowd-envent/model-name/predictions.jsonl \
  --results_file output/crowd-envent/model-name/results.json
```

Primary metrics use the event author's self-reported ratings/emotion.  The
report also includes secondary agreement with the five outside validators.
Rating/intensity metrics include MAE, RMSE, normalized MAE/RMSE, exact
accuracy, and Spearman.  Emotion metrics include accuracy/micro-F1, macro-F1,
weighted-F1, and per-label scores.  `emotion_candidate_hit` additionally
reports whether the author's gold label occurs anywhere in the complete
candidate set. `emotion_hit_at_k.by_k` reports Hit@1, Hit@2, and Hit@3 using
only `multilabel_emotion.labels` for direct multi-label runs and
`evaluation_only_emotion_ranking` for chain multi-label runs. No ranking is
inferred from intensity or dataset label order. Treat candidate-set metrics as
diagnostics, not classification accuracy, and report them together with
`mean_candidate_count`.

Missing or invalid predictions are penalized by default through a neutral
failure prediction: all appraisal ratings are set to 3, emotion is set to
`no-emotion`, and intensity is set to 1.  The report keeps valid coverage and
metric coverage separate.  Use `--missing_prediction_policy error` to require
complete coverage, or `--allow_incomplete`/`--missing_prediction_policy skip`
for debugging valid outputs only.

The older combined entry point remains available for quick compatibility runs:
`python crowd-envent_utils/evaluate_policy_crowd_envent.py --mode run ...`.

## 5. Optional API rubric Judge for natural-language reasoning

crowd-enVent has no human natural-language rationale, so BLEU, ROUGE, and
BERTScore are not valid.  The optional Judge evaluates reasoning against the
situation and the author's structured 21-rating/emotion reference.

```bash
python crowd-envent_utils/judge_crowd_envent_reasoning.py \
  --eval_file FirstPersonMethod/data/crowd_envent/test.json \
  --predictions_file output/crowd-envent/model-name/predictions.jsonl \
  --judge_provider glm \
  --judge_model glm-5.1 \
  --judge_thinking disabled \
  --judge_max_concurrency 8
```

Judge API/parse failures go to a separate invalid JSONL and are penalized by
default with minimum rubric scores.  The aggregate report keeps valid Judge
coverage and metric coverage separate.  Use `--missing_judgment_policy error`
to require complete Judge coverage, or `--allow_incomplete` /
`--missing_judgment_policy skip` for debugging valid judgments only.

Existing judgments can be aggregated without an API key or new requests:

```bash
python crowd-envent_utils/judge_crowd_envent_reasoning.py \
  --mode metrics \
  --predictions_file output/crowd-envent/model-name/predictions.jsonl
```
