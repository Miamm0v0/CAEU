# CHIARO evaluation for CAEU

This directory evaluates a local full model or PEFT adapter on the frozen
CHIARO test set. Generation and metric computation are separate.

## Experimental factors

`--generation_schema` controls whether appraisal reasoning is explicit:

- `direct`: one joint two-agent call per scene, without appraisal reasoning.
- `chain`: two calls per scene, one from each agent's perspective, with the five
  CAEU appraisal dimensions before emotion prediction.

`--emotion_mode` controls access to gold valence:

- `valence-constrained`: expose the five official same-valence MCQ options for
  each agent. `direct` uses the official CHIARO prompt and two-line response.
- `valence-free`: expose all 10 CHIARO labels without identifying the target
  agent's valence. Each agent emits CAREBench-style positive/negative label
  lists and intensities. Scoring selects the first ranked label from the side
  with greater intensity.

The four combinations should be written to different prediction files. JSON
output is directly compatible with CHIARO's `score_predictions.py`; JSONL is
also accepted by the local metric script.

## Generate

```bash
CUDA_VISIBLE_DEVICES=0 python chiaro_utils/generate_policy_chiaro.py \
  --generation_schema chain \
  --emotion_mode valence-free \
  --eval_file /data/liyanhong/Appraisal_emotion/Chiaro-main/data/chiaro_test.json \
  --model /data/liyanhong/Appraisal_emotion/output/model_trained/checkpoint-750 \
  --base_model /data/liyanhong/model/Qwen3.5-9B \
  --tokenizer /data/liyanhong/model/Qwen3.5-9B \
  --dtype bf16 --device cuda \
  --do_sample false --enable_thinking false \
  --predictions_file /data/liyanhong/Appraisal_emotion/output_chiaro/ours/chain/valence-free/predictions.json
```

For a quick smoke run, add `--max_samples 5`. Resume is automatic; use
`--overwrite_predictions` to intentionally replace an existing run.

For the official direct protocol, use:

```bash
CUDA_VISIBLE_DEVICES=0 python chiaro_utils/generate_policy_chiaro.py \
  --generation_schema direct \
  --emotion_mode valence-constrained \
  --eval_file /data/liyanhong/Appraisal_emotion/Chiaro-main/data/chiaro_test.json \
  --model /data/liyanhong/model/Qwen3.5-9B \
  --tokenizer /data/liyanhong/model/Qwen3.5-9B \
  --dtype bf16 --device cuda \
  --do_sample false --enable_thinking false \
  --predictions_file /data/liyanhong/Appraisal_emotion/output_chiaro/base/direct/valence-constrained/predictions.json
```

## Compute metrics

```bash
python chiaro_utils/compute_chiaro_metrics.py \
  --eval_file /data/liyanhong/Appraisal_emotion/Chiaro-main/data/chiaro_test.json \
  --predictions_file /data/liyanhong/Appraisal_emotion/output_chiaro/ours/chain/valence-free/predictions.json \
  --results_file /data/liyanhong/Appraisal_emotion/output_chiaro/ours/chain/valence-free/results.json
```

The report includes accuracy, 10-class macro-F1, per-emotion metrics,
physical/non-physical results, parse coverage, and two-agent pair accuracy.
