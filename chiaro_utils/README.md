# CHIARO evaluation for CAEU

This directory evaluates a local full model or PEFT adapter on the frozen
CHIARO test set. Generation and metric computation are separate.

## Experimental factors

`--generation_schema` controls whether agents are predicted jointly or
separately and whether appraisal reasoning is explicit:

- `direct`: one joint two-agent call per scene, without appraisal reasoning.
- `separate`: two independent calls per scene, one for each agent, without
  appraisal reasoning.
- `chain`: two calls per scene, one from each agent's perspective, with the five
  CAEU appraisal dimensions before emotion prediction.
- `chain-joint`: one joint call per scene whose JSON contains separate
  `agent_a` and `agent_b` appraisal-reasoning and emotion objects.

`--emotion_mode` controls access to gold valence:

- `valence-constrained`: expose the five official same-valence MCQ options for
  each agent. `direct` uses the official CHIARO prompt and two-line response;
  `chain` adds first-person appraisal reasoning before selecting one option.
- `valence-free`: expose all 10 CHIARO labels without identifying the target
  agent's valence. Each agent emits CAREBench-style positive/negative label
  lists and intensities. Scoring selects the first ranked label from the side
  with greater intensity. The native emotion object is preserved, and an
  `evaluation_only_emotion_ranking` field globally ranks exactly the labels in
  the two native arrays for Candidate Hit and Hit@k.

The eight combinations should be written to different prediction files. JSON
output is directly compatible with CHIARO's `score_predictions.py`; JSONL is
also accepted by the local metric script.

All four schemas write the same top-level prediction fields,
`llm_emotion_A` and `llm_emotion_B`, so they use the same metric command.
`separate`, `chain`, and `chain-joint` additionally retain structured
`agent_a_output` and `agent_b_output` fields for inspection.

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

Valence-free predictions created before prompt version `chiaro-caeu-0.6` do
not contain the explicit cross-valence ranking. Regenerate them into a new
prediction file (recommended) or pass `--overwrite_predictions` before
reporting Hit@k.

Use `--generation_schema separate` for two independent direct predictions, or
`--generation_schema chain-joint` for one joint appraisal-to-emotion call.

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
For valence-free runs it also reports Candidate Hit, mean candidate count,
Hit@1, Hit@2, Hit@3, and explicit-ranking coverage. Candidate Hit uses the
union of `positive_labels` and `negative_labels`; Hit@k uses only
`evaluation_only_emotion_ranking` and never infers a cross-valence order from
the two intensity scores.
