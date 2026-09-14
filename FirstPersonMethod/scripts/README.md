# Scripts Overview

This directory contains runnable scripts and prompt configs for the following paper:
**CAREBench: Evaluating LLMs' Emotion Understanding by Assessing Cognitive Appraisal Reasoning**

## Data
Data are available on [Huggingface](https://huggingface.co/datasets/zhaoyuesun/CAREBecnch) under the CC-BY-NC-ND 4.0 license.


## Scripts

### Inference Scripts:
- baseline_api.py: API-based baseline runner for step-by-step tasks (`appraisals`, emotion levels, emotion labels, `core-appraisals`, or `all`).
- baseline_api_with_appraisals.py: API-based runner that uses appraisal rating contexts to predict emotion tasks (`positive-level`, `negative-level`, `positive-labels`, `negative-labels`).
- baseline_api_with_cog_story.py: API-based runner that conditions on story with core-appraisal context for appraisal/emotion tasks.
- baseline_api_with_cog_story_and_appraisals.py: API-based runner that conditions on both core-appraisal and appraisal signals for emotion-task prediction.
- baseline_api_with_pred_cog_story.py: API-based runner that reads stories with model-generated core-appraisal reasonings (`final_scenario`) and runs appraisal/emotion tasks.
- counterfactual_api.py: API-based counterfactual evaluator that runs appraisal scoring over `data/counterfactual/<dimension>/*.json`.
- counterfactual_emotion_api.py: API-based counterfactual evaluator for emotion tasks only (levels and labels).
- summarise_pred_cog_stories.py: Builds coherent narrative from structured predicted core-appraisal answers.
- baseline_vllm_*.py: corresponding scripts for vLLM runner.
- counterfactual_prefix_transformers.py: paired assistant-prefix intervention
  runner. It uses native baseline prompts and continues an original or
  one-dimension-intervened appraisal prefix to ratings and/or emotion.


### Evaluation Scripts
- evaluate_counterfactual.py: Counterfactual appraisal evaluator that reports per-dimension correlation between model deltas and human deltas.
- evaluate_counterfactual_emotion.py: Counterfactual emotion evaluator for level/label deltas with per-dimension correlations.
- evaluate_human.py: Human-eval aggregator that compares third-person annotations to first-person gold and writes global evaluation outputs.
- evaluate_per_sample.py: First-person evaluator that outputs per-sample metrics instead of only aggregate summaries.


## Scripts/prompts

- baseline_prompt.toml: Main baseline prompt pack (appraisal statements, emotion levels, emotion labels, and core-appraisal question templates + label maps).
- baseline_with_appraisal_prompt.toml: Prompt pack for runs that inject appraisal context before predicting emotion tasks.
- baseline_with_single_core_appraisal.toml: Prompt pack for single-core-appraisal-conditioned runs plus emotion-task templates.
- counterfactual_prompt.toml: Prompt pack for counterfactual appraisal scoring (dimension statements + appraisal label map).
- summarsie_pred_cog_stories.toml: Prompt template used to shaping structured core-appraisal answers into coherent stories.

## Appraisal-prefix intervention

The current intervention protocol does not append appraisal text to the user
scenario. Run these stages in order:

1. Build native-situation files and separate third-person intervention fields:

```bash
python CAREBench/scripts/build_appraisal_counterfactual_data.py \
  --input <first-person-json> \
  --third_person_input <third-person-json> \
  --first_person_root <data-root>/origin \
  --counterfactual_root <data-root>/counterfactual \
  --replacement_granularity core \
  --original_context situation-only
```

2. Generate one normal model-authored appraisal trajectory per situation:

```bash
python FirstPersonMethod/scripts/baseline_transformers.py \
  --task chain-emotion \
  --source_folder <data-root>/origin \
  --target_folder <baseline-run> \
  --model <model-or-adapter> \
  --base_model <base-model-if-needed> \
  --do_sample false \
  --enable_thinking false
```

3. Continue paired original and intervened assistant prefixes:

```bash
python CAREBench/scripts/counterfactual_prefix_transformers.py \
  --outcome all \
  --first_person_root <data-root>/origin \
  --source_root <data-root>/counterfactual \
  --baseline_root <baseline-run> \
  --target_root <prefix-run> \
  --model <model-or-adapter> \
  --base_model <base-model-if-needed> \
  --do_sample false \
  --enable_thinking false
```

The runner writes evaluator-compatible original emotion predictions under
`<prefix-run>/origin/<task>/`, counterfactual predictions under
`<prefix-run>/counterfactual/<task>/<dimension>/`, complete rating readouts
under the corresponding `appraisals/` folders, and auditable prefixes and
continuations under `<prefix-run>/trajectories/`.

4. Evaluate the paired emotion continuations:

```bash
python CAREBench/scripts/evaluate_counterfactual_emotion.py \
  --first_person_root <data-root>/origin \
  --counterfactual_gold_root <data-root>/counterfactual \
  --baseline_root <prefix-run>/origin \
  --counterfactual_pred_root <prefix-run>/counterfactual \
  --prompt_path FirstPersonMethod/scripts/prompts/baseline_prompt.toml \
  --dimension_source gold \
  --output_file <prefix-run>/results_emotion.json
```

5. Evaluate the paired rating continuations. The evaluator automatically maps
each core intervention folder to its CAREBench rating dimensions:

```bash
python CAREBench/scripts/evaluate_counterfactual.py \
  --first_person_root <data-root>/origin \
  --counterfactual_gold_root <data-root>/counterfactual \
  --baseline_root <prefix-run>/origin \
  --counterfactual_pred_root <prefix-run>/counterfactual \
  --prompt_path FirstPersonMethod/scripts/prompts/baseline_prompt.toml \
  --dimension_source gold \
  --output_file <prefix-run>/results_ratings.json
```

`counterfactual_emotion_transformers.py` and
`counterfactual_transformers.py` are convenience entry points for emotion-only
and ratings-only runs. Use `counterfactual_prefix_transformers.py --outcome all`
for the fully paired experiment. The older vLLM scripts still implement the
legacy scenario-appending protocol and must not be mixed with these results.
