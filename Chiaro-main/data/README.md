# CHIARO

**Chiaroscuro for Emotions: A Two-Agent Contrastive Benchmark Grounded in Appraisal Theory**

CHIARO is a benchmark for *contrastive emotion inference*: given a sentence describing a single shared event involving two agents, predict one fine-grained emotion per agent, with the guarantee that one agent's emotion is positive and the other's is negative. No explicit affect words appear in any sentence; emotion must be recovered from situational context alone.

The benchmark is grounded in appraisal theory: two agents witnessing the same event under different goals or different agency should arrive at opposed emotions. CHIARO operationalizes that prediction.

---

## Overview

- **Task:** given a sentence and the short role descriptions of two agents in it, predict one emotion per agent from a 10-class taxonomy. Each scene is constructed so that exactly one agent's emotion is positive and the other's is negative.
- **Size:** 1,000 adjudicated sentences.
- **Source:** narratives drawn from the *r/AmItheAsshole* subreddit, rendered through a two-stage GPT-5.2 generation pipeline (draft + paired physical/non-physical render) with six validators and a repair loop.
- **Causal modes:** every scene is generated in two paired modes: a *physical* trigger (contact, force, object manipulation) and a *non-physical* trigger (overhearing, witnessing, announcing). Only one randomly chosen mode per scene is annotated and released.
- **Annotations:** every released sentence carries adjudicated gold labels for both agents, produced by two independent annotators with disagreements resolved through joint re-reading.

---

## Emotion taxonomy

Ten emotions, balanced across polarity, derived from GoEmotions and disambiguated under appraisal-theoretic criteria. Each emotion is paired with a *mandatory trigger* the generated scene must instantiate.

| Positive   | Negative      |
| ---------- | ------------- |
| joy        | anger         |
| pride      | sadness       |
| relief     | fear          |
| gratitude  | disgust       |
| excitement | embarrassment |

---

## Dataset statistics

| Statistic               | Value                          |
| ----------------------- | ------------------------------ |
| Released sentences      | 1,000                          |
| Physical / non-physical | 527 / 473                      |
| Agents per scene        | 2 (one positive, one negative) |
| Gold labels per scene   | 2 (one per agent)              |
| Annotators per scene    | 2 (independent + adjudicated)  |

### Per-emotion distribution (% of agent-emotion labels within polarity)

| Positive   |    % | Negative      |    % |
| ---------- | ---: | ------------- | ---: |
| gratitude  | 22.4 | anger         | 25.4 |
| relief     | 21.9 | embarrassment | 20.9 |
| joy        | 20.9 | fear          | 19.9 |
| excitement | 18.6 | sadness       | 17.4 |
| pride      | 16.1 | disgust       | 16.4 |

### Inter-annotator agreement (Cohen's κ)

| Slot                     |      Cohen's κ | Raw agreement |
| ------------------------ | --------------: | ------------: |
| Positive slot            |           0.798 |         83.9% |
| Negative slot            |           0.855 |         88.5% |
| **Average (κ̄)** | **0.827** |            - |

Neither annotator ever crossed polarity on either slot, so all residual disagreement is fine-grained within polarity.

---

## What will be in the release

Once unblinded, the release will contain (filenames indicative; subject to minor change at release time):

```
chiaro_data_release/
  README.md                     ← this file
  chiaro_full.json              ← 1,000 adjudicated scenes (the benchmark)
  chiaro_test.json              ← held-out test split used in the paper
  LICENSE                       ← license terms (TBD; see below)
  schema.md                     ← field-level documentation
```

### Schema (planned)

Each row will be a JSON object with the following fields:

| Field                      | Type   | Description                                              |
| -------------------------- | ------ | -------------------------------------------------------- |
| `id`                     | string | Stable scene identifier.                                 |
| `split`                  | string | `train` / `test` partition.                          |
| `version`                | string | `physical` or `non_physical`.                        |
| `sentence`               | string | The released sentence (no explicit affect words).        |
| `agent_a_role`           | string | Short identifier or role description for Agent A.        |
| `agent_b_role`           | string | Short identifier or role description for Agent B.        |
| `human_gold_a`           | string | Adjudicated gold emotion for Agent A.                    |
| `human_gold_b`           | string | Adjudicated gold emotion for Agent B.                    |
| `annotator_1_A` / `_B` | string | Pre-adjudication labels from annotator 1.                |
| `annotator_2_A` / `_B` | string | Pre-adjudication labels from annotator 2.                |
| `options_a`              | object | Polarity-filtered 5-option choice set shown for Agent A. |
| `options_b`              | object | Polarity-filtered 5-option choice set shown for Agent B. |
| `correct_a` / `_b`     | string | Correct option letter for each agent.                    |

---

## Benchmark headline (from the paper)

Seven frontier LLMs and four off-the-shelf emotion classifiers are evaluated under the same joint two-agent prompt that human annotators saw. Reported as macro-F1 against the adjudicated human gold:

| Model                      |       Macro-F1 |
| -------------------------- | -------------: |
| GPT-5.5                    | **67.3** |
| Qwen 3.6 Plus              |           66.9 |
| DeepSeek V4-Pro            |           66.5 |
| Qwen3.5-27B (open weights) |           66.3 |
| Llama 3.3 70B              |           66.3 |
| Gemini 3.5 Flash           |           64.1 |
| Qwen3.5-9B (open weights)  |           59.9 |

The strongest LLM sits well below the human inter-annotator ceiling (κ̄ = 0.827), with errors concentrated on the positive polarity: *joy* and *gratitude* are systematically mislabeled as *relief*. Off-the-shelf emotion classifiers transfer to CHIARO only at chance level. Full per-emotion analysis, physical vs non-physical breakdowns, and the RoBERTa fine-tuning experiments are reported in the paper.

---

## Intended use

CHIARO is intended for:

1. **Evaluation** of contrastive, role-conditioned emotion inference in language models, both as a standalone benchmark and as a diagnostic for appraisal-theoretic competence.
2. **Training signal** for fine-grained emotion classifiers, particularly in combination with single-experiencer emotion corpora; the paper reports gains on CHIARO itself and on six of ten external emotion benchmarks when CHIARO is combined with an existing emotion dataset.
3. **Interpretability and analysis** of how LLMs encode positive vs negative emotion subspaces, given the controlled paired structure (same event, two agents).
