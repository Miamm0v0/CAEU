# Evidence-grounded GRPO

GRPO prompts request `appraisal_reasoning`, `emotion_evidence`, and the native
CAREBench `emotion` object. Each selected dimension contains its own `evidence`
array first and `appraisal` text second:

```json
{
  "relevance": {
    "evidence": ["I passed my exam"],
    "appraisal": "I see this success as important to my plans."
  },
  "epistemic": {
    "evidence": [],
    "appraisal": "insufficient_information"
  }
}
```

Each evidence array contains up to two short verbatim situation quotes. Use
`[]` and `insufficient_information` when that dimension lacks evidence.
Emotion evidence contains `situation_quotes`, `appraisal_dimensions`, and a
concise `inference`. Existing warmup SFT targets retain their original schema;
an existing warmup adapter can initialize the new GRPO run.

Automatic checks allow whitespace normalization but not paraphrased quotes.
A nonexistent quote, unsupported empty citation, or quotation attached to an
abstention zeros that dimension's evidence score. The Judge separately scores
semantic support and whether abstention is appropriate. Invalid emotion quotes
or references to unavailable appraisals zero the linkage component. Merely
quoting text does not establish that the inference follows from it.

## Reward

- `A`: mean validity and experiencer-fidelity scores, divided by 4.
- `g_d`: per-dimension grounding score divided by 4, zeroed for invalid evidence.
- `G`: `0.7 * mean(g_d) + 0.3 * mean(lowest two g_d)`; a one-dimension run
  uses its single grounding score for both terms.
- `C`: coherence score divided by 4.
- `T`: linkage score divided by 4, subject to emotion-evidence checks.
- `L`: mean positive/negative gold-label set F1; empty/empty is 1.
- `I`: mean positive/negative `1 - abs(predicted_intensity - gold_intensity)/6`.

The default raw total is:

```text
R = 0.25*A + 0.20*G + 0.05*C + 0.10*(G*T) + 0.40*(0.8*L + 0.2*I)
```

The raw total is standardized once within each generation group. TRL-level
reward scaling is disabled. Component diagnostics all use the total's standard
deviation, so their logged weighted contributions sum to the training signal.
Equal-total groups produce zero signals. Judge failures are excluded; malformed
policy JSON receives the configured invalid component penalty.

Use a new output directory. Old caches are separated by the new rubric version
and the full Judge prompt. Existing commands with explicit weights override
the new defaults; use these settings for the proposed scheme:

```bash
--appraisal_reward_weight 0.25 --grounding_reward_weight 0.20 \
--coherence_reward_weight 0.05 --transition_reward_weight 0.10 \
--outcome_reward_weight 0.40 \
--outcome_label_weight 0.8 --outcome_intensity_weight 0.2
```

`--use_process_gate` now gates linkage by grounding, never gold outcome.
Its mode is `product`. `--no_process_gate` leaves linkage ungated.
Keep appraisal inferences and quotes concise to fit `--max_completion_length`.

## Audit artifacts

`api_judge_rubric.json` saves the policy prompts, evidence schema, Judge prompt,
and reward aggregation. `judge_cache.sqlite3` retains situation, parsed
candidate, hidden reference, evidence checks, and per-dimension Judge scores.
Completion logs expose `reward_contributions`, `evidence_diagnostics`, raw
reward components, and diagnostics centered using the total's standard
deviation. `raw_total_reward` is also logged as a training metric.

Quote provenance is automatic; semantic support still depends on Judge quality.
No new SFT targets or evidence annotations are fabricated from human references.
