# Generation pipeline

The complete two-stage pipeline that produced CHIARO, including every prompt.

## Files

- **`generate_stories_v2.py`**: the core pipeline. Stage 1 (draft): the system
  message is `DRAFT_PROMPT_TEMPLATE` (constraints, emotion-taxonomy trigger block,
  contrastive scenario type); the selected AITA post is passed separately as the
  **user** message, prefixed `"Story inspiration:"` and truncated to its first
  2,000 characters. Stage 2 (paired render): `PHYS_RENDER_PROMPT` and
  `NONPHYS_RENDER_PROMPT` (sharing `_RENDER_CORE`) turn each draft into a physical
  and a non-physical version. Six programmatic validators (valence contrast,
  lexical constraints, length, person-reference count, span consistency, role-head
  collision) gate every version, with a bounded repair loop (`_REPAIR_SYSTEM`).
- **`generate_stories_balanced.py`**: balanced-sampling driver: enforces per-class
  quotas by prepending a forced-target emotion pair to the draft prompt.
- **`generate_mcq_v2.py`**: builds the five-option MCQ per agent (polarity-filtered
  option sets, rotated correct-letter positions).
- **`noun_count_validator.py`**: the person-reference-count validator.

## Requirements

- `OPENAI_API_KEY` in the environment (or a `.env` file); the paper used
  `gpt-5.2` (`MODEL` env var).
- **Source posts are not redistributed.** Set `REDDIT_DUMP` to a JSONL file of
  posts collected via the Reddit API, one JSON object per line with at least
  `title` and `selftext` fields (r/AmItheAsshole in the paper). Keyword-based
  post selection is described in the paper's appendix.

Outputs are one JSON file per scene (`story_<idx>_*.json`) containing the draft,
both rendered versions, and validator metadata; runs are resume-safe.
