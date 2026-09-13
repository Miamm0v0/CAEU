# Evaluation

Everything here scores against the **adjudicated human gold** (`human_gold_a/b` in
`../data/chiaro_full.json`).

## Files

- **`score_predictions.py`**: canonical scorer. Recomputes the paper's Table 2
  (per-LLM macro-F1), Table 3 (per-emotion P/R/F1/support), Table 5 (physical vs
  non-physical macro-F1), the encoder table, and the human ceiling from the raw
  prediction files below. Run with no arguments for the full summary.
- **`eval_llm_joint.py`**: reference runner for the joint two-agent MCQ protocol
  against any OpenAI-compatible chat endpoint. Deterministic (temperature 0),
  resume-safe, and writes records `score_predictions.py` consumes directly.
- **`lr_baseline.py`**: TF-IDF + logistic-regression lexical baselines on the
  frozen train/val/test split (pair-input, sentence-only, and role-only variants,
  plus majority/random floors), scored with polarity-restricted prediction.
- **`raw_predictions/llm/`**: per-scene predictions for the seven LLMs benchmarked
  in the paper (GPT-5.5, Qwen 3.6 Plus, DeepSeek V4-Pro, Llama 3.3 70B,
  Gemini 3.5 Flash, Qwen3.5-27B, Qwen3.5-9B), including each model's raw response.
- **`raw_predictions/encoders/`**: per-scene predictions for the four off-the-shelf
  classifiers (ModernBERT-base/large GoEmotions, Emo Pillars contextless,
  Emollama-chat-7B). `llm_emotion_A/B` holds the polarity-restricted argmax over
  the five CHIARO emotions in the gold's polarity bucket, exactly as scored in the
  paper; unrestricted top-1 fields are preserved alongside.

## Protocol notes

- Joint prompt: both agents queried in a single call; the exact system/user messages
  are in the paper's Evaluation Prompt appendix and verbatim in `eval_llm_joint.py`.
- Option letters are rotated across scenes (each letter holds 19–21% of correct
  answers) and the positive agent occupies slot A in 52.6% of scenes, so neither
  slot order nor option position correlates with the answer.
- Records are joined to gold by sentence + agent role, so prediction files remain
  scoreable even if reordered.
