#!/usr/bin/env python3
"""CLI entry for configurable-appraisal Policy GRPO with an API LLM Judge.

Implementation is split across this package:

- ``spec.py``: configurable appraisal task contract and prompts
- ``data.py``: CAREBench data preparation and hidden Judge references
- ``parsing.py``: strict Policy/Judge parsing and grounded reward aggregation
- ``api_judge.py``: asynchronous API calls, retries, cache, and skip semantics
- ``runtime.py``: W&B, tokenizer, model, quantization, and PEFT helpers
- ``trainer.py``: GRPO configuration, training, and reproducibility artifacts
- ``config.py`` / ``cli.py``: command-line arguments and orchestration

Direct execution from the repository root remains supported::

    python FirstPersonMethod/scripts/train_grpo/train_policy_grpo_api_judge.py \
      --train_file train.json --preview_only

For a GLM Judge, set ``ZAI_API_KEY`` and select the provider::

    python FirstPersonMethod/scripts/train_grpo/train_policy_grpo_api_judge.py \
      --judge_provider glm --judge_model glm-4.5
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_grpo.cli import main
else:
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
