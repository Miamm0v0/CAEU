#!/usr/bin/env python3
"""Warm-up SFT for a configurable-appraisal Policy used by API-Judge GRPO.

This stage uses the exact Policy prompt and JSON contract from ``spec.py``.
CAREBench open-ended human appraisals are mapped to the canonical Policy schema:

- certainty -> epistemic
- congruence -> goal_congruence
- accountability -> agency_accountability
- control -> control_coping_potential

CAREBench has no open-ended norm/value annotation. When selected, that dimension
receives a conservative first-person target that explicitly avoids inventing a
value judgment. Use ``--missing_norm_value_policy error`` once a curated
norm/value target is available.

Preview one transformed sample without loading the training stack:

    python FirstPersonMethod/scripts/train_grpo/train_policy_warmup_sft.py \
      --train_file train.json --eval_ratio 0 --max_train_samples 8 \
      --preview_only

QLoRA warm-up:

    python FirstPersonMethod/scripts/train_grpo/train_policy_warmup_sft.py \
      --model_name_or_path Qwen/Qwen3-8B \
      --train_file train.json --eval_file dev.json \
      --appraisal_dimensions relevance,epistemic,goal_congruence,agency_accountability,control_coping_potential \
      --max_train_samples 100 --load_in_4bit --use_wandb \
      --output_dir FirstPersonMethod/output/policy_warmup_sft

The saved full model or PEFT adapter can be passed directly to GRPO through
``--model_name_or_path``; local adapters are detected automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from sft_common import (
        NEGATIVE_LABEL_MAP,
        NEGATIVE_LABELS,
        POSITIVE_LABEL_MAP,
        POSITIVE_LABELS,
        build_parser,
        run_sft,
    )
    from train_grpo.parsing import parse_policy_output
    from train_grpo.spec import (
        APPRAISAL_DIMENSIONS,
        CAREBENCH_REASONING_KEYS,
        RUBRIC_VERSION,
        build_policy_system_prompt,
        build_policy_user_prompt,
        resolve_appraisal_dimensions,
    )
else:
    from ..sft_common import (
        NEGATIVE_LABEL_MAP,
        NEGATIVE_LABELS,
        POSITIVE_LABEL_MAP,
        POSITIVE_LABELS,
        build_parser,
        run_sft,
    )
    from .parsing import parse_policy_output
    from .spec import (
        APPRAISAL_DIMENSIONS,
        CAREBENCH_REASONING_KEYS,
        RUBRIC_VERSION,
        build_policy_system_prompt,
        build_policy_user_prompt,
        resolve_appraisal_dimensions,
    )


LEGACY_REASONING_KEYS = CAREBENCH_REASONING_KEYS

DEFAULT_MISSING_NORM_VALUE_TARGET = (
    "I do not have enough evidence from this event to identify a clear "
    "alignment or conflict with my norms or values."
)


def require_nonempty_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def normalize_warmup_reasoning(
    record: dict[str, Any],
    missing_norm_value_policy: str,
    appraisal_dimensions: list[str] | str | None = None,
) -> dict[str, str]:
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    sample_id = str(record.get("id", "<unknown>"))
    raw = record.get("appraisal_reasoning")
    if not isinstance(raw, dict):
        raise ValueError(f"{sample_id}: appraisal_reasoning must be an object")

    reasoning: dict[str, str] = {}
    for policy_dimension in selected_dimensions:
        if policy_dimension == "norm_value_compatibility":
            continue
        legacy_key = LEGACY_REASONING_KEYS.get(
            policy_dimension,
            policy_dimension,
        )
        value = raw.get(policy_dimension, raw.get(legacy_key))
        reasoning[policy_dimension] = require_nonempty_text(
            value,
            f"{sample_id}: appraisal_reasoning.{legacy_key}",
        )

    if "norm_value_compatibility" in selected_dimensions:
        norm_value = raw.get("norm_value_compatibility")
        if isinstance(norm_value, str) and norm_value.strip():
            reasoning["norm_value_compatibility"] = norm_value.strip()
        elif missing_norm_value_policy == "neutral":
            reasoning["norm_value_compatibility"] = (
                DEFAULT_MISSING_NORM_VALUE_TARGET
            )
        elif missing_norm_value_policy == "error":
            raise ValueError(
                f"{sample_id}: appraisal_reasoning.norm_value_compatibility "
                "is required by --missing_norm_value_policy error"
            )
        else:
            raise ValueError(
                "missing_norm_value_policy must be either 'neutral' or 'error'"
            )

    if set(reasoning) != set(selected_dimensions):
        raise RuntimeError("Warm-up reasoning does not match the Policy schema")
    return reasoning


def require_bounded_int(
    value: Any,
    minimum: int,
    maximum: int,
    location: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{location} must be in [{minimum}, {maximum}]")
    return value


def normalize_warmup_emotion(record: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize labels by ontology, including misbucketed source labels."""
    sample_id = str(record.get("id", "<unknown>"))
    raw_emotion = record.get("emotion")
    if not isinstance(raw_emotion, dict):
        raise ValueError(f"{sample_id}: emotion must be an object")

    selected_positive: set[str] = set()
    selected_negative: set[str] = set()
    for source_field in ("positive_labels", "negative_labels"):
        labels = raw_emotion.get(source_field)
        if not isinstance(labels, list):
            raise ValueError(
                f"{sample_id}: emotion.{source_field} must be an array"
            )
        for raw_label in labels:
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError(
                    f"{sample_id}: emotion.{source_field} contains an "
                    "invalid label"
                )
            label = raw_label.strip()
            lowered = label.lower()
            if label in POSITIVE_LABELS:
                selected_positive.add(label)
            elif label in NEGATIVE_LABELS:
                selected_negative.add(label)
            elif lowered in POSITIVE_LABEL_MAP:
                selected_positive.add(POSITIVE_LABEL_MAP[lowered])
            elif lowered in NEGATIVE_LABEL_MAP:
                selected_negative.add(NEGATIVE_LABEL_MAP[lowered])
            else:
                raise ValueError(
                    f"{sample_id}: unknown CAREBench emotion label "
                    f"{raw_label!r}"
                )

    emotion = {
        "positive_intensity": require_bounded_int(
            raw_emotion.get("positive_intensity"),
            0,
            6,
            f"{sample_id}: emotion.positive_intensity",
        ),
        "negative_intensity": require_bounded_int(
            raw_emotion.get("negative_intensity"),
            0,
            6,
            f"{sample_id}: emotion.negative_intensity",
        ),
        "positive_labels": [
            label for label in POSITIVE_LABELS if label in selected_positive
        ],
        "negative_labels": [
            label for label in NEGATIVE_LABELS if label in selected_negative
        ],
    }
    if emotion["positive_intensity"] == 0 and emotion["positive_labels"]:
        raise ValueError(
            f"{sample_id}: positive labels are present at zero intensity"
        )
    if emotion["negative_intensity"] == 0 and emotion["negative_labels"]:
        raise ValueError(
            f"{sample_id}: negative labels are present at zero intensity"
        )
    return emotion


def build_warmup_example(
    record: dict[str, Any],
    missing_norm_value_policy: str = "neutral",
    enable_thinking: bool = False,
    appraisal_dimensions: list[str] | str | None = None,
) -> dict[str, Any]:
    selected_dimensions = resolve_appraisal_dimensions(
        appraisal_dimensions
    )
    if not isinstance(record, dict):
        raise ValueError("Each training record must be an object")
    sample_id = require_nonempty_text(record.get("id"), "id")
    situation = require_nonempty_text(
        record.get("situation"),
        f"{sample_id}: situation",
    )
    perspective = record.get("perspective")
    if perspective is not None and perspective != "first_person":
        raise ValueError(f"{sample_id}: expected first_person perspective")

    target = {
        "appraisal_reasoning": normalize_warmup_reasoning(
            record,
            missing_norm_value_policy,
            selected_dimensions,
        ),
        "emotion": normalize_warmup_emotion(record),
    }
    completion = json.dumps(target, ensure_ascii=False, indent=2)
    parse_policy_output(completion, selected_dimensions)
    return {
        "sample_id": sample_id,
        "prompt": [
            {
                "role": "system",
                "content": build_policy_system_prompt(selected_dimensions),
            },
            {
                "role": "user",
                "content": build_policy_user_prompt(
                    situation,
                    selected_dimensions,
                ),
            },
        ],
        "completion": [{"role": "assistant", "content": completion}],
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
        },
    }


def build_warmup_parser() -> argparse.ArgumentParser:
    parser = build_parser(
        description=(
            "Policy warm-up SFT: situation -> selected appraisals -> "
            "CAREBench emotion"
        ),
        default_output_dir="FirstPersonMethod/output/policy_warmup_sft",
        default_max_length=3072,
    )
    parser.set_defaults(
        train_file="train.json",
        eval_ratio=0.0,
        invalid_record_policy="skip",
    )
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(APPRAISAL_DIMENSIONS),
        metavar="DIM1,DIM2,...",
        help=(
            "Comma-separated appraisal dimensions, or 'all'. Input order is "
            f"canonicalized to: {','.join(APPRAISAL_DIMENSIONS)}"
        ),
    )
    parser.add_argument(
        "--missing_norm_value_policy",
        choices=["neutral", "error"],
        default="neutral",
        help=(
            "neutral inserts a conservative target when CAREBench lacks the "
            "norm/value appraisal; error requires a curated target"
        ),
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help=(
            "Enable a tokenizer's thinking-mode chat template. Keep disabled "
            "to match the default GRPO Policy configuration"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_warmup_parser().parse_args(argv)
    builder = partial(
        build_warmup_example,
        missing_norm_value_policy=args.missing_norm_value_policy,
        enable_thinking=args.enable_thinking,
        appraisal_dimensions=args.appraisal_dimensions,
    )
    return run_sft(
        args,
        method_name="policy_warmup_sft",
        example_builder=builder,
        manifest_extra={
            "rubric_version": RUBRIC_VERSION,
            "supported_appraisal_dimensions": APPRAISAL_DIMENSIONS,
            "policy_appraisal_dimensions": args.appraisal_dimensions,
            "legacy_reasoning_mapping": {
                dimension: LEGACY_REASONING_KEYS.get(dimension, dimension)
                for dimension in args.appraisal_dimensions
            },
            "missing_norm_value_policy": args.missing_norm_value_policy,
            "enable_thinking": args.enable_thinking,
            "default_missing_norm_value_target": (
                DEFAULT_MISSING_NORM_VALUE_TARGET
                if args.missing_norm_value_policy == "neutral"
                else None
            ),
            "policy_system_prompt": build_policy_system_prompt(
                args.appraisal_dimensions
            ),
            "target_schema": {
                "top_level": ["appraisal_reasoning", "emotion"],
                "strict_json": True,
            },
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
