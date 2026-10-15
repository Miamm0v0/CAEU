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

Generalization-oriented warm-up keeps the full output contract while reducing
CAREBench-specific outcome pressure and arbitrary multi-label order bias:

    --emotion_loss_weight 0.25 --emotion_label_order sample-shuffle

The saved full model or PEFT adapter can be passed directly to GRPO through
``--model_name_or_path``; local adapters are detected automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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


def _deterministically_shuffle_labels(
    labels: list[str],
    sample_id: str,
    valence: str,
    seed: int,
) -> list[str]:
    shuffled = list(labels)
    digest = hashlib.sha256(
        f"{seed}:{sample_id}:{valence}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(shuffled)
    return shuffled


def normalize_warmup_emotion(
    record: dict[str, Any],
    label_order: str = "canonical",
    label_order_seed: int = 42,
) -> dict[str, Any]:
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

    positive_labels = [
        label for label in POSITIVE_LABELS if label in selected_positive
    ]
    negative_labels = [
        label for label in NEGATIVE_LABELS if label in selected_negative
    ]
    if label_order == "sample-shuffle":
        positive_labels = _deterministically_shuffle_labels(
            positive_labels,
            sample_id,
            "positive",
            label_order_seed,
        )
        negative_labels = _deterministically_shuffle_labels(
            negative_labels,
            sample_id,
            "negative",
            label_order_seed,
        )
    elif label_order != "canonical":
        raise ValueError(
            "emotion label_order must be 'canonical' or 'sample-shuffle'"
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
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
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
    emotion_label_order: str = "canonical",
    label_order_seed: int = 42,
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
        "emotion": normalize_warmup_emotion(
            record,
            label_order=emotion_label_order,
            label_order_seed=label_order_seed,
        ),
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


def _bounded_emotion_loss_weight(value: str) -> float:
    try:
        weight = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "emotion loss weight must be a number"
        ) from exc
    if not 0.0 < weight <= 1.0:
        raise argparse.ArgumentTypeError(
            "emotion loss weight must be in (0, 1]"
        )
    return weight


def _find_last_subsequence_start(
    token_ids: list[int],
    pattern: list[int],
) -> int | None:
    if not pattern or len(pattern) > len(token_ids):
        return None
    last_start = None
    limit = len(token_ids) - len(pattern) + 1
    for start in range(limit):
        if token_ids[start : start + len(pattern)] == pattern:
            last_start = start
    return last_start


def _emotion_value_char_spans(serialized_emotion: str) -> list[tuple[int, int]]:
    """Locate only scalar intensity values and strings inside label arrays."""
    spans: list[tuple[int, int]] = []
    for field in ("positive_intensity", "negative_intensity"):
        match = re.search(
            rf'"{re.escape(field)}"\s*:\s*(-?\d+)',
            serialized_emotion,
        )
        if match is None:
            raise RuntimeError(f"Serialized emotion is missing {field}")
        spans.append(match.span(1))

    for field in ("positive_labels", "negative_labels"):
        array_match = re.search(
            rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]',
            serialized_emotion,
            flags=re.DOTALL,
        )
        if array_match is None:
            raise RuntimeError(f"Serialized emotion is missing {field}")
        array_start = array_match.start(1)
        label_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        for label_match in re.finditer(label_pattern, array_match.group(1)):
            spans.append(
                (
                    array_start + label_match.start(1),
                    array_start + label_match.end(1),
                )
            )
    return sorted(spans)


def _build_emotion_value_token_template(tokenizer, emotion: dict[str, Any]):
    serialized = json.dumps(
        {"emotion": emotion},
        ensure_ascii=False,
        indent=2,
    )
    char_spans = _emotion_value_char_spans(serialized)
    try:
        encoded = tokenizer(
            serialized,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
    except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "--emotion_loss_weight below 1.0 requires a fast tokenizer with "
            "offset mapping support"
        ) from exc

    weighted_positions = []
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        if any(
            token_end > value_start and token_start < value_end
            for value_start, value_end in char_spans
        ):
            weighted_positions.append(token_index)
    if len(weighted_positions) < 2:
        raise RuntimeError(
            "Could not map both emotion intensity values to tokenizer tokens"
        )
    return token_ids, weighted_positions


def _align_emotion_value_positions(
    row_token_ids: list[int],
    template_token_ids: list[int],
    template_value_positions: list[int],
) -> list[int] | None:
    """Align the largest template suffix that retains every target value."""
    first_value_position = min(template_value_positions)
    for trim in range(first_value_position + 1):
        pattern = template_token_ids[trim:]
        row_start = _find_last_subsequence_start(row_token_ids, pattern)
        if row_start is None:
            continue
        return [
            row_start + template_position - trim
            for template_position in template_value_positions
        ]
    return None


def build_emotion_weighted_trainer_class(stack, tokenizer, args):
    """Build a lazy TRL subclass so preview/tests need no ML dependencies."""
    base_trainer = stack["SFTTrainer"]
    torch = stack["torch"]
    emotion_loss_weight = args.emotion_loss_weight
    if emotion_loss_weight == 1.0:
        return base_trainer

    template_cache = {}

    class EmotionWeightedSFTTrainer(base_trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            labels = inputs.get("labels")
            if labels is None:
                raise RuntimeError(
                    "Emotion-weighted SFT requires token labels from SFTTrainer"
                )

            model_inputs = {
                key: value for key, value in inputs.items() if key != "labels"
            }
            outputs = model(**model_inputs)
            logits = outputs.logits
            token_weights = torch.ones_like(labels, dtype=logits.dtype)

            for row_index, row in enumerate(labels.detach().cpu().tolist()):
                supervised_ids = [token_id for token_id in row if token_id != -100]
                if not supervised_ids:
                    raise RuntimeError("SFT batch contains no supervised tokens")
                decoded = tokenizer.decode(
                    supervised_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                object_start = decoded.find("{")
                object_end = decoded.rfind("}")
                if object_start < 0 or object_end < object_start:
                    raise RuntimeError(
                        "Could not recover the supervised JSON completion. It may "
                        "have been truncated by --max_length."
                    )
                try:
                    target = json.loads(decoded[object_start : object_end + 1])
                    emotion = target["emotion"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise RuntimeError(
                        "Could not parse emotion from the supervised completion. "
                        "The sample may have been truncated by --max_length."
                    ) from exc
                serialized_emotion = json.dumps(
                    emotion,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                template = template_cache.get(serialized_emotion)
                if template is None:
                    template = _build_emotion_value_token_template(
                        tokenizer,
                        emotion,
                    )
                    template_cache[serialized_emotion] = template
                value_positions = _align_emotion_value_positions(
                    row,
                    template[0],
                    template[1],
                )
                if value_positions is None:
                    raise RuntimeError(
                        "Could not align emotion target values with the supervised "
                        "completion tokens. Check tokenizer offset behavior or "
                        "increase --max_length if the completion was truncated."
                    )
                for token_position in value_positions:
                    token_weights[row_index, token_position] = emotion_loss_weight

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = token_weights[..., 1:].contiguous()
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            flat_weights = shift_weights.view(-1)
            supervised_mask = shift_labels.view(-1).ne(-100)
            effective_weights = flat_weights * supervised_mask.to(
                flat_weights.dtype
            )
            denominator = effective_weights.sum()
            if denominator.item() <= 0:
                raise RuntimeError(
                    "Emotion-weighted SFT produced zero effective loss weight"
                )
            loss = (per_token_loss * effective_weights).sum() / denominator
            return (loss, outputs) if return_outputs else loss

    EmotionWeightedSFTTrainer.__name__ = "EmotionWeightedSFTTrainer"
    return EmotionWeightedSFTTrainer


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
    parser.add_argument(
        "--emotion_loss_weight",
        type=_bounded_emotion_loss_weight,
        default=1.0,
        help=(
            "Relative token-loss weight for emotion target values. 1.0 keeps "
            "the legacy objective. Values below 1.0 apply only to the two "
            "intensity values and label strings; JSON keys and structure retain "
            "weight 1.0."
        ),
    )
    parser.add_argument(
        "--emotion_label_order",
        choices=["canonical", "sample-shuffle"],
        default="canonical",
        help=(
            "canonical keeps legacy ontology order; sample-shuffle uses a "
            "deterministic per-sample order to avoid arbitrary first-label bias."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_warmup_parser().parse_args(argv)
    if args.emotion_loss_weight != 1.0 and args.packing:
        raise ValueError(
            "--packing is not supported with --emotion_loss_weight below 1.0"
        )
    builder = partial(
        build_warmup_example,
        missing_norm_value_policy=args.missing_norm_value_policy,
        enable_thinking=args.enable_thinking,
        appraisal_dimensions=args.appraisal_dimensions,
        emotion_label_order=args.emotion_label_order,
        label_order_seed=args.seed,
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
            "emotion_loss_weight": args.emotion_loss_weight,
            "emotion_loss_scope": (
                "emotion target values only: intensity scalars and label strings"
            ),
            "emotion_label_order": args.emotion_label_order,
            "emotion_label_order_seed": args.seed,
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
        trainer_class_factory=build_emotion_weighted_trainer_class,
    )


if __name__ == "__main__":
    raise SystemExit(main())
