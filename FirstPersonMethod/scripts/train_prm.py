#!/usr/bin/env python3
"""Train a stepwise Process Reward Model (PRM).

This script deliberately does not define the sample-level rubric.  Rubrics are
an upstream annotation specification: they judge candidate reasoning steps and
produce either a binary label or a scalar score for every step.  The PRM sees
the first-person event and the candidate trajectory, but not the gold rubric.

Two input layouts are accepted (JSON arrays or JSONL).

TRL-native layout::

    {
      "id": "candidate-0001",
      "sample_id": "carebench-source-id",
      "prompt": "First-person event: ...\nCandidate reasoning:\n",
      "completions": ["[relevance] ...", "[congruence] ..."],
      "labels": [true, false],
      "rubric": {"any": "metadata is allowed and ignored by the trainer"}
    }

Rubric-aware layout::

    {
      "id": "candidate-0001",
      "sample_id": "carebench-source-id",
      "situation": "I ...",
      "rubric": {"version": "to-be-designed"},
      "steps": [
        {"step_id": "relevance", "text": "...", "label": true},
        {"step_id": "congruence", "text": "...", "score": 0.25}
      ]
    }

Non-binary scores require ``--positive_threshold``.  Nested fields are
supported, for example ``--step_score_field rubric_result.overall``.  This
keeps the trainer independent of the eventual rubric dimensions and weights.

Preview and validate data without importing the training stack::

    python FirstPersonMethod/scripts/train_prm.py \
      --train_file prm_train.jsonl \
      --eval_file prm_val.jsonl \
      --positive_threshold 0.7 \
      --preview_only

QLoRA training example::

    accelerate launch FirstPersonMethod/scripts/train_prm.py \
      --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
      --train_file prm_train.jsonl \
      --eval_file prm_val.jsonl \
      --load_in_4bit \
      --use_wandb \
      --wandb_project first-person-appraisal-prm \
      --run_name qwen2.5-7b-prm-v1 \
      --output_dir FirstPersonMethod/output/prm

The implementation targets the current experimental TRL PRMTrainer.  That
trainer places a binary token-classification target at the separator following
each completion step, so the separator and step segmentation must also be used
when the PRM is later called from reinforcement learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PRM_PROMPT = """You are scoring a proposed first-person cognitive-appraisal trajectory.
At the end of each step, estimate whether that step is correct, grounded in the
event, internally consistent with the preceding steps, and useful for deriving
the person's emotion.

First-person event:
{situation}

Candidate appraisal-to-emotion trajectory:
"""

MISSING = object()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary stepwise PRM for first-person "
            "event -> cognitive appraisal -> emotion trajectories."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Base model id/path (required for training, not for preview)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="",
        help="Required PRM train JSON/JSONL containing candidate steps and labels",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="",
        help="Optional PRM validation JSON/JSONL; the test split must not be used here",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FirstPersonMethod/output/prm",
    )

    # The names can be changed after the rubric schema is finalized.
    parser.add_argument("--step_text_field", type=str, default="text")
    parser.add_argument("--step_label_field", type=str, default="label")
    parser.add_argument("--step_score_field", type=str, default="score")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=None,
        help=(
            "Convert a non-binary numeric rubric score to label 1 when score >= "
            "this value; omitted means labels must already be binary"
        ),
    )
    parser.add_argument(
        "--step_separator",
        type=str,
        default="\n",
        help=r"Separator appended after every step; literal \n and \t are decoded",
    )
    parser.add_argument(
        "--allow_single_class",
        action="store_true",
        help="Allow training data with only positive or only negative step labels",
    )
    parser.add_argument(
        "--invalid_record_policy",
        choices=["error", "skip"],
        default="error",
        help="Fail on malformed PRM supervision (recommended) or skip with a report",
    )

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=3072)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Default: 1e-4 for LoRA/QLoRA and 1e-5 for full fine-tuning",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train the complete token-classification model instead of LoRA",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use NF4 QLoRA; requires CUDA and bitsandbytes",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help="'all-linear' or a comma-separated module-name list",
    )
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataset_num_proc", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--train_on_last_step_only",
        action="store_true",
        help="Outcome-style ablation; normally leave disabled for a process model",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="'none' or comma-separated integrations such as wandb,tensorboard",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable the Transformers W&B callback (also implied by --wandb_project)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="W&B project; otherwise WANDB_PROJECT or the W&B default is used",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help="Optional W&B user/team entity",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default="",
        help="Group related runs, for example all PRM ablations",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated W&B tags",
    )
    parser.add_argument(
        "--wandb_notes",
        type=str,
        default="",
        help="Optional W&B run notes",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline"],
        default=None,
        help="Override WANDB_MODE; offline keeps local logs without syncing",
    )
    parser.add_argument(
        "--wandb_log_model",
        choices=["false", "end", "checkpoint"],
        default=None,
        help="Upload no model, the final model, or every saved checkpoint as Artifacts",
    )
    parser.add_argument(
        "--wandb_watch",
        choices=["false", "gradients", "parameters", "all"],
        default=None,
        help="Optional parameter/gradient histogram logging; may add overhead",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default="",
        help="Existing W&B run id when resuming the same tracked run",
    )
    parser.add_argument(
        "--wandb_resume",
        choices=["never", "allow", "must", "auto"],
        default=None,
        help="W&B resume policy; run id without this flag defaults to 'allow'",
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="",
        help="Optional DeepSpeed configuration path",
    )

    parser.add_argument(
        "--preview_only",
        action="store_true",
        help="Validate and print canonical PRM samples without training dependencies",
    )
    parser.add_argument("--preview_samples", type=int, default=1)
    parser.add_argument(
        "--print_schema",
        action="store_true",
        help="Print an adaptable rubric-aware data example and exit",
    )
    return parser


def schema_example() -> dict[str, Any]:
    return {
        "id": "source-id_candidate-0",
        "sample_id": "source-id",
        "participant_id": "optional-participant-id",
        "situation": "I experienced a first-person event ...",
        "rubric": {
            "version": "TBD",
            "note": "Rubric content is metadata; only its step verdict is supervised.",
        },
        "steps": [
            {
                "step_id": "relevance",
                "text": "[relevance] The event matters to me because ...",
                "label": True,
                "rubric_result": {"criterion_scores": "arbitrary metadata"},
            },
            {
                "step_id": "congruence",
                "text": "[congruence] The event conflicts with my goal because ...",
                "score": 0.25,
                "rubric_result": {"overall": 0.25},
            },
        ],
    }


def load_records(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PRM data file not found: {path}")

    records: list[Any]
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty JSON array or JSONL records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: item {index} must be an object")
    return records


def nested_get(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value.strip()


def require_prompt(value: Any, location: str) -> str:
    """Validate a prompt without discarding its separator-bearing suffix."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def normalize_binary_label(
    value: Any,
    location: str,
    positive_threshold: float | None,
) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{location} must be finite")
        if numeric in (0.0, 1.0):
            return int(numeric)
        if positive_threshold is None:
            raise ValueError(
                f"{location}={numeric} is not binary; provide --positive_threshold "
                "to convert rubric scores explicitly"
            )
        return int(numeric >= positive_threshold)

    if isinstance(value, str):
        normalized = value.strip().lower()
        positive = {"1", "true", "correct", "pass", "positive", "yes"}
        negative = {"0", "false", "incorrect", "fail", "negative", "no"}
        if normalized in positive:
            return 1
        if normalized in negative:
            return 0

    raise ValueError(
        f"{location} must be binary bool/0/1 or a numeric rubric score with "
        "--positive_threshold"
    )


def record_identity(record: dict[str, Any], index: int) -> tuple[str, str, str]:
    trajectory_id = record.get("trajectory_id", record.get("id"))
    trajectory_id = require_string(trajectory_id, f"item {index}.id")
    sample_id = record.get("sample_id", record.get("source_id", trajectory_id))
    sample_id = require_string(sample_id, f"{trajectory_id}.sample_id")
    participant_id = record.get("participant_id")
    if participant_id is None:
        participant_id = sample_id.split("_", 1)[0]
    participant_id = require_string(
        participant_id, f"{trajectory_id}.participant_id"
    )
    return trajectory_id, sample_id, participant_id


def resolve_step_label(
    step: dict[str, Any],
    trajectory_id: str,
    step_index: int,
    args: argparse.Namespace,
) -> int:
    label = nested_get(step, args.step_label_field)
    if label is not MISSING:
        return normalize_binary_label(
            label,
            f"{trajectory_id}.steps[{step_index}].{args.step_label_field}",
            args.positive_threshold,
        )

    score = nested_get(step, args.step_score_field)
    if score is MISSING:
        raise ValueError(
            f"{trajectory_id}.steps[{step_index}] has neither "
            f"'{args.step_label_field}' nor '{args.step_score_field}'"
        )
    if args.positive_threshold is None:
        raise ValueError(
            f"{trajectory_id}.steps[{step_index}].{args.step_score_field} is a "
            "rubric score; provide --positive_threshold"
        )
    return normalize_binary_label(
        score,
        f"{trajectory_id}.steps[{step_index}].{args.step_score_field}",
        args.positive_threshold,
    )


def normalize_record(
    record: dict[str, Any], index: int, args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory_id, sample_id, participant_id = record_identity(record, index)

    raw_prompt = record.get("prompt", MISSING)
    if raw_prompt is not MISSING:
        if isinstance(raw_prompt, list):
            raise ValueError(
                f"{trajectory_id}.prompt is conversational; PRMTrainer requires a "
                "standard string prompt. Render the chat template upstream."
            )
        prompt = require_prompt(raw_prompt, f"{trajectory_id}.prompt")
    else:
        situation = record.get("situation", record.get("event"))
        situation = require_string(situation, f"{trajectory_id}.situation")
        prompt = DEFAULT_PRM_PROMPT.format(situation=situation)
    # PRMTrainer concatenates prompt tokens directly with the first completion;
    # unlike later steps, it does not insert the separator at this boundary.
    if not prompt.endswith(args.step_separator):
        prompt += args.step_separator

    step_types: list[str] = []
    if "completions" in record or "labels" in record:
        completions = record.get("completions")
        raw_labels = record.get("labels")
        if not isinstance(completions, list) or not completions:
            raise ValueError(f"{trajectory_id}.completions must be a non-empty list")
        if not isinstance(raw_labels, list):
            raise ValueError(f"{trajectory_id}.labels must be a list")
        if len(completions) != len(raw_labels):
            raise ValueError(
                f"{trajectory_id}: completions/labels length mismatch "
                f"({len(completions)} != {len(raw_labels)})"
            )
        completions = [
            require_string(text, f"{trajectory_id}.completions[{step_index}]")
            for step_index, text in enumerate(completions)
        ]
        labels = [
            normalize_binary_label(
                label,
                f"{trajectory_id}.labels[{step_index}]",
                args.positive_threshold,
            )
            for step_index, label in enumerate(raw_labels)
        ]
        raw_step_types = record.get("step_types", record.get("step_ids", []))
        if raw_step_types:
            if not isinstance(raw_step_types, list) or len(raw_step_types) != len(labels):
                raise ValueError(
                    f"{trajectory_id}.step_types must match completions length"
                )
            step_types = [str(value) for value in raw_step_types]
        else:
            step_types = [f"step_{index + 1}" for index in range(len(labels))]
    else:
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                f"{trajectory_id}.steps must be non-empty, or provide "
                "completions and labels"
            )
        completions = []
        labels = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(
                    f"{trajectory_id}.steps[{step_index}] must be an object"
                )
            text = nested_get(step, args.step_text_field)
            completions.append(
                require_string(
                    text,
                    f"{trajectory_id}.steps[{step_index}].{args.step_text_field}",
                )
            )
            labels.append(resolve_step_label(step, trajectory_id, step_index, args))
            step_type = step.get("step_type", step.get("step_id", f"step_{step_index + 1}"))
            step_types.append(str(step_type))

    canonical = {
        "prompt": prompt,
        "completions": completions,
        "labels": labels,
    }
    metadata = {
        "trajectory_id": trajectory_id,
        "sample_id": sample_id,
        "participant_id": participant_id,
        "step_types": step_types,
        "has_rubric_metadata": "rubric" in record
        or any(
            isinstance(step, dict)
            and ("rubric" in step or "rubric_result" in step)
            for step in (record.get("steps") or [])
        ),
    }
    return canonical, metadata


def stable_limit(
    records: list[dict[str, Any]], limit: int | None, seed: int, namespace: str
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records

    def key(item: tuple[int, dict[str, Any]]) -> str:
        index, record = item
        identity = record.get("trajectory_id", record.get("id", index))
        raw = f"{namespace}:{seed}:{identity}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    return [record for _, record in sorted(enumerate(records), key=key)[:limit]]


def summarize_examples(
    examples: list[dict[str, Any]], metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    label_counts: Counter[int] = Counter()
    type_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for example, item_metadata in zip(examples, metadata):
        label_counts.update(example["labels"])
        for step_type, label in zip(
            item_metadata["step_types"], example["labels"]
        ):
            type_counts[step_type][label] += 1

    total_steps = sum(label_counts.values())
    return {
        "trajectories": len(examples),
        "source_samples": len({item["sample_id"] for item in metadata}),
        "participants": len({item["participant_id"] for item in metadata}),
        "steps": total_steps,
        "positive_steps": label_counts[1],
        "negative_steps": label_counts[0],
        "positive_rate": (label_counts[1] / total_steps) if total_steps else 0.0,
        "trajectories_with_rubric_metadata": sum(
            bool(item["has_rubric_metadata"]) for item in metadata
        ),
        "per_step_type": {
            step_type: {
                "positive": counts[1],
                "negative": counts[0],
            }
            for step_type, counts in sorted(type_counts.items())
        },
    }


def prepare_split(
    records: list[dict[str, Any]], split_name: str, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_trajectory_ids: set[str] = set()

    for index, record in enumerate(records):
        try:
            example, item_metadata = normalize_record(record, index, args)
            trajectory_id = item_metadata["trajectory_id"]
            if trajectory_id in seen_trajectory_ids:
                raise ValueError(
                    f"{split_name}: duplicate trajectory id '{trajectory_id}'"
                )
            seen_trajectory_ids.add(trajectory_id)
            examples.append(example)
            metadata.append(item_metadata)
        except (TypeError, ValueError) as exc:
            if args.invalid_record_policy == "error":
                raise
            errors.append(
                {
                    "item": str(index),
                    "id": str(record.get("trajectory_id", record.get("id", "<unknown>"))),
                    "error": str(exc),
                }
            )

    if errors:
        print(f"[data] split={split_name} skipped_invalid={len(errors)}")
        for error in errors[:10]:
            print(f"[data] skipped id={error['id']} error={error['error']}")
        if len(errors) > 10:
            print(f"[data] ... {len(errors) - 10} additional invalid records")
    if not examples:
        raise ValueError(f"No valid {split_name} PRM samples remain")
    return examples, metadata, errors


def reject_split_overlap(
    train_metadata: list[dict[str, Any]], eval_metadata: list[dict[str, Any]]
) -> None:
    if not eval_metadata:
        return
    train_trajectories = {item["trajectory_id"] for item in train_metadata}
    eval_trajectories = {item["trajectory_id"] for item in eval_metadata}
    trajectory_overlap = sorted(train_trajectories & eval_trajectories)
    if trajectory_overlap:
        raise ValueError(
            "Train/eval trajectory overlap detected (first ids: {})".format(
                trajectory_overlap[:10]
            )
        )

    train_sources = {item["sample_id"] for item in train_metadata}
    eval_sources = {item["sample_id"] for item in eval_metadata}
    source_overlap = sorted(train_sources & eval_sources)
    if source_overlap:
        raise ValueError(
            "Train/eval source-sample overlap detected (first ids: {})".format(
                source_overlap[:10]
            )
        )

    train_participants = {item["participant_id"] for item in train_metadata}
    eval_participants = {item["participant_id"] for item in eval_metadata}
    participant_overlap = sorted(train_participants & eval_participants)
    if participant_overlap:
        raise ValueError(
            "Train/eval participant overlap detected (first ids: {})".format(
                participant_overlap[:10]
            )
        )


def prepare_data(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not args.train_file:
        raise ValueError("--train_file is required")
    if args.positive_threshold is not None and not math.isfinite(args.positive_threshold):
        raise ValueError("--positive_threshold must be finite")

    train_records = stable_limit(
        load_records(args.train_file), args.max_train_samples, args.seed, "train-limit"
    )
    eval_records = (
        stable_limit(
            load_records(args.eval_file), args.max_eval_samples, args.seed, "eval-limit"
        )
        if args.eval_file
        else []
    )

    train_examples, train_metadata, train_errors = prepare_split(
        train_records, "train", args
    )
    if eval_records:
        eval_examples, eval_metadata, eval_errors = prepare_split(
            eval_records, "eval", args
        )
    else:
        eval_examples, eval_metadata, eval_errors = [], [], []

    reject_split_overlap(train_metadata, eval_metadata)
    train_summary = summarize_examples(train_examples, train_metadata)
    if not args.allow_single_class and (
        train_summary["positive_steps"] == 0 or train_summary["negative_steps"] == 0
    ):
        raise ValueError(
            "PRM train data must contain both correct and incorrect steps. Gold-only "
            "chains cannot train a discriminative process reward model; add rubric-"
            "labeled candidate/negative trajectories or use --allow_single_class "
            "only for a pipeline smoke test."
        )

    summary = {
        "train": train_summary,
        "eval": summarize_examples(eval_examples, eval_metadata)
        if eval_examples
        else None,
        "train_skipped_invalid": len(train_errors),
        "eval_skipped_invalid": len(eval_errors),
        "rubric_adapter": {
            "step_text_field": args.step_text_field,
            "step_label_field": args.step_label_field,
            "step_score_field": args.step_score_field,
            "positive_threshold": args.positive_threshold,
        },
    }
    return train_examples, eval_examples, summary


def print_preview(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    summary: dict[str, Any],
    count: int,
) -> None:
    print("[preview] canonical PRM data")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for index, example in enumerate(train_examples[: max(0, count)]):
        print(f"\n===== TRAIN PRM SAMPLE {index} =====")
        print(json.dumps(example, ensure_ascii=False, indent=2))
    if eval_examples and count > 0:
        print("\n===== EVAL PRM SAMPLE 0 =====")
        print(json.dumps(eval_examples[0], ensure_ascii=False, indent=2))


def decode_separator(value: str) -> str:
    replacements = {r"\n": "\n", r"\t": "\t", r"\r": "\r"}
    decoded = value
    for escaped, actual in replacements.items():
        decoded = decoded.replace(escaped, actual)
    if not decoded:
        raise ValueError("--step_separator must not be empty")
    return decoded


def parse_report_to(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized or normalized.lower() == "none":
        return []
    return [part.strip() for part in normalized.split(",") if part.strip()]


def wandb_requested(args: argparse.Namespace, reporters: list[str]) -> bool:
    configured_values = (
        args.wandb_project,
        args.wandb_entity,
        args.wandb_group,
        args.wandb_tags,
        args.wandb_notes,
        args.wandb_mode,
        args.wandb_log_model,
        args.wandb_watch,
        args.wandb_run_id,
        args.wandb_resume,
    )
    return (
        args.use_wandb
        or any(reporter.lower() == "wandb" for reporter in reporters)
        or any(value not in (None, "") for value in configured_values)
    )


def resolve_reporters(args: argparse.Namespace) -> list[str]:
    reporters = parse_report_to(args.report_to)
    if wandb_requested(args, reporters) and not any(
        reporter.lower() == "wandb" for reporter in reporters
    ):
        reporters.append("wandb")
    return reporters


def configure_wandb(
    args: argparse.Namespace, reporters: list[str]
) -> dict[str, Any]:
    """Configure the Trainer-managed W&B run without handling API keys."""
    enabled = any(reporter.lower() == "wandb" for reporter in reporters)
    if not enabled:
        return {"enabled": False}

    cli_environment = {
        "WANDB_PROJECT": args.wandb_project.strip(),
        "WANDB_ENTITY": args.wandb_entity.strip(),
        "WANDB_RUN_GROUP": args.wandb_group.strip(),
        "WANDB_TAGS": ",".join(
            tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()
        ),
        "WANDB_NOTES": args.wandb_notes.strip(),
        "WANDB_MODE": args.wandb_mode or "",
        "WANDB_LOG_MODEL": args.wandb_log_model or "",
        "WANDB_WATCH": args.wandb_watch or "",
        "WANDB_RUN_ID": args.wandb_run_id.strip(),
    }
    for variable, value in cli_environment.items():
        if value:
            os.environ[variable] = value

    existing_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
    resume_policy = args.wandb_resume
    if resume_policy in {"allow", "must"} and not existing_run_id:
        raise ValueError(
            f"--wandb_resume {resume_policy} requires --wandb_run_id or "
            "the WANDB_RUN_ID environment variable"
        )
    if resume_policy:
        os.environ["WANDB_RESUME"] = resume_policy
    elif args.wandb_run_id and "WANDB_RESUME" not in os.environ:
        os.environ["WANDB_RESUME"] = "allow"

    os.environ.setdefault("WANDB_JOB_TYPE", "prm-training")
    if (
        args.resume_from_checkpoint
        and not existing_run_id
        and os.environ.get("WANDB_RESUME", "").lower() != "auto"
        and os.environ.get("LOCAL_RANK", "-1") in {"-1", "0"}
    ):
        print(
            "[wandb] warning: model/optimizer state will resume from a local "
            "checkpoint, but W&B will create a new run. Pass --wandb_run_id "
            "to continue the original tracked run."
        )

    summary = {
        "enabled": True,
        "project": os.environ.get("WANDB_PROJECT") or None,
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "group": os.environ.get("WANDB_RUN_GROUP") or None,
        "tags": [
            tag.strip()
            for tag in os.environ.get("WANDB_TAGS", "").split(",")
            if tag.strip()
        ],
        "mode": os.environ.get("WANDB_MODE", "online"),
        "log_model": os.environ.get("WANDB_LOG_MODEL", "false"),
        "watch": os.environ.get("WANDB_WATCH", "false"),
        "run_id": existing_run_id or None,
        "resume": os.environ.get("WANDB_RESUME") or None,
        "run_name": args.run_name or None,
    }
    if os.environ.get("LOCAL_RANK", "-1") in {"-1", "0"}:
        print(
            "[wandb] enabled project={} run_name={} mode={} log_model={}".format(
                summary["project"] or "<wandb-default>",
                summary["run_name"] or "<auto>",
                summary["mode"],
                summary["log_model"],
            )
        )
    return summary


def parse_target_modules(value: str) -> str | list[str]:
    normalized = value.strip()
    if normalized == "all-linear":
        return normalized
    modules = [part.strip() for part in normalized.split(",") if part.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must not be empty")
    return modules


def resolve_resume(value: str) -> str | bool | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"true", "last", "yes"}:
        return True
    return normalized


def import_training_stack(
    load_in_4bit: bool,
    use_peft: bool,
    use_wandb: bool,
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "PRM training requires Python >= 3.10; active interpreter is "
            f"{sys.version.split()[0]}"
        )
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        try:
            from trl.experimental.prm import PRMConfig, PRMTrainer
        except ImportError:
            # Compatibility with TRL releases where PRM was not yet moved under
            # the experimental namespace.
            from trl import PRMConfig, PRMTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install the training stack: torch transformers datasets accelerate "
            "and trl"
        ) from exc

    LoraConfig = None
    if use_peft:
        try:
            from peft import LoraConfig
        except ImportError as exc:
            raise RuntimeError("LoRA training requires peft: pip install peft") from exc

    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requires wandb: pip install wandb, then run wandb login "
                "or set WANDB_API_KEY outside this script"
            ) from exc

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load_in_4bit requires bitsandbytes: pip install bitsandbytes"
            ) from exc

    return {
        "np": np,
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "PRMConfig": PRMConfig,
        "PRMTrainer": PRMTrainer,
        "LoraConfig": LoraConfig,
    }


def choose_precision(torch: Any, requested: str) -> tuple[Any, bool, bool]:
    if requested == "bf16":
        return torch.bfloat16, True, False
    if requested == "fp16":
        return torch.float16, False, True
    if requested == "fp32":
        return torch.float32, False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def build_compute_metrics(np: Any):
    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, labels = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100
        gold = labels[mask]
        predicted = predictions[mask]
        if gold.size == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        true_positive = int(((predicted == 1) & (gold == 1)).sum())
        false_positive = int(((predicted == 1) & (gold == 0)).sum())
        false_negative = int(((predicted == 0) & (gold == 1)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        return {
            "accuracy": float((predicted == gold).mean()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "gold_positive_rate": float((gold == 1).mean()),
            "predicted_positive_rate": float((predicted == 1).mean()),
        }

    return compute_metrics


def run_training(
    args: argparse.Namespace,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    data_summary: dict[str, Any],
) -> None:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required for training")
    if args.load_in_4bit and args.full_finetune:
        raise ValueError("--load_in_4bit requires LoRA; remove --full_finetune")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.max_completion_length == 0 or args.max_completion_length < -1:
        raise ValueError("--max_completion_length must be positive or -1 for unlimited")

    reporters = resolve_reporters(args)
    wandb_summary = configure_wandb(args, reporters)
    stack = import_training_stack(
        args.load_in_4bit,
        use_peft=not args.full_finetune,
        use_wandb=wandb_summary["enabled"],
    )
    torch = stack["torch"]
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("--load_in_4bit requires a CUDA GPU")

    model_dtype, use_bf16, use_fp16 = choose_precision(torch, args.dtype)
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "id2label": {0: "incorrect", 1: "correct"},
        "label2id": {"incorrect": 0, "correct": 1},
        "trust_remote_code": args.trust_remote_code,
        "dtype": model_dtype,
    }
    if args.load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
        )
        model_kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = stack["AutoModelForTokenClassification"].from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False if not args.no_gradient_checkpointing else True

    peft_config = None
    if not args.full_finetune:
        peft_config = stack["LoraConfig"](
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=parse_target_modules(args.lora_target_modules),
            task_type="TOKEN_CLS",
        )

    train_dataset = stack["Dataset"].from_list(train_examples)
    eval_dataset = (
        stack["Dataset"].from_list(eval_examples) if eval_examples else None
    )
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-5 if args.full_finetune else 1e-4

    config_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "max_completion_length": (
            None if args.max_completion_length == -1 else args.max_completion_length
        ),
        "step_separator": decode_separator(args.step_separator),
        "train_on_last_step_only": args.train_on_last_step_only,
        "dataset_num_proc": args.dataset_num_proc,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "seed": args.seed,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "report_to": reporters,
        "run_name": args.run_name or None,
    }
    if eval_dataset is not None:
        config_kwargs["eval_steps"] = args.eval_steps
    if args.deepspeed:
        config_kwargs["deepspeed"] = args.deepspeed

    training_config = stack["PRMConfig"](**config_kwargs)
    trainer = stack["PRMTrainer"](
        model=model,
        args=training_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        compute_metrics=build_compute_metrics(stack["np"]),
    )

    train_result = trainer.train(
        resume_from_checkpoint=resolve_resume(args.resume_from_checkpoint)
    )
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "method": "first_person_stepwise_prm",
        "model_name_or_path": args.model_name_or_path,
        "objective": "binary step correctness",
        "step_separator": decode_separator(args.step_separator),
        "full_finetune": args.full_finetune,
        "load_in_4bit": args.load_in_4bit,
        "wandb": wandb_summary,
        "data": data_summary,
    }
    with (output_dir / "prm_run_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.print_schema:
        print(json.dumps(schema_example(), ensure_ascii=False, indent=2))
        return 0

    args.step_separator = decode_separator(args.step_separator)
    train_examples, eval_examples, summary = prepare_data(args)
    if args.preview_only:
        print_preview(
            train_examples,
            eval_examples,
            summary,
            count=args.preview_samples,
        )
        return 0

    run_training(args, train_examples, eval_examples, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
