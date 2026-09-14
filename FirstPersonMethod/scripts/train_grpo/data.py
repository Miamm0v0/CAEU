"""Data loading, validation, hidden references, and split leakage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from ..sft_common import normalize_emotion
except (ImportError, ValueError):
    from sft_common import normalize_emotion

from .spec import (
    CAREBENCH_REASONING_KEYS,
    build_policy_system_prompt,
    build_policy_user_prompt,
)


def load_records(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
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
        raise ValueError(f"{path} must contain a non-empty JSON array or JSONL")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every item in {path} must be a JSON object")
    return records


def stable_limit(
    records: list[dict[str, Any]],
    limit: int | None,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records

    def key(record: dict[str, Any]) -> str:
        identity = record.get("id", record.get("sample_id", "<missing>"))
        value = f"{namespace}:{seed}:{identity}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return sorted(records, key=key)[:limit]


def participant_id(record: dict[str, Any], sample_id: str) -> str:
    explicit = record.get("participant_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return sample_id.split("_", 1)[0]


def build_hidden_reference(
    record: dict[str, Any],
    appraisal_dimensions: list[str],
    include_appraisal_reference: bool = True,
) -> dict[str, Any]:
    # Gold emotion is always required by the deterministic outcome reward. The
    # judge_reference_mode flag controls only whether appraisal annotations are
    # shown to the API Judge; it must never disable the local outcome target.
    reference: dict[str, Any] = {
        "gold_emotion": normalize_emotion(record),
    }
    raw_reasoning = record.get("appraisal_reasoning")
    if include_appraisal_reference and isinstance(raw_reasoning, dict):
        reasoning: dict[str, str] = {}
        for dimension in appraisal_dimensions:
            legacy_key = CAREBENCH_REASONING_KEYS.get(
                dimension,
                dimension,
            )
            value = raw_reasoning.get(
                dimension,
                raw_reasoning.get(legacy_key),
            )
            if isinstance(value, str) and value.strip():
                reasoning[dimension] = value.strip()
        if reasoning:
            reference["legacy_human_appraisal_reasoning"] = reasoning
    return reference


def normalize_policy_record(
    record: dict[str, Any],
    index: int,
    reference_mode: str,
    appraisal_dimensions: list[str],
) -> dict[str, Any]:
    sample_id = record.get("id", record.get("sample_id"))
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"item {index}: id must be a non-empty string")
    sample_id = sample_id.strip()
    situation = record.get("situation", record.get("event"))
    if not isinstance(situation, str) or not situation.strip():
        raise ValueError(f"{sample_id}: situation must be a non-empty string")
    situation = situation.strip()
    perspective = record.get("perspective")
    if perspective is not None and perspective != "first_person":
        raise ValueError(f"{sample_id}: expected first_person perspective")

    reference = build_hidden_reference(
        record,
        appraisal_dimensions,
        include_appraisal_reference=reference_mode != "none",
    )
    reference_reasoning = reference.get(
        "legacy_human_appraisal_reasoning",
        {},
    )
    if reference_mode == "required" and (
        not isinstance(reference_reasoning, dict)
        or set(reference_reasoning) != set(appraisal_dimensions)
    ):
        raise ValueError(
            f"{sample_id}: --judge_reference_mode required needs valid human "
            "appraisal annotations for every selected dimension"
        )
    return {
        "sample_id": sample_id,
        "participant_id": participant_id(record, sample_id),
        "situation": situation,
        "reference_json": json.dumps(
            reference,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "prompt": [
            {
                "role": "system",
                "content": build_policy_system_prompt(appraisal_dimensions),
            },
            {
                "role": "user",
                "content": build_policy_user_prompt(
                    situation,
                    appraisal_dimensions,
                ),
            },
        ],
    }


def prepare_split(
    records: list[dict[str, Any]],
    split_name: str,
    invalid_policy: str,
    reference_mode: str,
    appraisal_dimensions: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            example = normalize_policy_record(
                record,
                index,
                reference_mode,
                appraisal_dimensions,
            )
            if example["sample_id"] in seen:
                raise ValueError(
                    f"{split_name}: duplicate id {example['sample_id']}"
                )
            seen.add(example["sample_id"])
            examples.append(example)
        except (TypeError, ValueError) as exc:
            if invalid_policy == "error":
                raise
            errors.append(
                {
                    "item": str(index),
                    "id": str(
                        record.get("id", record.get("sample_id", "<unknown>"))
                    ),
                    "error": str(exc),
                }
            )
    if not examples:
        raise ValueError(f"No valid {split_name} records remain")
    if errors:
        print(f"[data] split={split_name} skipped_invalid={len(errors)}")
        for error in errors[:10]:
            print(f"[data] skipped id={error['id']} error={error['error']}")
    return examples, errors


def reject_split_overlap(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    allow_participant_overlap: bool,
) -> None:
    if not evaluation:
        return
    sample_overlap = sorted(
        {item["sample_id"] for item in train}
        & {item["sample_id"] for item in evaluation}
    )
    if sample_overlap:
        raise ValueError(
            f"Train/eval sample overlap detected (first ids: "
            f"{sample_overlap[:10]})"
        )
    participant_overlap = sorted(
        {item["participant_id"] for item in train}
        & {item["participant_id"] for item in evaluation}
    )
    if participant_overlap and not allow_participant_overlap:
        raise ValueError(
            "Train/eval participant overlap detected (first ids: {}). Re-split "
            "by participant; use --allow_participant_overlap only for a smoke "
            "test.".format(participant_overlap[:10])
        )


def prepare_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records = stable_limit(
        load_records(args.train_file),
        args.max_train_samples,
        args.seed,
        "train",
    )
    eval_records = (
        stable_limit(
            load_records(args.eval_file),
            args.max_eval_samples,
            args.seed,
            "eval",
        )
        if args.eval_file
        else []
    )
    train, train_errors = prepare_split(
        train_records,
        "train",
        args.invalid_record_policy,
        args.judge_reference_mode,
        args.appraisal_dimensions,
    )
    if eval_records:
        evaluation, eval_errors = prepare_split(
            eval_records,
            "eval",
            args.invalid_record_policy,
            args.judge_reference_mode,
            args.appraisal_dimensions,
        )
    else:
        evaluation, eval_errors = [], []
    reject_split_overlap(train, evaluation, args.allow_participant_overlap)
    summary = {
        "train_samples": len(train),
        "eval_samples": len(evaluation),
        "train_participants": len({item["participant_id"] for item in train}),
        "eval_participants": len(
            {item["participant_id"] for item in evaluation}
        ),
        "train_with_judge_reference": sum(
            "legacy_human_appraisal_reasoning"
            in json.loads(item["reference_json"])
            for item in train
        ),
        "eval_with_judge_reference": sum(
            "legacy_human_appraisal_reasoning"
            in json.loads(item["reference_json"])
            for item in evaluation
        ),
        "train_with_gold_emotion": sum(
            "gold_emotion" in json.loads(item["reference_json"])
            for item in train
        ),
        "eval_with_gold_emotion": sum(
            "gold_emotion" in json.loads(item["reference_json"])
            for item in evaluation
        ),
        "train_skipped_invalid": len(train_errors),
        "eval_skipped_invalid": len(eval_errors),
        "judge_reference_mode": args.judge_reference_mode,
        "appraisal_dimensions": args.appraisal_dimensions,
    }
    return train, evaluation, summary
