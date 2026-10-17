#!/usr/bin/env python3
"""Prepare paired CAREBench appraisal-to-emotion training data.

The script emits:

* ``ae_train.jsonl``: valid conditional appraisal -> emotion examples from
  first-person authors and third-person observers assigned to the train split.
* ``ae_pairs_train.jsonl``: same-situation first-person/observer pairs whose
  appraisals and emotions both differ clearly.
* ``ae_val.jsonl``: conditional examples assigned to the validation split.
* ``ae_data_report.json``: counts, filtering reasons, and emotion distributions.

Third-person annotations are valid conditional supervision. They are never
written as unlabelled negatives: each appraisal is kept together with the
emotion reported by the same annotator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


if __package__ in {None, ""}:
    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_appraisal_counterfactual_data as counterfactual
    from sft_common import (
        NEGATIVE_LABEL_DESCRIPTIONS,
        NEGATIVE_LABELS,
        POSITIVE_LABEL_DESCRIPTIONS,
        POSITIVE_LABELS,
    )
    from train_grpo.spec import CAREBENCH_REASONING_KEYS
else:
    from .. import build_appraisal_counterfactual_data as counterfactual
    from ..sft_common import (
        NEGATIVE_LABEL_DESCRIPTIONS,
        NEGATIVE_LABELS,
        POSITIVE_LABEL_DESCRIPTIONS,
        POSITIVE_LABELS,
    )
    from ..train_grpo.spec import CAREBENCH_REASONING_KEYS


REPORT_NAME = "ae_data_report.json"
TRAIN_NAME = "ae_train.jsonl"
PAIR_TRAIN_NAME = "ae_pairs_train.jsonl"
PAIR_VAL_NAME = "ae_pairs_val.jsonl"
VAL_NAME = "ae_val.jsonl"

POLICY_DIMENSIONS = tuple(CAREBENCH_REASONING_KEYS)
LEGACY_CORE_TO_POLICY = {
    legacy: policy for policy, legacy in CAREBENCH_REASONING_KEYS.items()
}


@dataclass(frozen=True)
class ConditionalSample:
    record_type: str
    sample_id: str
    source_sample_id: str
    annotator_id: str
    annotator_type: str
    situation: str
    appraisal_reasoning: dict[str, str]
    appraisal_scores: dict[str, float]
    appraisal_rating_scores: dict[str, int]
    emotion: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert first-person and third-person CAREBench annotations into "
            "conditional and paired appraisal-to-emotion JSONL data."
        )
    )
    parser.add_argument(
        "--first_person_input",
        required=True,
        type=Path,
        help=(
            "First-person JSON used as the complete input or as the train split "
            "when --first_person_val_input is provided."
        ),
    )
    parser.add_argument(
        "--first_person_val_input",
        type=Path,
        default=None,
        help=(
            "Optional existing validation split. Without it, situations from "
            "--first_person_input are split deterministically."
        ),
    )
    parser.add_argument(
        "--third_person_input",
        type=Path,
        default=None,
        help=(
            "Third-person JSON shaped as source_sample_id -> annotator records. "
            "Defaults to the repository third_person.json."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/appraisal_emotion_train"),
    )
    parser.add_argument(
        "--prompt_config",
        type=Path,
        default=counterfactual.DEFAULT_PROMPT_CONFIG,
        help="Prompt TOML used to resolve the 22 CAREBench rating statements.",
    )
    parser.add_argument(
        "--compact_scale",
        choices=("auto", "zero-based", "one-based"),
        default="auto",
    )
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_opposite_core_dimensions",
        type=int,
        default=1,
        help="Minimum number of five-dimensional appraisal groups with opposition.",
    )
    parser.add_argument(
        "--min_opposite_items",
        type=int,
        default=1,
        help="Minimum opposite rating items within an eligible core dimension.",
    )
    parser.add_argument(
        "--min_opposite_fraction",
        type=float,
        default=0.25,
        help="Minimum opposite-item fraction within an eligible core dimension.",
    )
    parser.add_argument(
        "--min_emotion_label_distance",
        type=float,
        default=0.5,
        help="Minimum Jaccard distance between valence-tagged emotion labels.",
    )
    parser.add_argument(
        "--min_emotion_intensity_gap",
        type=int,
        default=2,
        help="Minimum gap in either positive or negative intensity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.first_person_input.is_file():
        raise FileNotFoundError(args.first_person_input)
    if args.first_person_val_input is not None and not args.first_person_val_input.is_file():
        raise FileNotFoundError(args.first_person_val_input)
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation_ratio must be in (0, 1)")
    if args.min_opposite_core_dimensions < 1:
        raise ValueError("--min_opposite_core_dimensions must be positive")
    if args.min_opposite_items < 1:
        raise ValueError("--min_opposite_items must be positive")
    if not 0.0 <= args.min_opposite_fraction <= 1.0:
        raise ValueError("--min_opposite_fraction must be in [0, 1]")
    if not 0.0 <= args.min_emotion_label_distance <= 1.0:
        raise ValueError("--min_emotion_label_distance must be in [0, 1]")
    if not 0 <= args.min_emotion_intensity_gap <= 6:
        raise ValueError("--min_emotion_intensity_gap must be in [0, 6]")


def load_json(path: Path) -> Any:
    return counterfactual.load_json(path)


def failure_bucket(
    failures: Counter[str],
    examples: dict[str, list[str]],
    reason: str,
    identity: str,
    error: Exception | str,
) -> None:
    failures[reason] += 1
    if len(examples[reason]) < 10:
        examples[reason].append(f"{identity}: {error}")


def normalize_first_records(
    payload: Any,
    compact_scale: str,
    failures: Counter[str],
    failure_examples: dict[str, list[str]],
) -> list[counterfactual.NormalizedRecord]:
    output: list[counterfactual.NormalizedRecord] = []
    seen: set[str] = set()
    for source_id, raw in counterfactual.iter_source_records(payload):
        identity = str(raw.get("id") or raw.get("sample_id") or source_id)
        try:
            if identity in seen:
                raise ValueError("duplicate source_sample_id")
            seen.add(identity)
            labels, scores = counterfactual.extract_appraisal_ratings(
                raw,
                identity,
                compact_scale,
            )
            output.append(
                counterfactual.NormalizedRecord(
                    sample_id=identity,
                    participant_id=str(raw.get("participant_id") or identity),
                    situation=counterfactual.extract_situation(raw),
                    appraisal_ratings=labels,
                    appraisal_scores=scores,
                    appraisal_reasoning=counterfactual.extract_reasoning(raw),
                    emotion_labels=counterfactual.normalize_emotion_labels(raw),
                    raw=raw,
                )
            )
        except Exception as exc:  # noqa: BLE001 - malformed data is reported.
            failure_bucket(
                failures,
                failure_examples,
                "invalid_first_person_record",
                identity,
                exc,
            )
    return output


def normalize_observer_records(
    payload: Any,
    compact_scale: str,
    failures: Counter[str],
    failure_examples: dict[str, list[str]],
) -> dict[str, list[counterfactual.ThirdPersonRecord]]:
    output: dict[str, list[counterfactual.ThirdPersonRecord]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for source_id, index, raw in counterfactual.iter_third_person_records(payload):
        annotator_id = str(
            raw.get("participant_id")
            or raw.get("id")
            or f"annotator_{index:03d}"
        )
        identity = f"{source_id}[{index}]/{annotator_id}"
        try:
            key = (source_id, annotator_id)
            if key in seen:
                raise ValueError("duplicate observer for this source_sample_id")
            seen.add(key)
            labels, scores = counterfactual.extract_appraisal_ratings(
                raw,
                identity,
                compact_scale,
            )
            output[source_id].append(
                counterfactual.ThirdPersonRecord(
                    source_sample_id=source_id,
                    participant_id=annotator_id,
                    appraisal_ratings=labels,
                    appraisal_scores=scores,
                    appraisal_reasoning=counterfactual.extract_reasoning(raw),
                    emotion_labels=counterfactual.normalize_emotion_labels(raw),
                    raw=raw,
                )
            )
        except Exception as exc:  # noqa: BLE001 - malformed data is reported.
            failure_bucket(
                failures,
                failure_examples,
                "invalid_observer_record",
                identity,
                exc,
            )
    return dict(output)


def normalize_reasoning(raw: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for policy_dimension, legacy_dimension in CAREBENCH_REASONING_KEYS.items():
        value = raw.get(policy_dimension, raw.get(legacy_dimension))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing appraisal reasoning for {policy_dimension}")
        output[policy_dimension] = value.strip()
    return output


def core_appraisal_scores(rating_scores: Mapping[str, int]) -> dict[str, float]:
    output: dict[str, float] = {}
    for legacy_core in counterfactual.CORE_DIMENSION_ORDER:
        policy_dimension = LEGACY_CORE_TO_POLICY[legacy_core]
        values = [
            int(rating_scores[spec.dimension])
            for spec in counterfactual.SPECS_BY_CORE_DIMENSION[legacy_core]
            if spec.dimension in rating_scores
        ]
        if not values:
            raise ValueError(f"missing appraisal ratings for {policy_dimension}")
        output[policy_dimension] = sum(values) / len(values)
    return output


def build_label_aliases(descriptions: Mapping[str, str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical, description in descriptions.items():
        aliases[canonical.casefold()] = canonical
        aliases[description.casefold()] = canonical
        for synonym in description.split(","):
            aliases[synonym.strip().casefold()] = canonical
    return aliases


POSITIVE_ALIASES = build_label_aliases(POSITIVE_LABEL_DESCRIPTIONS)
NEGATIVE_ALIASES = build_label_aliases(NEGATIVE_LABEL_DESCRIPTIONS)


def parse_intensity(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer from 0 to 6")
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str):
        match = re.match(r"^\s*([0-6])(?:\s|$|-)", value)
        if match is None:
            raise ValueError(f"cannot parse {field}: {value!r}")
        score = int(match.group(1))
    else:
        raise ValueError(f"cannot parse {field}: {value!r}")
    if not 0 <= score <= 6:
        raise ValueError(f"{field} must be in [0, 6]")
    return score


def normalize_label_list(
    values: Any,
    aliases: Mapping[str, str],
    canonical_order: Sequence[str],
    field: str,
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be an array")
    selected: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field} contains a non-string label")
        canonical = aliases.get(value.strip().casefold())
        if canonical is None:
            raise ValueError(f"unknown label {value!r} in {field}")
        selected.add(canonical)
    return [label for label in canonical_order if label in selected]


def normalize_emotion(raw: Mapping[str, Any]) -> dict[str, Any]:
    direct = raw.get("emotion")
    if isinstance(direct, dict):
        positive_intensity = parse_intensity(
            direct.get("positive_intensity"),
            "emotion.positive_intensity",
        )
        negative_intensity = parse_intensity(
            direct.get("negative_intensity"),
            "emotion.negative_intensity",
        )
        positive_raw = direct.get("positive_labels")
        negative_raw = direct.get("negative_labels")
    else:
        legacy = raw.get("emotion_labels")
        if not isinstance(legacy, dict):
            raise ValueError("missing emotion/emotion_labels object")
        positive_intensity = parse_intensity(
            legacy.get("positive_level"),
            "emotion_labels.positive_level",
        )
        negative_intensity = parse_intensity(
            legacy.get("negative_level"),
            "emotion_labels.negative_level",
        )
        positive_raw = legacy.get("positive_emotion_labels")
        negative_raw = legacy.get("negative_emotion_labels")

    positive_labels = normalize_label_list(
        positive_raw,
        POSITIVE_ALIASES,
        POSITIVE_LABELS,
        "positive emotion labels",
    )
    negative_labels = normalize_label_list(
        negative_raw,
        NEGATIVE_ALIASES,
        NEGATIVE_LABELS,
        "negative emotion labels",
    )
    if positive_intensity == 0 and positive_labels:
        raise ValueError("positive labels are present at zero intensity")
    if negative_intensity == 0 and negative_labels:
        raise ValueError("negative labels are present at zero intensity")
    return {
        "positive_intensity": positive_intensity,
        "negative_intensity": negative_intensity,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
    }


def conditional_sample(
    *,
    source_sample_id: str,
    annotator_id: str,
    annotator_type: str,
    annotation_index: int,
    situation: str,
    appraisal_reasoning: Mapping[str, Any],
    appraisal_rating_scores: Mapping[str, int],
    raw: Mapping[str, Any],
) -> ConditionalSample:
    suffix = "first" if annotator_type == "first_person" else f"observer-{annotation_index:03d}"
    return ConditionalSample(
        record_type="conditional",
        sample_id=f"{source_sample_id}::{suffix}::{annotator_id}",
        source_sample_id=source_sample_id,
        annotator_id=annotator_id,
        annotator_type=annotator_type,
        situation=situation,
        appraisal_reasoning=normalize_reasoning(appraisal_reasoning),
        appraisal_scores=core_appraisal_scores(appraisal_rating_scores),
        appraisal_rating_scores={
            dimension: int(score)
            for dimension, score in appraisal_rating_scores.items()
        },
        emotion=normalize_emotion(raw),
    )


def situation_fingerprint(situation: str) -> str:
    normalized = " ".join(situation.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_split(
    records: Sequence[counterfactual.NormalizedRecord],
    validation_ratio: float,
    seed: int,
) -> tuple[list[counterfactual.NormalizedRecord], list[counterfactual.NormalizedRecord]]:
    groups: dict[str, list[counterfactual.NormalizedRecord]] = defaultdict(list)
    for record in records:
        groups[situation_fingerprint(record.situation)].append(record)
    if len(groups) < 2:
        raise ValueError("At least two distinct situations are required for a split")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
    )
    val_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_ratio)))
    val_keys = set(ordered[:val_count])
    train = [record for key, values in groups.items() if key not in val_keys for record in values]
    val = [record for key, values in groups.items() if key in val_keys for record in values]
    return train, val


def assert_disjoint_splits(
    train: Sequence[counterfactual.NormalizedRecord],
    val: Sequence[counterfactual.NormalizedRecord],
) -> None:
    train_ids = {record.sample_id for record in train}
    val_ids = {record.sample_id for record in val}
    duplicate_ids = train_ids & val_ids
    if duplicate_ids:
        raise ValueError(
            "First-person train/validation source_sample_id overlap: "
            + ", ".join(sorted(duplicate_ids)[:10])
        )
    train_situations = {situation_fingerprint(record.situation) for record in train}
    val_situations = {situation_fingerprint(record.situation) for record in val}
    if train_situations & val_situations:
        raise ValueError(
            "Train/validation contain duplicate normalized situations; observer "
            "annotations would leak across splits"
        )


def valenced_label_set(emotion: Mapping[str, Any]) -> set[str]:
    return {
        *(f"positive:{label}" for label in emotion["positive_labels"]),
        *(f"negative:{label}" for label in emotion["negative_labels"]),
    }


def emotion_difference(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_labels = valenced_label_set(left)
    right_labels = valenced_label_set(right)
    union = left_labels | right_labels
    intersection = left_labels & right_labels
    label_distance = 1.0 - len(intersection) / len(union) if union else 0.0
    positive_gap = abs(int(left["positive_intensity"]) - int(right["positive_intensity"]))
    negative_gap = abs(int(left["negative_intensity"]) - int(right["negative_intensity"]))
    return {
        "label_jaccard_distance": label_distance,
        "positive_intensity_gap": positive_gap,
        "negative_intensity_gap": negative_gap,
        "max_intensity_gap": max(positive_gap, negative_gap),
        "left_only_labels": sorted(left_labels - right_labels),
        "right_only_labels": sorted(right_labels - left_labels),
    }


def appraisal_difference(
    first: counterfactual.NormalizedRecord,
    observer: counterfactual.ThirdPersonRecord,
    args: argparse.Namespace,
) -> dict[str, Any]:
    details = [
        counterfactual.core_opposition_details(first, observer, core)
        for core in counterfactual.CORE_DIMENSION_ORDER
    ]
    eligible = [
        detail
        for detail in details
        if int(detail["opposite_count"]) >= args.min_opposite_items
        and float(detail["opposite_fraction"]) >= args.min_opposite_fraction
    ]
    return {
        "opposite_core_dimensions": [
            LEGACY_CORE_TO_POLICY[detail["core_dimension"]]
            for detail in eligible
        ],
        "core_details": details,
    }


def contrast_weight(
    appraisal: Mapping[str, Any],
    emotion: Mapping[str, Any],
) -> float:
    appraisal_strength = min(
        1.0,
        len(appraisal["opposite_core_dimensions"])
        / len(counterfactual.CORE_DIMENSION_ORDER),
    )
    emotion_strength = max(
        float(emotion["label_jaccard_distance"]),
        float(emotion["max_intensity_gap"]) / 6.0,
    )
    return math.sqrt(max(0.0, appraisal_strength * emotion_strength))


def pair_payload(
    first_sample: ConditionalSample,
    observer_sample: ConditionalSample,
    first_record: counterfactual.NormalizedRecord,
    observer_record: counterfactual.ThirdPersonRecord,
    observer_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None]:
    appraisal = appraisal_difference(first_record, observer_record, args)
    if len(appraisal["opposite_core_dimensions"]) < args.min_opposite_core_dimensions:
        return None, "pair_appraisal_not_distinct"
    emotion = emotion_difference(first_sample.emotion, observer_sample.emotion)
    if (
        emotion["label_jaccard_distance"] < args.min_emotion_label_distance
        and emotion["max_intensity_gap"] < args.min_emotion_intensity_gap
    ):
        return None, "pair_emotion_not_distinct"
    return {
        "record_type": "pair",
        "pair_id": (
            f"{first_record.sample_id}::first-vs-observer-{observer_index:03d}"
        ),
        "source_sample_id": first_record.sample_id,
        "situation": first_record.situation,
        "left": asdict(first_sample),
        "right": asdict(observer_sample),
        "difference": {
            "appraisal": appraisal,
            "emotion": emotion,
            "contrast_weight": contrast_weight(appraisal, emotion),
        },
    }, None


def build_split_samples(
    first_records: Sequence[counterfactual.NormalizedRecord],
    observers_by_source: Mapping[str, Sequence[counterfactual.ThirdPersonRecord]],
    split: str,
    args: argparse.Namespace,
    failures: Counter[str],
    failure_examples: dict[str, list[str]],
) -> tuple[list[ConditionalSample], list[dict[str, Any]]]:
    conditional: list[ConditionalSample] = []
    pairs: list[dict[str, Any]] = []
    for first_record in first_records:
        try:
            first_sample = conditional_sample(
                source_sample_id=first_record.sample_id,
                annotator_id=first_record.participant_id,
                annotator_type="first_person",
                annotation_index=0,
                situation=first_record.situation,
                appraisal_reasoning=first_record.appraisal_reasoning,
                appraisal_rating_scores=first_record.appraisal_scores,
                raw=first_record.raw,
            )
        except Exception as exc:  # noqa: BLE001
            failure_bucket(
                failures,
                failure_examples,
                f"{split}_invalid_first_person_conditional",
                first_record.sample_id,
                exc,
            )
            continue
        conditional.append(first_sample)

        observers = observers_by_source.get(first_record.sample_id, ())
        if not observers:
            failures[f"{split}_missing_observers"] += 1
        for observer_index, observer in enumerate(observers):
            identity = f"{first_record.sample_id}/{observer.participant_id}"
            try:
                observer_sample = conditional_sample(
                    source_sample_id=first_record.sample_id,
                    annotator_id=observer.participant_id,
                    annotator_type="observer",
                    annotation_index=observer_index,
                    situation=first_record.situation,
                    appraisal_reasoning=observer.appraisal_reasoning,
                    appraisal_rating_scores=observer.appraisal_scores,
                    raw=observer.raw,
                )
            except Exception as exc:  # noqa: BLE001
                failure_bucket(
                    failures,
                    failure_examples,
                    f"{split}_invalid_observer_conditional",
                    identity,
                    exc,
                )
                continue
            conditional.append(observer_sample)
            pair, reason = pair_payload(
                first_sample,
                observer_sample,
                first_record,
                observer,
                observer_index,
                args,
            )
            if pair is None:
                failures[f"{split}_{reason}"] += 1
            else:
                pairs.append(pair)
    return conditional, pairs


def emotion_distribution(samples: Iterable[ConditionalSample]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    positive_intensity: Counter[str] = Counter()
    negative_intensity: Counter[str] = Counter()
    annotator_types: Counter[str] = Counter()
    count = 0
    for sample in samples:
        count += 1
        annotator_types[sample.annotator_type] += 1
        for label in sample.emotion["positive_labels"]:
            labels[f"positive:{label}"] += 1
        for label in sample.emotion["negative_labels"]:
            labels[f"negative:{label}"] += 1
        positive_intensity[str(sample.emotion["positive_intensity"])] += 1
        negative_intensity[str(sample.emotion["negative_intensity"])] += 1
    return {
        "samples": count,
        "annotator_types": dict(sorted(annotator_types.items())),
        "labels": dict(sorted(labels.items())),
        "positive_intensity": dict(sorted(positive_intensity.items())),
        "negative_intensity": dict(sorted(negative_intensity.items())),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    counterfactual.configure_specs(args.prompt_config)
    third_path = counterfactual.resolve_third_person_input(args.third_person_input)

    failures: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = defaultdict(list)
    first_payload = load_json(args.first_person_input)
    first_records = normalize_first_records(
        first_payload,
        args.compact_scale,
        failures,
        failure_examples,
    )
    if args.first_person_val_input is not None:
        train_records = first_records
        val_records = normalize_first_records(
            load_json(args.first_person_val_input),
            args.compact_scale,
            failures,
            failure_examples,
        )
        split_mode = "provided"
    else:
        train_records, val_records = deterministic_split(
            first_records,
            args.validation_ratio,
            args.seed,
        )
        split_mode = "deterministic_situation"
    if not train_records or not val_records:
        raise ValueError("Both train and validation splits must contain records")
    assert_disjoint_splits(train_records, val_records)

    observers = normalize_observer_records(
        load_json(third_path),
        args.compact_scale,
        failures,
        failure_examples,
    )
    known_sources = {
        record.sample_id for record in [*train_records, *val_records]
    }
    orphan_observers = sum(
        len(records)
        for source_id, records in observers.items()
        if source_id not in known_sources
    )
    if orphan_observers:
        failures["observer_source_not_in_first_person_splits"] += orphan_observers

    train_samples, train_pairs = build_split_samples(
        train_records,
        observers,
        "train",
        args,
        failures,
        failure_examples,
    )
    val_samples, val_pairs = build_split_samples(
        val_records,
        observers,
        "val",
        args,
        failures,
        failure_examples,
    )
    if not train_samples or not val_samples:
        raise ValueError("No valid conditional samples remain in one or both splits")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "train": args.output_dir / TRAIN_NAME,
        "pairs_train": args.output_dir / PAIR_TRAIN_NAME,
        "pairs_val": args.output_dir / PAIR_VAL_NAME,
        "val": args.output_dir / VAL_NAME,
        "report": args.output_dir / REPORT_NAME,
    }
    if not args.overwrite:
        existing_outputs = [path for path in output_paths.values() if path.exists()]
        if existing_outputs:
            formatted = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(
                f"Output files already exist: {formatted}; pass --overwrite to replace them"
            )
    write_jsonl(
        output_paths["train"],
        (asdict(sample) for sample in train_samples),
        args.overwrite,
    )
    write_jsonl(output_paths["pairs_train"], train_pairs, args.overwrite)
    write_jsonl(output_paths["pairs_val"], val_pairs, args.overwrite)
    write_jsonl(
        output_paths["val"],
        (asdict(sample) for sample in val_samples),
        args.overwrite,
    )

    pair_dimensions: Counter[str] = Counter()
    for pair in train_pairs:
        pair_dimensions.update(
            pair["difference"]["appraisal"]["opposite_core_dimensions"]
        )
    report: dict[str, Any] = {
        "schema_version": "carebench-ae-data-v1",
        "inputs": {
            "first_person_input": str(args.first_person_input),
            "first_person_val_input": (
                str(args.first_person_val_input)
                if args.first_person_val_input is not None
                else None
            ),
            "third_person_input": str(third_path),
            "prompt_config": str(args.prompt_config),
        },
        "split": {
            "mode": split_mode,
            "seed": args.seed,
            "validation_ratio": (
                None if args.first_person_val_input is not None else args.validation_ratio
            ),
            "train_source_records": len(train_records),
            "val_source_records": len(val_records),
            "train_source_situations": len(
                {situation_fingerprint(record.situation) for record in train_records}
            ),
            "val_source_situations": len(
                {situation_fingerprint(record.situation) for record in val_records}
            ),
        },
        "filters": {
            "compact_scale": args.compact_scale,
            "min_opposite_core_dimensions": args.min_opposite_core_dimensions,
            "min_opposite_items": args.min_opposite_items,
            "min_opposite_fraction": args.min_opposite_fraction,
            "min_emotion_label_distance": args.min_emotion_label_distance,
            "min_emotion_intensity_gap": args.min_emotion_intensity_gap,
        },
        "counts": {
            "normalized_observer_annotations": sum(map(len, observers.values())),
            "train_conditional_samples": len(train_samples),
            "train_pairs": len(train_pairs),
            "val_conditional_samples": len(val_samples),
            "val_pairs": len(val_pairs),
        },
        "filter_reasons": dict(sorted(failures.items())),
        "filter_examples": dict(sorted(failure_examples.items())),
        "emotion_distribution": {
            "train": emotion_distribution(train_samples),
            "val": emotion_distribution(val_samples),
        },
        "train_pair_opposite_dimensions": dict(sorted(pair_dimensions.items())),
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }
    write_json(output_paths["report"], report, args.overwrite)
    return report


def main() -> int:
    report = run(parse_args())
    counts = report["counts"]
    print(
        "[ae-data] "
        f"train={counts['train_conditional_samples']} "
        f"pairs={counts['train_pairs']} "
        f"val={counts['val_conditional_samples']}"
    )
    print(f"[ae-data] report={report['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
