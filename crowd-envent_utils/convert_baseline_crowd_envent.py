#!/usr/bin/env python3
"""Convert baseline_transformers task outputs to crowd-enVent JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "FirstPersonMethod" / "scripts"
for import_path in (UTILS_DIR, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from carebench_aligned_projection import (
    CROWD_ENVENT_NEGATIVE_LABELS,
    CROWD_ENVENT_NO_EMOTION,
    CROWD_ENVENT_POSITIVE_LABELS,
    normalize_carebench_appraisal_ratings,
    project_crowd_envent_rating,
)
from crowd_envent_schema import (
    APPRAISAL_FIELDS,
    EMOTION_LABELS,
    normalize_emotion,
    normalize_ratings,
    normalize_reasoning,
)
from evaluate_policy_crowd_envent import load_evaluation_file, select_samples


EVALUATION_ONLY_EMOTION_RANKING_FIELD = "evaluation_only_emotion_ranking"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _require_intensity(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 6:
        raise ValueError(f"{field} must be an integer in [0, 6]")
    return value


def _normalize_labels(value: Any, allowed: tuple[str, ...], field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    allowed_without_none = tuple(
        label for label in allowed if label != CROWD_ENVENT_NO_EMOTION
    )
    selected: list[str] = []
    for raw_label in value:
        if not isinstance(raw_label, str):
            raise ValueError(f"{field} contains a non-string label")
        label = raw_label.strip().lower()
        if label not in allowed_without_none:
            raise ValueError(
                f"{field} has invalid label {raw_label!r}; "
                f"allowed={list(allowed_without_none)}"
            )
        if label not in selected:
            selected.append(label)
    return selected


def _scale_intensity(score: int) -> int:
    return max(1, min(5, int(math.floor(1.0 + score * (4.0 / 6.0) + 0.5))))


def _native_emotion(raw_emotion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(raw_emotion, dict):
        raise ValueError("chain-emotion output has no emotion object")
    expected = {
        "positive_intensity",
        "negative_intensity",
        "positive_labels",
        "negative_labels",
    }
    if set(raw_emotion) != expected:
        raise ValueError(
            "emotion keys mismatch: "
            f"missing={sorted(expected - set(raw_emotion))}, "
            f"extra={sorted(set(raw_emotion) - expected)}"
        )
    positive_score = _require_intensity(
        raw_emotion["positive_intensity"], "emotion.positive_intensity"
    )
    negative_score = _require_intensity(
        raw_emotion["negative_intensity"], "emotion.negative_intensity"
    )
    positive_labels = _normalize_labels(
        raw_emotion["positive_labels"],
        CROWD_ENVENT_POSITIVE_LABELS,
        "emotion.positive_labels",
    )
    negative_labels = _normalize_labels(
        raw_emotion["negative_labels"],
        CROWD_ENVENT_NEGATIVE_LABELS,
        "emotion.negative_labels",
    )
    if positive_score == 0 and positive_labels:
        raise ValueError("positive_labels must be empty when positive_intensity is 0")
    if negative_score == 0 and negative_labels:
        raise ValueError("negative_labels must be empty when negative_intensity is 0")

    positive_label = positive_labels[0] if positive_labels else CROWD_ENVENT_NO_EMOTION
    negative_label = negative_labels[0] if negative_labels else CROWD_ENVENT_NO_EMOTION
    valenced_emotion = {
        "positive": {
            "label": positive_label,
            "intensity": _scale_intensity(positive_score) if positive_labels else 1,
        },
        "negative": {
            "label": negative_label,
            "intensity": _scale_intensity(negative_score) if negative_labels else 1,
        },
    }
    if positive_labels and negative_labels:
        selected_side = "positive" if positive_score >= negative_score else "negative"
        native = dict(valenced_emotion[selected_side])
    elif positive_labels:
        native = dict(valenced_emotion["positive"])
    elif negative_labels:
        native = dict(valenced_emotion["negative"])
    else:
        native = {"label": CROWD_ENVENT_NO_EMOTION, "intensity": 1}
    return valenced_emotion, native


def _normalize_external_emotion(
    raw_emotion: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if isinstance(raw_emotion, dict) and set(raw_emotion) == {"label", "intensity"}:
        return normalize_emotion(raw_emotion), None
    if isinstance(raw_emotion, dict) and set(raw_emotion) == {"labels", "intensity"}:
        ranked = _normalize_ranked_multilabel_emotion(raw_emotion)
        return {
            "label": ranked["labels"][0],
            "intensity": ranked["intensity"],
        }, None
    legacy_valenced, native = _native_emotion(raw_emotion)
    return native, legacy_valenced


def _normalize_ranked_multilabel_emotion(raw_emotion: Any) -> dict[str, Any]:
    if not isinstance(raw_emotion, dict):
        raise ValueError("multi-label emotion output must be a JSON object")
    if set(raw_emotion) != {"labels", "intensity"}:
        raise ValueError("multi-label emotion output requires labels and intensity")
    raw_labels = raw_emotion["labels"]
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("emotion.labels must contain at least one label")
    labels: list[str] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, str):
            raise ValueError("emotion.labels contains a non-string label")
        label = raw_label.strip().lower()
        if label in {"none", "no emotion"}:
            label = CROWD_ENVENT_NO_EMOTION
        if label not in EMOTION_LABELS:
            raise ValueError(
                f"emotion.labels has invalid label {raw_label!r}; "
                f"allowed={list(EMOTION_LABELS)}"
            )
        if label not in labels:
            labels.append(label)
    if CROWD_ENVENT_NO_EMOTION in labels and len(labels) != 1:
        raise ValueError("no-emotion cannot be combined with another emotion label")
    normalized_primary = normalize_emotion(
        {"label": labels[0], "intensity": raw_emotion["intensity"]}
    )
    return {"labels": labels, "intensity": normalized_primary["intensity"]}


def _emotion_candidate_labels(raw_emotion: Any) -> list[str]:
    """Preserve the complete pre-conversion label set for hit-rate analysis."""
    if isinstance(raw_emotion, dict) and set(raw_emotion) == {
        "label",
        "intensity",
    }:
        return [normalize_emotion(raw_emotion)["label"]]
    if isinstance(raw_emotion, dict) and set(raw_emotion) == {
        "labels",
        "intensity",
    }:
        ranked = _normalize_ranked_multilabel_emotion(raw_emotion)
        selected = set(ranked["labels"])
        return [label for label in EMOTION_LABELS if label in selected]
    if not isinstance(raw_emotion, dict):
        raise ValueError("chain-emotion output has no emotion object")
    positive = _normalize_labels(
        raw_emotion.get("positive_labels"),
        CROWD_ENVENT_POSITIVE_LABELS,
        "emotion.positive_labels",
    )
    negative = _normalize_labels(
        raw_emotion.get("negative_labels"),
        CROWD_ENVENT_NEGATIVE_LABELS,
        "emotion.negative_labels",
    )
    selected = set(positive) | set(negative)
    merged = [label for label in EMOTION_LABELS if label in selected]
    return merged or [CROWD_ENVENT_NO_EMOTION]


def _normalize_evaluation_only_emotion_ranking(
    raw_ranking: Any,
    raw_emotion: Any,
) -> list[str]:
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError(
            f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} must be a non-empty array"
        )
    ranking: list[str] = []
    for raw_label in raw_ranking:
        if not isinstance(raw_label, str):
            raise ValueError(
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} contains a non-string label"
            )
        label = raw_label.strip().lower()
        if label in {"none", "no emotion"}:
            label = CROWD_ENVENT_NO_EMOTION
        if label not in EMOTION_LABELS:
            raise ValueError(
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} has invalid label "
                f"{raw_label!r}; allowed={list(EMOTION_LABELS)}"
            )
        if label in ranking:
            raise ValueError(
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} contains duplicate "
                f"label {label!r}"
            )
        ranking.append(label)
    expected = set(_emotion_candidate_labels(raw_emotion))
    if set(ranking) != expected:
        raise ValueError(
            f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} must contain exactly the "
            "native positive_labels and negative_labels candidates"
        )
    return ranking


def convert_outputs(
    eval_file: Path,
    baseline_output_folder: Path,
    predictions_file: Path,
    invalid_file: Path | None = None,
    max_samples: int | None = None,
    sample_seed: int = 42,
    generation_schema: str = "chain",
    emotion_output_mode: str = "single-label",
) -> tuple[int, int]:
    if generation_schema not in {"chain", "official-direct"}:
        raise ValueError(
            "generation_schema must be 'chain' or 'official-direct'"
        )
    if emotion_output_mode not in {"single-label", "multi-label"}:
        raise ValueError(
            "emotion_output_mode must be 'single-label' or 'multi-label'"
        )
    evaluation = load_evaluation_file(eval_file)
    samples = select_samples(evaluation["samples"], max_samples, sample_seed)
    records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["id"]
        if generation_schema == "chain":
            appraisal_task = "chain-appraisals"
            emotion_task = "chain-emotion"
        else:
            appraisal_task = "direct-appraisals"
            emotion_task = "direct-emotion"
        appraisal_path = baseline_output_folder / appraisal_task / f"{sample_id}.json"
        emotion_path = baseline_output_folder / emotion_task / f"{sample_id}.json"
        try:
            appraisal_trajectory = _read_json(appraisal_path)
            emotion_trajectory = _read_json(emotion_path)
            raw_emotion = emotion_trajectory.get("emotion")
            if emotion_output_mode == "single-label":
                expected_emotion_keys = {"label", "intensity"}
            elif generation_schema == "chain":
                expected_emotion_keys = {
                    "positive_intensity",
                    "negative_intensity",
                    "positive_labels",
                    "negative_labels",
                }
            else:
                expected_emotion_keys = {"labels", "intensity"}
            if not isinstance(raw_emotion, dict) or set(raw_emotion) != expected_emotion_keys:
                raise ValueError(
                    f"{emotion_task} emotion keys do not match "
                    f"--emotion_output_mode {emotion_output_mode}: "
                    f"expected={sorted(expected_emotion_keys)}"
                )
            native_emotion, legacy_carebench_emotion = _normalize_external_emotion(
                raw_emotion
            )
            emotion_candidate_labels = _emotion_candidate_labels(raw_emotion)
            evaluation_only_emotion_ranking = None
            if generation_schema == "chain" and emotion_output_mode == "multi-label":
                evaluation_only_emotion_ranking = (
                    _normalize_evaluation_only_emotion_ranking(
                        emotion_trajectory.get(
                            EVALUATION_ONLY_EMOTION_RANKING_FIELD
                        ),
                        raw_emotion,
                    )
                )
            if generation_schema == "chain":
                raw_ratings = appraisal_trajectory.get("appraisals")
                if not isinstance(raw_ratings, dict):
                    raise ValueError(
                        "chain-appraisals output has no appraisals object"
                    )
                carebench_ratings = normalize_carebench_appraisal_ratings(
                    {
                        key: value.get("score") if isinstance(value, dict) else None
                        for key, value in raw_ratings.items()
                    }
                )
                native_ratings = {
                    field: project_crowd_envent_rating(field, carebench_ratings)
                    for field in APPRAISAL_FIELDS
                }
                record = {
                    "sample_id": sample_id,
                    "source_id": sample.get("source_id"),
                    "generation_schema": "carebench",
                    "appraisal_reasoning": {
                        "appraisals": normalize_reasoning(
                            appraisal_trajectory.get("appraisal_reasoning")
                        ),
                        "emotion": normalize_reasoning(
                            emotion_trajectory.get("appraisal_reasoning")
                        ),
                    },
                    "carebench_appraisal_ratings": carebench_ratings,
                    "appraisal_ratings": native_ratings,
                    "emotion": native_emotion,
                    "emotion_candidate_labels": emotion_candidate_labels,
                    "projection": {
                        "appraisals": "carebench_statement_projection_v1",
                        "emotion": (
                            "native_crowd_envent_single_label"
                            if legacy_carebench_emotion is None
                            else "baseline_split_intensity_to_crowd_envent_v1"
                        ),
                    },
                    "baseline_outputs": {
                        "appraisals": str(appraisal_path),
                        "emotion": str(emotion_path),
                    },
                }
                if legacy_carebench_emotion is not None:
                    record["carebench_emotion"] = legacy_carebench_emotion
                if evaluation_only_emotion_ranking is not None:
                    record[EVALUATION_ONLY_EMOTION_RANKING_FIELD] = (
                        evaluation_only_emotion_ranking
                    )
            else:
                if legacy_carebench_emotion is not None:
                    raise ValueError(
                        "direct-emotion output must use one native crowd-enVent label"
                    )
                native_ratings = normalize_ratings(
                    appraisal_trajectory.get("appraisal_ratings")
                )
                record = {
                    "sample_id": sample_id,
                    "source_id": sample.get("source_id"),
                    "generation_schema": "direct",
                    "appraisal_ratings": native_ratings,
                    "emotion": native_emotion,
                    "emotion_candidate_labels": emotion_candidate_labels,
                    "projection": {
                        "appraisals": "native_crowd_envent_direct",
                        "emotion": (
                            "ranked_multilabel_first_label"
                            if emotion_output_mode == "multi-label"
                            else "native_crowd_envent_single_label"
                        ),
                    },
                    "baseline_outputs": {
                        "appraisals": str(appraisal_path),
                        "emotion": str(emotion_path),
                    },
                }
                if emotion_output_mode == "multi-label":
                    record["multilabel_emotion"] = (
                        _normalize_ranked_multilabel_emotion(raw_emotion)
                    )
            records.append(record)
        except (FileNotFoundError, OSError, KeyError, TypeError, ValueError) as exc:
            invalid_records.append(
                {
                    "sample_id": sample_id,
                    "source_id": sample.get("source_id"),
                    "error": str(exc),
                    "required_outputs": [str(appraisal_path), str(emotion_path)],
                }
            )

    _write_jsonl_atomic(predictions_file, records)
    invalid_path = invalid_file or predictions_file.with_name(
        predictions_file.stem + ".invalid.jsonl"
    )
    _write_jsonl_atomic(invalid_path, invalid_records)
    print(
        f"[convert] dataset=crowd-envent valid={len(records)} "
        f"invalid={len(invalid_records)} predictions={predictions_file}"
    )
    return len(records), len(invalid_records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert baseline_transformers outputs to crowd-enVent JSONL"
    )
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--baseline_output_folder", type=Path, required=True)
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--invalid_file", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument(
        "--generation_schema",
        choices=["chain", "official-direct"],
        default="chain",
        help="Layout of the saved Transformers task outputs.",
    )
    parser.add_argument(
        "--emotion_output_mode",
        choices=["single-label", "multi-label"],
        default="single-label",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert_outputs(
        args.eval_file,
        args.baseline_output_folder,
        args.predictions_file,
        args.invalid_file,
        args.max_samples,
        args.sample_seed,
        generation_schema=args.generation_schema,
        emotion_output_mode=args.emotion_output_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
