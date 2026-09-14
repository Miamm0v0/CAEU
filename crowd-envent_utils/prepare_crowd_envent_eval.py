#!/usr/bin/env python3
"""Prepare a leakage-controlled first-person crowd-enVent evaluation set."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from crowd_envent_schema import APPRAISAL_FIELDS, EMOTION_LABELS


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
DEFAULT_GENERATION = REPO_ROOT / "crowd-enVent2023/corpus/crowd-enVent_generation.tsv"
DEFAULT_VALIDATION = REPO_ROOT / "crowd-enVent2023/corpus/crowd-enVent_validation.tsv"
DEFAULT_OUTPUT = REPO_ROOT / "FirstPersonMethod/data/crowd_envent/test.json"

CONFIRMED_REAL = "The event really happened in my life."
CONFIRMED_IMAGINED = (
    "I never experienced that event, but I really imagined how it would make me feel."
)
SUBSETS = ("strict-validated", "strict-all", "all-except-imagined", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare crowd-enVent for first-person Policy evaluation"
    )
    parser.add_argument("--generation_tsv", type=Path, default=DEFAULT_GENERATION)
    parser.add_argument("--validation_tsv", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subset", choices=list(SUBSETS), default="strict-validated")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--no_mask_emotion_words", action="store_true")
    parser.add_argument(
        "--first_person_prefix",
        default="I experienced the following situation:",
        help="Prefix added without revealing the gold emotion",
    )
    parser.add_argument("--indent", type=int, default=2)
    return parser


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no TSV header")
        return [dict(row) for row in reader]


def parse_rating(value: str, location: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be an integer") from exc
    if not 1 <= parsed <= 5:
        raise ValueError(f"{location} must be in [1, 5]")
    return parsed


def mask_emotion_cues(text: str) -> str:
    """Apply one uniform mask to dataset ellipses and all class-name tokens."""
    masked = re.sub(r"\.{3,}", " [EMOTION] ", text)
    label_words = [label for label in EMOTION_LABELS if label != "no-emotion"]
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(label) for label in label_words) + r")\b",
        re.IGNORECASE,
    )
    masked = pattern.sub("[EMOTION]", masked)
    masked = re.sub(r"(?:\s*\[EMOTION\]\s*){2,}", " [EMOTION] ", masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def first_person_situation(text: str, prefix: str, mask_words: bool) -> str:
    cleaned = mask_emotion_cues(text) if mask_words else re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("hidden_emo_text is empty after preprocessing")
    normalized_prefix = prefix.strip()
    return f"{normalized_prefix} {cleaned}" if normalized_prefix else cleaned


def validation_index(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("text_id", "")).strip()].append(row)
    return dict(grouped)


def select_generation_rows(
    rows: list[dict[str, str]],
    validations: dict[str, list[dict[str, str]]],
    subset: str,
) -> list[dict[str, str]]:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}")
    selected: list[dict[str, str]] = []
    for row in rows:
        reality = row.get("did_you_lie?", "").strip()
        text_id = row.get("text_id", "").strip()
        if subset == "strict-validated":
            include = reality == CONFIRMED_REAL and text_id in validations
        elif subset == "strict-all":
            include = reality == CONFIRMED_REAL
        elif subset == "all-except-imagined":
            include = reality != CONFIRMED_IMAGINED
        else:
            include = True
        if include:
            selected.append(row)
    return selected


def majority_labels(labels: list[str]) -> list[str]:
    if not labels:
        return []
    counts = Counter(labels)
    maximum = max(counts.values())
    return sorted(label for label, count in counts.items() if count == maximum)


def mean_validation_ratings(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        field: sum(parse_rating(row[field], f"validation.{field}") for row in rows)
        / len(rows)
        for field in APPRAISAL_FIELDS
    }


def build_sample(
    row: dict[str, str],
    validation_rows: list[dict[str, str]],
    *,
    mask_words: bool,
    prefix: str,
) -> dict[str, Any]:
    text_id = row.get("text_id", "").strip()
    if not text_id:
        raise ValueError("generation row has no text_id")
    emotion = row.get("emotion", "").strip().lower()
    if emotion not in EMOTION_LABELS:
        raise ValueError(f"{text_id}: invalid emotion {emotion!r}")
    hidden_text = row.get("hidden_emo_text", "").strip()
    if not hidden_text:
        raise ValueError(f"{text_id}: missing hidden_emo_text")
    ratings = {
        field: parse_rating(row.get(field, ""), f"{text_id}.{field}")
        for field in APPRAISAL_FIELDS
    }
    validator_emotions = [
        item.get("emotion", "").strip().lower()
        for item in validation_rows
        if item.get("emotion", "").strip().lower() in EMOTION_LABELS
    ]
    return {
        "id": f"crowd-envent-{text_id}",
        "source_id": text_id,
        "situation": first_person_situation(hidden_text, prefix, mask_words),
        "reference": {
            "appraisal_ratings": ratings,
            "emotion": {
                "label": emotion,
                "intensity": parse_rating(
                    row.get("intensity", ""), f"{text_id}.intensity"
                ),
            },
        },
        "validation_reference": {
            "annotation_count": len(validation_rows),
            "emotion_labels": validator_emotions,
            "majority_emotion_labels": majority_labels(validator_emotions),
            "mean_appraisal_ratings": mean_validation_ratings(validation_rows),
        },
        "metadata": {
            "round_number": row.get("round_number", ""),
            "confirmed_real": row.get("did_you_lie?", "").strip() == CONFIRMED_REAL,
            "reality_response": row.get("did_you_lie?", "").strip(),
            "emotion_words_masked": mask_words,
            "original_hidden_emo_text": hidden_text,
        },
    }


def validate_unique_authors(samples: list[dict[str, Any]], source_rows: list[dict[str, str]]) -> int:
    by_text_id = {row["text_id"]: row for row in source_rows}
    authors = [by_text_id[sample["source_id"]].get("prolific_id", "") for sample in samples]
    return len(set(authors))


def write_json_atomic(path: Path, payload: dict[str, Any], indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    temporary.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    require_file(args.generation_tsv, "crowd-enVent generation TSV")
    require_file(args.validation_tsv, "crowd-enVent validation TSV")
    generation = load_tsv(args.generation_tsv)
    validation = load_tsv(args.validation_tsv)
    validations = validation_index(validation)
    selected_rows = select_generation_rows(generation, validations, args.subset)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        if args.max_samples < len(selected_rows):
            selected_rows = random.Random(args.sample_seed).sample(
                selected_rows, args.max_samples
            )
    samples = [
        build_sample(
            row,
            validations.get(row["text_id"], []),
            mask_words=not args.no_mask_emotion_words,
            prefix=args.first_person_prefix,
        )
        for row in selected_rows
    ]
    if not samples:
        raise ValueError("Selection produced no crowd-enVent samples")
    payload = {
        "dataset": "crowd-enVent-2023",
        "schema_version": "first-person-policy-eval-v1",
        "subset": args.subset,
        "source": {
            "generation_tsv": str(args.generation_tsv),
            "validation_tsv": str(args.validation_tsv),
        },
        "preprocessing": {
            "uses_hidden_emo_text": True,
            "mask_all_emotion_class_words": not args.no_mask_emotion_words,
            "first_person_prefix": args.first_person_prefix,
            "sample_seed": args.sample_seed,
        },
        "stats": {
            "samples": len(samples),
            "unique_authors": validate_unique_authors(samples, selected_rows),
            "validated_samples": sum(
                sample["validation_reference"]["annotation_count"] > 0
                for sample in samples
            ),
            "confirmed_real_samples": sum(
                sample["metadata"]["confirmed_real"] for sample in samples
            ),
            "emotion_distribution": dict(
                sorted(Counter(sample["reference"]["emotion"]["label"] for sample in samples).items())
            ),
        },
        "samples": samples,
    }
    write_json_atomic(args.output_file, payload, args.indent)
    print(json.dumps(payload["stats"], ensure_ascii=False, indent=2))
    print(f"[write] {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
