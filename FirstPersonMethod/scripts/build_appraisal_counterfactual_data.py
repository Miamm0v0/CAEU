#!/usr/bin/env python3
"""Build appraisal-intervention inputs for CAREBench counterfactual scripts.

The CAREBench prefix-intervention runners expect files shaped like:

    data/counterfactual/<appraisal_dimension>/<sample_id>.json

where each JSON file is a non-empty list of items and each item contains the
native ``situation``, a separate ``appraisal_reasoning`` profile, and gold
``appraisal_ratings``.

This builder creates a prefix-intervention version of that format:

* original first-person file:
    the native situation, with the original appraisal profile stored separately
* counterfactual file:
    the same native situation, with a separate appraisal profile in which one
    core appraisal dimension is replaced by a real third-person annotation

The appraisal profile is not concatenated into ``final_scenario``. A prefix
intervention runner applies the baseline prompt to the native situation, then
continues from either the original or intervened assistant appraisal prefix.

By default, third-person annotations are read from ``--third_person_input`` and
filtered by scale side within the rating items belonging to the same core
dimension. For example, if any relevance rating is first-person 1/2 but
third-person 4/5, or vice versa, the relevance appraisal can be replaced.
Ratings 1 and 2 are the same low side, ratings 4 and 5 are the same high side,
and within-side differences are not treated as counterfactual opposites.
Use ``--counterfactual_source synthetic`` only for a controlled constructed
ablation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ImportError:  # pragma: no cover - for Python < 3.11
    import tomli as tomllib


DEFAULT_PROMPT_CONFIG = Path(__file__).resolve().parent / "prompts" / "counterfactual_prompt.toml"

# Organized FirstPersonMethod/CAREBench files use compact keys and a 0-4 scale.
# Counterfactual/evaluation scripts use the original CAREBench dimensions and
# prompt statements.
APPRAISAL_KEY_MAP = [
    ("A1_relevance_general", "relevance.general"),
    ("A2_relevance_urgency", "relevance.urgency"),
    ("A3_relevance_goals", "relevance.goals"),
    ("A4_relevance_physical_wellbeing", "relevance.bodily motives"),
    ("A5_relevance_people", "relevance.social motives"),
    ("A6_relevance_identity", "relevance.identity motives"),
    ("B1_certainty_clarity", "certainty.construal"),
    ("B2_certainty_next", "certainty.outlook"),
    ("B3_certainty_expected", "certainty.predictability"),
    ("B4_certainty_novelty", "certainty.novelty"),
    ("C1_congruence_good", "congruence.general"),
    ("C2_congruence_improvement", "congruence.outlook"),
    ("C3_congruence_better_than_expected", "congruence.positive prediction error"),
    ("C4_congruence_worse_than_expected", "congruence.negative prediction error"),
    ("D1_control_general", "control.general"),
    ("D2_control_stay_or_leave", "control.select"),
    ("D3_control_others", "control.vicarious"),
    ("D4_control_effort", "control.effortful"),
    ("E1_accountability_self", "accountability.self"),
    ("E2_accountability_others", "accountability.other"),
    ("E3_accountability_intentional", "accountability.intentionality"),
    ("E4_accountability_fairness", "accountability.fairness"),
]

CORE_DIMENSION_ORDER = [
    "relevance",
    "certainty",
    "congruence",
    "control",
    "accountability",
]

CORE_DIMENSION_TO_RATING_DIMENSIONS = {
    "relevance": [
        "relevance.general",
        "relevance.urgency",
        "relevance.goals",
        "relevance.bodily motives",
        "relevance.social motives",
        "relevance.identity motives",
    ],
    "certainty": [
        "certainty.construal",
        "certainty.outlook",
        "certainty.predictability",
        "certainty.novelty",
    ],
    "congruence": [
        "congruence.general",
        "congruence.outlook",
        "congruence.positive prediction error",
        "congruence.negative prediction error",
    ],
    "control": [
        "control.general",
        "control.select",
        "control.vicarious",
        "control.effortful",
    ],
    "accountability": [
        "accountability.self",
        "accountability.other",
        "accountability.intentionality",
        "accountability.fairness",
    ],
}

FALLBACK_DIMENSION_TO_STATEMENT = {
    "relevance.general": "This situation matters to me.",
    "relevance.urgency": "I need to do something about this situation right away.",
    "relevance.goals": "This situation relates to my goals and plans.",
    "relevance.bodily motives": "This situation relates to my physical well-being.",
    "relevance.social motives": "This situation involves people who matter to me.",
    "relevance.identity motives": "This situation concerns who I am and what I stand for.",
    "certainty.construal": "It is clear to me what is going on in this situation.",
    "certainty.outlook": "I know what will come next in this situation.",
    "certainty.predictability": "I saw this situation coming.",
    "certainty.novelty": "This is a new kind of situation for me.",
    "congruence.general": "This is a good situation.",
    "congruence.outlook": "This situation will get better with time.",
    "congruence.positive prediction error": "This situation is better than I expected.",
    "congruence.negative prediction error": "This situation is worse than I expected.",
    "control.general": "This situation is under my control.",
    "control.select": "I can decide whether to stay in this situation or leave it.",
    "control.vicarious": "Someone can handle this situation for me.",
    "control.effortful": "I have to exert effort in this situation.",
    "accountability.self": "I am responsible for this situation.",
    "accountability.other": "Someone else is responsible for this situation.",
    "accountability.intentionality": "This situation was caused intentionally.",
    "accountability.fairness": "This situation is fair and deserved.",
}

LABEL_BY_SCORE = {
    1: "Strongly disagree",
    2: "Somewhat disagree",
    3: "Neither agree nor disagree",
    4: "Somewhat agree",
    5: "Strongly agree",
}

SCORE_BY_LABEL = {label.lower(): score for score, label in LABEL_BY_SCORE.items()}

POSITIVE_LEVEL_LABELS = {
    0: "Not at all positive",
    1: "Very slightly positive",
    2: "Slightly positive",
    3: "Moderately positive",
    4: "Quite a bit positive",
    5: "Very positive",
    6: "Extremely positive",
}

NEGATIVE_LEVEL_LABELS = {
    0: "Not at all negative",
    1: "Very slightly negative",
    2: "Slightly negative",
    3: "Moderately negative",
    4: "Quite a bit negative",
    5: "Very negative",
    6: "Extremely negative",
}

POSITIVE_EMOTION_LABELS = {
    "hopeful": "Hopeful, optimistic, encouraged",
    "grateful": "Grateful, appreciative, thankful",
    "glad": "Glad, happy, joyful",
    "love": "Love, closeness, trust",
    "amused": "Amused, entertained",
    "sympathy": "Sympathy, concern, compassion",
    "inspiration": "Inspiration, admiration, appreciation",
    "calm": "Calm, peaceful, relaxed",
    "proud": "Proud, confident, determined",
    "excited": "Excited, enthusiastic, elated",
}

NEGATIVE_EMOTION_LABELS = {
    "angry": "Angry, frustrated, annoyed",
    "worried": "Worried, nervous, fearful",
    "sad": "Sad, downhearted, unhappy",
    "disgust": "Disgust, distaste, revulsion",
    "despair": "Despair, hopelessness, sorrow",
    "ashamed": "Ashamed, humiliated, embarrassed",
    "lonely": "Lonely, isolated, disconnected from others",
    "panicked": "Panicked, alarmed, freaked out",
    "guilty": "Guilty, blameworthy, repentant",
    "confused": "Confused, disoriented, surprised",
}


@dataclass(frozen=True)
class AppraisalSpec:
    compact_key: str
    dimension: str
    statement: str


@dataclass
class NormalizedRecord:
    sample_id: str
    participant_id: str
    situation: str
    appraisal_ratings: dict[str, str]
    appraisal_scores: dict[str, int]
    appraisal_reasoning: dict[str, str]
    emotion_labels: dict[str, Any] | None
    raw: dict[str, Any]


@dataclass
class ThirdPersonRecord:
    source_sample_id: str
    participant_id: str
    appraisal_ratings: dict[str, str]
    appraisal_scores: dict[str, int]
    appraisal_reasoning: dict[str, str]
    emotion_labels: dict[str, Any] | None
    raw: dict[str, Any]


def load_dimension_to_statement(prompt_config: Path) -> dict[str, str]:
    if prompt_config.exists():
        with prompt_config.open("rb") as handle:
            prompt_cfg = tomllib.load(handle)
        mapping = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
        if isinstance(mapping, dict) and mapping:
            return {str(key): str(value) for key, value in mapping.items()}

    if prompt_config == DEFAULT_PROMPT_CONFIG:
        return dict(FALLBACK_DIMENSION_TO_STATEMENT)

    raise FileNotFoundError(
        f"Cannot load appraisals.dimension_to_statement from {prompt_config}"
    )


def configure_specs(prompt_config: Path) -> None:
    global SPECS, SPEC_BY_DIMENSION, SPEC_BY_STATEMENT, SPEC_BY_COMPACT_KEY
    global SPECS_BY_CORE_DIMENSION

    dim_to_statement = load_dimension_to_statement(prompt_config)
    missing = [dimension for _, dimension in APPRAISAL_KEY_MAP if dimension not in dim_to_statement]
    if missing:
        raise ValueError(
            "Prompt config is missing statements for dimensions: "
            + ", ".join(missing)
        )

    SPECS = [
        AppraisalSpec(compact_key, dimension, dim_to_statement[dimension])
        for compact_key, dimension in APPRAISAL_KEY_MAP
    ]
    SPEC_BY_DIMENSION = {spec.dimension: spec for spec in SPECS}
    SPEC_BY_STATEMENT = {spec.statement: spec for spec in SPECS}
    SPEC_BY_COMPACT_KEY = {spec.compact_key: spec for spec in SPECS}
    SPECS_BY_CORE_DIMENSION = {
        core_dimension: [
            SPEC_BY_DIMENSION[dimension]
            for dimension in rating_dimensions
            if dimension in SPEC_BY_DIMENSION
        ]
        for core_dimension, rating_dimensions in CORE_DIMENSION_TO_RATING_DIMENSIONS.items()
    }


SPECS: list[AppraisalSpec] = []
SPEC_BY_DIMENSION: dict[str, AppraisalSpec] = {}
SPEC_BY_STATEMENT: dict[str, AppraisalSpec] = {}
SPEC_BY_COMPACT_KEY: dict[str, AppraisalSpec] = {}
SPECS_BY_CORE_DIMENSION: dict[str, list[AppraisalSpec]] = {}
configure_specs(DEFAULT_PROMPT_CONFIG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct first-person and appraisal-counterfactual JSON files "
            "for CAREBench counterfactual_*.py scripts."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "First-person source JSON. Supports a list of organized CAREBench "
            "samples or a raw CAREBench id->sample mapping."
        ),
    )
    parser.add_argument(
        "--third_person_input",
        type=Path,
        default=None,
        help=(
            "Third-person JSON used to construct counterfactual entries. "
            "Expected shape is sample_id -> [annotator objects]. If omitted "
            "with --counterfactual_source third-person, common locations such "
            "as third_person.json are tried."
        ),
    )
    parser.add_argument(
        "--counterfactual_source",
        choices=("third-person", "synthetic"),
        default="third-person",
        help=(
            "third-person filters real third_person annotations; synthetic "
            "constructs opposite-side ratings from the first-person record."
        ),
    )
    parser.add_argument(
        "--replacement_granularity",
        choices=("core", "rating"),
        default="core",
        help=(
            "core replaces one of the five appraisal dimensions using its "
            "grouped rating items; rating keeps the older one-rating-item "
            "counterfactual format."
        ),
    )
    parser.add_argument(
        "--prompt_config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG,
        help="CAREBench prompt TOML that defines appraisals.dimension_to_statement.",
    )
    parser.add_argument(
        "--first_person_root",
        type=Path,
        default=Path("data/first_person_appraisal_counterfactual"),
        help="Output folder for original first-person files.",
    )
    parser.add_argument(
        "--counterfactual_root",
        type=Path,
        default=Path("data/counterfactual_appraisal"),
        help="Output folder for counterfactual appraisal files.",
    )
    parser.add_argument(
        "--original_context",
        choices=("all-appraisals", "situation-only"),
        default="situation-only",
        help=(
            "Context written to original final_scenario. The default "
            "situation-only is required for prefix intervention; "
            "all-appraisals is retained only for reproducing legacy runs."
        ),
    )
    parser.add_argument(
        "--compact_scale",
        choices=("auto", "zero-based", "one-based"),
        default="auto",
        help=(
            "Scale for compact organized appraisal keys. FirstPersonMethod "
            "organized files use zero-based 0-4; CAREBench-compatible scores "
            "usually use one-based 1-5."
        ),
    )
    parser.add_argument(
        "--opposite_policy",
        choices=("opposite-side", "mirror", "nearest"),
        default="opposite-side",
        help=(
            "How to move ratings to the other side of the scale. "
            "opposite-side uses 1/2->[4,5] and 4/5->[1,2]; "
            "mirror uses 1<->5 and 2<->4; nearest uses 1/2->4 and 4/5->2."
        ),
    )
    parser.add_argument(
        "--min_opposite_count",
        type=int,
        default=1,
        help=(
            "For --replacement_granularity core, require at least this many "
            "rating items in the core dimension to be on opposite sides."
        ),
    )
    parser.add_argument(
        "--min_opposite_fraction",
        type=float,
        default=0.0,
        help=(
            "For --replacement_granularity core, require this fraction of "
            "comparable non-neutral rating items to be opposite-side. Use 1.0 "
            "to require all comparable items."
        ),
    )
    parser.add_argument(
        "--neutral_policy",
        choices=("skip", "low", "high"),
        default="skip",
        help=(
            "How to handle neutral original ratings. skip omits them; low "
            "changes 3->2; high changes 3->4."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing JSON files in the output folders.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for a JSON summary of constructed files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite if you want to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_source_records(payload: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        for index, record in enumerate(payload):
            if isinstance(record, dict):
                sample_id = str(
                    record.get("id")
                    or record.get("sample_id")
                    or record.get("participant_id")
                    or f"sample_{index:05d}"
                )
                yield sample_id, record
        return

    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        for index, record in enumerate(payload["samples"]):
            if isinstance(record, dict):
                sample_id = str(
                    record.get("id")
                    or record.get("sample_id")
                    or record.get("participant_id")
                    or f"sample_{index:05d}"
                )
                yield sample_id, record
        return

    if isinstance(payload, dict):
        for sample_id, record in payload.items():
            if isinstance(record, dict):
                record = dict(record)
                record.setdefault("id", str(sample_id))
                yield str(sample_id), record


def sanitize_filename(value: str) -> str:
    value = value.strip() or "sample"
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value)
    value = re.sub(r"\s+", "_", value)
    return value[:160]


def extract_situation(record: dict[str, Any]) -> str:
    direct_fields = ("situation", "scenario", "event", "text")
    for field in direct_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for container_name in ("story_collection", "cognitive_questions"):
        container = record.get(container_name)
        if isinstance(container, dict):
            value = container.get("final_scenario")
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise ValueError("missing situation/final_scenario")


def numeric_score(value: Any, *, zero_based: bool | None = None) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        score = int(value)
        if zero_based is True and 0 <= score <= 4:
            return score + 1
        if 1 <= score <= 5:
            return score
        if zero_based is None and 0 <= score <= 4:
            return score + 1
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        lowered = stripped.lower()
        if lowered in SCORE_BY_LABEL:
            return SCORE_BY_LABEL[lowered]
        match = re.match(r"^([0-5])(?:\b|\s*[-:])", stripped)
        if match:
            score = int(match.group(1))
            if zero_based is True and 0 <= score <= 4:
                return score + 1
            if 1 <= score <= 5:
                return score
            if zero_based is None and 0 <= score <= 4:
                return score + 1
    return None


def infer_compact_zero_based(appraisal_ratings: dict[str, Any]) -> bool | None:
    compact_values: list[int] = []
    for spec in SPECS:
        value = appraisal_ratings.get(spec.compact_key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            compact_values.append(int(value))
        elif isinstance(value, str) and re.fullmatch(r"\s*[0-5]\s*", value):
            compact_values.append(int(value.strip()))

    if not compact_values:
        return None
    if min(compact_values) >= 0 and max(compact_values) <= 4:
        return True
    return False


def extract_appraisal_ratings(
    record: dict[str, Any],
    sample_id: str,
    compact_scale: str,
) -> tuple[dict[str, str], dict[str, int]]:
    raw_ratings = record.get("appraisal_ratings")
    if not isinstance(raw_ratings, dict):
        raise ValueError(f"{sample_id}: missing appraisal_ratings object")

    if compact_scale == "zero-based":
        zero_based = True
    elif compact_scale == "one-based":
        zero_based = False
    else:
        zero_based = infer_compact_zero_based(raw_ratings)
    labels: dict[str, str] = {}
    scores: dict[str, int] = {}

    for spec in SPECS:
        candidates = (
            (spec.statement, None),
            (spec.dimension, None),
            (spec.compact_key, zero_based),
        )
        value = None
        value_zero_based: bool | None = None
        for key, key_zero_based in candidates:
            if key in raw_ratings:
                value = raw_ratings[key]
                value_zero_based = key_zero_based
                break
        if value is None:
            continue
        score = numeric_score(value, zero_based=value_zero_based)
        if score is None or score not in LABEL_BY_SCORE:
            raise ValueError(
                f"{sample_id}: invalid rating for {spec.dimension}: {value!r}"
            )
        scores[spec.dimension] = score
        labels[spec.statement] = LABEL_BY_SCORE[score]

    if not scores:
        raise ValueError(f"{sample_id}: no recognizable CAREBench appraisal ratings")
    return labels, scores


def extract_reasoning(record: dict[str, Any]) -> dict[str, str]:
    reasoning: dict[str, str] = {}

    direct = record.get("appraisal_reasoning")
    if isinstance(direct, dict):
        for key, value in direct.items():
            if isinstance(value, str) and value.strip():
                reasoning[str(key)] = value.strip()

    cognitive_questions = record.get("cognitive_questions")
    if isinstance(cognitive_questions, dict):
        summary_answers = cognitive_questions.get("summary_answers")
        if isinstance(summary_answers, dict):
            for key, value in summary_answers.items():
                if isinstance(value, str) and value.strip():
                    reasoning[str(key)] = value.strip()

    return reasoning


def normalize_emotion_labels(record: dict[str, Any]) -> dict[str, Any] | None:
    raw = record.get("emotion_labels")
    if isinstance(raw, dict):
        return raw

    emotion = record.get("emotion")
    if not isinstance(emotion, dict):
        return None

    labels: dict[str, Any] = {}
    if "positive_intensity" in emotion:
        score = int(emotion["positive_intensity"])
        if score in POSITIVE_LEVEL_LABELS:
            labels["positive_level"] = f"{score} - {POSITIVE_LEVEL_LABELS[score]}"
    if "negative_intensity" in emotion:
        score = int(emotion["negative_intensity"])
        if score in NEGATIVE_LEVEL_LABELS:
            labels["negative_level"] = f"{score} - {NEGATIVE_LEVEL_LABELS[score]}"

    positive_labels = emotion.get("positive_labels")
    if isinstance(positive_labels, list):
        labels["positive_emotion_labels"] = [
            POSITIVE_EMOTION_LABELS.get(str(label), str(label))
            for label in positive_labels
        ]

    negative_labels = emotion.get("negative_labels")
    if isinstance(negative_labels, list):
        labels["negative_emotion_labels"] = [
            NEGATIVE_EMOTION_LABELS.get(str(label), str(label))
            for label in negative_labels
        ]

    return labels or None


def normalize_records(payload: Any, compact_scale: str) -> list[NormalizedRecord]:
    records: list[NormalizedRecord] = []
    errors: list[str] = []

    for sample_id, record in iter_source_records(payload):
        try:
            situation = extract_situation(record)
            labels, scores = extract_appraisal_ratings(record, sample_id, compact_scale)
            participant_id = str(record.get("participant_id") or sample_id)
            records.append(
                NormalizedRecord(
                    sample_id=str(record.get("id") or record.get("sample_id") or sample_id),
                    participant_id=participant_id,
                    situation=situation,
                    appraisal_ratings=labels,
                    appraisal_scores=scores,
                    appraisal_reasoning=extract_reasoning(record),
                    emotion_labels=normalize_emotion_labels(record),
                    raw=record,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report all malformed samples together.
            errors.append(str(exc))

    if not records:
        joined = "\n".join(errors[:20])
        raise ValueError(f"No valid source records were found.\n{joined}")

    if errors:
        print(f"Skipped {len(errors)} malformed records. First issues:")
        for error in errors[:10]:
            print(f"  - {error}")

    return records


def resolve_third_person_input(path: Path | None) -> Path:
    if path is not None:
        return path

    candidates = [
        Path("third_person.json"),
        Path(__file__).resolve().parents[2] / "third_person.json",
        Path(__file__).resolve().parents[1] / "data" / "third_person.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Missing --third_person_input and no default third_person.json was found."
    )


def iter_third_person_records(payload: Any) -> Iterable[tuple[str, int, dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise ValueError("Third-person JSON must be an object: sample_id -> list")

    for sample_id, value in payload.items():
        if isinstance(value, list):
            items = value
        elif isinstance(value, dict):
            items = [value]
        else:
            continue

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            yield str(sample_id), index, item


def normalize_third_person_records(
    payload: Any,
    compact_scale: str,
) -> dict[str, list[ThirdPersonRecord]]:
    records_by_sample: dict[str, list[ThirdPersonRecord]] = {}
    errors: list[str] = []

    for source_sample_id, index, record in iter_third_person_records(payload):
        try:
            labels, scores = extract_appraisal_ratings(
                record,
                f"{source_sample_id}[{index}]",
                compact_scale,
            )
            participant_id = str(
                record.get("participant_id")
                or record.get("id")
                or f"annotator_{index:03d}"
            )
            records_by_sample.setdefault(source_sample_id, []).append(
                ThirdPersonRecord(
                    source_sample_id=source_sample_id,
                    participant_id=participant_id,
                    appraisal_ratings=labels,
                    appraisal_scores=scores,
                    appraisal_reasoning=extract_reasoning(record),
                    emotion_labels=normalize_emotion_labels(record),
                    raw=record,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report malformed annotations together.
            errors.append(str(exc))

    if errors:
        print(f"Skipped {len(errors)} malformed third-person records. First issues:")
        for error in errors[:10]:
            print(f"  - {error}")

    return records_by_sample


def opposite_scores(score: int, policy: str, neutral_policy: str) -> list[int]:
    if score == 3:
        if neutral_policy == "skip":
            return []
        return [2 if neutral_policy == "low" else 4]

    if policy == "opposite-side":
        if score < 3:
            return [4, 5]
        return [1, 2]

    if policy == "mirror":
        return [6 - score]

    if score < 3:
        return [4]
    return [2]


def score_side(score: int) -> str:
    if score < 3:
        return "low"
    if score > 3:
        return "high"
    return "neutral"


def core_dimension_for_rating(rating_dimension: str) -> str:
    for core_dimension, rating_dimensions in CORE_DIMENSION_TO_RATING_DIMENSIONS.items():
        if rating_dimension in rating_dimensions:
            return core_dimension
    raise ValueError(f"No core appraisal dimension for rating: {rating_dimension}")


def is_opposite_side(first_person_score: int, third_person_score: int) -> bool:
    first_side = score_side(first_person_score)
    third_side = score_side(third_person_score)
    return (
        first_side in {"low", "high"}
        and third_side in {"low", "high"}
        and first_side != third_side
    )


def core_opposition_details(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    core_dimension: str,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    comparable_count = 0
    neutral_or_missing_count = 0

    for spec in SPECS_BY_CORE_DIMENSION.get(core_dimension, []):
        first_score = record.appraisal_scores.get(spec.dimension)
        third_score = third_person.appraisal_scores.get(spec.dimension)
        if first_score is None or third_score is None:
            neutral_or_missing_count += 1
            continue
        if score_side(first_score) == "neutral" or score_side(third_score) == "neutral":
            neutral_or_missing_count += 1
            continue
        comparable_count += 1
        if is_opposite_side(first_score, third_score):
            details.append(
                {
                    "dimension": spec.dimension,
                    "statement": spec.statement,
                    "first_person_score": first_score,
                    "first_person_label": LABEL_BY_SCORE[first_score],
                    "third_person_score": third_score,
                    "third_person_label": LABEL_BY_SCORE[third_score],
                }
            )

    opposite_count = len(details)
    opposite_fraction = (
        opposite_count / comparable_count if comparable_count > 0 else 0.0
    )
    return {
        "core_dimension": core_dimension,
        "opposite_count": opposite_count,
        "comparable_count": comparable_count,
        "neutral_or_missing_count": neutral_or_missing_count,
        "opposite_fraction": opposite_fraction,
        "opposite_items": details,
    }


def core_replacement_is_eligible(
    opposition: dict[str, Any],
    min_opposite_count: int,
    min_opposite_fraction: float,
) -> bool:
    comparable_count = int(opposition["comparable_count"])
    opposite_count = int(opposition["opposite_count"])
    if comparable_count <= 0:
        return False
    if opposite_count < min_opposite_count:
        return False
    return float(opposition["opposite_fraction"]) >= min_opposite_fraction


def replaced_core_ratings(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    core_dimension: str,
) -> dict[str, str]:
    ratings = dict(record.appraisal_ratings)
    for spec in SPECS_BY_CORE_DIMENSION.get(core_dimension, []):
        third_label = third_person.appraisal_ratings.get(spec.statement)
        if third_label:
            ratings[spec.statement] = third_label
    return ratings


def replaced_core_reasoning(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    core_dimension: str,
) -> dict[str, str]:
    reasoning = dict(record.appraisal_reasoning)
    third_text = third_person.appraisal_reasoning.get(core_dimension)
    if third_text:
        reasoning[core_dimension] = third_text
    return reasoning


def format_core_heading(core_dimension: str) -> str:
    return f"{core_dimension.capitalize()}:"


def append_core_profile_lines(
    lines: list[str],
    *,
    core_dimension: str,
    reasoning: str,
) -> None:
    lines.append(format_core_heading(core_dimension))
    if reasoning:
        lines.append(reasoning)


def append_appraisal_profile_lines(
    lines: list[str],
    record: NormalizedRecord,
    *,
    replacement_core_dimension: str | None = None,
    replacement_third_person: ThirdPersonRecord | None = None,
) -> None:
    replaced_reasoning: dict[str, str] | None = None
    if replacement_core_dimension and replacement_third_person:
        replaced_reasoning = replaced_core_reasoning(
            record,
            replacement_third_person,
            replacement_core_dimension,
        )

    for core_dimension in CORE_DIMENSION_ORDER:
        is_replaced = core_dimension == replacement_core_dimension
        reasoning = (
            replaced_reasoning.get(core_dimension, "")
            if is_replaced and replaced_reasoning
            else record.appraisal_reasoning.get(core_dimension, "")
        )
        append_core_profile_lines(
            lines,
            core_dimension=core_dimension,
            reasoning=reasoning,
        )


def build_original_scenario(record: NormalizedRecord, context_mode: str) -> str:
    if context_mode == "situation-only":
        return record.situation

    lines = [
        "Original situation:",
        record.situation,
        "",
        "Appraisal profile:",
    ]
    append_appraisal_profile_lines(lines, record)
    return "\n".join(lines)


def build_counterfactual_scenario(
    record: NormalizedRecord,
    spec: AppraisalSpec,
    counterfactual_label: str,
) -> str:
    del spec, counterfactual_label
    return record.situation


def build_core_counterfactual_scenario(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    core_dimension: str,
) -> str:
    del third_person, core_dimension
    return record.situation


def first_person_payload(record: NormalizedRecord, context_mode: str) -> dict[str, Any]:
    scenario = build_original_scenario(record, context_mode)
    payload: dict[str, Any] = {
        "id": record.sample_id,
        "participant_id": record.participant_id,
        "situation": record.situation,
        "story_collection": {"final_scenario": scenario},
        "cognitive_questions": {
            "final_scenario": scenario,
            "summary_answers": record.appraisal_reasoning,
        },
        "appraisal_reasoning": record.appraisal_reasoning,
        "first_person_appraisal_reasoning": record.appraisal_reasoning,
        "appraisal_ratings": record.appraisal_ratings,
        "metadata": {
            "source_sample_id": record.sample_id,
            "construction": (
                "native_situation_with_separate_first_person_appraisal"
                if context_mode == "situation-only"
                else "legacy_situation_plus_first_person_appraisal"
            ),
        },
    }
    if record.emotion_labels is not None:
        payload["emotion_labels"] = record.emotion_labels
    return payload


def third_person_core_counterfactual_payload(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    core_dimension: str,
    opposition: dict[str, Any],
) -> dict[str, Any]:
    scenario = build_core_counterfactual_scenario(
        record,
        third_person,
        core_dimension,
    )
    ratings = replaced_core_ratings(record, third_person, core_dimension)
    reasoning = replaced_core_reasoning(record, third_person, core_dimension)

    payload: dict[str, Any] = {
        "id": (
            f"{record.sample_id}__{core_dimension}"
            f"__third_{sanitize_filename(third_person.participant_id)}"
        ),
        "participant_id": third_person.participant_id,
        "situation": record.situation,
        "story_collection": {"final_scenario": scenario},
        "cognitive_questions": {
            "final_scenario": scenario,
            "summary_answers": reasoning,
        },
        "appraisal_reasoning": reasoning,
        "first_person_appraisal_reasoning": record.appraisal_reasoning,
        "third_person_appraisal_reasoning": third_person.appraisal_reasoning,
        "appraisal_ratings": ratings,
        "metadata": {
            "source_sample_id": record.sample_id,
            "third_person_participant_id": third_person.participant_id,
            "counterfactual_dimension": core_dimension,
            "replacement_granularity": "core",
            "construction": (
                "native_situation_with_one_separate_core_appraisal_replaced_"
                "by_real_third_person"
            ),
            "opposite_count": opposition["opposite_count"],
            "comparable_count": opposition["comparable_count"],
            "opposite_fraction": opposition["opposite_fraction"],
            "neutral_or_missing_count": opposition["neutral_or_missing_count"],
            "opposite_items": opposition["opposite_items"],
        },
    }
    if third_person.emotion_labels is not None:
        payload["emotion_labels"] = third_person.emotion_labels
    return payload


def counterfactual_payload(
    record: NormalizedRecord,
    spec: AppraisalSpec,
    counterfactual_score: int,
) -> dict[str, Any]:
    original_score = record.appraisal_scores[spec.dimension]
    original_label = LABEL_BY_SCORE[original_score]
    counterfactual_label = LABEL_BY_SCORE[counterfactual_score]
    ratings = dict(record.appraisal_ratings)
    ratings[spec.statement] = counterfactual_label
    scenario = build_counterfactual_scenario(record, spec, counterfactual_label)

    return {
        "id": f"{record.sample_id}__{spec.dimension}__score_{counterfactual_score}",
        "participant_id": (
            f"{record.participant_id}__cf_{spec.dimension}__score_{counterfactual_score}"
        ),
        "situation": record.situation,
        "story_collection": {"final_scenario": scenario},
        "cognitive_questions": {
            "final_scenario": scenario,
            "summary_answers": record.appraisal_reasoning,
        },
        "appraisal_reasoning": record.appraisal_reasoning,
        "appraisal_ratings": ratings,
        "metadata": {
            "source_sample_id": record.sample_id,
            "counterfactual_dimension": spec.dimension,
            "counterfactual_statement": spec.statement,
            "original_score": original_score,
            "original_label": original_label,
            "counterfactual_score": counterfactual_score,
            "counterfactual_label": counterfactual_label,
            "construction": "native_situation_with_separate_synthetic_rating_change",
        },
    }


def third_person_counterfactual_payload(
    record: NormalizedRecord,
    third_person: ThirdPersonRecord,
    spec: AppraisalSpec,
    counterfactual_score: int,
) -> dict[str, Any]:
    original_score = record.appraisal_scores[spec.dimension]
    original_label = LABEL_BY_SCORE[original_score]
    counterfactual_label = LABEL_BY_SCORE[counterfactual_score]
    scenario = build_counterfactual_scenario(record, spec, counterfactual_label)
    core_dimension = core_dimension_for_rating(spec.dimension)
    reasoning = replaced_core_reasoning(record, third_person, core_dimension)

    payload: dict[str, Any] = {
        "id": (
            f"{record.sample_id}__{spec.dimension}"
            f"__third_{sanitize_filename(third_person.participant_id)}"
        ),
        "participant_id": third_person.participant_id,
        "situation": record.situation,
        "story_collection": {"final_scenario": scenario},
        "cognitive_questions": {
            "final_scenario": scenario,
            "summary_answers": reasoning,
        },
        "appraisal_reasoning": reasoning,
        "first_person_appraisal_reasoning": record.appraisal_reasoning,
        "third_person_appraisal_reasoning": third_person.appraisal_reasoning,
        "appraisal_ratings": third_person.appraisal_ratings,
        "metadata": {
            "source_sample_id": record.sample_id,
            "third_person_participant_id": third_person.participant_id,
            "counterfactual_dimension": spec.dimension,
            "counterfactual_core_dimension": core_dimension,
            "counterfactual_statement": spec.statement,
            "original_score": original_score,
            "original_label": original_label,
            "third_person_score": counterfactual_score,
            "third_person_label": counterfactual_label,
            "construction": (
                "native_situation_with_one_separate_core_appraisal_replaced_"
                "by_real_third_person"
            ),
        },
    }
    if third_person.emotion_labels is not None:
        payload["emotion_labels"] = third_person.emotion_labels
    return payload


def main() -> None:
    args = parse_args()
    if args.min_opposite_count <= 0:
        raise ValueError("--min_opposite_count must be >= 1")
    if not 0.0 <= args.min_opposite_fraction <= 1.0:
        raise ValueError("--min_opposite_fraction must be between 0 and 1")
    if (
        args.counterfactual_source == "synthetic"
        and args.replacement_granularity == "core"
    ):
        raise ValueError(
            "--counterfactual_source synthetic currently supports "
            "--replacement_granularity rating only"
        )

    configure_specs(args.prompt_config)
    payload = load_json(args.input)
    records = normalize_records(payload, args.compact_scale)
    third_person_input: Path | None = None
    third_person_by_sample: dict[str, list[ThirdPersonRecord]] = {}
    if args.counterfactual_source == "third-person":
        third_person_input = resolve_third_person_input(args.third_person_input)
        third_person_by_sample = normalize_third_person_records(
            load_json(third_person_input),
            args.compact_scale,
        )

    first_person_written = 0
    counterfactual_files_written = 0
    counterfactual_items_written = 0
    skipped_neutral = 0
    skipped_missing_dimension = 0
    skipped_missing_third_person_sample = 0
    skipped_no_opposite_third_person = 0
    output_dimensions = (
        CORE_DIMENSION_ORDER
        if args.replacement_granularity == "core"
        else [spec.dimension for spec in SPECS]
    )
    per_dimension_files: dict[str, int] = {dimension: 0 for dimension in output_dimensions}
    per_dimension_items: dict[str, int] = {dimension: 0 for dimension in output_dimensions}

    for record in records:
        file_stem = sanitize_filename(record.sample_id)
        original_path = args.first_person_root / f"{file_stem}.json"
        dump_json(
            original_path,
            first_person_payload(record, args.original_context),
            args.overwrite,
        )
        first_person_written += 1

        if args.replacement_granularity == "core":
            for core_dimension in CORE_DIMENSION_ORDER:
                core_specs = SPECS_BY_CORE_DIMENSION.get(core_dimension, [])
                if not core_specs:
                    skipped_missing_dimension += 1
                    continue
                has_first_person_side = any(
                    score_side(record.appraisal_scores.get(spec.dimension, 3))
                    in {"low", "high"}
                    for spec in core_specs
                )
                if not has_first_person_side:
                    skipped_neutral += 1
                    continue

                third_person_records = third_person_by_sample.get(record.sample_id, [])
                if not third_person_records:
                    skipped_missing_third_person_sample += 1
                    continue

                cf_rows: list[dict[str, Any]] = []
                for third_person in third_person_records:
                    opposition = core_opposition_details(
                        record,
                        third_person,
                        core_dimension,
                    )
                    if core_replacement_is_eligible(
                        opposition,
                        args.min_opposite_count,
                        args.min_opposite_fraction,
                    ):
                        cf_rows.append(
                            third_person_core_counterfactual_payload(
                                record,
                                third_person,
                                core_dimension,
                                opposition,
                            )
                        )
                if not cf_rows:
                    skipped_no_opposite_third_person += 1
                    continue

                cf_path = args.counterfactual_root / core_dimension / f"{file_stem}.json"
                dump_json(cf_path, cf_rows, args.overwrite)
                counterfactual_files_written += 1
                counterfactual_items_written += len(cf_rows)
                per_dimension_files[core_dimension] += 1
                per_dimension_items[core_dimension] += len(cf_rows)
        else:
            for spec in SPECS:
                original_score = record.appraisal_scores.get(spec.dimension)
                if original_score is None:
                    skipped_missing_dimension += 1
                    continue
                cf_scores = opposite_scores(
                    original_score,
                    policy=args.opposite_policy,
                    neutral_policy=args.neutral_policy,
                )
                if not cf_scores:
                    skipped_neutral += 1
                    continue

                cf_path = args.counterfactual_root / spec.dimension / f"{file_stem}.json"
                if args.counterfactual_source == "third-person":
                    third_person_records = third_person_by_sample.get(record.sample_id, [])
                    if not third_person_records:
                        skipped_missing_third_person_sample += 1
                        continue
                    cf_score_set = set(cf_scores)
                    cf_rows = []
                    for third_person in third_person_records:
                        third_person_score = third_person.appraisal_scores.get(spec.dimension)
                        if third_person_score in cf_score_set:
                            cf_rows.append(
                                third_person_counterfactual_payload(
                                    record,
                                    third_person,
                                    spec,
                                    third_person_score,
                                )
                            )
                    if not cf_rows:
                        skipped_no_opposite_third_person += 1
                        continue
                else:
                    cf_rows = [
                        counterfactual_payload(record, spec, cf_score)
                        for cf_score in cf_scores
                    ]
                dump_json(cf_path, cf_rows, args.overwrite)
                counterfactual_files_written += 1
                counterfactual_items_written += len(cf_rows)
                per_dimension_files[spec.dimension] += 1
                per_dimension_items[spec.dimension] += len(cf_rows)

    manifest = {
        "protocol": "carebench-appraisal-prefix-intervention-v1",
        "input": str(args.input),
        "third_person_input": str(third_person_input) if third_person_input else None,
        "prompt_config": str(args.prompt_config),
        "first_person_root": str(args.first_person_root),
        "counterfactual_root": str(args.counterfactual_root),
        "counterfactual_source": args.counterfactual_source,
        "replacement_granularity": args.replacement_granularity,
        "records_loaded": len(records),
        "third_person_samples_loaded": len(third_person_by_sample),
        "third_person_records_loaded": sum(
            len(items) for items in third_person_by_sample.values()
        ),
        "first_person_files": first_person_written,
        "counterfactual_files": counterfactual_files_written,
        "counterfactual_items": counterfactual_items_written,
        "skipped_neutral_dimensions": skipped_neutral,
        "skipped_missing_dimensions": skipped_missing_dimension,
        "skipped_missing_third_person_sample_dimensions": skipped_missing_third_person_sample,
        "skipped_no_opposite_third_person_dimensions": skipped_no_opposite_third_person,
        "compact_scale": args.compact_scale,
        "opposite_policy": args.opposite_policy,
        "min_opposite_count": args.min_opposite_count,
        "min_opposite_fraction": args.min_opposite_fraction,
        "neutral_policy": args.neutral_policy,
        "per_dimension_files": per_dimension_files,
        "per_dimension_items": per_dimension_items,
    }

    if args.manifest is not None:
        dump_json(args.manifest, manifest, args.overwrite)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
