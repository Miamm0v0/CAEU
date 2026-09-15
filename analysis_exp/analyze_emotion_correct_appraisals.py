#!/usr/bin/env python3
"""Analyze CAREBench appraisals conditional on strong emotion predictions.

The primary condition is sample-level joint label F1 at or above a configurable
threshold. The script then evaluates either the same samples' 22 appraisal
ratings or their five free-text core appraisals. It reads per-sample task
outputs, not an aggregate ``results.json`` file.

Example:
  python analysis_exp/analyze_emotion_correct_appraisals.py \
    --gold FirstPersonMethod/data/first_person \
    --pred_root output/first_person/my_run \
    --prompt_path FirstPersonMethod/scripts/prompts/baseline_prompt.toml
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    REPO_ROOT / "FirstPersonMethod" / "scripts" / "prompts" / "baseline_prompt.toml"
)
EMOTION_TASKS = ("positive-labels", "negative-labels")
RATING_TASK = "appraisals"
REASONING_TASK = "core-appraisals"
REQUIRED_TASKS = (RATING_TASK, *EMOTION_TASKS)
CORE_APPRAISAL_DIMENSIONS = (
    "relevance",
    "congruence",
    "accountability",
    "control",
    "certainty",
)
LEXICAL_REASONING_METRICS = ("bleu", "rouge_1", "rouge_2", "rouge_l")
REASONING_METRICS = (*LEXICAL_REASONING_METRICS, "bertscore")
DEFAULT_EMOTION_F1_THRESHOLD = 0.8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure appraisal quality on CAREBench samples whose emotion "
            "labels were predicted correctly."
        )
    )
    parser.add_argument(
        "--gold",
        "--gold_folder",
        dest="gold",
        type=Path,
        required=True,
        help="Per-sample gold folder or monolithic first_person.json.",
    )
    parser.add_argument(
        "--pred_root",
        "--pred_folder",
        dest="pred_root",
        type=Path,
        required=True,
        help=(
            "One prediction run containing appraisals/, positive-labels/, "
            "and negative-labels/."
        ),
    )
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=DEFAULT_PROMPT,
        help="CAREBench prompt TOML containing rating and label mappings.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to a target-specific analysis folder "
            "under pred_root."
        ),
    )
    parser.add_argument(
        "--appraisal_target",
        "--appraisal_type",
        choices=("ratings", "appraisal"),
        default="ratings",
        help=(
            "ratings evaluates the 22 ordinal appraisal ratings; appraisal "
            "evaluates the five free-text core appraisals with BLEU-4, "
            "ROUGE-1/2/L, and optional BERTScore F1."
        ),
    )
    parser.add_argument(
        "--emotion_f1_threshold",
        type=float,
        default=DEFAULT_EMOTION_F1_THRESHOLD,
        help=(
            "Keep samples whose joint positive/negative emotion-label F1 is "
            "at least this value. Default: 0.8."
        ),
    )
    parser.add_argument(
        "--min_appraisal_coverage",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of the configured appraisal dimensions required "
            "for a sample to enter the conditional analysis."
        ),
    )
    parser.add_argument(
        "--low_appraisal_accuracy_threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold used to count emotion-correct samples with low "
            "sample-level appraisal accuracy."
        ),
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="",
        help=(
            "Optional model id or local path used to add per-sample appraisal "
            "reasoning BERTScore F1. When omitted, lexical metrics are still "
            "computed and the BERTScore CSV column is left empty."
        ),
    )
    parser.add_argument(
        "--bertscore_num_layers",
        type=int,
        default=None,
        help=(
            "Optional explicit BERTScore layer count. This is normally needed "
            "when --bertscore_model is a local path, for example 17 for "
            "roberta-large."
        ),
    )
    parser.add_argument(
        "--bertscore_batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--bertscore_device",
        type=str,
        default="auto",
        help="BERTScore device: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
        help="Bootstrap resamples for 95%% confidence intervals; 0 disables.",
    )
    parser.add_argument(
        "--permutation_samples",
        type=int,
        default=5000,
        help="Permutations for the correct-vs-incorrect comparison; 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_prompt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {path}")
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt TOML must contain an object: {path}")
    return payload


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if path.is_file():
        payload = load_json(path)
        if isinstance(payload, dict):
            for sample_id, item in payload.items():
                if isinstance(item, dict):
                    records[str(sample_id)] = item
        elif isinstance(payload, list):
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                sample_id = item.get("id", item.get("sample_id"))
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"Gold list item {index} has no id/sample_id")
                records[sample_id.strip()] = item
        else:
            raise ValueError(f"Unsupported gold JSON structure: {path}")
    elif path.is_dir():
        for item_path in sorted(path.glob("*.json")):
            payload = load_json(item_path)
            if not isinstance(payload, dict):
                raise ValueError(f"Gold sample must be an object: {item_path}")
            records[item_path.stem] = payload
    else:
        raise FileNotFoundError(f"gold path not found: {path}")

    if not records:
        raise ValueError(f"No gold samples found in: {path}")
    return records


def required_tasks(appraisal_target: str) -> tuple[str, ...]:
    appraisal_task = RATING_TASK if appraisal_target == "ratings" else REASONING_TASK
    return (appraisal_task, *EMOTION_TASKS)


def has_task_folders(path: Path, tasks: Sequence[str]) -> bool:
    return path.is_dir() and all((path / task).is_dir() for task in tasks)


def resolve_prediction_root(path: Path, tasks: Sequence[str]) -> Path:
    if has_task_folders(path, tasks):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"pred_root not found: {path}")
    candidates = [
        child
        for child in sorted(path.iterdir())
        if has_task_folders(child, tasks)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No prediction run with {', '.join(tasks)} under: {path}. "
            "Aggregate results.json files are insufficient for this analysis."
        )
    raise ValueError(
        "Multiple prediction runs found; point --pred_root at one run: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def safe_mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def extract_situation(record: dict[str, Any]) -> str:
    direct = record.get("situation")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for section_name in ("story_collection", "cognitive_questions"):
        section = record.get(section_name)
        if not isinstance(section, dict):
            continue
        scenario = section.get("final_scenario")
        if isinstance(scenario, str) and scenario.strip():
            return scenario.strip()
    return ""


def label_set(value: Any, allowed: set[str]) -> set[str] | None:
    if not isinstance(value, list):
        return None
    return {
        label.strip()
        for label in value
        if isinstance(label, str) and label.strip() in allowed
    }


def load_predicted_label_set(
    prediction_root: Path,
    task: str,
    sample_id: str,
    allowed: set[str],
) -> set[str] | None:
    path = prediction_root / task / f"{sample_id}.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    value = payload.get("labels")
    if value is None:
        value = payload.get(task.replace("-", "_"))
    return label_set(value, allowed)


def prediction_appraisals(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("appraisals")
    return nested if isinstance(nested, dict) else payload


def predicted_appraisal_score(entry: Any) -> float | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if 1.0 <= score <= 5.0 else None


def set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    true_positive = len(predicted & gold)
    precision = safe_div(true_positive, len(predicted))
    recall = safe_div(true_positive, len(gold))
    return safe_div(2.0 * precision * recall, precision + recall)


def is_opposite_side(predicted: float, gold: float) -> bool:
    return (predicted <= 2.0 and gold >= 4.0) or (
        predicted >= 4.0 and gold <= 2.0
    )


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def make_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [
        tuple(tokens[index : index + n])
        for index in range(len(tokens) - n + 1)
    ]


def bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    reference_tokens = tokenize_text(reference)
    prediction_tokens = tokenize_text(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        prediction_ngrams = make_ngrams(prediction_tokens, n)
        reference_ngrams = make_ngrams(reference_tokens, n)
        if not prediction_ngrams:
            return 0.0
        prediction_counts = collections.Counter(prediction_ngrams)
        reference_counts = collections.Counter(reference_ngrams)
        overlap = sum(
            min(count, reference_counts[gram])
            for gram, count in prediction_counts.items()
        )
        precision = safe_div(overlap, sum(prediction_counts.values()))
        if precision <= 0.0:
            return 0.0
        precisions.append(precision)

    brevity_penalty = (
        1.0
        if len(prediction_tokens) > len(reference_tokens)
        else math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_n
    )


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    reference_ngrams = make_ngrams(tokenize_text(reference), n)
    prediction_ngrams = make_ngrams(tokenize_text(prediction), n)
    if not reference_ngrams or not prediction_ngrams:
        return 0.0
    reference_counts = collections.Counter(reference_ngrams)
    prediction_counts = collections.Counter(prediction_ngrams)
    overlap = sum(
        min(count, prediction_counts[gram])
        for gram, count in reference_counts.items()
    )
    precision = safe_div(overlap, sum(prediction_counts.values()))
    recall = safe_div(overlap, sum(reference_counts.values()))
    return safe_div(2.0 * precision * recall, precision + recall)


def lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    if not first or not second:
        return 0
    previous = [0] * (len(second) + 1)
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = tokenize_text(reference)
    prediction_tokens = tokenize_text(prediction)
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = lcs_length(reference_tokens, prediction_tokens)
    precision = safe_div(overlap, len(prediction_tokens))
    recall = safe_div(overlap, len(reference_tokens))
    return safe_div(2.0 * precision * recall, precision + recall)


def lexical_reasoning_scores(
    reference: str, prediction: str
) -> dict[str, float]:
    return {
        "bleu": bleu_score(reference, prediction),
        "rouge_1": rouge_n_f1(reference, prediction, 1),
        "rouge_2": rouge_n_f1(reference, prediction, 2),
        "rouge_l": rouge_l_f1(reference, prediction),
    }


def add_reasoning_bertscore(
    rows: Sequence[dict[str, Any]],
    *,
    model_type: str,
    num_layers: int | None,
    batch_size: int,
    device: str,
) -> None:
    try:
        from bert_score import BERTScorer
    except ImportError as exc:
        raise RuntimeError(
            "BERTScore requested but bert-score is not installed"
        ) from exc

    resolved_device = device
    if device == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"

    scorer_kwargs: dict[str, Any] = {
        "model_type": model_type,
        "lang": "en",
        "device": resolved_device,
    }
    if num_layers is not None:
        scorer_kwargs["num_layers"] = num_layers
    scorer = BERTScorer(**scorer_kwargs)

    pending = [
        observation
        for row in rows
        for observation in row.get("_reasoning_observations", [])
    ]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        candidates = [item["predicted_text"] for item in batch]
        references = [item["gold_text"] for item in batch]
        _, _, f1 = scorer.score(
            candidates,
            references,
            verbose=False,
            batch_size=batch_size,
        )
        for observation, score in zip(batch, f1):
            observation["bertscore"] = float(score)


def add_sample_reasoning_metrics(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        observations = row.get("_reasoning_observations")
        if not isinstance(observations, list):
            continue
        for metric in REASONING_METRICS:
            values = [
                float(observation[metric])
                for observation in observations
                if observation.get(metric) is not None
            ]
            row[f"appraisal_reasoning_{metric}"] = safe_mean(values)


def gold_core_appraisal(record: dict[str, Any], dimension: str) -> str | None:
    cognitive_questions = record.get("cognitive_questions")
    if not isinstance(cognitive_questions, dict):
        return None
    summary_answers = cognitive_questions.get("summary_answers")
    if not isinstance(summary_answers, dict):
        return None
    value = summary_answers.get(dimension)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def predicted_core_appraisal(payload: Any, dimension: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(dimension)
    if isinstance(value, dict):
        value = value.get("answer")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def joint_emotion_label_f1(
    predicted_positive: set[str],
    gold_positive: set[str],
    predicted_negative: set[str],
    gold_negative: set[str],
) -> float:
    predicted = {f"positive:{label}" for label in predicted_positive} | {
        f"negative:{label}" for label in predicted_negative
    }
    gold = {f"positive:{label}" for label in gold_positive} | {
        f"negative:{label}" for label in gold_negative
    }
    return set_f1(predicted, gold)


def analyze_samples(
    *,
    gold_records: dict[str, dict[str, Any]],
    prediction_root: Path,
    dimension_to_statement: dict[str, str],
    appraisal_label_map: dict[str, float],
    positive_labels: set[str],
    negative_labels: set[str],
    emotion_f1_threshold: float,
    min_appraisal_coverage: float,
    low_accuracy_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    coverage = {
        "gold_samples": len(gold_records),
        "complete_samples": 0,
        "excluded_missing_emotion_prediction": 0,
        "excluded_invalid_emotion_gold": 0,
        "excluded_insufficient_appraisal_coverage": 0,
        "missing_appraisal_prediction_file": 0,
    }
    required_dimensions = max(
        1, math.ceil(len(dimension_to_statement) * min_appraisal_coverage)
    )

    for sample_id, gold in sorted(gold_records.items()):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "situation": extract_situation(gold),
            "included": False,
            "exclusion_reason": "",
        }
        emotion_labels = gold.get("emotion_labels")
        if not isinstance(emotion_labels, dict):
            coverage["excluded_invalid_emotion_gold"] += 1
            row["exclusion_reason"] = "invalid_emotion_gold"
            rows.append(row)
            continue

        gold_positive = label_set(
            emotion_labels.get("positive_emotion_labels"), positive_labels
        )
        gold_negative = label_set(
            emotion_labels.get("negative_emotion_labels"), negative_labels
        )
        if gold_positive is None or gold_negative is None:
            coverage["excluded_invalid_emotion_gold"] += 1
            row["exclusion_reason"] = "invalid_emotion_gold"
            rows.append(row)
            continue

        predicted_positive = load_predicted_label_set(
            prediction_root, "positive-labels", sample_id, positive_labels
        )
        predicted_negative = load_predicted_label_set(
            prediction_root, "negative-labels", sample_id, negative_labels
        )
        if predicted_positive is None or predicted_negative is None:
            coverage["excluded_missing_emotion_prediction"] += 1
            row["exclusion_reason"] = "missing_or_invalid_emotion_prediction"
            rows.append(row)
            continue

        appraisals_path = prediction_root / "appraisals" / f"{sample_id}.json"
        if appraisals_path.is_file():
            predicted_ratings = prediction_appraisals(load_json(appraisals_path))
        else:
            coverage["missing_appraisal_prediction_file"] += 1
            predicted_ratings = None
        gold_ratings = gold.get("appraisal_ratings")
        observations: list[dict[str, Any]] = []
        if isinstance(gold_ratings, dict) and isinstance(predicted_ratings, dict):
            for dimension, statement in dimension_to_statement.items():
                gold_label = gold_ratings.get(statement)
                if not isinstance(gold_label, str) or gold_label not in appraisal_label_map:
                    continue
                predicted_score = predicted_appraisal_score(
                    predicted_ratings.get(dimension)
                )
                if predicted_score is None:
                    continue
                gold_score = float(appraisal_label_map[gold_label])
                error = predicted_score - gold_score
                observations.append(
                    {
                        "dimension": dimension,
                        "core_dimension": dimension.split(".", 1)[0],
                        "gold_score": gold_score,
                        "predicted_score": predicted_score,
                        "exact": error == 0.0,
                        "absolute_error": abs(error),
                        "squared_error": error * error,
                        "within_one": abs(error) <= 1.0,
                        "opposite_side": is_opposite_side(
                            predicted_score, gold_score
                        ),
                    }
                )

        appraisal_coverage = safe_div(
            len(observations), len(dimension_to_statement)
        )
        if len(observations) < required_dimensions:
            coverage["excluded_insufficient_appraisal_coverage"] += 1
            row.update(
                {
                    "exclusion_reason": "insufficient_appraisal_coverage",
                    "appraisal_dimensions_evaluated": len(observations),
                    "appraisal_coverage": appraisal_coverage,
                }
            )
            rows.append(row)
            continue

        positive_exact = predicted_positive == gold_positive
        negative_exact = predicted_negative == gold_negative
        exact_count = sum(int(item["exact"]) for item in observations)
        opposite_count = sum(int(item["opposite_side"]) for item in observations)
        appraisal_accuracy = safe_div(exact_count, len(observations))
        absolute_errors = [float(item["absolute_error"]) for item in observations]
        squared_errors = [float(item["squared_error"]) for item in observations]
        emotion_label_f1 = joint_emotion_label_f1(
            predicted_positive,
            gold_positive,
            predicted_negative,
            gold_negative,
        )
        correct = emotion_label_f1 >= emotion_f1_threshold

        row.update(
            {
                "included": True,
                "positive_labels_exact": positive_exact,
                "negative_labels_exact": negative_exact,
                "joint_labels_exact": positive_exact and negative_exact,
                "emotion_correct": correct,
                "emotion_label_f1": emotion_label_f1,
                "positive_labels_f1": set_f1(predicted_positive, gold_positive),
                "negative_labels_f1": set_f1(predicted_negative, gold_negative),
                "gold_positive_labels": sorted(gold_positive),
                "predicted_positive_labels": sorted(predicted_positive),
                "gold_negative_labels": sorted(gold_negative),
                "predicted_negative_labels": sorted(predicted_negative),
                "appraisal_dimensions_evaluated": len(observations),
                "appraisal_coverage": appraisal_coverage,
                "appraisal_exact_count": exact_count,
                "appraisal_accuracy": appraisal_accuracy,
                "appraisal_mae": safe_mean(absolute_errors),
                "appraisal_normalized_mae": (
                    safe_mean(absolute_errors) / 4.0 if absolute_errors else None
                ),
                "appraisal_normalized_rmse": (
                    math.sqrt(sum(squared_errors) / len(squared_errors)) / 4.0
                    if squared_errors
                    else None
                ),
                "appraisal_within_one_accuracy": safe_div(
                    sum(int(item["within_one"]) for item in observations),
                    len(observations),
                ),
                "appraisal_opposite_side_errors": opposite_count,
                "all_appraisals_exact": exact_count == len(observations),
                "any_appraisal_error": exact_count != len(observations),
                "has_opposite_side_appraisal_error": opposite_count > 0,
                "low_appraisal_accuracy": (
                    appraisal_accuracy < low_accuracy_threshold
                ),
                "appraisal_error_dimensions": [
                    item["dimension"] for item in observations if not item["exact"]
                ],
                "opposite_side_error_dimensions": [
                    item["dimension"]
                    for item in observations
                    if item["opposite_side"]
                ],
                "_appraisal_observations": observations,
            }
        )
        coverage["complete_samples"] += 1
        rows.append(row)

    return rows, coverage


def analyze_reasoning_samples(
    *,
    gold_records: dict[str, dict[str, Any]],
    prediction_root: Path,
    positive_labels: set[str],
    negative_labels: set[str],
    emotion_f1_threshold: float,
    min_appraisal_coverage: float,
    low_score_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    coverage = {
        "gold_samples": len(gold_records),
        "complete_samples": 0,
        "excluded_missing_emotion_prediction": 0,
        "excluded_invalid_emotion_gold": 0,
        "excluded_insufficient_appraisal_coverage": 0,
        "missing_core_appraisal_prediction_file": 0,
    }
    required_dimensions = max(
        1,
        math.ceil(len(CORE_APPRAISAL_DIMENSIONS) * min_appraisal_coverage),
    )

    for sample_id, gold in sorted(gold_records.items()):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "situation": extract_situation(gold),
            "included": False,
            "exclusion_reason": "",
        }
        emotion_labels = gold.get("emotion_labels")
        if not isinstance(emotion_labels, dict):
            coverage["excluded_invalid_emotion_gold"] += 1
            row["exclusion_reason"] = "invalid_emotion_gold"
            rows.append(row)
            continue

        gold_positive = label_set(
            emotion_labels.get("positive_emotion_labels"), positive_labels
        )
        gold_negative = label_set(
            emotion_labels.get("negative_emotion_labels"), negative_labels
        )
        if gold_positive is None or gold_negative is None:
            coverage["excluded_invalid_emotion_gold"] += 1
            row["exclusion_reason"] = "invalid_emotion_gold"
            rows.append(row)
            continue

        predicted_positive = load_predicted_label_set(
            prediction_root, "positive-labels", sample_id, positive_labels
        )
        predicted_negative = load_predicted_label_set(
            prediction_root, "negative-labels", sample_id, negative_labels
        )
        if predicted_positive is None or predicted_negative is None:
            coverage["excluded_missing_emotion_prediction"] += 1
            row["exclusion_reason"] = "missing_or_invalid_emotion_prediction"
            rows.append(row)
            continue

        prediction_path = (
            prediction_root / REASONING_TASK / f"{sample_id}.json"
        )
        prediction_payload = None
        if prediction_path.is_file():
            prediction_payload = load_json(prediction_path)
        else:
            coverage["missing_core_appraisal_prediction_file"] += 1

        observations: list[dict[str, Any]] = []
        for dimension in CORE_APPRAISAL_DIMENSIONS:
            gold_text = gold_core_appraisal(gold, dimension)
            predicted_text = predicted_core_appraisal(
                prediction_payload, dimension
            )
            if gold_text is None or predicted_text is None:
                continue
            observations.append(
                {
                    "dimension": dimension,
                    "gold_text": gold_text,
                    "predicted_text": predicted_text,
                    **lexical_reasoning_scores(gold_text, predicted_text),
                }
            )

        appraisal_coverage = safe_div(
            len(observations), len(CORE_APPRAISAL_DIMENSIONS)
        )
        if len(observations) < required_dimensions:
            coverage["excluded_insufficient_appraisal_coverage"] += 1
            row.update(
                {
                    "exclusion_reason": "insufficient_appraisal_coverage",
                    "appraisal_dimensions_evaluated": len(observations),
                    "appraisal_coverage": appraisal_coverage,
                }
            )
            rows.append(row)
            continue

        positive_exact = predicted_positive == gold_positive
        negative_exact = predicted_negative == gold_negative
        scores = [float(item["rouge_l"]) for item in observations]
        mean_score = float(safe_mean(scores) or 0.0)
        emotion_label_f1 = joint_emotion_label_f1(
            predicted_positive,
            gold_positive,
            predicted_negative,
            gold_negative,
        )
        correct = emotion_label_f1 >= emotion_f1_threshold
        row.update(
            {
                "included": True,
                "positive_labels_exact": positive_exact,
                "negative_labels_exact": negative_exact,
                "joint_labels_exact": positive_exact and negative_exact,
                "emotion_correct": correct,
                "emotion_label_f1": emotion_label_f1,
                "positive_labels_f1": set_f1(predicted_positive, gold_positive),
                "negative_labels_f1": set_f1(predicted_negative, gold_negative),
                "gold_positive_labels": sorted(gold_positive),
                "predicted_positive_labels": sorted(predicted_positive),
                "gold_negative_labels": sorted(gold_negative),
                "predicted_negative_labels": sorted(predicted_negative),
                "appraisal_dimensions_evaluated": len(observations),
                "appraisal_coverage": appraisal_coverage,
                "low_appraisal_score": mean_score < low_score_threshold,
                "_reasoning_observations": observations,
            }
        )
        for metric in LEXICAL_REASONING_METRICS:
            row[f"appraisal_reasoning_{metric}"] = safe_mean(
                [float(item[metric]) for item in observations]
            )
        coverage["complete_samples"] += 1
        rows.append(row)

    return rows, coverage


def summarize_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    observations = [
        observation
        for row in rows
        for observation in row["_appraisal_observations"]
    ]
    accuracies = [float(row["appraisal_accuracy"]) for row in rows]
    absolute_errors = [float(item["absolute_error"]) for item in observations]
    squared_errors = [float(item["squared_error"]) for item in observations]
    return {
        "samples": len(rows),
        "appraisal_pairs": len(observations),
        "appraisal_exact_accuracy": safe_div(
            sum(int(item["exact"]) for item in observations), len(observations)
        ),
        "mean_sample_appraisal_accuracy": safe_mean(accuracies),
        "median_sample_appraisal_accuracy": (
            statistics.median(accuracies) if accuracies else None
        ),
        "appraisal_mae": safe_mean(absolute_errors),
        "appraisal_normalized_mae": (
            safe_mean(absolute_errors) / 4.0 if absolute_errors else None
        ),
        "appraisal_normalized_rmse": (
            math.sqrt(sum(squared_errors) / len(squared_errors)) / 4.0
            if squared_errors
            else None
        ),
        "appraisal_within_one_accuracy": safe_div(
            sum(int(item["within_one"]) for item in observations),
            len(observations),
        ),
        "opposite_side_appraisal_pair_rate": safe_div(
            sum(int(item["opposite_side"]) for item in observations),
            len(observations),
        ),
        "all_appraisals_exact_sample_rate": safe_div(
            sum(int(row["all_appraisals_exact"]) for row in rows), len(rows)
        ),
        "any_appraisal_error_sample_rate": safe_div(
            sum(int(row["any_appraisal_error"]) for row in rows), len(rows)
        ),
        "opposite_side_error_sample_rate": safe_div(
            sum(int(row["has_opposite_side_appraisal_error"]) for row in rows),
            len(rows),
        ),
        "low_appraisal_accuracy_sample_rate": safe_div(
            sum(int(row["low_appraisal_accuracy"]) for row in rows), len(rows)
        ),
    }


def summarize_reasoning_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    observations = [
        observation
        for row in rows
        for observation in row["_reasoning_observations"]
    ]
    output: dict[str, Any] = {
        "samples": len(rows),
        "appraisal_pairs": len(observations),
        "low_appraisal_score_sample_rate": safe_div(
            sum(int(row["low_appraisal_score"]) for row in rows), len(rows)
        ),
    }
    for metric in REASONING_METRICS:
        pair_scores = [
            float(item[metric])
            for item in observations
            if item.get(metric) is not None
        ]
        sample_field = f"appraisal_reasoning_{metric}"
        sample_scores = [
            float(row[sample_field])
            for row in rows
            if row.get(sample_field) is not None
        ]
        output[sample_field] = safe_mean(pair_scores)
        output[f"mean_sample_{sample_field}"] = safe_mean(sample_scores)
        output[f"median_sample_{sample_field}"] = (
            statistics.median(sample_scores) if sample_scores else None
        )
    return output


def observation_summary(observations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    absolute_errors = [float(item["absolute_error"]) for item in observations]
    return {
        "pairs": len(observations),
        "accuracy": safe_div(
            sum(int(item["exact"]) for item in observations), len(observations)
        ),
        "mae": safe_mean(absolute_errors),
        "normalized_mae": (
            safe_mean(absolute_errors) / 4.0 if absolute_errors else None
        ),
        "within_one_accuracy": safe_div(
            sum(int(item["within_one"]) for item in observations),
            len(observations),
        ),
        "opposite_side_error_rate": safe_div(
            sum(int(item["opposite_side"]) for item in observations),
            len(observations),
        ),
    }


def breakdown(
    rows: Sequence[dict[str, Any]],
    dimensions: Iterable[str],
    key: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in dimensions:
        matching = [
            observation
            for row in rows
            for observation in row["_appraisal_observations"]
            if observation[key] == value
        ]
        output[value] = observation_summary(matching)
    return output


def reasoning_breakdown(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for dimension in CORE_APPRAISAL_DIMENSIONS:
        matching = [
            observation
            for row in rows
            for observation in row["_reasoning_observations"]
            if observation["dimension"] == dimension
        ]
        summary: dict[str, Any] = {"pairs": len(matching)}
        for metric in REASONING_METRICS:
            scores = [
                float(observation[metric])
                for observation in matching
                if observation.get(metric) is not None
            ]
            output_name = (
                "bertscore_f1"
                if metric == "bertscore"
                else f"{metric}_f1"
                if metric.startswith("rouge_")
                else metric
            )
            summary[output_name] = safe_mean(scores)
        output[dimension] = summary
    return output


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int,
    rng: random.Random,
) -> list[float | None]:
    if not values or samples <= 0:
        return [None, None]
    estimates = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def bootstrap_difference_ci(
    first: Sequence[float],
    second: Sequence[float],
    samples: int,
    rng: random.Random,
) -> list[float | None]:
    if not first or not second or samples <= 0:
        return [None, None]
    estimates = []
    for _ in range(samples):
        first_mean = sum(rng.choice(first) for _ in first) / len(first)
        second_mean = sum(rng.choice(second) for _ in second) / len(second)
        estimates.append(first_mean - second_mean)
    estimates.sort()
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def permutation_p_value(
    first: Sequence[float],
    second: Sequence[float],
    samples: int,
    rng: random.Random,
) -> float | None:
    if not first or not second or samples <= 0:
        return None
    observed = abs(sum(first) / len(first) - sum(second) / len(second))
    combined = list(first) + list(second)
    extreme = 0
    for _ in range(samples):
        shuffled = list(combined)
        rng.shuffle(shuffled)
        difference = abs(
            sum(shuffled[: len(first)]) / len(first)
            - sum(shuffled[len(first) :]) / len(second)
        )
        if difference + 1e-12 >= observed:
            extreme += 1
    return (extreme + 1.0) / (samples + 1.0)


def pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    mean_x = sum(values_x) / len(values_x)
    mean_y = sum(values_y) / len(values_y)
    centered_x = [value - mean_x for value in values_x]
    centered_y = [value - mean_y for value in values_y]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_reasoning_analysis(
    *,
    args: argparse.Namespace,
    gold_records: dict[str, dict[str, Any]],
    prediction_root: Path,
    output_dir: Path,
    positive_labels: set[str],
    negative_labels: set[str],
) -> int:
    all_rows, coverage = analyze_reasoning_samples(
        gold_records=gold_records,
        prediction_root=prediction_root,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        emotion_f1_threshold=args.emotion_f1_threshold,
        min_appraisal_coverage=args.min_appraisal_coverage,
        low_score_threshold=args.low_appraisal_accuracy_threshold,
    )
    if args.bertscore_model:
        add_reasoning_bertscore(
            all_rows,
            model_type=args.bertscore_model,
            num_layers=args.bertscore_num_layers,
            batch_size=args.bertscore_batch_size,
            device=args.bertscore_device,
        )
    add_sample_reasoning_metrics(all_rows)
    included = [row for row in all_rows if row["included"]]
    emotion_correct = [row for row in included if row["emotion_correct"]]
    emotion_incorrect = [row for row in included if not row["emotion_correct"]]
    correct_scores = [
        float(row["appraisal_reasoning_rouge_l"]) for row in emotion_correct
    ]
    incorrect_scores = [
        float(row["appraisal_reasoning_rouge_l"])
        for row in emotion_incorrect
    ]
    correct_mean = safe_mean(correct_scores)
    incorrect_mean = safe_mean(incorrect_scores)
    difference = (
        correct_mean - incorrect_mean
        if correct_mean is not None and incorrect_mean is not None
        else None
    )
    rng = random.Random(args.seed)
    comparison = {
        "metric": "sample_mean_core_appraisal_rouge_l_f1",
        "emotion_correct_mean": correct_mean,
        "emotion_correct_bootstrap_95_ci": bootstrap_mean_ci(
            correct_scores, args.bootstrap_samples, rng
        ),
        "emotion_incorrect_mean": incorrect_mean,
        "emotion_incorrect_bootstrap_95_ci": bootstrap_mean_ci(
            incorrect_scores, args.bootstrap_samples, rng
        ),
        "correct_minus_incorrect": difference,
        "difference_bootstrap_95_ci": bootstrap_difference_ci(
            correct_scores, incorrect_scores, args.bootstrap_samples, rng
        ),
        "two_sided_permutation_p": permutation_p_value(
            correct_scores,
            incorrect_scores,
            args.permutation_samples,
            rng,
        ),
        "point_biserial_correlation": pearson(
            [1.0 if row["emotion_correct"] else 0.0 for row in included],
            [float(row["appraisal_reasoning_rouge_l"]) for row in included],
        ),
    }

    correct_group = summarize_reasoning_group(emotion_correct)
    dimension_all = reasoning_breakdown(included)
    dimension_correct = reasoning_breakdown(emotion_correct)
    dimension_incorrect = reasoning_breakdown(emotion_incorrect)
    headline = {
        "emotion_correct_samples": len(emotion_correct),
        "emotion_correct_rate": safe_div(len(emotion_correct), len(included)),
        "emotion_correct_but_low_appraisal_score_samples": sum(
            int(row["low_appraisal_score"]) for row in emotion_correct
        ),
        "emotion_correct_but_low_appraisal_score_rate": correct_group[
            "low_appraisal_score_sample_rate"
        ],
    }
    for metric in REASONING_METRICS:
        field = f"appraisal_reasoning_{metric}"
        headline[f"{field}_given_emotion_correct"] = correct_group[field]
    summary = {
        "meta": {
            "gold": str(args.gold),
            "prediction_root": str(prediction_root),
            "prompt_path": str(args.prompt_path),
            "appraisal_target": "appraisal",
            "appraisal_representation": "five-dimensional free-text reasoning",
            "appraisal_metrics": [
                "BLEU-4",
                "ROUGE-1 F1",
                "ROUGE-2 F1",
                "ROUGE-L F1",
                *(["BERTScore F1"] if args.bertscore_model else []),
            ],
            "appraisal_metric": "ROUGE-L F1",
            "primary_appraisal_metric": "ROUGE-L F1",
            "bertscore_model": args.bertscore_model or None,
            "bertscore_num_layers": (
                args.bertscore_num_layers if args.bertscore_model else None
            ),
            "bertscore_device": (
                args.bertscore_device if args.bertscore_model else None
            ),
            "emotion_selection_metric": "sample-level joint emotion-label F1",
            "emotion_f1_threshold": args.emotion_f1_threshold,
            "appraisal_dimensions": len(CORE_APPRAISAL_DIMENSIONS),
            "min_appraisal_coverage": args.min_appraisal_coverage,
            "low_appraisal_score_threshold": (
                args.low_appraisal_accuracy_threshold
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
        },
        "coverage": coverage,
        "headline": headline,
        "groups": {
            "all_complete": summarize_reasoning_group(included),
            "emotion_correct": correct_group,
            "emotion_incorrect": summarize_reasoning_group(emotion_incorrect),
        },
        "correct_vs_incorrect": comparison,
        "dimension": {
            dimension: {
                "all_complete": dimension_all[dimension],
                "emotion_correct": dimension_correct[dimension],
                "emotion_incorrect": dimension_incorrect[dimension],
            }
            for dimension in CORE_APPRAISAL_DIMENSIONS
        },
        "interpretation_guardrail": (
            "Low reference-overlap on emotion-correct samples shows that correct "
            "emotion labels do not guarantee reference-aligned appraisal text. "
            "Automatic text-similarity metrics are not complete measures of "
            "psychological reasoning."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    public_rows = [
        {
            key: csv_value(value)
            for key, value in row.items()
            if not key.startswith("_")
        }
        for row in all_rows
    ]
    sample_fields = (
        "sample_id",
        "situation",
        "included",
        "exclusion_reason",
        "emotion_correct",
        "emotion_label_f1",
        "joint_labels_exact",
        "positive_labels_exact",
        "negative_labels_exact",
        "positive_labels_f1",
        "negative_labels_f1",
        "appraisal_dimensions_evaluated",
        "appraisal_coverage",
        "appraisal_reasoning_bleu",
        "appraisal_reasoning_rouge_1",
        "appraisal_reasoning_rouge_2",
        "appraisal_reasoning_rouge_l",
        "appraisal_reasoning_bertscore",
        "low_appraisal_score",
        "gold_positive_labels",
        "predicted_positive_labels",
        "gold_negative_labels",
        "predicted_negative_labels",
    )
    write_csv(output_dir / "samples.csv", sample_fields, public_rows)

    dimension_rows = []
    for dimension in CORE_APPRAISAL_DIMENSIONS:
        row: dict[str, Any] = {"dimension": dimension}
        for group_name, group_values in summary["dimension"][dimension].items():
            for metric, value in group_values.items():
                row[f"{group_name}_{metric}"] = value
        dimension_rows.append(row)
    write_csv(
        output_dir / "dimensions.csv",
        list(dimension_rows[0]),
        dimension_rows,
    )

    pair_rows = []
    for row in emotion_correct:
        for observation in row["_reasoning_observations"]:
            pair_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "situation": row["situation"],
                    "dimension": observation["dimension"],
                    "bleu": observation["bleu"],
                    "rouge_1_f1": observation["rouge_1"],
                    "rouge_2_f1": observation["rouge_2"],
                    "rouge_l_f1": observation["rouge_l"],
                    "bertscore_f1": observation.get("bertscore"),
                    "below_threshold": (
                        observation["rouge_l"]
                        < args.low_appraisal_accuracy_threshold
                    ),
                    "gold_appraisal": observation["gold_text"],
                    "predicted_appraisal": observation["predicted_text"],
                }
            )
    pair_rows.sort(key=lambda row: float(row["rouge_l_f1"]))
    pair_fields = (
        "sample_id",
        "situation",
        "dimension",
        "bleu",
        "rouge_1_f1",
        "rouge_2_f1",
        "rouge_l_f1",
        "bertscore_f1",
        "below_threshold",
        "gold_appraisal",
        "predicted_appraisal",
    )
    write_csv(
        output_dir / "emotion_correct_appraisal_reasoning.csv",
        pair_fields,
        pair_rows,
    )
    write_csv(
        output_dir / "headline.csv",
        ("metric", "value"),
        ({"metric": key, "value": value} for key, value in headline.items()),
    )
    score = headline["appraisal_reasoning_rouge_l_given_emotion_correct"]
    score_text = "n/a" if score is None else f"{float(score):.6f}"
    print(
        "[done] target=appraisal complete={} emotion_correct={} "
        "rouge_l={} output={}".format(
            len(included), len(emotion_correct), score_text, output_dir
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.min_appraisal_coverage <= 1.0:
        raise ValueError("--min_appraisal_coverage must be between 0 and 1")
    if not 0.0 <= args.low_appraisal_accuracy_threshold <= 1.0:
        raise ValueError(
            "--low_appraisal_accuracy_threshold must be between 0 and 1"
        )
    if not 0.0 <= args.emotion_f1_threshold <= 1.0:
        raise ValueError("--emotion_f1_threshold must be between 0 and 1")
    if args.bootstrap_samples < 0 or args.permutation_samples < 0:
        raise ValueError("Resample counts must be non-negative")
    if args.bertscore_batch_size <= 0:
        raise ValueError("--bertscore_batch_size must be positive")
    if args.bertscore_num_layers is not None and args.bertscore_num_layers <= 0:
        raise ValueError("--bertscore_num_layers must be positive")

    prediction_root = resolve_prediction_root(
        args.pred_root, required_tasks(args.appraisal_target)
    )
    output_dir = args.output_dir or (
        prediction_root
        / f"analysis_emotion_correct_appraisal_{args.appraisal_target}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args.prompt_path)
    raw_dimension_to_statement = prompt.get("appraisals", {}).get(
        "dimension_to_statement"
    )
    raw_appraisal_map = prompt.get("label_maps", {}).get("appraisals")
    raw_positive = prompt.get("label_options", {}).get("positive-labels", {})
    raw_negative = prompt.get("label_options", {}).get("negative-labels", {})
    if not isinstance(raw_dimension_to_statement, dict) or not raw_dimension_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt")
    if not isinstance(raw_appraisal_map, dict) or not raw_appraisal_map:
        raise ValueError("Missing [label_maps.appraisals] in prompt")
    positive_values = raw_positive.get("values") if isinstance(raw_positive, dict) else None
    negative_values = raw_negative.get("values") if isinstance(raw_negative, dict) else None
    if not isinstance(positive_values, list) or not isinstance(negative_values, list):
        raise ValueError("Missing CAREBench positive/negative label options in prompt")

    dimension_to_statement = {
        str(dimension): str(statement)
        for dimension, statement in raw_dimension_to_statement.items()
    }
    appraisal_label_map = {
        str(label): float(score) for label, score in raw_appraisal_map.items()
    }
    positive_labels = {str(label) for label in positive_values}
    negative_labels = {str(label) for label in negative_values}
    gold_records = load_gold(args.gold)

    if args.appraisal_target == "appraisal":
        return write_reasoning_analysis(
            args=args,
            gold_records=gold_records,
            prediction_root=prediction_root,
            output_dir=output_dir,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )

    all_rows, coverage = analyze_samples(
        gold_records=gold_records,
        prediction_root=prediction_root,
        dimension_to_statement=dimension_to_statement,
        appraisal_label_map=appraisal_label_map,
        positive_labels=positive_labels,
        negative_labels=negative_labels,
        emotion_f1_threshold=args.emotion_f1_threshold,
        min_appraisal_coverage=args.min_appraisal_coverage,
        low_accuracy_threshold=args.low_appraisal_accuracy_threshold,
    )
    included = [row for row in all_rows if row["included"]]
    emotion_correct = [row for row in included if row["emotion_correct"]]
    emotion_incorrect = [row for row in included if not row["emotion_correct"]]

    correct_accuracies = [
        float(row["appraisal_accuracy"]) for row in emotion_correct
    ]
    incorrect_accuracies = [
        float(row["appraisal_accuracy"]) for row in emotion_incorrect
    ]
    correct_mean = safe_mean(correct_accuracies)
    incorrect_mean = safe_mean(incorrect_accuracies)
    difference = (
        correct_mean - incorrect_mean
        if correct_mean is not None and incorrect_mean is not None
        else None
    )
    rng = random.Random(args.seed)
    comparison = {
        "metric": "sample_appraisal_accuracy",
        "emotion_correct_mean": correct_mean,
        "emotion_correct_bootstrap_95_ci": bootstrap_mean_ci(
            correct_accuracies, args.bootstrap_samples, rng
        ),
        "emotion_incorrect_mean": incorrect_mean,
        "emotion_incorrect_bootstrap_95_ci": bootstrap_mean_ci(
            incorrect_accuracies, args.bootstrap_samples, rng
        ),
        "correct_minus_incorrect": difference,
        "difference_bootstrap_95_ci": bootstrap_difference_ci(
            correct_accuracies, incorrect_accuracies, args.bootstrap_samples, rng
        ),
        "two_sided_permutation_p": permutation_p_value(
            correct_accuracies,
            incorrect_accuracies,
            args.permutation_samples,
            rng,
        ),
        "point_biserial_correlation": pearson(
            [1.0 if row["emotion_correct"] else 0.0 for row in included],
            [float(row["appraisal_accuracy"]) for row in included],
        ),
    }

    dimensions = list(dimension_to_statement)
    core_dimensions = list(dict.fromkeys(dim.split(".", 1)[0] for dim in dimensions))
    dimension_all = breakdown(included, dimensions, "dimension")
    dimension_correct = breakdown(emotion_correct, dimensions, "dimension")
    dimension_incorrect = breakdown(emotion_incorrect, dimensions, "dimension")
    core_all = breakdown(included, core_dimensions, "core_dimension")
    core_correct = breakdown(emotion_correct, core_dimensions, "core_dimension")
    core_incorrect = breakdown(emotion_incorrect, core_dimensions, "core_dimension")

    correct_group = summarize_group(emotion_correct)
    headline = {
        "emotion_correct_samples": len(emotion_correct),
        "emotion_correct_rate": safe_div(len(emotion_correct), len(included)),
        "appraisal_accuracy_given_emotion_correct": correct_group[
            "appraisal_exact_accuracy"
        ],
        "emotion_correct_but_any_appraisal_error_samples": sum(
            int(row["any_appraisal_error"]) for row in emotion_correct
        ),
        "emotion_correct_but_any_appraisal_error_rate": correct_group[
            "any_appraisal_error_sample_rate"
        ],
        "emotion_correct_but_low_appraisal_accuracy_samples": sum(
            int(row["low_appraisal_accuracy"]) for row in emotion_correct
        ),
        "emotion_correct_but_low_appraisal_accuracy_rate": correct_group[
            "low_appraisal_accuracy_sample_rate"
        ],
        "emotion_correct_with_opposite_side_appraisal_error_samples": sum(
            int(row["has_opposite_side_appraisal_error"])
            for row in emotion_correct
        ),
        "emotion_correct_with_opposite_side_appraisal_error_rate": correct_group[
            "opposite_side_error_sample_rate"
        ],
    }
    summary = {
        "meta": {
            "gold": str(args.gold),
            "prediction_root": str(prediction_root),
            "prompt_path": str(args.prompt_path),
            "appraisal_target": "ratings",
            "appraisal_metric": "exact 1-5 rating accuracy",
            "emotion_selection_metric": "sample-level joint emotion-label F1",
            "emotion_f1_threshold": args.emotion_f1_threshold,
            "appraisal_dimensions": len(dimensions),
            "min_appraisal_coverage": args.min_appraisal_coverage,
            "low_appraisal_accuracy_threshold": (
                args.low_appraisal_accuracy_threshold
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "seed": args.seed,
        },
        "coverage": coverage,
        "headline": headline,
        "groups": {
            "all_complete": summarize_group(included),
            "emotion_correct": correct_group,
            "emotion_incorrect": summarize_group(emotion_incorrect),
        },
        "correct_vs_incorrect": comparison,
        "dimension": {
            dimension: {
                "all_complete": dimension_all[dimension],
                "emotion_correct": dimension_correct[dimension],
                "emotion_incorrect": dimension_incorrect[dimension],
            }
            for dimension in dimensions
        },
        "core_dimension": {
            core: {
                "all_complete": core_all[core],
                "emotion_correct": core_correct[core],
                "emotion_incorrect": core_incorrect[core],
            }
            for core in core_dimensions
        },
        "interpretation_guardrail": (
            "Conditional appraisal errors are evidence that correct emotion-label "
            "prediction does not guarantee correct appraisal inference. They do "
            "not by themselves prove the absence of all latent appraisal reasoning."
        ),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    public_sample_rows = []
    for row in all_rows:
        public_row = {
            key: csv_value(value)
            for key, value in row.items()
            if not key.startswith("_")
        }
        public_sample_rows.append(public_row)
    sample_fields = [
        "sample_id",
        "situation",
        "included",
        "exclusion_reason",
        "emotion_correct",
        "emotion_label_f1",
        "joint_labels_exact",
        "positive_labels_exact",
        "negative_labels_exact",
        "positive_labels_f1",
        "negative_labels_f1",
        "appraisal_dimensions_evaluated",
        "appraisal_coverage",
        "appraisal_exact_count",
        "appraisal_accuracy",
        "appraisal_mae",
        "appraisal_normalized_mae",
        "appraisal_normalized_rmse",
        "appraisal_within_one_accuracy",
        "appraisal_opposite_side_errors",
        "all_appraisals_exact",
        "any_appraisal_error",
        "low_appraisal_accuracy",
        "has_opposite_side_appraisal_error",
        "appraisal_error_dimensions",
        "opposite_side_error_dimensions",
        "gold_positive_labels",
        "predicted_positive_labels",
        "gold_negative_labels",
        "predicted_negative_labels",
    ]
    write_csv(output_dir / "samples.csv", sample_fields, public_sample_rows)

    error_rows = []
    for row in emotion_correct:
        for observation in row["_appraisal_observations"]:
            if observation["exact"]:
                continue
            error_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "situation": row["situation"],
                    "dimension": observation["dimension"],
                    "core_dimension": observation["core_dimension"],
                    "statement": dimension_to_statement[
                        observation["dimension"]
                    ],
                    "gold_score": observation["gold_score"],
                    "predicted_score": observation["predicted_score"],
                    "absolute_error": observation["absolute_error"],
                    "opposite_side": observation["opposite_side"],
                    "gold_positive_labels": json.dumps(
                        row["gold_positive_labels"], ensure_ascii=False
                    ),
                    "predicted_positive_labels": json.dumps(
                        row["predicted_positive_labels"], ensure_ascii=False
                    ),
                    "gold_negative_labels": json.dumps(
                        row["gold_negative_labels"], ensure_ascii=False
                    ),
                    "predicted_negative_labels": json.dumps(
                        row["predicted_negative_labels"], ensure_ascii=False
                    ),
                }
            )
    error_fields = (
        "sample_id",
        "situation",
        "dimension",
        "core_dimension",
        "statement",
        "gold_score",
        "predicted_score",
        "absolute_error",
        "opposite_side",
        "gold_positive_labels",
        "predicted_positive_labels",
        "gold_negative_labels",
        "predicted_negative_labels",
    )
    write_csv(
        output_dir / "emotion_correct_appraisal_errors.csv",
        error_fields,
        error_rows,
    )

    dimension_rows = []
    for dimension in dimensions:
        row: dict[str, Any] = {
            "dimension": dimension,
            "core_dimension": dimension.split(".", 1)[0],
            "statement": dimension_to_statement[dimension],
        }
        for group_name, group_values in summary["dimension"][dimension].items():
            for metric, value in group_values.items():
                row[f"{group_name}_{metric}"] = value
        dimension_rows.append(row)
    dimension_fields = list(dimension_rows[0]) if dimension_rows else []
    write_csv(output_dir / "dimensions.csv", dimension_fields, dimension_rows)

    core_rows = []
    for core in core_dimensions:
        row = {"core_dimension": core}
        for group_name, group_values in summary["core_dimension"][core].items():
            for metric, value in group_values.items():
                row[f"{group_name}_{metric}"] = value
        core_rows.append(row)
    core_fields = list(core_rows[0]) if core_rows else []
    write_csv(output_dir / "core_dimensions.csv", core_fields, core_rows)

    write_csv(
        output_dir / "headline.csv",
        ("metric", "value"),
        ({"metric": key, "value": value} for key, value in headline.items()),
    )
    print(
        "[done] target=ratings complete={} emotion_correct={} "
        "appraisal_accuracy={:.6f} output={}".format(
            len(included),
            len(emotion_correct),
            float(headline["appraisal_accuracy_given_emotion_correct"]),
            output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
