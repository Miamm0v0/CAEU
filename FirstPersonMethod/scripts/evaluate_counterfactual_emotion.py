#!/usr/bin/env python3
"""
Usage:
  /home/ubuntu/anaconda3/bin/python scripts/evaluate_counterfactual_emotion.py \
    --first_person_root data/first_person \
    --counterfactual_gold_root data/counterfactual \
    --baseline_root output/first_person/baseline_with_cog_story/Qwen-3.5-9B \
    --counterfactual_pred_root output/first_person/counterfactual_emotion/Qwen-3.5-9B \
    --prompt_path scripts/prompts/counterfactual_emotion_prompt.toml \
    --output_file results.json

Notes:
- Correlations are computed per appraisal dimension.
- For level tasks:
    delta_model = mean(third_person_model_levels) - mean(first_person_model_levels)
    delta_human = mean(third_person_human_levels) - first_person_human_level
- For each emotion category label:
    delta_model = third_person_model_probability - first_person_model_probability
    delta_human = third_person_human_probability - first_person_human_probability
- For samples with multiple third-person entries, third-person level and category occurrence
  are averaged first, then used in sample-level correlation.
- Output is written to:
    <counterfactual_pred_root>/<output_file>
- Prefix-intervention runs may also contain a sibling ``first-person-appraisal``
  branch. When present, it is evaluated against the original first-person
  emotion gold and compared directly with the ``origin`` branch.
"""

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # for python < 3.11


LEVEL_TASKS = ["positive-level", "negative-level"]
LABEL_TASKS = ["positive-labels", "negative-labels"]
APPRAISAL_DIRECTIONS = ["low_to_high", "high_to_low", "mixed", "unknown"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        payload = tomllib.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level TOML must be object: {path}")
    return payload


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def score_side(value: Any) -> str:
    score = numeric_score(value)
    if score is None:
        return "unknown"
    if score < 3:
        return "low"
    if score > 3:
        return "high"
    return "neutral"


def score_direction(first_person_score: Any, counterfactual_score: Any) -> str:
    first_side = score_side(first_person_score)
    counterfactual_side = score_side(counterfactual_score)
    if first_side == "low" and counterfactual_side == "high":
        return "low_to_high"
    if first_side == "high" and counterfactual_side == "low":
        return "high_to_low"
    return "unknown"


def appraisal_direction(counterfactual_item: Dict[str, Any]) -> str:
    metadata = counterfactual_item.get("metadata", {})
    if not isinstance(metadata, dict):
        return "unknown"

    directions: Set[str] = set()
    opposite_items = metadata.get("opposite_items")
    if isinstance(opposite_items, list):
        for item in opposite_items:
            if not isinstance(item, dict):
                continue
            direction = score_direction(
                item.get("first_person_score"),
                item.get("third_person_score"),
            )
            if direction in {"low_to_high", "high_to_low"}:
                directions.add(direction)

    if not directions:
        direction = score_direction(
            metadata.get("original_score"),
            metadata.get("third_person_score", metadata.get("counterfactual_score")),
        )
        if direction in {"low_to_high", "high_to_low"}:
            directions.add(direction)

    if directions == {"low_to_high"}:
        return "low_to_high"
    if directions == {"high_to_low"}:
        return "high_to_low"
    if directions:
        return "mixed"
    return "unknown"


def parse_level_value(text: Any) -> Tuple[Optional[int], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    cleaned = text.strip()
    if not cleaned:
        return None, None

    match = re.match(r"^\s*(\d+)\s*-\s*(.+?)\s*$", cleaned)
    if match:
        return int(match.group(1)), match.group(2).strip()

    digits = re.findall(r"\d+", cleaned)
    score = int(digits[0]) if digits else None
    return score, cleaned


def rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (float(i + 1) + float(j + 1)) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[str]]:
    n = len(x)
    if n != len(y):
        return None, "Length mismatch"
    if n < 2:
        return None, "Need at least 2 samples"

    mean_x = safe_mean(x)
    mean_y = safe_mean(y)
    if mean_x is None or mean_y is None:
        return None, "Empty input"

    centered_x = [v - mean_x for v in x]
    centered_y = [v - mean_y for v in y]
    var_x = sum(v * v for v in centered_x)
    var_y = sum(v * v for v in centered_y)
    if var_x == 0.0 or var_y == 0.0:
        return None, "Zero variance"

    cov = sum(a * b for a, b in zip(centered_x, centered_y))
    denom = math.sqrt(var_x * var_y)
    if denom == 0.0:
        return None, "Zero denominator"
    return cov / denom, None


def spearman_corr(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[str]]:
    if len(x) != len(y):
        return None, "Length mismatch"
    if len(x) < 2:
        return None, "Need at least 2 samples"
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson_corr(rx, ry)


def run_sort_key(path: Path) -> int:
    match = re.search(r"run_(\d+)$", path.name)
    return int(match.group(1)) if match else 10**9


def resolve_baseline_run_roots(model_root: Path) -> List[Path]:
    direct_task = model_root / "positive-level"
    if direct_task.exists() and direct_task.is_dir():
        return [model_root]

    run_roots = sorted(
        [p for p in model_root.glob("run_*") if p.is_dir()],
        key=run_sort_key,
    )
    if run_roots:
        return run_roots

    raise FileNotFoundError(
        f"Cannot find baseline prediction folders under: {model_root}. "
        "Expected either direct task folders or run_*/task folders."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual emotion deltas")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model name stored in result metadata; paths no longer append it.",
    )
    parser.add_argument(
        "--first_person_root",
        type=str,
        default="data/first_person",
        help="First-person gold root",
    )
    parser.add_argument(
        "--counterfactual_gold_root",
        type=str,
        default="data/counterfactual",
        help="Counterfactual human gold root",
    )
    parser.add_argument(
        "--baseline_root",
        type=str,
        default="output/first_person/baseline_with_cog_story",
        help="Baseline prediction folder containing task folders or run_*/task folders.",
    )
    parser.add_argument(
        "--counterfactual_pred_root",
        type=str,
        default="output/first_person/counterfactual_emotion",
        help="Counterfactual emotion prediction folder containing task/dimension folders.",
    )
    parser.add_argument(
        "--first_person_appraisal_pred_root",
        type=str,
        default="",
        help=(
            "Optional first-person-appraisal prediction folder containing task "
            "folders. If omitted, a sibling of a third-person-appraisal root "
            "is detected automatically."
        ),
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/counterfactual_emotion_prompt.toml",
        help="Prompt TOML for emotion label options",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output filename under counterfactual_pred_root, or an absolute output path.",
    )
    parser.add_argument(
        "--dimension_source",
        choices=["gold", "prompt"],
        default="gold",
        help=(
            "gold evaluates the actual folders under counterfactual_gold_root "
            "(works for core-dimension folders such as relevance/control); "
            "prompt preserves the original behavior and iterates all prompt "
            "appraisal rating dimensions."
        ),
    )
    return parser


def resolve_first_person_appraisal_root(
    explicit_root: str,
    counterfactual_model_root: Path,
) -> Optional[Path]:
    stripped = explicit_root.strip()
    if stripped:
        root = Path(stripped)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(
                f"first_person_appraisal_pred_root not found: {root}"
            )
        return root

    candidates = [
        counterfactual_model_root.parent / "first-person-appraisal",
        counterfactual_model_root / "first-person-appraisal",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def resolve_output_path(counterfactual_root: Path, output_file: str) -> Path:
    stripped = output_file.strip()
    if not stripped:
        raise ValueError("output_file cannot be empty")
    output = Path(stripped)
    if output.is_absolute():
        if output.suffix == "" or (output.exists() and output.is_dir()):
            return output / "results.json"
        return output
    return counterfactual_root / output


def _load_first_person_human_level(first_person_payload: Dict[str, Any], task: str) -> Optional[float]:
    key = "positive_level" if task == "positive-level" else "negative_level"
    emotion_labels = first_person_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    value, _ = parse_level_value(emotion_labels.get(key))
    return float(value) if isinstance(value, int) else None


def _load_first_person_human_label_prob(
    first_person_payload: Dict[str, Any],
    task: str,
    label: str,
) -> Optional[float]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    emotion_labels = first_person_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    labels_raw = emotion_labels.get(key)
    if not isinstance(labels_raw, list):
        return None
    labels_set = {v for v in labels_raw if isinstance(v, str)}
    return 1.0 if label in labels_set else 0.0


def _load_first_person_human_labels(
    first_person_payload: Dict[str, Any],
    task: str,
    allowed_labels: List[str],
) -> Optional[Set[str]]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    emotion_labels = first_person_payload.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    labels_raw = emotion_labels.get(key)
    if not isinstance(labels_raw, list):
        return None
    allowed = set(allowed_labels)
    return {v for v in labels_raw if isinstance(v, str) and v in allowed}


def _load_first_person_model_level_from_runs(run_roots: List[Path], task: str, sample_name: str) -> Optional[float]:
    values: List[float] = []
    for run_root in run_roots:
        path = run_root / task / sample_name
        if not path.exists() or not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        score = numeric_score(payload.get("score"))
        if score is not None:
            values.append(score)
    return safe_mean(values)


def _load_first_person_model_label_prob_from_runs(
    run_roots: List[Path],
    task: str,
    sample_name: str,
    label: str,
) -> Optional[float]:
    probs: List[float] = []
    for run_root in run_roots:
        path = run_root / task / sample_name
        if not path.exists() or not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        labels_raw = payload.get("labels")
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _load_first_person_model_labels_from_runs(
    run_roots: List[Path],
    task: str,
    sample_name: str,
    allowed_labels: List[str],
) -> Optional[Set[str]]:
    allowed = set(allowed_labels)
    counts: collections.Counter[str] = collections.Counter()
    valid_runs = 0
    for run_root in run_roots:
        path = run_root / task / sample_name
        if not path.exists() or not path.is_file():
            continue
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        labels_raw = payload.get("labels")
        if not isinstance(labels_raw, list):
            continue
        valid_runs += 1
        for label in labels_raw:
            if isinstance(label, str) and label in allowed:
                counts[label] += 1
    if valid_runs == 0:
        return None
    return {label for label, count in counts.items() if count / valid_runs >= 0.5}


def _load_third_person_human_level(gold_counterfactual_payload: Any, task: str) -> Optional[float]:
    key = "positive_level" if task == "positive-level" else "negative_level"
    if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
        return None

    values: List[float] = []
    for item in gold_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        emotion_labels = item.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        score, _ = parse_level_value(emotion_labels.get(key))
        if isinstance(score, int):
            values.append(float(score))
    return safe_mean(values)


def _load_human_level_from_item(item: Dict[str, Any], task: str) -> Optional[float]:
    key = "positive_level" if task == "positive-level" else "negative_level"
    emotion_labels = item.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    score, _ = parse_level_value(emotion_labels.get(key))
    return float(score) if isinstance(score, int) else None


def _load_third_person_model_level(pred_counterfactual_payload: Any) -> Optional[float]:
    if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
        return None

    values: List[float] = []
    for item in pred_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        score = numeric_score(item.get("score"))
        if score is not None:
            values.append(score)
    return safe_mean(values)


def _load_model_level_from_item(item: Dict[str, Any]) -> Optional[float]:
    return numeric_score(item.get("score"))


def _load_third_person_human_label_prob(
    gold_counterfactual_payload: Any,
    task: str,
    label: str,
) -> Optional[float]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
        return None

    probs: List[float] = []
    for item in gold_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        emotion_labels = item.get("emotion_labels", {})
        if not isinstance(emotion_labels, dict):
            continue
        labels_raw = emotion_labels.get(key)
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _load_human_label_set_from_item(
    item: Dict[str, Any],
    task: str,
    allowed_labels: List[str],
) -> Optional[Set[str]]:
    key = "positive_emotion_labels" if task == "positive-labels" else "negative_emotion_labels"
    emotion_labels = item.get("emotion_labels", {})
    if not isinstance(emotion_labels, dict):
        return None
    labels_raw = emotion_labels.get(key)
    if not isinstance(labels_raw, list):
        return None
    allowed = set(allowed_labels)
    return {v for v in labels_raw if isinstance(v, str) and v in allowed}


def _load_third_person_model_label_prob(pred_counterfactual_payload: Any, label: str) -> Optional[float]:
    if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
        return None

    probs: List[float] = []
    for item in pred_counterfactual_payload:
        if not isinstance(item, dict):
            continue
        labels_raw = item.get("labels")
        if not isinstance(labels_raw, list):
            continue
        labels_set = {v for v in labels_raw if isinstance(v, str)}
        probs.append(1.0 if label in labels_set else 0.0)
    return safe_mean(probs)


def _load_model_label_set_from_item(
    item: Dict[str, Any],
    allowed_labels: List[str],
) -> Optional[Set[str]]:
    labels_raw = item.get("labels")
    if not isinstance(labels_raw, list):
        return None
    allowed = set(allowed_labels)
    return {v for v in labels_raw if isinstance(v, str) and v in allowed}


def _prediction_file_count(run_roots: List[Path], task: str) -> int:
    return sum(
        1
        for run_root in run_roots
        for path in (run_root / task).glob("*.json")
        if path.is_file()
    )


def _mean_squared_error(gold: List[float], predicted: List[float]) -> Optional[float]:
    if not gold or len(gold) != len(predicted):
        return None
    return sum((pred - target) ** 2 for target, pred in zip(gold, predicted)) / len(gold)


def _summarize_level_predictions(
    human_values: List[float],
    model_values: List[float],
) -> Dict[str, Any]:
    if not human_values or len(human_values) != len(model_values):
        return {
            "items": 0,
            "mean_human": None,
            "mean_model": None,
            "mae": None,
            "rmse": None,
            "exact_accuracy": None,
            "pearson": None,
            "spearman": None,
            "warnings": ["No aligned first-person gold/model levels"],
        }

    errors = [
        abs(predicted - human)
        for human, predicted in zip(human_values, model_values)
    ]
    mse = _mean_squared_error(human_values, model_values)
    pearson_value, pearson_reason = pearson_corr(model_values, human_values)
    spearman_value, spearman_reason = spearman_corr(model_values, human_values)
    warnings: List[str] = []
    if pearson_reason is not None:
        warnings.append(f"Pearson unavailable: {pearson_reason}")
    if spearman_reason is not None:
        warnings.append(f"Spearman unavailable: {spearman_reason}")
    return {
        "items": len(human_values),
        "mean_human": safe_mean(human_values),
        "mean_model": safe_mean(model_values),
        "mae": safe_mean(errors),
        "rmse": math.sqrt(mse) if mse is not None else None,
        "exact_accuracy": sum(
            predicted == human
            for human, predicted in zip(human_values, model_values)
        )
        / len(human_values),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "warnings": warnings,
    }


def _summarize_numeric_changes(values: List[float]) -> Dict[str, Any]:
    items = len(values)
    counts = collections.Counter(delta_sign(value) for value in values)
    return {
        "items": items,
        "mean_delta": safe_mean(values),
        "mean_absolute_delta": safe_mean([abs(value) for value in values]),
        "increase": int(counts["increase"]),
        "decrease": int(counts["decrease"]),
        "no_change": int(counts["no_change"]),
        "increase_rate": counts["increase"] / items if items else None,
        "decrease_rate": counts["decrease"] / items if items else None,
        "no_change_rate": counts["no_change"] / items if items else None,
    }


def _summarize_delta_alignment(
    model_deltas: List[float],
    human_deltas: List[float],
) -> Dict[str, Any]:
    if not model_deltas or len(model_deltas) != len(human_deltas):
        return {
            "items": 0,
            "mean_model_delta": None,
            "mean_human_delta": None,
            "pearson": None,
            "spearman": None,
            "same_direction_count": 0,
            "same_direction_rate": None,
            "warnings": ["No aligned model/human counterfactual deltas"],
        }
    pearson_value, pearson_reason = pearson_corr(model_deltas, human_deltas)
    spearman_value, spearman_reason = spearman_corr(model_deltas, human_deltas)
    same_direction = sum(
        delta_sign(model) == delta_sign(human)
        for model, human in zip(model_deltas, human_deltas)
    )
    warnings: List[str] = []
    if pearson_reason is not None:
        warnings.append(f"Pearson unavailable: {pearson_reason}")
    if spearman_reason is not None:
        warnings.append(f"Spearman unavailable: {spearman_reason}")
    return {
        "items": len(model_deltas),
        "mean_model_delta": safe_mean(model_deltas),
        "mean_human_delta": safe_mean(human_deltas),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "same_direction_count": same_direction,
        "same_direction_rate": same_direction / len(model_deltas),
        "warnings": warnings,
    }


def _evaluate_first_person_appraisal_level_task(
    task: str,
    first_person_root: Path,
    origin_run_roots: List[Path],
    first_person_appraisal_run_roots: List[Path],
) -> Dict[str, Any]:
    gold_files = sorted(path for path in first_person_root.glob("*.json") if path.is_file())
    stats = {
        "gold_files": len(gold_files),
        "origin_pred_files": _prediction_file_count(origin_run_roots, task),
        "first_person_appraisal_pred_files": _prediction_file_count(
            first_person_appraisal_run_roots, task
        ),
        "evaluated_samples": 0,
        "skipped_invalid_first_person_human": 0,
        "skipped_missing_origin": 0,
        "skipped_missing_first_person_appraisal": 0,
    }
    human_values: List[float] = []
    origin_values: List[float] = []
    first_person_appraisal_values: List[float] = []
    absolute_error_deltas: List[float] = []
    prediction_deltas: List[float] = []
    outcomes = collections.Counter()

    for gold_file in gold_files:
        payload = load_json(gold_file)
        if not isinstance(payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        human = _load_first_person_human_level(payload, task)
        if human is None:
            stats["skipped_invalid_first_person_human"] += 1
            continue
        origin = _load_first_person_model_level_from_runs(
            origin_run_roots, task, gold_file.name
        )
        if origin is None:
            stats["skipped_missing_origin"] += 1
            continue
        intervention = _load_first_person_model_level_from_runs(
            first_person_appraisal_run_roots, task, gold_file.name
        )
        if intervention is None:
            stats["skipped_missing_first_person_appraisal"] += 1
            continue

        human_values.append(human)
        origin_values.append(origin)
        first_person_appraisal_values.append(intervention)
        prediction_deltas.append(intervention - origin)
        error_delta = abs(intervention - human) - abs(origin - human)
        absolute_error_deltas.append(error_delta)
        if error_delta < 0:
            outcomes["improved"] += 1
        elif error_delta > 0:
            outcomes["worsened"] += 1
        else:
            outcomes["tied"] += 1
        stats["evaluated_samples"] += 1

    items = len(human_values)
    return {
        "stats": stats,
        "origin": _summarize_level_predictions(human_values, origin_values),
        "first_person_appraisal": _summarize_level_predictions(
            human_values, first_person_appraisal_values
        ),
        "comparison": {
            "mean_prediction_delta": safe_mean(prediction_deltas),
            "mean_absolute_error_delta": safe_mean(absolute_error_deltas),
            "absolute_error_delta_definition": (
                "first_person_appraisal absolute error minus origin absolute "
                "error; negative values indicate improvement"
            ),
            "improved": int(outcomes["improved"]),
            "tied": int(outcomes["tied"]),
            "worsened": int(outcomes["worsened"]),
            "improved_rate": outcomes["improved"] / items if items else None,
            "tied_rate": outcomes["tied"] / items if items else None,
            "worsened_rate": outcomes["worsened"] / items if items else None,
        },
    }


def _sample_label_scores(gold: Set[str], predicted: Set[str]) -> Dict[str, float]:
    true_positive = len(gold & predicted)
    if not gold and not predicted:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        2.0 * true_positive / (len(gold) + len(predicted))
        if gold or predicted
        else 1.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _summarize_label_predictions(
    human_labels: List[Set[str]],
    model_labels: List[Set[str]],
    allowed_labels: List[str],
) -> Dict[str, Any]:
    items = len(human_labels)
    if not items or items != len(model_labels):
        return {
            "items": 0,
            "example_precision": None,
            "example_recall": None,
            "example_f1": None,
            "exact_match": None,
            "micro_precision": None,
            "micro_recall": None,
            "micro_f1": None,
            "macro_f1": None,
            "per_label": {},
        }

    sample_scores = [
        _sample_label_scores(gold, predicted)
        for gold, predicted in zip(human_labels, model_labels)
    ]
    total_tp = total_fp = total_fn = 0
    per_label: Dict[str, Dict[str, Any]] = {}
    for label in allowed_labels:
        tp = sum(label in gold and label in predicted for gold, predicted in zip(human_labels, model_labels))
        fp = sum(label not in gold and label in predicted for gold, predicted in zip(human_labels, model_labels))
        fn = sum(label in gold and label not in predicted for gold, predicted in zip(human_labels, model_labels))
        tn = items - tp - fp - fn
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "accuracy": (tp + tn) / items,
        }

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2.0 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn
        else 0.0
    )
    return {
        "items": items,
        "example_precision": safe_mean([score["precision"] for score in sample_scores]),
        "example_recall": safe_mean([score["recall"] for score in sample_scores]),
        "example_f1": safe_mean([score["f1"] for score in sample_scores]),
        "exact_match": sum(gold == predicted for gold, predicted in zip(human_labels, model_labels)) / items,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": safe_mean([metrics["f1"] for metrics in per_label.values()]),
        "per_label": per_label,
    }


def _evaluate_first_person_appraisal_label_task(
    task: str,
    allowed_labels: List[str],
    first_person_root: Path,
    origin_run_roots: List[Path],
    first_person_appraisal_run_roots: List[Path],
) -> Dict[str, Any]:
    gold_files = sorted(path for path in first_person_root.glob("*.json") if path.is_file())
    stats = {
        "gold_files": len(gold_files),
        "origin_pred_files": _prediction_file_count(origin_run_roots, task),
        "first_person_appraisal_pred_files": _prediction_file_count(
            first_person_appraisal_run_roots, task
        ),
        "evaluated_samples": 0,
        "skipped_invalid_first_person_human": 0,
        "skipped_missing_origin": 0,
        "skipped_missing_first_person_appraisal": 0,
    }
    human_sets: List[Set[str]] = []
    origin_sets: List[Set[str]] = []
    first_person_appraisal_sets: List[Set[str]] = []
    f1_deltas: List[float] = []
    outcomes = collections.Counter()
    transitions = new_label_change_accumulator()

    for gold_file in gold_files:
        payload = load_json(gold_file)
        if not isinstance(payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        human = _load_first_person_human_labels(payload, task, allowed_labels)
        if human is None:
            stats["skipped_invalid_first_person_human"] += 1
            continue
        origin = _load_first_person_model_labels_from_runs(
            origin_run_roots, task, gold_file.name, allowed_labels
        )
        if origin is None:
            stats["skipped_missing_origin"] += 1
            continue
        intervention = _load_first_person_model_labels_from_runs(
            first_person_appraisal_run_roots,
            task,
            gold_file.name,
            allowed_labels,
        )
        if intervention is None:
            stats["skipped_missing_first_person_appraisal"] += 1
            continue

        human_sets.append(human)
        origin_sets.append(origin)
        first_person_appraisal_sets.append(intervention)
        origin_f1 = _sample_label_scores(human, origin)["f1"]
        intervention_f1 = _sample_label_scores(human, intervention)["f1"]
        delta = intervention_f1 - origin_f1
        f1_deltas.append(delta)
        if delta > 0:
            outcomes["improved"] += 1
        elif delta < 0:
            outcomes["worsened"] += 1
        else:
            outcomes["tied"] += 1
        update_label_change_accumulator(
            transitions, origin, intervention, allowed_labels
        )
        stats["evaluated_samples"] += 1

    items = len(human_sets)
    return {
        "stats": stats,
        "origin": _summarize_label_predictions(
            human_sets, origin_sets, allowed_labels
        ),
        "first_person_appraisal": _summarize_label_predictions(
            human_sets, first_person_appraisal_sets, allowed_labels
        ),
        "comparison": {
            "mean_example_f1_delta": safe_mean(f1_deltas),
            "example_f1_delta_definition": (
                "first_person_appraisal example F1 minus origin example F1; "
                "positive values indicate improvement"
            ),
            "improved": int(outcomes["improved"]),
            "tied": int(outcomes["tied"]),
            "worsened": int(outcomes["worsened"]),
            "improved_rate": outcomes["improved"] / items if items else None,
            "tied_rate": outcomes["tied"] / items if items else None,
            "worsened_rate": outcomes["worsened"] / items if items else None,
            "origin_to_first_person_appraisal_transitions": (
                finalize_label_change_accumulator(transitions, allowed_labels)
            ),
        },
    }


def _build_base_stats(gold_files: int, pred_files: int) -> Dict[str, int]:
    return {
        "gold_files": gold_files,
        "pred_files": pred_files,
        "evaluated_samples": 0,
        "skipped_missing_first_person": 0,
        "skipped_missing_baseline": 0,
        "skipped_missing_counterfactual_pred_file": 0,
        "skipped_invalid_first_person_human": 0,
        "skipped_invalid_baseline": 0,
        "skipped_invalid_human_counterfactual": 0,
        "skipped_invalid_model_counterfactual": 0,
    }


def label_signature(labels: Set[str]) -> str:
    if not labels:
        return "None"
    return " ; ".join(sorted(labels))


def new_label_change_accumulator() -> Dict[str, Any]:
    return {
        "items": 0,
        "changed": 0,
        "unchanged": 0,
        "transition_counts": collections.Counter(),
        "added": collections.Counter(),
        "removed": collections.Counter(),
        "kept_present": collections.Counter(),
        "kept_absent": collections.Counter(),
        "original_label_count_sum": 0.0,
        "counterfactual_label_count_sum": 0.0,
    }


def update_label_change_accumulator(
    accumulator: Dict[str, Any],
    original_labels: Set[str],
    counterfactual_labels: Set[str],
    allowed_labels: List[str],
) -> None:
    accumulator["items"] += 1
    accumulator["original_label_count_sum"] += len(original_labels)
    accumulator["counterfactual_label_count_sum"] += len(counterfactual_labels)
    if original_labels == counterfactual_labels:
        accumulator["unchanged"] += 1
    else:
        accumulator["changed"] += 1
    accumulator["transition_counts"][
        (label_signature(original_labels), label_signature(counterfactual_labels))
    ] += 1
    for label in allowed_labels:
        before = label in original_labels
        after = label in counterfactual_labels
        if before and after:
            accumulator["kept_present"][label] += 1
        elif before and not after:
            accumulator["removed"][label] += 1
        elif not before and after:
            accumulator["added"][label] += 1
        else:
            accumulator["kept_absent"][label] += 1


def finalize_label_change_accumulator(
    accumulator: Dict[str, Any],
    allowed_labels: List[str],
    top_k: int = 20,
) -> Dict[str, Any]:
    items = int(accumulator["items"])
    per_label: Dict[str, Dict[str, Any]] = {}
    for label in allowed_labels:
        added = int(accumulator["added"][label])
        removed = int(accumulator["removed"][label])
        kept_present = int(accumulator["kept_present"][label])
        kept_absent = int(accumulator["kept_absent"][label])
        per_label[label] = {
            "added": added,
            "removed": removed,
            "kept_present": kept_present,
            "kept_absent": kept_absent,
            "added_rate": added / items if items else None,
            "removed_rate": removed / items if items else None,
            "net_change": added - removed,
            "net_change_rate": (added - removed) / items if items else None,
        }

    transitions = [
        {
            "from": before,
            "to": after,
            "count": int(count),
            "rate": int(count) / items if items else None,
        }
        for (before, after), count in accumulator["transition_counts"].most_common(top_k)
    ]
    return {
        "items": items,
        "changed": int(accumulator["changed"]),
        "unchanged": int(accumulator["unchanged"]),
        "change_rate": int(accumulator["changed"]) / items if items else None,
        "mean_original_label_count": (
            accumulator["original_label_count_sum"] / items if items else None
        ),
        "mean_counterfactual_label_count": (
            accumulator["counterfactual_label_count_sum"] / items if items else None
        ),
        "per_label": per_label,
        "top_transitions": transitions,
    }


def new_intensity_change_accumulator() -> Dict[str, Any]:
    return {
        "items": 0,
        "original_human_sum": 0.0,
        "counterfactual_human_sum": 0.0,
        "human_delta_sum": 0.0,
        "original_model_sum": 0.0,
        "counterfactual_model_sum": 0.0,
        "model_delta_sum": 0.0,
        "same_direction": 0,
        "human_delta_signs": collections.Counter(),
        "model_delta_signs": collections.Counter(),
    }


def delta_sign(value: float) -> str:
    if value > 0:
        return "increase"
    if value < 0:
        return "decrease"
    return "no_change"


def update_intensity_change_accumulator(
    accumulator: Dict[str, Any],
    original_human: float,
    counterfactual_human: float,
    original_model: float,
    counterfactual_model: float,
) -> None:
    human_delta = counterfactual_human - original_human
    model_delta = counterfactual_model - original_model
    accumulator["items"] += 1
    accumulator["original_human_sum"] += original_human
    accumulator["counterfactual_human_sum"] += counterfactual_human
    accumulator["human_delta_sum"] += human_delta
    accumulator["original_model_sum"] += original_model
    accumulator["counterfactual_model_sum"] += counterfactual_model
    accumulator["model_delta_sum"] += model_delta
    human_sign = delta_sign(human_delta)
    model_sign = delta_sign(model_delta)
    accumulator["human_delta_signs"][human_sign] += 1
    accumulator["model_delta_signs"][model_sign] += 1
    if human_sign == model_sign:
        accumulator["same_direction"] += 1


def finalize_intensity_change_accumulator(accumulator: Dict[str, Any]) -> Dict[str, Any]:
    items = int(accumulator["items"])
    return {
        "items": items,
        "mean_original_human": accumulator["original_human_sum"] / items if items else None,
        "mean_counterfactual_human": (
            accumulator["counterfactual_human_sum"] / items if items else None
        ),
        "mean_human_delta": accumulator["human_delta_sum"] / items if items else None,
        "mean_original_model": accumulator["original_model_sum"] / items if items else None,
        "mean_counterfactual_model": (
            accumulator["counterfactual_model_sum"] / items if items else None
        ),
        "mean_model_delta": accumulator["model_delta_sum"] / items if items else None,
        "same_direction_count": int(accumulator["same_direction"]),
        "same_direction_rate": (
            int(accumulator["same_direction"]) / items if items else None
        ),
        "human_delta_direction_counts": dict(accumulator["human_delta_signs"]),
        "model_delta_direction_counts": dict(accumulator["model_delta_signs"]),
    }


def direction_accumulator(
    accumulators: Dict[str, Dict[str, Any]],
    direction: str,
    factory: Any,
) -> Dict[str, Any]:
    if direction not in accumulators:
        accumulators[direction] = factory()
    return accumulators[direction]


def _evaluate_level_task_for_dimension(
    task: str,
    gold_dim_dir: Path,
    pred_dim_dir: Path,
    first_person_root: Path,
    baseline_run_roots: List[Path],
    first_person_appraisal_run_roots: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    gold_files = sorted([p for p in gold_dim_dir.glob("*.json") if p.is_file()])
    pred_files = len([p for p in pred_dim_dir.glob("*.json") if p.is_file()]) if pred_dim_dir.exists() else 0
    stats = _build_base_stats(gold_files=len(gold_files), pred_files=pred_files)

    model_deltas: List[float] = []
    human_deltas: List[float] = []
    intensity_all = new_intensity_change_accumulator()
    intensity_by_direction: Dict[str, Dict[str, Any]] = {}
    first_to_third_intensity_all = new_intensity_change_accumulator()
    first_to_third_intensity_by_direction: Dict[str, Dict[str, Any]] = {}
    appraisal_direction_counts: collections.Counter[str] = collections.Counter()
    three_way_stats = {
        "available": bool(first_person_appraisal_run_roots),
        "evaluated_samples": 0,
        "skipped_missing_first_person_appraisal": 0,
    }
    three_first_person_human: List[float] = []
    three_third_person_human: List[float] = []
    three_origin_model: List[float] = []
    three_first_person_model: List[float] = []
    three_third_person_model: List[float] = []

    for gold_file in gold_files:
        sample_name = gold_file.name
        first_person_file = first_person_root / sample_name
        pred_file = pred_dim_dir / sample_name

        if not first_person_file.exists() or not first_person_file.is_file():
            stats["skipped_missing_first_person"] += 1
            continue
        if not pred_file.exists() or not pred_file.is_file():
            stats["skipped_missing_counterfactual_pred_file"] += 1
            continue

        first_person_payload = load_json(first_person_file)
        gold_counterfactual_payload = load_json(gold_file)
        pred_counterfactual_payload = load_json(pred_file)

        if not isinstance(first_person_payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue

        first_person_human_level = _load_first_person_human_level(first_person_payload, task)
        if first_person_human_level is None:
            stats["skipped_invalid_first_person_human"] += 1
            continue

        first_person_model_level = _load_first_person_model_level_from_runs(baseline_run_roots, task, sample_name)
        if first_person_model_level is None:
            stats["skipped_missing_baseline"] += 1
            continue

        third_person_human_level = _load_third_person_human_level(gold_counterfactual_payload, task)
        if third_person_human_level is None:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue

        third_person_model_level = _load_third_person_model_level(pred_counterfactual_payload)
        if third_person_model_level is None:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        human_deltas.append(third_person_human_level - first_person_human_level)
        model_deltas.append(third_person_model_level - first_person_model_level)
        stats["evaluated_samples"] += 1

        first_person_appraisal_model_level: Optional[float] = None
        if first_person_appraisal_run_roots:
            first_person_appraisal_model_level = (
                _load_first_person_model_level_from_runs(
                    first_person_appraisal_run_roots,
                    task,
                    sample_name,
                )
            )
            if first_person_appraisal_model_level is None:
                three_way_stats["skipped_missing_first_person_appraisal"] += 1
            else:
                three_first_person_human.append(first_person_human_level)
                three_third_person_human.append(third_person_human_level)
                three_origin_model.append(first_person_model_level)
                three_first_person_model.append(first_person_appraisal_model_level)
                three_third_person_model.append(third_person_model_level)
                three_way_stats["evaluated_samples"] += 1

        for gold_item, pred_item in zip(gold_counterfactual_payload, pred_counterfactual_payload):
            if not isinstance(gold_item, dict) or not isinstance(pred_item, dict):
                continue
            item_human_level = _load_human_level_from_item(gold_item, task)
            item_model_level = _load_model_level_from_item(pred_item)
            if item_human_level is None or item_model_level is None:
                continue
            direction = appraisal_direction(gold_item)
            appraisal_direction_counts[direction] += 1
            update_intensity_change_accumulator(
                intensity_all,
                first_person_human_level,
                item_human_level,
                first_person_model_level,
                item_model_level,
            )
            if first_person_appraisal_model_level is not None:
                update_intensity_change_accumulator(
                    first_to_third_intensity_all,
                    first_person_human_level,
                    item_human_level,
                    first_person_appraisal_model_level,
                    item_model_level,
                )
                update_intensity_change_accumulator(
                    direction_accumulator(
                        first_to_third_intensity_by_direction,
                        direction,
                        new_intensity_change_accumulator,
                    ),
                    first_person_human_level,
                    item_human_level,
                    first_person_appraisal_model_level,
                    item_model_level,
                )
            update_intensity_change_accumulator(
                direction_accumulator(
                    intensity_by_direction,
                    direction,
                    new_intensity_change_accumulator,
                ),
                first_person_human_level,
                item_human_level,
                first_person_model_level,
                item_model_level,
            )

    pearson_value, pearson_reason = pearson_corr(model_deltas, human_deltas)
    spearman_value, spearman_reason = spearman_corr(model_deltas, human_deltas)

    warnings: List[str] = []
    if pearson_reason is not None:
        warnings.append(f"Pearson unavailable: {pearson_reason}")
    if spearman_reason is not None:
        warnings.append(f"Spearman unavailable: {spearman_reason}")

    first_minus_origin = [
        first - origin
        for first, origin in zip(three_first_person_model, three_origin_model)
    ]
    third_minus_origin = [
        third - origin
        for third, origin in zip(three_third_person_model, three_origin_model)
    ]
    third_minus_first = [
        third - first
        for third, first in zip(
            three_third_person_model, three_first_person_model
        )
    ]
    three_human_deltas = [
        third - first
        for third, first in zip(
            three_third_person_human, three_first_person_human
        )
    ]

    return {
        "pearson": pearson_value,
        "spearman": spearman_value,
        "stats": stats,
        "intensity_changes": {
            "all": finalize_intensity_change_accumulator(intensity_all),
            "by_appraisal_direction": {
                direction: finalize_intensity_change_accumulator(accumulator)
                for direction, accumulator in sorted(intensity_by_direction.items())
            },
            "appraisal_direction_counts": dict(appraisal_direction_counts),
        },
        "three_way_comparison": {
            "stats": three_way_stats,
            "matched_gold_performance": {
                "origin": _summarize_level_predictions(
                    three_first_person_human,
                    three_origin_model,
                ),
                "first_person_appraisal": _summarize_level_predictions(
                    three_first_person_human,
                    three_first_person_model,
                ),
                "third_person_appraisal": _summarize_level_predictions(
                    three_third_person_human,
                    three_third_person_model,
                ),
            },
            "pairwise_model_changes": {
                "first_person_appraisal_minus_origin": (
                    _summarize_numeric_changes(first_minus_origin)
                ),
                "third_person_appraisal_minus_origin": (
                    _summarize_numeric_changes(third_minus_origin)
                ),
                "third_person_appraisal_minus_first_person_appraisal": (
                    _summarize_numeric_changes(third_minus_first)
                ),
            },
            "counterfactual_delta_alignment": {
                "third_person_appraisal_minus_origin": (
                    _summarize_delta_alignment(
                        third_minus_origin,
                        three_human_deltas,
                    )
                ),
                "third_person_appraisal_minus_first_person_appraisal": (
                    _summarize_delta_alignment(
                        third_minus_first,
                        three_human_deltas,
                    )
                ),
            },
            "first_person_to_third_person_intensity_changes": {
                "all": finalize_intensity_change_accumulator(
                    first_to_third_intensity_all
                ),
                "by_appraisal_direction": {
                    direction: finalize_intensity_change_accumulator(accumulator)
                    for direction, accumulator in sorted(
                        first_to_third_intensity_by_direction.items()
                    )
                },
            },
        },
        "warnings": warnings,
    }


def _evaluate_label_task_for_dimension(
    task: str,
    allowed_labels: List[str],
    gold_dim_dir: Path,
    pred_dim_dir: Path,
    first_person_root: Path,
    baseline_run_roots: List[Path],
    first_person_appraisal_run_roots: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    gold_files = sorted([p for p in gold_dim_dir.glob("*.json") if p.is_file()])
    pred_files = len([p for p in pred_dim_dir.glob("*.json") if p.is_file()]) if pred_dim_dir.exists() else 0
    stats = _build_base_stats(gold_files=len(gold_files), pred_files=pred_files)

    model_deltas_by_label: Dict[str, List[float]] = {label: [] for label in allowed_labels}
    human_deltas_by_label: Dict[str, List[float]] = {label: [] for label in allowed_labels}
    model_transition_all = new_label_change_accumulator()
    human_transition_all = new_label_change_accumulator()
    model_transition_by_direction: Dict[str, Dict[str, Any]] = {}
    human_transition_by_direction: Dict[str, Dict[str, Any]] = {}
    appraisal_direction_counts: collections.Counter[str] = collections.Counter()
    three_way_stats = {
        "available": bool(first_person_appraisal_run_roots),
        "evaluated_samples": 0,
        "evaluated_third_person_entries": 0,
        "skipped_missing_first_person_appraisal": 0,
    }
    three_first_person_human_sets: List[Set[str]] = []
    three_origin_model_sets: List[Set[str]] = []
    three_first_person_model_sets: List[Set[str]] = []
    three_third_person_human_sets: List[Set[str]] = []
    three_third_person_model_sets: List[Set[str]] = []
    origin_to_first_transitions = new_label_change_accumulator()
    three_origin_to_third_transitions = new_label_change_accumulator()
    first_to_third_transitions = new_label_change_accumulator()
    first_to_third_transitions_by_direction: Dict[str, Dict[str, Any]] = {}
    three_human_deltas_by_label: Dict[str, List[float]] = {
        label: [] for label in allowed_labels
    }
    three_origin_deltas_by_label: Dict[str, List[float]] = {
        label: [] for label in allowed_labels
    }
    three_first_deltas_by_label: Dict[str, List[float]] = {
        label: [] for label in allowed_labels
    }

    for gold_file in gold_files:
        sample_name = gold_file.name
        first_person_file = first_person_root / sample_name
        pred_file = pred_dim_dir / sample_name

        if not first_person_file.exists() or not first_person_file.is_file():
            stats["skipped_missing_first_person"] += 1
            continue
        if not pred_file.exists() or not pred_file.is_file():
            stats["skipped_missing_counterfactual_pred_file"] += 1
            continue

        first_person_payload = load_json(first_person_file)
        gold_counterfactual_payload = load_json(gold_file)
        pred_counterfactual_payload = load_json(pred_file)

        if not isinstance(first_person_payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        if not isinstance(gold_counterfactual_payload, list) or not gold_counterfactual_payload:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue
        if not isinstance(pred_counterfactual_payload, list) or not pred_counterfactual_payload:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        first_person_human_labels = _load_first_person_human_labels(
            first_person_payload,
            task,
            allowed_labels,
        )
        if first_person_human_labels is None:
            stats["skipped_invalid_first_person_human"] += 1
            continue

        first_person_model_labels = _load_first_person_model_labels_from_runs(
            baseline_run_roots,
            task,
            sample_name,
            allowed_labels,
        )
        if first_person_model_labels is None:
            stats["skipped_missing_baseline"] += 1
            continue

        sample_model_deltas: Dict[str, float] = {}
        sample_human_deltas: Dict[str, float] = {}
        sample_origin_model_probs: Dict[str, float] = {}
        sample_third_model_probs: Dict[str, float] = {}

        for label in allowed_labels:
            first_person_human_prob = _load_first_person_human_label_prob(first_person_payload, task, label)
            if first_person_human_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            first_person_model_prob = _load_first_person_model_label_prob_from_runs(
                baseline_run_roots,
                task,
                sample_name,
                label,
            )
            if first_person_model_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            third_person_human_prob = _load_third_person_human_label_prob(gold_counterfactual_payload, task, label)
            if third_person_human_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            third_person_model_prob = _load_third_person_model_label_prob(pred_counterfactual_payload, label)
            if third_person_model_prob is None:
                sample_model_deltas = {}
                sample_human_deltas = {}
                break

            sample_human_deltas[label] = third_person_human_prob - first_person_human_prob
            sample_model_deltas[label] = third_person_model_prob - first_person_model_prob
            sample_origin_model_probs[label] = first_person_model_prob
            sample_third_model_probs[label] = third_person_model_prob

        if not sample_model_deltas or not sample_human_deltas:
            # Align skip reasons to dominant failure type by checking baseline/human/model components once.
            # This keeps behavior deterministic while preserving focused counters.
            probe_label = allowed_labels[0] if allowed_labels else None
            if probe_label is None:
                stats["skipped_invalid_first_person_human"] += 1
                continue
            if _load_first_person_human_label_prob(first_person_payload, task, probe_label) is None:
                stats["skipped_invalid_first_person_human"] += 1
                continue
            if _load_first_person_model_label_prob_from_runs(
                baseline_run_roots, task, sample_name, probe_label
            ) is None:
                stats["skipped_missing_baseline"] += 1
                continue
            if _load_third_person_human_label_prob(gold_counterfactual_payload, task, probe_label) is None:
                stats["skipped_invalid_human_counterfactual"] += 1
                continue
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        for label in allowed_labels:
            human_deltas_by_label[label].append(sample_human_deltas[label])
            model_deltas_by_label[label].append(sample_model_deltas[label])

        stats["evaluated_samples"] += 1

        first_person_appraisal_model_labels: Optional[Set[str]] = None
        first_person_appraisal_model_probs: Dict[str, float] = {}
        if first_person_appraisal_run_roots:
            first_person_appraisal_model_labels = (
                _load_first_person_model_labels_from_runs(
                    first_person_appraisal_run_roots,
                    task,
                    sample_name,
                    allowed_labels,
                )
            )
            if first_person_appraisal_model_labels is not None:
                for label in allowed_labels:
                    probability = _load_first_person_model_label_prob_from_runs(
                        first_person_appraisal_run_roots,
                        task,
                        sample_name,
                        label,
                    )
                    if probability is None:
                        first_person_appraisal_model_probs = {}
                        first_person_appraisal_model_labels = None
                        break
                    first_person_appraisal_model_probs[label] = probability
            if first_person_appraisal_model_labels is None:
                three_way_stats["skipped_missing_first_person_appraisal"] += 1
            else:
                three_first_person_human_sets.append(first_person_human_labels)
                three_origin_model_sets.append(first_person_model_labels)
                three_first_person_model_sets.append(
                    first_person_appraisal_model_labels
                )
                update_label_change_accumulator(
                    origin_to_first_transitions,
                    first_person_model_labels,
                    first_person_appraisal_model_labels,
                    allowed_labels,
                )
                for label in allowed_labels:
                    three_human_deltas_by_label[label].append(
                        sample_human_deltas[label]
                    )
                    three_origin_deltas_by_label[label].append(
                        sample_model_deltas[label]
                    )
                    three_first_deltas_by_label[label].append(
                        sample_third_model_probs[label]
                        - first_person_appraisal_model_probs[label]
                    )
                three_way_stats["evaluated_samples"] += 1

        for gold_item, pred_item in zip(gold_counterfactual_payload, pred_counterfactual_payload):
            if not isinstance(gold_item, dict) or not isinstance(pred_item, dict):
                continue
            counterfactual_human_labels = _load_human_label_set_from_item(
                gold_item,
                task,
                allowed_labels,
            )
            counterfactual_model_labels = _load_model_label_set_from_item(
                pred_item,
                allowed_labels,
            )
            if counterfactual_human_labels is None or counterfactual_model_labels is None:
                continue
            direction = appraisal_direction(gold_item)
            appraisal_direction_counts[direction] += 1
            update_label_change_accumulator(
                human_transition_all,
                first_person_human_labels,
                counterfactual_human_labels,
                allowed_labels,
            )
            update_label_change_accumulator(
                model_transition_all,
                first_person_model_labels,
                counterfactual_model_labels,
                allowed_labels,
            )
            update_label_change_accumulator(
                direction_accumulator(
                    human_transition_by_direction,
                    direction,
                    new_label_change_accumulator,
                ),
                first_person_human_labels,
                counterfactual_human_labels,
                allowed_labels,
            )
            update_label_change_accumulator(
                direction_accumulator(
                    model_transition_by_direction,
                    direction,
                    new_label_change_accumulator,
                ),
                first_person_model_labels,
                counterfactual_model_labels,
                allowed_labels,
            )
            if first_person_appraisal_model_labels is not None:
                three_third_person_human_sets.append(counterfactual_human_labels)
                three_third_person_model_sets.append(counterfactual_model_labels)
                update_label_change_accumulator(
                    three_origin_to_third_transitions,
                    first_person_model_labels,
                    counterfactual_model_labels,
                    allowed_labels,
                )
                update_label_change_accumulator(
                    first_to_third_transitions,
                    first_person_appraisal_model_labels,
                    counterfactual_model_labels,
                    allowed_labels,
                )
                update_label_change_accumulator(
                    direction_accumulator(
                        first_to_third_transitions_by_direction,
                        direction,
                        new_label_change_accumulator,
                    ),
                    first_person_appraisal_model_labels,
                    counterfactual_model_labels,
                    allowed_labels,
                )
                three_way_stats["evaluated_third_person_entries"] += 1

    categories: Dict[str, Any] = {}
    for label in allowed_labels:
        pearson_value, pearson_reason = pearson_corr(model_deltas_by_label[label], human_deltas_by_label[label])
        spearman_value, spearman_reason = spearman_corr(model_deltas_by_label[label], human_deltas_by_label[label])

        warnings: List[str] = []
        if pearson_reason is not None:
            warnings.append(f"Pearson unavailable: {pearson_reason}")
        if spearman_reason is not None:
            warnings.append(f"Spearman unavailable: {spearman_reason}")

        categories[label] = {
            "pearson": pearson_value,
            "spearman": spearman_value,
            "stats": dict(stats),
            "warnings": warnings,
        }

    three_way_delta_correlations: Dict[str, Any] = {}
    for label in allowed_labels:
        three_way_delta_correlations[label] = {
            "third_person_appraisal_minus_origin": _summarize_delta_alignment(
                three_origin_deltas_by_label[label],
                three_human_deltas_by_label[label],
            ),
            "third_person_appraisal_minus_first_person_appraisal": (
                _summarize_delta_alignment(
                    three_first_deltas_by_label[label],
                    three_human_deltas_by_label[label],
                )
            ),
        }

    return {
        "stats": stats,
        "categories": categories,
        "label_transitions": {
            "model": {
                "all": finalize_label_change_accumulator(
                    model_transition_all,
                    allowed_labels,
                ),
                "by_appraisal_direction": {
                    direction: finalize_label_change_accumulator(
                        accumulator,
                        allowed_labels,
                    )
                    for direction, accumulator in sorted(
                        model_transition_by_direction.items()
                    )
                },
            },
            "human": {
                "all": finalize_label_change_accumulator(
                    human_transition_all,
                    allowed_labels,
                ),
                "by_appraisal_direction": {
                    direction: finalize_label_change_accumulator(
                        accumulator,
                        allowed_labels,
                    )
                    for direction, accumulator in sorted(
                        human_transition_by_direction.items()
                    )
                },
            },
            "appraisal_direction_counts": dict(appraisal_direction_counts),
        },
        "three_way_comparison": {
            "stats": three_way_stats,
            "matched_gold_performance": {
                "origin": _summarize_label_predictions(
                    three_first_person_human_sets,
                    three_origin_model_sets,
                    allowed_labels,
                ),
                "first_person_appraisal": _summarize_label_predictions(
                    three_first_person_human_sets,
                    three_first_person_model_sets,
                    allowed_labels,
                ),
                "third_person_appraisal": _summarize_label_predictions(
                    three_third_person_human_sets,
                    three_third_person_model_sets,
                    allowed_labels,
                ),
            },
            "pairwise_model_transitions": {
                "origin_to_first_person_appraisal": (
                    finalize_label_change_accumulator(
                        origin_to_first_transitions,
                        allowed_labels,
                    )
                ),
                "origin_to_third_person_appraisal": (
                    finalize_label_change_accumulator(
                        three_origin_to_third_transitions,
                        allowed_labels,
                    )
                ),
                "first_person_appraisal_to_third_person_appraisal": (
                    finalize_label_change_accumulator(
                        first_to_third_transitions,
                        allowed_labels,
                    )
                ),
                "first_person_appraisal_to_third_person_by_direction": {
                    direction: finalize_label_change_accumulator(
                        accumulator,
                        allowed_labels,
                    )
                    for direction, accumulator in sorted(
                        first_to_third_transitions_by_direction.items()
                    )
                },
            },
            "counterfactual_delta_alignment_by_label": (
                three_way_delta_correlations
            ),
        },
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    first_person_root = Path(args.first_person_root)
    counterfactual_gold_root = Path(args.counterfactual_gold_root)
    requested_baseline_root = Path(args.baseline_root)
    requested_counterfactual_root = Path(args.counterfactual_pred_root)
    baseline_model_root = requested_baseline_root
    counterfactual_model_root = requested_counterfactual_root
    prompt_path = Path(args.prompt_path)

    if (
        (baseline_model_root / "origin").is_dir()
        and not (baseline_model_root / "positive-level").is_dir()
    ):
        baseline_model_root = baseline_model_root / "origin"
    if (counterfactual_model_root / "third-person-appraisal").is_dir():
        counterfactual_model_root = (
            counterfactual_model_root / "third-person-appraisal"
        )
    first_person_appraisal_model_root = resolve_first_person_appraisal_root(
        args.first_person_appraisal_pred_root,
        requested_counterfactual_root,
    )

    if not first_person_root.exists() or not first_person_root.is_dir():
        raise FileNotFoundError(f"first_person_root not found: {first_person_root}")
    if not counterfactual_gold_root.exists() or not counterfactual_gold_root.is_dir():
        raise FileNotFoundError(f"counterfactual_gold_root not found: {counterfactual_gold_root}")
    if not baseline_model_root.exists() or not baseline_model_root.is_dir():
        raise FileNotFoundError(f"baseline model folder not found: {baseline_model_root}")
    if not counterfactual_model_root.exists() or not counterfactual_model_root.is_dir():
        raise FileNotFoundError(f"counterfactual model folder not found: {counterfactual_model_root}")
    if not prompt_path.exists() or not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    baseline_run_roots = resolve_baseline_run_roots(baseline_model_root)
    first_person_appraisal_run_roots = (
        resolve_baseline_run_roots(first_person_appraisal_model_root)
        if first_person_appraisal_model_root is not None
        else []
    )
    prompt_cfg = load_toml(prompt_path)

    dim_to_statement = prompt_cfg.get("appraisals", {}).get("dimension_to_statement", {})
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")

    pos_label_options = prompt_cfg.get("label_options", {}).get("positive-labels", {}).get("values")
    neg_label_options = prompt_cfg.get("label_options", {}).get("negative-labels", {}).get("values")
    if not isinstance(pos_label_options, list) or not pos_label_options:
        raise ValueError("Missing [label_options.positive-labels.values] in prompt TOML")
    if not isinstance(neg_label_options, list) or not neg_label_options:
        raise ValueError("Missing [label_options.negative-labels.values] in prompt TOML")

    positive_labels = [str(v) for v in pos_label_options if isinstance(v, str)]
    negative_labels = [str(v) for v in neg_label_options if isinstance(v, str)]
    if not positive_labels or not negative_labels:
        raise ValueError("Emotion label options are empty after filtering string values")

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model.strip() or None,
            "first_person_root": str(first_person_root),
            "counterfactual_gold_root": str(counterfactual_gold_root),
            "baseline_model_root": str(baseline_model_root),
            "counterfactual_model_root": str(counterfactual_model_root),
            "baseline_run_roots": [str(p) for p in baseline_run_roots],
            "first_person_appraisal_model_root": (
                str(first_person_appraisal_model_root)
                if first_person_appraisal_model_root is not None
                else None
            ),
            "first_person_appraisal_run_roots": [
                str(path) for path in first_person_appraisal_run_roots
            ],
            "prompt_path": str(prompt_path),
            "tasks": LEVEL_TASKS + LABEL_TASKS,
        },
        "dimension": {},
    }

    if first_person_appraisal_run_roots:
        results["first_person_appraisal"] = {
            "available": True,
            "description": (
                "Origin and first-person-gold appraisal-prefix continuations "
                "evaluated against the original first-person emotion gold."
            ),
            "positive-level": _evaluate_first_person_appraisal_level_task(
                "positive-level",
                first_person_root,
                baseline_run_roots,
                first_person_appraisal_run_roots,
            ),
            "negative-level": _evaluate_first_person_appraisal_level_task(
                "negative-level",
                first_person_root,
                baseline_run_roots,
                first_person_appraisal_run_roots,
            ),
            "positive-labels": _evaluate_first_person_appraisal_label_task(
                "positive-labels",
                positive_labels,
                first_person_root,
                baseline_run_roots,
                first_person_appraisal_run_roots,
            ),
            "negative-labels": _evaluate_first_person_appraisal_label_task(
                "negative-labels",
                negative_labels,
                first_person_root,
                baseline_run_roots,
                first_person_appraisal_run_roots,
            ),
        }
    else:
        results["first_person_appraisal"] = {
            "available": False,
            "warning": (
                "No first-person-appraisal prediction folder was provided or "
                "found beside counterfactual_pred_root."
            ),
        }

    if args.dimension_source == "gold":
        selected_dimensions = [
            path.name
            for path in sorted(counterfactual_gold_root.iterdir())
            if path.is_dir()
        ]
    else:
        selected_dimensions = list(dim_to_statement.keys())

    if not selected_dimensions:
        raise ValueError(
            f"No dimension folders found under counterfactual_gold_root: {counterfactual_gold_root}"
        )

    results["meta"]["dimension_source"] = args.dimension_source
    results["meta"]["dimensions"] = selected_dimensions

    for dimension in selected_dimensions:
        gold_dim_dir = counterfactual_gold_root / dimension
        if not gold_dim_dir.exists() or not gold_dim_dir.is_dir():
            results["dimension"][dimension] = {
                "positive-level": {
                    "pearson": None,
                    "spearman": None,
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                },
                "negative-level": {
                    "pearson": None,
                    "spearman": None,
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                },
                "positive-labels": {
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "categories": {
                        label: {
                            "pearson": None,
                            "spearman": None,
                            "stats": _build_base_stats(gold_files=0, pred_files=0),
                            "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                        }
                        for label in positive_labels
                    },
                },
                "negative-labels": {
                    "stats": _build_base_stats(gold_files=0, pred_files=0),
                    "categories": {
                        label: {
                            "pearson": None,
                            "spearman": None,
                            "stats": _build_base_stats(gold_files=0, pred_files=0),
                            "warnings": [f"Missing human counterfactual folder: {gold_dim_dir}"],
                        }
                        for label in negative_labels
                    },
                },
            }
            continue

        dim_obj: Dict[str, Any] = {}

        for level_task in LEVEL_TASKS:
            pred_dim_dir = counterfactual_model_root / level_task / dimension
            dim_obj[level_task] = _evaluate_level_task_for_dimension(
                level_task,
                gold_dim_dir,
                pred_dim_dir,
                first_person_root,
                baseline_run_roots,
                first_person_appraisal_run_roots,
            )

        dim_obj["positive-labels"] = _evaluate_label_task_for_dimension(
            "positive-labels",
            positive_labels,
            gold_dim_dir,
            counterfactual_model_root / "positive-labels" / dimension,
            first_person_root,
            baseline_run_roots,
            first_person_appraisal_run_roots,
        )
        dim_obj["negative-labels"] = _evaluate_label_task_for_dimension(
            "negative-labels",
            negative_labels,
            gold_dim_dir,
            counterfactual_model_root / "negative-labels" / dimension,
            first_person_root,
            baseline_run_roots,
            first_person_appraisal_run_roots,
        )

        results["dimension"][dimension] = dim_obj

    output_path = resolve_output_path(requested_counterfactual_root, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
