#!/usr/bin/env python3
"""
Usage:
  python scripts/evaluate_counterfactual.py \
    --first_person_root data/first_person \
    --counterfactual_gold_root data/counterfactual \
    --baseline_root output/prefix_run/origin \
    --counterfactual_pred_root output/prefix_run/counterfactual \
    --prompt_path scripts/prompts/baseline_prompt.toml \
    --output_file results.json

Notes:
- For each appraisal dimension, this script computes correlation between:
    model_delta = mean(third_person_model_scores) - first_person_model_score
    human_delta = mean(third_person_human_scores) - first_person_human_score
- Pearson and Spearman are both reported per dimension.
- Both legacy rating-dimension folders and the appraisal-prefix protocol's
  core-dimension folders are supported.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # for python < 3.11


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
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


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def numeric_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


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


def resolve_baseline_appraisals_dir(model_root: Path) -> Path:
    explicit_candidates = (
        model_root / "appraisals",
        model_root / "origin" / "appraisals",
    )
    for candidate in explicit_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    if model_root.exists() and model_root.is_dir():
        if any(path.is_file() for path in model_root.glob("*.json")):
            return model_root

    nested_candidates: List[Path] = []
    if model_root.exists() and model_root.is_dir():
        for child in sorted(model_root.iterdir()):
            if child.is_dir() and (child / "appraisals").is_dir():
                nested_candidates.append(child / "appraisals")

    if not nested_candidates:
        raise FileNotFoundError(f"Cannot find baseline appraisals dir under: {model_root}")
    return nested_candidates[0]


def resolve_counterfactual_appraisals_dir(
    root: Path,
    intervention_dimensions: List[str],
) -> Path:
    candidates = (
        root / "counterfactual" / "appraisals",
        root / "appraisals",
        root,
    )
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any((candidate / dimension).is_dir() for dimension in intervention_dimensions):
            return candidate
    raise FileNotFoundError(
        f"Cannot find counterfactual appraisal folders under: {root}"
    )


def resolve_with_legacy_model(
    root: Path,
    model: str,
    resolver: Any,
    *resolver_args: Any,
) -> Path:
    try:
        return resolver(root, *resolver_args)
    except FileNotFoundError as direct_error:
        if model:
            legacy_root = root / model
            try:
                return resolver(legacy_root, *resolver_args)
            except FileNotFoundError:
                pass
        raise direct_error


def extract_model_appraisal_score(payload: Dict[str, Any], dimension: str) -> Optional[float]:
    appraisals = payload.get("appraisals")
    container = appraisals if isinstance(appraisals, dict) else payload
    entry = container.get(dimension)
    if not isinstance(entry, dict):
        return None
    return numeric_score(entry.get("score"))


def rating_dimensions_for_intervention(
    intervention_dimension: str,
    dimension_to_statement: Dict[str, Any],
) -> List[str]:
    if intervention_dimension in dimension_to_statement:
        return [intervention_dimension]
    prefix = f"{intervention_dimension}."
    return [
        dimension
        for dimension in dimension_to_statement
        if dimension.startswith(prefix)
    ]


def resolve_output_path(root: Path, output_file: str) -> Path:
    stripped = output_file.strip()
    if not stripped:
        raise ValueError("output_file cannot be empty")
    output = Path(stripped)
    if output.is_absolute():
        if output.suffix == "" or (output.exists() and output.is_dir()):
            return output / "results.json"
        return output
    return root / output


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual appraisal deltas")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help=(
            "Optional model metadata. Paths are treated as direct roots; the "
            "legacy root/model layout is used only as a fallback."
        ),
    )
    parser.add_argument("--first_person_root", type=str, default="data/first_person", help="First-person gold root")
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
        help="Baseline prediction root",
    )
    parser.add_argument(
        "--counterfactual_pred_root",
        type=str,
        default="output/first_person/counterfactual",
        help="Counterfactual prediction root",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="scripts/prompts/counterfactual_prompt.toml",
        help="Prompt TOML for dimension/label mapping",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results.json",
        help="Output filename under counterfactual_pred_root, or an absolute path",
    )
    parser.add_argument(
        "--dimension_source",
        choices=["gold", "prompt"],
        default="gold",
        help=(
            "gold evaluates actual intervention folders (including five core "
            "dimensions); prompt preserves the legacy 22-folder behavior."
        ),
    )
    return parser


def evaluate_rating_dimension(
    *,
    intervention_dimension: str,
    rating_dimension: str,
    statement: str,
    first_person_root: Path,
    gold_dimension_dir: Path,
    prediction_dimension_dir: Path,
    baseline_appraisals_dir: Path,
    label_map: Dict[str, Any],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "gold_files": 0,
        "pred_files": 0,
        "evaluated_samples": 0,
        "prediction_count": 0,
        "human_counterfactual_count": 0,
        "skipped_missing_first_person": 0,
        "skipped_missing_baseline": 0,
        "skipped_missing_counterfactual_pred_file": 0,
        "skipped_invalid_first_person_human": 0,
        "skipped_invalid_baseline": 0,
        "skipped_invalid_human_counterfactual": 0,
        "skipped_invalid_model_counterfactual": 0,
    }
    model_deltas: List[float] = []
    human_deltas: List[float] = []

    if not gold_dimension_dir.exists() or not gold_dimension_dir.is_dir():
        return {
            "intervention_dimension": intervention_dimension,
            "statement": statement,
            "pearson": None,
            "spearman": None,
            "stats": stats,
            "warnings": [
                f"Missing human counterfactual folder: {gold_dimension_dir}"
            ],
        }

    gold_files = sorted(
        path for path in gold_dimension_dir.glob("*.json") if path.is_file()
    )
    stats["gold_files"] = len(gold_files)
    if prediction_dimension_dir.exists() and prediction_dimension_dir.is_dir():
        stats["pred_files"] = len(
            [
                path
                for path in prediction_dimension_dir.glob("*.json")
                if path.is_file()
            ]
        )

    for gold_file in gold_files:
        sample_name = gold_file.name
        first_person_file = first_person_root / sample_name
        baseline_file = baseline_appraisals_dir / sample_name
        prediction_file = prediction_dimension_dir / sample_name

        if not first_person_file.is_file():
            stats["skipped_missing_first_person"] += 1
            continue
        if not baseline_file.is_file():
            stats["skipped_missing_baseline"] += 1
            continue
        if not prediction_file.is_file():
            stats["skipped_missing_counterfactual_pred_file"] += 1
            continue

        first_person_payload = load_json(first_person_file)
        baseline_payload = load_json(baseline_file)
        gold_payload = load_json(gold_file)
        prediction_payload = load_json(prediction_file)

        if not isinstance(first_person_payload, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        if not isinstance(baseline_payload, dict):
            stats["skipped_invalid_baseline"] += 1
            continue
        if not isinstance(gold_payload, list) or not gold_payload:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue
        if not isinstance(prediction_payload, list) or not prediction_payload:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        first_person_ratings = first_person_payload.get("appraisal_ratings")
        if not isinstance(first_person_ratings, dict):
            stats["skipped_invalid_first_person_human"] += 1
            continue
        first_person_label = first_person_ratings.get(statement)
        if not isinstance(first_person_label, str) or first_person_label not in label_map:
            stats["skipped_invalid_first_person_human"] += 1
            continue
        first_person_human_score = float(label_map[first_person_label])

        first_person_model_score = extract_model_appraisal_score(
            baseline_payload, rating_dimension
        )
        if first_person_model_score is None:
            stats["skipped_invalid_baseline"] += 1
            continue

        human_third_scores: List[float] = []
        for item in gold_payload:
            if not isinstance(item, dict):
                continue
            ratings = item.get("appraisal_ratings")
            if not isinstance(ratings, dict):
                continue
            label = ratings.get(statement)
            if isinstance(label, str) and label in label_map:
                human_third_scores.append(float(label_map[label]))
        if not human_third_scores:
            stats["skipped_invalid_human_counterfactual"] += 1
            continue

        model_third_scores = [
            score
            for item in prediction_payload
            if isinstance(item, dict)
            for score in [extract_model_appraisal_score(item, rating_dimension)]
            if score is not None
        ]
        if not model_third_scores:
            stats["skipped_invalid_model_counterfactual"] += 1
            continue

        human_third_mean = safe_mean(human_third_scores)
        model_third_mean = safe_mean(model_third_scores)
        if human_third_mean is None or model_third_mean is None:
            continue

        human_deltas.append(human_third_mean - first_person_human_score)
        model_deltas.append(model_third_mean - first_person_model_score)
        stats["human_counterfactual_count"] += len(human_third_scores)
        stats["prediction_count"] += len(model_third_scores)
        stats["evaluated_samples"] += 1

    pearson_value, pearson_reason = pearson_corr(model_deltas, human_deltas)
    spearman_value, spearman_reason = spearman_corr(model_deltas, human_deltas)
    warnings: List[str] = []
    if pearson_reason is not None:
        warnings.append(f"Pearson unavailable: {pearson_reason}")
    if spearman_reason is not None:
        warnings.append(f"Spearman unavailable: {spearman_reason}")

    return {
        "intervention_dimension": intervention_dimension,
        "statement": statement,
        "pearson": pearson_value,
        "spearman": spearman_value,
        "stats": stats,
        "warnings": warnings,
    }


def main() -> int:
    args = build_argument_parser().parse_args()

    first_person_root = Path(args.first_person_root)
    counterfactual_gold_root = Path(args.counterfactual_gold_root)
    requested_baseline_root = Path(args.baseline_root)
    requested_prediction_root = Path(args.counterfactual_pred_root)
    prompt_path = Path(args.prompt_path)

    if not first_person_root.is_dir():
        raise FileNotFoundError(f"first_person_root not found: {first_person_root}")
    if not counterfactual_gold_root.is_dir():
        raise FileNotFoundError(
            f"counterfactual_gold_root not found: {counterfactual_gold_root}"
        )
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    prompt_cfg = load_toml(prompt_path)
    dim_to_statement = prompt_cfg.get("appraisals", {}).get(
        "dimension_to_statement", {}
    )
    if not isinstance(dim_to_statement, dict) or not dim_to_statement:
        raise ValueError("Missing [appraisals.dimension_to_statement] in prompt TOML")
    label_map = prompt_cfg.get("label_maps", {}).get("appraisals", {})
    if not isinstance(label_map, dict) or not label_map:
        raise ValueError("Missing [label_maps.appraisals] in prompt TOML")

    if args.dimension_source == "gold":
        intervention_dimensions = sorted(
            path.name
            for path in counterfactual_gold_root.iterdir()
            if path.is_dir()
        )
    else:
        intervention_dimensions = list(dim_to_statement)
    if not intervention_dimensions:
        raise ValueError(
            f"No intervention dimensions found under: {counterfactual_gold_root}"
        )

    baseline_appraisals_dir = resolve_with_legacy_model(
        requested_baseline_root,
        args.model,
        resolve_baseline_appraisals_dir,
    )
    counterfactual_appraisals_dir = resolve_with_legacy_model(
        requested_prediction_root,
        args.model,
        resolve_counterfactual_appraisals_dir,
        intervention_dimensions,
    )

    results: Dict[str, Any] = {
        "meta": {
            "model": args.model or None,
            "dimension_source": args.dimension_source,
            "first_person_root": str(first_person_root),
            "counterfactual_gold_root": str(counterfactual_gold_root),
            "baseline_appraisals_dir": str(baseline_appraisals_dir),
            "counterfactual_appraisals_dir": str(
                counterfactual_appraisals_dir
            ),
            "prompt_path": str(prompt_path),
        },
        "dimension": {},
        "intervention_dimension": {},
    }

    for intervention_dimension in intervention_dimensions:
        rating_dimensions = rating_dimensions_for_intervention(
            intervention_dimension, dim_to_statement
        )
        intervention_results: Dict[str, Any] = {
            "rating_dimensions": rating_dimensions,
            "ratings": {},
            "macro": {
                "pearson": None,
                "spearman": None,
                "rating_count": len(rating_dimensions),
                "evaluated_rating_count": 0,
            },
            "warnings": [],
        }
        if not rating_dimensions:
            intervention_results["warnings"].append(
                "No appraisal rating dimensions map to this intervention folder"
            )
            results["intervention_dimension"][
                intervention_dimension
            ] = intervention_results
            continue

        pearsons: List[float] = []
        spearmans: List[float] = []
        for rating_dimension in rating_dimensions:
            statement = dim_to_statement.get(rating_dimension)
            if not isinstance(statement, str) or not statement.strip():
                continue
            metric = evaluate_rating_dimension(
                intervention_dimension=intervention_dimension,
                rating_dimension=rating_dimension,
                statement=statement,
                first_person_root=first_person_root,
                gold_dimension_dir=(
                    counterfactual_gold_root / intervention_dimension
                ),
                prediction_dimension_dir=(
                    counterfactual_appraisals_dir / intervention_dimension
                ),
                baseline_appraisals_dir=baseline_appraisals_dir,
                label_map=label_map,
            )
            results["dimension"][rating_dimension] = metric
            intervention_results["ratings"][rating_dimension] = metric
            if metric["stats"]["evaluated_samples"] > 0:
                intervention_results["macro"]["evaluated_rating_count"] += 1
            if metric["pearson"] is not None:
                pearsons.append(float(metric["pearson"]))
            if metric["spearman"] is not None:
                spearmans.append(float(metric["spearman"]))

        intervention_results["macro"]["pearson"] = safe_mean(pearsons)
        intervention_results["macro"]["spearman"] = safe_mean(spearmans)
        results["intervention_dimension"][
            intervention_dimension
        ] = intervention_results

    output_root = requested_prediction_root
    legacy_output_root = requested_prediction_root / args.model if args.model else None
    if (
        legacy_output_root is not None
        and legacy_output_root.exists()
        and counterfactual_appraisals_dir.is_relative_to(legacy_output_root)
    ):
        output_root = legacy_output_root
    output_path = resolve_output_path(output_root, args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
