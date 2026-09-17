#!/usr/bin/env python3
"""QC, aggregate, and measure agreement for blinded human annotations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_grpo.parsing import (  # noqa: E402
    aggregate_judgment,
    parse_judge_output,
)
from train_grpo.spec import (  # noqa: E402
    APPRAISAL_CRITERIA,
    COHERENCE_CRITERION,
    RUBRIC_VERSION,
    TRANSITION_CRITERION,
    resolve_appraisal_dimensions,
)


DEFAULT_DIMENSIONS = resolve_appraisal_dimensions(
    "relevance,epistemic,goal_congruence,agency_accountability,"
    "control_coping_potential"
)
SCORE_CATEGORIES = tuple(range(5))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate 2--3 human raters and compute agreement"
    )
    parser.add_argument("--master_file", type=Path, required=True)
    parser.add_argument(
        "--annotation_files", type=Path, nargs="+", required=True
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument("--min_raters", type=int, default=2)
    parser.add_argument(
        "--strict_qc",
        action="store_true",
        help="Fail instead of excluding invalid/duplicate/unknown rows",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row["_source_file"] = str(path)
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def rubric_item_names(dimensions: Sequence[str]) -> list[str]:
    return [
        f"appraisals.{dimension}.{criterion}"
        for dimension in dimensions
        for criterion in APPRAISAL_CRITERIA
    ] + [COHERENCE_CRITERION, TRANSITION_CRITERION]


def extract_scores(
    judgment: Mapping[str, Any], dimensions: Sequence[str]
) -> dict[str, int | float]:
    scores = {
        f"appraisals.{dimension}.{criterion}": judgment["appraisals"][
            dimension
        ][criterion]["score"]
        for dimension in dimensions
        for criterion in APPRAISAL_CRITERIA
    }
    scores[COHERENCE_CRITERION] = judgment["coherence"]["score"]
    scores[TRANSITION_CRITERION] = judgment["transition"]["score"]
    return scores


def score_item(
    judgment: Mapping[str, Any], item_name: str
) -> Mapping[str, Any]:
    if item_name == COHERENCE_CRITERION:
        return judgment["coherence"]
    if item_name == TRANSITION_CRITERION:
        return judgment["transition"]
    _, dimension, criterion = item_name.split(".", maxsplit=2)
    return judgment["appraisals"][dimension][criterion]


def krippendorff_alpha_ordinal(
    units: Mapping[str, Mapping[str, int]],
) -> float | None:
    """Krippendorff's alpha with the standard ordinal disagreement metric.

    Each unit maps annotator IDs to 0--4 scores. Missing raters are naturally
    supported; units with fewer than two valid ratings do not contribute.
    """
    coincidence = {
        left: {right: 0.0 for right in SCORE_CATEGORIES}
        for left in SCORE_CATEGORIES
    }
    total = 0.0
    for ratings in units.values():
        values = [int(value) for value in ratings.values()]
        count = len(values)
        if count < 2:
            continue
        total += count
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                if left_index != right_index:
                    coincidence[left][right] += 1.0 / (count - 1)
    if total < 2:
        return None
    marginals = {
        category: sum(coincidence[category].values())
        for category in SCORE_CATEGORIES
    }

    def ordinal_distance(left: int, right: int) -> float:
        if left == right:
            return 0.0
        low, high = sorted((left, right))
        interval_mass = sum(
            marginals[category] for category in range(low, high + 1)
        ) - (marginals[low] + marginals[high]) / 2.0
        return interval_mass * interval_mass

    observed = sum(
        coincidence[left][right] * ordinal_distance(left, right)
        for left in SCORE_CATEGORIES
        for right in SCORE_CATEGORIES
    ) / total
    expected = sum(
        marginals[left]
        * marginals[right]
        * ordinal_distance(left, right)
        for left in SCORE_CATEGORIES
        for right in SCORE_CATEGORIES
    ) / (total * (total - 1.0))
    if expected <= 0:
        return None
    return float(1.0 - observed / expected)


def quadratic_weighted_kappa(
    left: Sequence[int], right: Sequence[int]
) -> float | None:
    if len(left) != len(right):
        raise ValueError("kappa vectors must have equal length")
    if not left:
        return None
    observed = [[0.0] * 5 for _ in range(5)]
    left_counts = [0.0] * 5
    right_counts = [0.0] * 5
    for left_score, right_score in zip(left, right):
        observed[left_score][right_score] += 1.0
        left_counts[left_score] += 1.0
        right_counts[right_score] += 1.0
    count = float(len(left))

    def weight(a: int, b: int) -> float:
        return ((a - b) / 4.0) ** 2

    observed_disagreement = sum(
        weight(a, b) * observed[a][b]
        for a in SCORE_CATEGORIES
        for b in SCORE_CATEGORIES
    ) / count
    expected_disagreement = sum(
        weight(a, b) * left_counts[a] * right_counts[b]
        for a in SCORE_CATEGORIES
        for b in SCORE_CATEGORIES
    ) / (count * count)
    if expected_disagreement <= 0:
        return None
    return float(1.0 - observed_disagreement / expected_disagreement)


def pairwise_kappas(
    units: Mapping[str, Mapping[str, int]], annotator_ids: Sequence[str]
) -> dict[str, Any]:
    values: dict[str, float | None] = {}
    defined: list[float] = []
    for left_id, right_id in itertools.combinations(annotator_ids, 2):
        left_scores: list[int] = []
        right_scores: list[int] = []
        for ratings in units.values():
            if left_id in ratings and right_id in ratings:
                left_scores.append(ratings[left_id])
                right_scores.append(ratings[right_id])
        kappa = quadratic_weighted_kappa(left_scores, right_scores)
        values[f"{left_id}__{right_id}"] = kappa
        if kappa is not None:
            defined.append(kappa)
    return {
        "pairwise_quadratic_weighted_kappa": values,
        "mean_pairwise_quadratic_weighted_kappa": (
            statistics.mean(defined) if defined else None
        ),
    }


def consensus_judgment(
    annotations: Sequence[tuple[str, Mapping[str, Any]]],
    dimensions: Sequence[str],
) -> dict[str, Any]:
    def consensus_item(item_name: str) -> dict[str, Any]:
        values = [
            float(score_item(judgment, item_name)["score"])
            for _, judgment in annotations
        ]
        rationales = [
            f"[{annotator_id}] {score_item(judgment, item_name)['rationale']}"
            for annotator_id, judgment in annotations
        ]
        return {
            "score": statistics.mean(values),
            "rationale": " || ".join(rationales),
        }

    result = {
        "appraisals": {
            dimension: {
                criterion: consensus_item(
                    f"appraisals.{dimension}.{criterion}"
                )
                for criterion in APPRAISAL_CRITERIA
            }
            for dimension in dimensions
        },
        "coherence": consensus_item(COHERENCE_CRITERION),
        "transition": consensus_item(TRANSITION_CRITERION),
        "overall_feedback": " || ".join(
            f"[{annotator_id}] {judgment['overall_feedback']}"
            for annotator_id, judgment in annotations
        ),
    }
    return result


def mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(defined) if defined else None


def main() -> int:
    args = build_parser().parse_args()
    dimensions = list(args.appraisal_dimensions)
    if len(args.annotation_files) not in {2, 3}:
        raise ValueError("Provide exactly 2 or 3 --annotation_files")
    if args.min_raters < 2 or args.min_raters > len(args.annotation_files):
        raise ValueError("--min_raters must be between 2 and the file count")

    master_rows = read_jsonl(args.master_file)
    master = {str(row.get("candidate_id", "")): row for row in master_rows}
    if "" in master or len(master) != len(master_rows):
        raise ValueError("master candidate IDs must be non-empty and unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus_path = args.output_dir / "human_consensus.jsonl"
    agreement_path = args.output_dir / "human_agreement.json"
    qc_path = args.output_dir / "human_qc_report.json"
    if any(path.exists() for path in (consensus_path, agreement_path, qc_path)):
        if not args.overwrite:
            raise FileExistsError(
                "Human aggregation output exists; pass --overwrite"
            )

    annotations: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    annotator_ids: list[str] = []
    invalid_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    unknown_rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    for path in args.annotation_files:
        rows = read_jsonl(path)
        file_ids = {
            str(row.get("annotator_id", "")).strip()
            for row in rows
            if str(row.get("annotator_id", "")).strip()
        }
        if len(file_ids) != 1:
            raise ValueError(
                f"{path}: must contain exactly one non-empty annotator_id"
            )
        annotator_id = next(iter(file_ids))
        if annotator_id in annotator_ids:
            raise ValueError(f"duplicate annotator file for {annotator_id}")
        annotator_ids.append(annotator_id)
        accepted = 0
        for row in rows:
            candidate_id = str(row.get("candidate_id", ""))
            location = f"{row['_source_file']}:{row['_line_number']}"
            if candidate_id not in master:
                unknown_rows.append(
                    {"location": location, "candidate_id": candidate_id}
                )
                continue
            if annotator_id in annotations[candidate_id]:
                duplicate_rows.append(
                    {"location": location, "candidate_id": candidate_id}
                )
                continue
            try:
                judgment = parse_judge_output(
                    json.dumps(row.get("judgment"), ensure_ascii=False),
                    dimensions,
                )
            except (TypeError, ValueError) as exc:
                invalid_rows.append(
                    {
                        "location": location,
                        "candidate_id": candidate_id,
                        "annotator_id": annotator_id,
                        "error": str(exc),
                    }
                )
                continue
            annotations[candidate_id][annotator_id] = judgment
            accepted += 1
        file_summaries.append(
            {
                "file": str(path),
                "annotator_id": annotator_id,
                "row_count": len(rows),
                "accepted_count": accepted,
            }
        )

    qc_defects = len(invalid_rows) + len(duplicate_rows) + len(unknown_rows)
    if args.strict_qc and qc_defects:
        raise ValueError(
            f"Strict QC failed: {qc_defects} invalid/duplicate/unknown rows"
        )

    missing_by_annotator = {
        annotator_id: sorted(
            candidate_id
            for candidate_id in master
            if annotator_id not in annotations.get(candidate_id, {})
        )
        for annotator_id in annotator_ids
    }
    below_minimum = sorted(
        candidate_id
        for candidate_id in master
        if len(annotations.get(candidate_id, {})) < args.min_raters
    )

    consensus_rows: list[dict[str, Any]] = []
    for candidate_id, source in master.items():
        candidate_annotations = annotations.get(candidate_id, {})
        if len(candidate_annotations) < args.min_raters:
            continue
        ordered = [
            (annotator_id, candidate_annotations[annotator_id])
            for annotator_id in annotator_ids
            if annotator_id in candidate_annotations
        ]
        consensus = consensus_judgment(ordered, dimensions)
        rewards = aggregate_judgment(consensus, dimensions)
        consensus_rows.append(
            {
                "schema_version": "judge-human-consensus-v1",
                "rubric_version": RUBRIC_VERSION,
                "pair_id": source.get("pair_id"),
                "candidate_id": candidate_id,
                "candidate_slot": source.get("candidate_slot"),
                "source_sample_id": source.get("source_sample_id"),
                "annotator_ids": [item[0] for item in ordered],
                "annotator_count": len(ordered),
                "consensus_judgment": consensus,
                **rewards,
            }
        )
    write_jsonl(consensus_path, consensus_rows)

    item_units: dict[str, dict[str, dict[str, int]]] = {
        item_name: {} for item_name in rubric_item_names(dimensions)
    }
    pooled_units: dict[str, dict[str, int]] = {}
    for candidate_id, by_annotator in annotations.items():
        for item_name in item_units:
            ratings = {
                annotator_id: int(
                    extract_scores(judgment, dimensions)[item_name]
                )
                for annotator_id, judgment in by_annotator.items()
            }
            item_units[item_name][candidate_id] = ratings
            pooled_units[f"{candidate_id}::{item_name}"] = ratings

    by_item: dict[str, Any] = {}
    for item_name, units in item_units.items():
        by_item[item_name] = {
            "unit_count_with_two_or_more_raters": sum(
                len(ratings) >= 2 for ratings in units.values()
            ),
            "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(units),
            **pairwise_kappas(units, annotator_ids),
        }
    pooled_kappas = pairwise_kappas(pooled_units, annotator_ids)
    agreement = {
        "schema_version": "judge-human-agreement-v1",
        "rubric_version": RUBRIC_VERSION,
        "appraisal_dimensions": dimensions,
        "annotator_ids": annotator_ids,
        "criterion_scale": [0, 4],
        "by_rubric_item": by_item,
        "macro": {
            "mean_krippendorff_alpha_ordinal": mean_defined(
                value["krippendorff_alpha_ordinal"] for value in by_item.values()
            ),
            "mean_pairwise_quadratic_weighted_kappa": mean_defined(
                value["mean_pairwise_quadratic_weighted_kappa"]
                for value in by_item.values()
            ),
        },
        "pooled": {
            "krippendorff_alpha_ordinal": krippendorff_alpha_ordinal(
                pooled_units
            ),
            **pooled_kappas,
        },
    }
    write_json(agreement_path, agreement)
    write_json(
        qc_path,
        {
            "schema_version": "judge-human-qc-v1",
            "rubric_version": RUBRIC_VERSION,
            "master_candidate_count": len(master),
            "consensus_candidate_count": len(consensus_rows),
            "min_raters": args.min_raters,
            "file_summaries": file_summaries,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "unknown_candidate_rows": unknown_rows,
            "missing_by_annotator": missing_by_annotator,
            "candidates_below_minimum_raters": below_minimum,
        },
    )
    print(
        f"Aggregated {len(consensus_rows)}/{len(master)} candidates; "
        f"pooled alpha={agreement['pooled']['krippendorff_alpha_ordinal']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
