#!/usr/bin/env python3
"""Build a blinded, paired Base/CAEU sample set for Judge validation.

Candidate outputs are validated by the GRPO ``parse_policy_output`` function.
The hidden Judge reference is produced from the organized CAREBench test split
by the same ``normalize_policy_record``/``build_hidden_reference`` path used in
GRPO.  The human packets expose the same appraisal reference seen by the API
Judge, but never expose the internal gold emotion that the API Judge also does
not receive in its request message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_grpo.parsing import parse_policy_output  # noqa: E402
from train_grpo.data import load_records, normalize_policy_record  # noqa: E402
from train_grpo.spec import (  # noqa: E402
    CAREBENCH_REASONING_KEYS,
    RUBRIC_VERSION,
    resolve_appraisal_dimensions,
)


DEFAULT_DIMENSIONS = resolve_appraisal_dimensions(
    "relevance,epistemic,goal_congruence,agency_accountability,"
    "control_coping_potential"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare paired, blinded Base/CAEU CAREBench candidates"
    )
    parser.add_argument("--source_folder", type=Path, required=True)
    parser.add_argument(
        "--reference_file",
        type=Path,
        required=True,
        help=(
            "Organized CAREBench test JSON/JSONL used by GRPO, containing id, "
            "situation, appraisal_reasoning, and emotion"
        ),
    )
    parser.add_argument("--base_pred_folder", type=Path, required=True)
    parser.add_argument("--caeu_pred_folder", type=Path, required=True)
    parser.add_argument(
        "--base_prediction_task", type=str, default="chain-emotion"
    )
    parser.add_argument(
        "--caeu_prediction_task", type=str, default="chain-emotion"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of paired CAREBench situations",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judge_reference_mode",
        choices=["none", "available", "required"],
        default="available",
        help="Same semantics and default as train_grpo/config.py",
    )
    parser.add_argument(
        "--invalid_policy",
        choices=["skip", "error"],
        default="skip",
        help="Skip an entire pair or fail when either candidate is invalid",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_situations(source_folder: Path) -> list[dict[str, str]]:
    if not source_folder.is_dir():
        raise FileNotFoundError(f"CAREBench folder not found: {source_folder}")
    rows: list[dict[str, str]] = []
    for path in sorted(source_folder.glob("*.json")):
        payload = read_json_object(path)
        story = payload.get("story_collection")
        situation = (
            story.get("final_scenario") if isinstance(story, dict) else None
        )
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError(
                f"{path}: story_collection.final_scenario must be non-empty"
            )
        rows.append({"sample_id": path.stem, "situation": situation.strip()})
    if not rows:
        raise ValueError(f"No JSON samples found in {source_folder}")
    return rows


def load_grpo_reference_examples(
    reference_file: Path,
    dimensions: list[str],
    reference_mode: str,
) -> dict[str, dict[str, Any]]:
    """Run the exact GRPO record normalization/reference construction path."""
    records = load_records(str(reference_file))
    examples: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        example = normalize_policy_record(
            record,
            index,
            reference_mode,
            dimensions,
        )
        sample_id = example["sample_id"]
        if sample_id in examples:
            raise ValueError(
                f"{reference_file}: duplicate sample id {sample_id}"
            )
        examples[sample_id] = example
    return examples


def prediction_path(root: Path, task: str, sample_id: str) -> Path:
    nested = root / task / f"{sample_id}.json"
    direct = root / f"{sample_id}.json"
    if nested.is_file():
        return nested
    if direct.is_file():
        return direct
    raise FileNotFoundError(
        f"missing both {nested} and direct-file fallback {direct}"
    )


def normalize_candidate(
    payload: Mapping[str, Any], dimensions: Sequence[str]
) -> dict[str, Any]:
    raw_reasoning = payload.get("appraisal_reasoning")
    if not isinstance(raw_reasoning, dict):
        raise ValueError("appraisal_reasoning must be an object")
    reasoning: dict[str, Any] = {}
    for dimension in dimensions:
        legacy_key = CAREBENCH_REASONING_KEYS.get(dimension, dimension)
        reasoning[dimension] = raw_reasoning.get(
            dimension, raw_reasoning.get(legacy_key)
        )
    minimal = {
        "appraisal_reasoning": reasoning,
        "emotion": payload.get("emotion"),
    }
    return parse_policy_output(
        json.dumps(minimal, ensure_ascii=False), dimensions
    )


def stable_digest(*parts: Any, length: int = 20) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def select_pairs(
    pairs: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    if maximum is None:
        return pairs
    if maximum <= 0:
        raise ValueError("--max_samples must be positive")
    ordered = sorted(
        pairs,
        key=lambda row: stable_digest(
            "judge-human-sample", seed, row["source_sample_id"], length=64
        ),
    )
    return ordered[:maximum]


def main() -> int:
    args = build_parser().parse_args()
    dimensions = list(args.appraisal_dimensions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    master_path = args.output_dir / "master_samples.jsonl"
    key_path = args.output_dir / "identity_key.json"
    report_path = args.output_dir / "prepare_report.json"
    existing = [path for path in (master_path, key_path, report_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite"
        )

    valid_pairs: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    situations = load_situations(args.source_folder)
    reference_examples = load_grpo_reference_examples(
        args.reference_file,
        dimensions,
        args.judge_reference_mode,
    )
    systems = (
        ("base", args.base_pred_folder, args.base_prediction_task),
        ("caeu", args.caeu_pred_folder, args.caeu_prediction_task),
    )
    for source in situations:
        loaded: dict[str, dict[str, Any]] = {}
        pair_errors: list[str] = []
        reference_example = reference_examples.get(source["sample_id"])
        if reference_example is None:
            pair_errors.append(
                "reference: sample is missing from organized --reference_file"
            )
        elif reference_example["situation"] != source["situation"]:
            pair_errors.append(
                "reference: situation differs between raw and organized test data"
            )
        for system, root, task in systems:
            try:
                path = prediction_path(root, task, source["sample_id"])
                loaded[system] = {
                    "candidate": normalize_candidate(
                        read_json_object(path), dimensions
                    ),
                    "source_path": str(path.resolve()),
                    "prediction_task": task,
                }
            except (FileNotFoundError, TypeError, ValueError) as exc:
                pair_errors.append(f"{system}: {exc}")
        if pair_errors:
            rejected.append(
                {
                    "source_sample_id": source["sample_id"],
                    "error": "; ".join(pair_errors),
                }
            )
            if args.invalid_policy == "error":
                raise ValueError(
                    f"{source['sample_id']}: " + "; ".join(pair_errors)
                )
            continue
        valid_pairs.append(
            {
                "source_sample_id": source["sample_id"],
                "situation": source["situation"],
                "reference_json": reference_example["reference_json"],
                "systems": loaded,
            }
        )

    selected = select_pairs(valid_pairs, args.max_samples, args.seed)
    master_rows: list[dict[str, Any]] = []
    identity_pairs: list[dict[str, Any]] = []
    for pair in selected:
        sample_id = pair["source_sample_id"]
        pair_id = "pair_" + stable_digest(
            "judge-human-pair-v1", args.seed, sample_id
        )
        order = ["base", "caeu"]
        random.Random(
            int(stable_digest("blind-slot", args.seed, sample_id, length=16), 16)
        ).shuffle(order)
        candidates: dict[str, Any] = {}
        for slot, system in zip(("A", "B"), order):
            candidate_id = "candidate_" + stable_digest(
                "judge-human-candidate-v1", args.seed, sample_id, slot
            )
            master_rows.append(
                {
                    "schema_version": "judge-human-master-v2",
                    "rubric_version": RUBRIC_VERSION,
                    "pair_id": pair_id,
                    "candidate_id": candidate_id,
                    "candidate_slot": slot,
                    "source_sample_id": sample_id,
                    "situation": pair["situation"],
                    "candidate": pair["systems"][system]["candidate"],
                    # 03 exposes only legacy_human_appraisal_reasoning, which
                    # is exactly what APIJudge.request_messages() exposes. It
                    # never copies gold_emotion into annotator packets.
                    "reference_json": pair["reference_json"],
                }
            )
            candidates[slot] = {
                "candidate_id": candidate_id,
                "system": system,
                "prediction_task": pair["systems"][system]["prediction_task"],
                "source_path": pair["systems"][system]["source_path"],
            }
        identity_pairs.append(
            {
                "pair_id": pair_id,
                "source_sample_id": sample_id,
                "candidates": candidates,
            }
        )

    master_rows.sort(
        key=lambda row: stable_digest(
            "master-row-order", args.seed, row["candidate_id"], length=64
        )
    )
    report = {
        "rubric_version": RUBRIC_VERSION,
        "appraisal_dimensions": dimensions,
        "source_situation_count": len(situations),
        "complete_valid_pair_count": len(valid_pairs),
        "selected_pair_count": len(selected),
        "candidate_count": len(master_rows),
        "rejected_pair_count": len(rejected),
        "rejected_pairs": rejected,
        "reference_file": str(args.reference_file.resolve()),
        "judge_reference_mode": args.judge_reference_mode,
        "reference_record_count": len(reference_examples),
        "selected_with_appraisal_reference_count": sum(
            "legacy_human_appraisal_reasoning"
            in json.loads(pair["reference_json"])
            for pair in selected
        ),
        "selected_with_gold_emotion_count": sum(
            "gold_emotion" in json.loads(pair["reference_json"])
            for pair in selected
        ),
        "gold_annotations_read_for_grpo_reference": True,
        "appraisal_reference_available_to_human_packet_builder": True,
        "gold_emotion_exposed_to_api_or_human_judge": False,
        "master_file": str(master_path.resolve()),
        "private_identity_key": str(key_path.resolve()),
    }
    if not master_rows:
        write_json(report_path, report)
        raise ValueError(
            "No complete valid Base/CAEU pairs were available; see "
            f"{report_path} for per-pair errors"
        )
    write_jsonl(master_path, master_rows)
    write_json(
        key_path,
        {
            "schema_version": "judge-human-identity-key-v1",
            "rubric_version": RUBRIC_VERSION,
            "appraisal_dimensions": dimensions,
            "seed": args.seed,
            "warning": (
                "PRIVATE UNBLINDING KEY. Do not provide this file to annotators."
            ),
            "pairs": identity_pairs,
        },
    )
    write_json(report_path, report)
    print(
        f"Prepared {len(selected)} blinded pairs ({len(master_rows)} candidates) "
        f"in {master_path}"
    )
    print(f"Keep the unblinding key private: {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
