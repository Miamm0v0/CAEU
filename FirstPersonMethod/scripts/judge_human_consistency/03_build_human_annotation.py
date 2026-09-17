#!/usr/bin/env python3
"""Create blinded human-annotation packets from ``master_samples.jsonl``.

The instructions extract the scoring-rubric portion from the exact text
returned by the current training ``build_judge_system_prompt``. Consequently
the human task cannot silently drift away from the Judge rubric or its 0--4
anchors, while API-only rationale/JSON requirements are not imposed on people.
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

from train_grpo.spec import (  # noqa: E402
    RUBRIC_VERSION,
    build_judge_system_prompt,
    judge_output_example,
    resolve_appraisal_dimensions,
)


DEFAULT_DIMENSIONS = resolve_appraisal_dimensions(
    "relevance,epistemic,goal_congruence,agency_accountability,"
    "control_coping_potential"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build 2--3 blinded human annotation files"
    )
    parser.add_argument("--master_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--annotator_ids",
        type=str,
        default="annotator_1,annotator_2,annotator_3",
        help="Comma-separated identifiers for two or three annotators",
    )
    parser.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_DIMENSIONS),
    )
    parser.add_argument("--seed", type=int, default=42)
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
            rows.append(row)
    return rows


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


def parse_annotator_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if len(ids) not in {2, 3}:
        raise ValueError("--annotator_ids must contain exactly 2 or 3 IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("--annotator_ids contains duplicates")
    for annotator_id in ids:
        if any(character in annotator_id for character in "/\\"):
            raise ValueError("annotator IDs cannot contain path separators")
    return ids


def blank_judgment(dimensions: Sequence[str]) -> dict[str, Any]:
    # Derive every scoring field from the training template. Human annotators
    # provide the same 17 scores, but are not burdened with 17 rationales.
    template = judge_output_example(dimensions)
    for criteria in template["appraisals"].values():
        for item in criteria.values():
            item["score"] = None
            item.pop("rationale", None)
    for section in ("coherence", "transition"):
        template[section]["score"] = None
        template[section].pop("rationale", None)
    template.pop("overall_feedback", None)
    template["overall_notes"] = ""
    return template


def shuffled_for_annotator(
    rows: Sequence[dict[str, Any]], annotator_id: str, seed: int
) -> list[dict[str, Any]]:
    copied = list(rows)
    digest = hashlib.sha256(
        f"human-annotation-order|{seed}|{annotator_id}".encode("utf-8")
    ).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(copied)
    return copied


def main() -> int:
    args = build_parser().parse_args()
    dimensions = list(args.appraisal_dimensions)
    annotator_ids = parse_annotator_ids(args.annotator_ids)
    master_rows = read_jsonl(args.master_file)
    if not master_rows:
        raise ValueError("master file is empty")
    candidate_ids = [str(row.get("candidate_id", "")) for row in master_rows]
    if any(not value for value in candidate_ids) or len(candidate_ids) != len(
        set(candidate_ids)
    ):
        raise ValueError("master candidate_id values must be non-empty and unique")
    reference_by_candidate: dict[str, dict[str, Any]] = {}
    for row in master_rows:
        raw_reference = row.get("reference_json")
        if not isinstance(raw_reference, str) or not raw_reference.strip():
            raise ValueError(
                "master contains no reference_json; rerun 01_prepare_samples.py"
            )
        try:
            reference = json.loads(raw_reference)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{row['candidate_id']}: invalid reference_json: {exc.msg}"
            ) from exc
        if not isinstance(reference, dict):
            raise ValueError(
                f"{row['candidate_id']}: reference_json must be an object"
            )
        reference_by_candidate[row["candidate_id"]] = reference

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        args.output_dir / f"annotations_{annotator_id}.jsonl"
        for annotator_id in annotator_ids
    ]
    instructions_path = args.output_dir / "annotation_instructions.md"
    manifest_path = args.output_dir / "annotation_manifest.json"
    existing = [
        path
        for path in [*output_paths, instructions_path, manifest_path]
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite"
        )

    exact_prompt = build_judge_system_prompt(dimensions)
    prompt_sha = hashlib.sha256(exact_prompt.encode("utf-8")).hexdigest()
    rubric_text, separator, _ = exact_prompt.partition("OUTPUT FORMAT IS STRICT:")
    if not separator:
        raise RuntimeError(
            "Could not isolate the scoring rubric from the training Judge prompt"
        )
    instructions = f"""# Human annotation instructions

This packet is blinded. Do not try to infer model identity, and do not consult
the private `identity_key.json`. Score the displayed situation, human appraisal
reference, and candidate. This is the same semantic evidence supplied to the
training API Judge. The internal gold emotion is not shown because the API
Judge does not receive it either.

Fill all 17 `judgment.*.score` fields with integers from 0 to 4. You do **not**
need to write a rationale for every score. `overall_notes` is optional and may
be left empty. Do not change IDs, situations, references, candidates, key
names, or row order.

The following is the **verbatim scoring-rubric and anchor portion** of the
current GRPO training Judge prompt (`rubric_version={RUBRIC_VERSION}`, full
prompt SHA-256 `{prompt_sha}`). The automated JSON/rationale output contract is
intentionally omitted; only its human-irrelevant formatting requirement differs.

```text
{rubric_text.strip()}
```
"""
    instructions_path.write_text(instructions, encoding="utf-8")

    for annotator_id, output_path in zip(annotator_ids, output_paths):
        packet: list[dict[str, Any]] = []
        for row in shuffled_for_annotator(master_rows, annotator_id, args.seed):
            reference = reference_by_candidate[row["candidate_id"]]
            appraisal_reference = reference.get(
                "legacy_human_appraisal_reasoning"
            )
            packet.append(
                {
                    "schema_version": "judge-human-annotation-v2",
                    "rubric_version": RUBRIC_VERSION,
                    "rubric_prompt_sha256": prompt_sha,
                    "annotator_id": annotator_id,
                    "annotation_id": row["candidate_id"],
                    "pair_id": row.get("pair_id"),
                    "candidate_id": row["candidate_id"],
                    "candidate_slot": row.get("candidate_slot"),
                    "source_sample_id": row.get("source_sample_id"),
                    "situation": row.get("situation"),
                    "optional_human_appraisal_reference": (
                        appraisal_reference
                        if isinstance(appraisal_reference, dict)
                        else None
                    ),
                    "candidate": row.get("candidate"),
                    "judgment": blank_judgment(dimensions),
                }
            )
        write_jsonl(output_path, packet)

    write_json(
        manifest_path,
        {
            "schema_version": "judge-human-annotation-manifest-v2",
            "rubric_version": RUBRIC_VERSION,
            "rubric_prompt_sha256": prompt_sha,
            "appraisal_dimensions": dimensions,
            "annotator_ids": annotator_ids,
            "candidate_count_per_annotator": len(master_rows),
            "assignment_mode": "all_annotators_rate_all_candidates",
            "model_identity_included": False,
            "appraisal_reference_included": True,
            "appraisal_reference_candidate_count": sum(
                isinstance(
                    reference.get("legacy_human_appraisal_reasoning"), dict
                )
                for reference in reference_by_candidate.values()
            ),
            "gold_emotion_included": False,
            "rationales_required": False,
            "required_score_count_per_candidate": (
                len(dimensions) * 3 + 2
            ),
            "annotation_files": [str(path.resolve()) for path in output_paths],
            "instructions_file": str(instructions_path.resolve()),
        },
    )
    print(
        f"Built {len(output_paths)} blinded annotation packets with "
        f"{len(master_rows)} candidates each in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
