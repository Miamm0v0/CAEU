#!/usr/bin/env python3
"""Convert organized first-person records to chosen-chain records.

Input record:
    {
      "id": "...",
      "perspective": "first_person",
      "situation": "...",
      "appraisal_reasoning": {...},
      "appraisal_ratings": {...},
      "emotion": {...}
    }

Output record:
    {
      "situation": "...",
      "chain": {
        "appraisal_reasoning": {...},
        "appraisal_ratings": {...},
        "emotion": {...}
      },
      "label": "chosen"
    }

Examples:
    python build_chosen_chains.py
    python build_chosen_chains.py train.json -o train_chosen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHAIN_FIELDS = (
    "appraisal_reasoning",
    "appraisal_ratings",
    "emotion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chosen appraisal-chain records from organized data."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("first_person_organized.json"),
        help="input JSON array (default: first_person_organized.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("first_person_chosen.json"),
        help="output JSON file (default: first_person_chosen.json)",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    if not isinstance(records, list):
        raise ValueError("the top level of the input JSON must be an array")
    return records


def convert_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"record {index} must be an object")

    required_fields = ("situation", *CHAIN_FIELDS)
    missing_fields = [field for field in required_fields if field not in record]
    if missing_fields:
        record_id = record.get("id", index)
        raise ValueError(
            f"record {record_id!r} is missing fields: {', '.join(missing_fields)}"
        )

    if not isinstance(record["situation"], str):
        raise ValueError(f"record {index}: situation must be a string")
    if not isinstance(record["appraisal_reasoning"], dict):
        raise ValueError(f"record {index}: appraisal_reasoning must be an object")
    if not isinstance(record["appraisal_ratings"], dict):
        raise ValueError(f"record {index}: appraisal_ratings must be an object")
    if not isinstance(record["emotion"], dict):
        raise ValueError(f"record {index}: emotion must be an object")

    return {
        "situation": record["situation"],
        "chain": {
            field: record[field]
            for field in CHAIN_FIELDS
        },
        "label": "chosen",
    }


def main() -> None:
    args = parse_args()
    source_records = load_records(args.input)
    output_records = [
        convert_record(record, index)
        for index, record in enumerate(source_records)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output_records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"Converted {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()
