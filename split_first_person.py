#!/usr/bin/env python3
"""Split first_person_organized.json into train/dev/test JSON files.

Examples:
    python split_first_person.py
    python split_first_person.py --train-ratio 0.8 --dev-ratio 0.1 --seed 42
    python split_first_person.py data.json --output-dir splits

The test ratio is calculated as: 1 - train_ratio - dev_ratio.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly split organized first-person data into train/dev/test sets."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("first_person_organized.json"),
        help="input JSON array (default: first_person_organized.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory for split files (default: current directory)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="training-set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="development-set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for reproducible shuffling (default: 42)",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, dev_ratio: float) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0 <= dev_ratio < 1:
        raise ValueError("--dev-ratio must be between 0 and 1")
    if train_ratio + dev_ratio >= 1:
        raise ValueError(
            "--train-ratio + --dev-ratio must be less than 1 "
            "so that the test set is not empty"
        )


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    if not isinstance(records, list):
        raise ValueError("the top level of the input JSON must be an array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every item in the input JSON must be an object")
    if len(records) < 3:
        raise ValueError("at least 3 records are required for train/dev/test splitting")

    ids = [record.get("id") for record in records]
    if any(record_id is None for record_id in ids):
        raise ValueError("every record must contain an id field")
    if len(ids) != len(set(ids)):
        raise ValueError("record ids must be unique")

    return records


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = records.copy()
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_size = int(total * train_ratio)
    dev_size = int(total * dev_ratio)

    # Keep every split non-empty even for small valid datasets.
    train_size = max(1, train_size)
    dev_size = max(1, dev_size)
    if train_size + dev_size >= total:
        dev_size = 1
        train_size = total - 2

    train_end = train_size
    dev_end = train_end + dev_size

    train = shuffled[:train_end]
    dev = shuffled[train_end:dev_end]
    test = shuffled[dev_end:]
    return train, dev, test


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(records, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.dev_ratio)
    records = load_records(args.input)

    train, dev, test = split_records(
        records,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "dev.json", dev)
    write_json(args.output_dir / "test.json", test)

    print(
        f"Split {len(records)} records: "
        f"train={len(train)}, dev={len(dev)}, test={len(test)}"
    )
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
