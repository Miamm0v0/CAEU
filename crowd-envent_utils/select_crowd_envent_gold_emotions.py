#!/usr/bin/env python3
"""Select crowd-enVent events by the author's gold emotion label."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "crowd-envent_test.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "crowd-envent_test_selected.json"
SELECTED_LABELS = (
    "joy",
    "pride",
    "anger",
    "sadness",
    "fear",
    "shame",
    "guilt",
)


def _gold_label(sample: dict[str, Any]) -> str:
    try:
        label = sample["reference"]["emotion"]["label"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Sample {sample.get('id', '<unknown>')} has no gold emotion") from exc
    if not isinstance(label, str):
        raise ValueError(f"Sample {sample.get('id', '<unknown>')} has an invalid gold emotion")
    return label.strip().lower()


def select_samples(input_file: Path, output_file: Path) -> None:
    with input_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"{input_file} must contain a JSON object with a samples array")

    samples = payload["samples"]
    selected = [sample for sample in samples if _gold_label(sample) in SELECTED_LABELS]
    distribution = Counter(_gold_label(sample) for sample in selected)

    output = dict(payload)
    output["samples"] = selected
    output["selection"] = {
        "field": "reference.emotion.label",
        "labels": list(SELECTED_LABELS),
        "source_samples": len(samples),
    }
    stats = dict(payload.get("stats", {}))
    stats.update(
        {
            "samples": len(selected),
            "unique_authors": len(selected),
            "validated_samples": sum(
                isinstance(sample.get("validation_reference"), dict)
                for sample in selected
            ),
            "confirmed_real_samples": sum(
                sample.get("metadata", {}).get("confirmed_real") is True
                for sample in selected
            ),
            "emotion_distribution": {
                label: distribution.get(label, 0) for label in SELECTED_LABELS
            },
        }
    )
    output["stats"] = stats

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Selected {len(selected)}/{len(samples)} samples -> {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_file", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    select_samples(args.input_file, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
