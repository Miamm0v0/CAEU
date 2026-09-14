"""Lightweight CLI orchestration; heavy ML libraries load only for training."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .config import build_parser
from .data import prepare_data
from .spec import judge_output_schema, policy_output_example
from .trainer import run_training


def print_preview(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    summary: dict[str, Any],
    count: int,
    appraisal_dimensions: list[str],
) -> None:
    print(
        "[preview] configurable-appraisal API-Judge GRPO data "
        f"dimensions={','.join(appraisal_dimensions)}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n===== POLICY OUTPUT EXAMPLE =====")
    print(
        json.dumps(
            policy_output_example(appraisal_dimensions),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n===== JUDGE RUBRIC OUTPUT SCHEMA =====")
    print(
        json.dumps(
            judge_output_schema(appraisal_dimensions),
            ensure_ascii=False,
            indent=2,
        )
    )
    for index, example in enumerate(train_examples[: max(0, count)]):
        visible = {
            "sample_id": example["sample_id"],
            "situation": example["situation"],
            "prompt": example["prompt"],
        }
        print(f"\n===== TRAIN POLICY SAMPLE {index} =====")
        print(json.dumps(visible, ensure_ascii=False, indent=2))
        print("\n===== HIDDEN REWARD REFERENCE (GOLD OUTCOME + OPTIONAL JUDGE APPRAISALS) =====")
        print(
            json.dumps(
                json.loads(example["reference_json"]),
                ensure_ascii=False,
                indent=2,
            )
        )
    if eval_examples and count > 0:
        print("\n===== EVAL POLICY SAMPLE 0 =====")
        print(
            json.dumps(
                {
                    "sample_id": eval_examples[0]["sample_id"],
                    "situation": eval_examples[0]["situation"],
                    "prompt": eval_examples[0]["prompt"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_output_schema:
        print(
            json.dumps(
                policy_output_example(args.appraisal_dimensions),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.print_judge_schema:
        print(
            json.dumps(
                judge_output_schema(args.appraisal_dimensions),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    train_examples, eval_examples, summary = prepare_data(args)
    if args.preview_only:
        print_preview(
            train_examples,
            eval_examples,
            summary,
            args.preview_samples,
            args.appraisal_dimensions,
        )
        return 0
    run_training(args, train_examples, eval_examples, summary)
    return 0
