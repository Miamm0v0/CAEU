#!/usr/bin/env python3
"""Run the shared Transformers baseline on crowd-enVent, then convert ratings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "FirstPersonMethod" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import baseline_transformers
from convert_baseline_crowd_envent import convert_outputs


DEFAULT_EVAL_FILE = REPO_ROOT / "FirstPersonMethod/data/crowd_envent/test.json"


def build_parser() -> argparse.ArgumentParser:
    parser = baseline_transformers.build_argument_parser(model_required=False)
    parser.description = (
        "Generate crowd-enVent predictions with either appraisal-mediated chains "
        "or separate official-style text-to-ratings and text-to-emotion branches"
    )
    parser.add_argument(
        "--eval_file",
        dest="dataset_file",
        type=str,
        default=str(DEFAULT_EVAL_FILE),
        help="crowd-enVent evaluation JSON; alias for --dataset_file.",
    )
    parser.add_argument("--predictions_file", type=Path, default=None)
    parser.add_argument("--invalid_file", type=Path, default=None)
    parser.add_argument(
        "--generation_schema",
        choices=["chain", "official-direct"],
        default="chain",
        help=(
            "chain: situation -> appraisal reasoning -> target; official-direct: "
            "separate situation -> 21 ratings and situation -> emotion calls."
        ),
    )
    parser.add_argument(
        "--emotion_output_mode",
        choices=["single-label", "multi-label"],
        default="single-label",
        help=(
            "single-label emits one of 13 crowd-enVent classes; multi-label "
            "keeps the native CAEU valenced output plus an evaluation-only "
            "ranking for chain, and uses a ranked label list for official-direct."
        ),
    )
    parser.add_argument(
        "--overwrite_predictions",
        action="store_true",
        help="Accepted for compatibility; converted JSONL is rebuilt atomically.",
    )
    parser.set_defaults(
        dataset_format="crowd-envent",
        dataset_file=str(DEFAULT_EVAL_FILE),
        mode="generate",
        task="chain-all",
        chain_all_emotion_mode="chain-emotion",
        chain_core_appraisals_source="chain-emotion",
    )
    return parser


def _configure_generation_flow(args: argparse.Namespace) -> None:
    if not args.model:
        raise ValueError("--model is required")
    if args.predictions_file is None:
        raise ValueError("--predictions_file is required")
    if args.dataset_format != "crowd-envent":
        raise ValueError(
            "generate_policy_crowd_envent.py requires "
            "--dataset_format crowd-envent"
        )
    if args.generation_schema == "chain":
        args.task = "chain-all"
        args.chain_all_emotion_mode = "chain-emotion"
        args.chain_core_appraisals_source = "chain-emotion"
    else:
        args.task = "official-direct"
        args.chain_all_emotion_mode = "separate"
        args.chain_core_appraisals_source = "separate"


def main() -> int:
    args = build_parser().parse_args()
    _configure_generation_flow(args)
    if args.target_folder is None:
        args.target_folder = str(args.predictions_file.parent / "baseline_outputs")
    status = baseline_transformers.execute(args)
    if status:
        return status
    convert_outputs(
        Path(args.dataset_file),
        Path(args.target_folder),
        args.predictions_file,
        args.invalid_file,
        args.max_samples,
        args.sample_seed,
        generation_schema=args.generation_schema,
        emotion_output_mode=args.emotion_output_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
