#!/usr/bin/env python3
"""Supervised fine-tuning for event-to-appraisal-to-emotion chains.

The target is one structured JSON trajectory containing five first-person
appraisal explanations, 22 CAREBench appraisal ratings, and the final emotion.

Preview without training dependencies:
  python scripts/train_chain_sft.py --preview_only

QLoRA training example (run from the workspace root):
  python FirstPersonMethod/scripts/train_chain_sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file train.json \
    --eval_file dev.json \
    --load_in_4bit \
    --output_dir FirstPersonMethod/output/chain_sft
"""

from sft_common import build_chain_example, build_parser, run_sft


def main():
    parser = build_parser(
        description="Chain SFT: first-person event -> appraisal chain -> emotion",
        default_output_dir="FirstPersonMethod/output/chain_sft",
        default_max_length=3072,
    )
    args = parser.parse_args()
    return run_sft(args, method_name="chain_sft", example_builder=build_chain_example)


if __name__ == "__main__":
    raise SystemExit(main())

