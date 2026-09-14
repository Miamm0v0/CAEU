#!/usr/bin/env python3
"""Supervised fine-tuning for the direct event-to-emotion baseline.

The model sees only the first-person event and learns to emit emotion intensity
and multi-label emotion JSON.  Appraisal reasoning and ratings are deliberately
excluded from both the prompt and completion.

Preview without training dependencies:
  python scripts/train_direct_sft.py --preview_only

LoRA training example (run from the workspace root):
  python FirstPersonMethod/scripts/train_direct_sft.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file train.json \
    --eval_file dev.json \
    --use_wandb \
    --wandb_project first-person-appraisal-sft \
    --run_name direct-sft \
    --output_dir FirstPersonMethod/output/direct_sft
"""

from sft_common import build_direct_example, build_parser, run_sft


def main():
    parser = build_parser(
        description="Direct SFT: first-person event -> emotion",
        default_output_dir="FirstPersonMethod/output/direct_sft",
        default_max_length=1536,
    )
    args = parser.parse_args()
    return run_sft(args, method_name="direct_sft", example_builder=build_direct_example)


if __name__ == "__main__":
    raise SystemExit(main())
