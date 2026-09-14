#!/usr/bin/env python3
"""Shared local-Transformers helpers for CAREBench counterfactual runners."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_PERSON_SCRIPTS = REPO_ROOT / "FirstPersonMethod" / "scripts"

if not FIRST_PERSON_SCRIPTS.exists():
    raise FileNotFoundError(
        "Cannot find FirstPersonMethod/scripts for the Transformers backend: "
        f"{FIRST_PERSON_SCRIPTS}"
    )
if str(FIRST_PERSON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIRST_PERSON_SCRIPTS))

from baseline_transformers_backend import TransformersClient  # noqa: E402


def add_transformers_arguments(
    parser: argparse.ArgumentParser,
    *,
    str2bool: Callable[[str], bool],
    max_tokens_default: int = 512,
) -> None:
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model id, full-model directory, or PEFT adapter directory",
    )
    parser.add_argument(
        "--base_model",
        "--base_model_name_or_path",
        dest="base_model",
        type=str,
        default="",
        help=(
            "Optional base model override for a PEFT/LoRA --model. Local "
            "adapters otherwise use base_model_name_or_path from adapter_config.json."
        ),
    )
    parser.add_argument(
        "--model_is_adapter",
        action="store_true",
        help="Force --model to load as a PEFT adapter.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help=(
            "Optional tokenizer id/path; defaults to a tokenizer saved with "
            "--model, then to the PEFT base model."
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=max_tokens_default)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        type=str2bool,
        default=True,
        help="Set false for deterministic greedy decoding.",
    )
    parser.add_argument(
        "--enable_thinking",
        type=str2bool,
        default=False,
        help="Qwen3 thinking mode; disabled by default for label parsing.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:N; ignored for dispatched device maps.",
    )
    parser.add_argument(
        "--device_map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
        help="Accelerate device map for multi-GPU/offload; omit to use --device.",
    )
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load_in_4bit", action="store_true")
    quantization.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)


def build_transformers_client(args: Any) -> TransformersClient:
    return TransformersClient(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        revision=args.revision,
        seed=args.seed,
        base_model_name_or_path=args.base_model,
        model_is_adapter=args.model_is_adapter,
    )


def default_model_dir(
    *,
    script_file: str,
    target_root: Optional[str],
    model: str,
    family: str,
    sanitize_model_name: Callable[[str], str],
    target_root_is_model_dir: bool,
) -> Path:
    if target_root:
        base = Path(target_root)
        return base if target_root_is_model_dir else base / sanitize_model_name(model)

    carebench_root = Path(script_file).resolve().parents[1]
    return (
        carebench_root
        / "output"
        / "first_person"
        / family
        / sanitize_model_name(model)
    )


def warn_if_threaded(num_threads: int) -> None:
    if num_threads > 1:
        print(
            "[warning] local model.generate calls are serialized; --num_threads > 1 "
            "only overlaps JSON parsing and file writes, not GPU generation."
        )
