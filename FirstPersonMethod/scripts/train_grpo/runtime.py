"""Lazy training-stack imports, W&B setup, tokenizer, model, and PEFT helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

try:
    from peft_adapter_compat import load_peft_adapter_with_compat
except ModuleNotFoundError:  # package-style imports used by the unit tests
    from ..peft_adapter_compat import load_peft_adapter_with_compat


def parse_reporters(value: str) -> list[str]:
    if not value.strip() or value.strip().lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_wandb(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    reporters = parse_reporters(args.report_to)
    requested = args.use_wandb or any(
        reporter.lower() == "wandb" for reporter in reporters
    ) or any(
        value
        for value in (
            args.wandb_project,
            args.wandb_entity,
            args.wandb_group,
            args.wandb_tags,
            args.wandb_notes,
            args.wandb_mode,
            args.wandb_log_model,
            args.wandb_run_id,
            args.wandb_resume,
        )
    )
    if requested and not any(
        reporter.lower() == "wandb" for reporter in reporters
    ):
        reporters.append("wandb")
    if not requested:
        return reporters, {"enabled": False}

    variables = {
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_RUN_GROUP": args.wandb_group,
        "WANDB_TAGS": ",".join(
            tag.strip()
            for tag in args.wandb_tags.split(",")
            if tag.strip()
        ),
        "WANDB_NOTES": args.wandb_notes,
        "WANDB_MODE": args.wandb_mode or "",
        "WANDB_LOG_MODEL": args.wandb_log_model or "",
        "WANDB_RUN_ID": args.wandb_run_id,
    }
    for name, value in variables.items():
        if value:
            os.environ[name] = value
    if args.wandb_resume:
        os.environ["WANDB_RESUME"] = args.wandb_resume
    elif args.wandb_run_id and "WANDB_RESUME" not in os.environ:
        os.environ["WANDB_RESUME"] = "allow"
    if (
        args.wandb_resume in {"allow", "must"}
        and not os.environ.get("WANDB_RUN_ID")
    ):
        raise ValueError(
            f"--wandb_resume {args.wandb_resume} requires --wandb_run_id"
        )
    os.environ.setdefault("WANDB_JOB_TYPE", "policy-grpo-api-judge")
    return reporters, {
        "enabled": True,
        "project": os.environ.get("WANDB_PROJECT") or None,
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "group": os.environ.get("WANDB_RUN_GROUP") or None,
        "mode": os.environ.get("WANDB_MODE", "online"),
        "run_id": os.environ.get("WANDB_RUN_ID") or None,
        "resume": os.environ.get("WANDB_RESUME") or None,
    }


def import_training_stack(
    use_wandb: bool,
    need_bitsandbytes: bool,
) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from openai import AsyncOpenAI
        from peft import (
            LoraConfig,
            PeftConfig,
            PeftModel,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training requires torch, transformers, datasets, peft, trl, and "
            "openai"
        ) from exc
    if need_bitsandbytes:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load_in_4bit requires bitsandbytes"
            ) from exc
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("W&B logging requires wandb") from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "AsyncOpenAI": AsyncOpenAI,
        "LoraConfig": LoraConfig,
        "PeftConfig": PeftConfig,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def choose_dtype(torch: Any, requested: str) -> tuple[Any, bool, bool]:
    if requested == "bf16":
        return torch.bfloat16, True, False
    if requested == "fp16":
        return torch.float16, False, True
    if requested == "fp32":
        return torch.float32, False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def parse_target_modules(value: str) -> str | list[str]:
    if value.strip() == "all-linear":
        return "all-linear"
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must not be empty")
    return modules


def quantization_config(
    stack: dict[str, Any],
    dtype: Any,
    enabled: bool,
) -> Any:
    if not enabled:
        return None
    return stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def local_adapter_checkpoint(path_value: str) -> bool:
    path = Path(path_value)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def load_tokenizer(
    stack: dict[str, Any],
    source: str,
    trust_remote_code: bool,
) -> Any:
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        source,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {source} has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_chat_template_override(
    tokenizer: Any,
    template_path: str,
    stack: dict[str, Any],
    trust_remote_code: bool,
) -> None:
    if not template_path:
        return
    path = Path(template_path)
    if path.is_file():
        tokenizer.chat_template = path.read_text(encoding="utf-8")
        return
    template_tokenizer = load_tokenizer(
        stack,
        template_path,
        trust_remote_code,
    )
    if not getattr(template_tokenizer, "chat_template", None):
        raise ValueError(f"{template_path} tokenizer has no chat template")
    tokenizer.chat_template = template_tokenizer.chat_template


def validate_prompt_lengths(
    tokenizer: Any,
    examples: list[dict[str, Any]],
    split_name: str,
    max_prompt_length: int,
    enable_thinking: bool,
) -> dict[str, float | int]:
    lengths: list[int] = []
    over_limit: list[tuple[str, int]] = []
    for example in examples:
        token_ids = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        length = len(token_ids)
        lengths.append(length)
        if length > max_prompt_length:
            over_limit.append((example["sample_id"], length))
    if over_limit:
        raise ValueError(
            f"{split_name} contains {len(over_limit)} rendered prompts longer "
            f"than --max_prompt_length={max_prompt_length}; first samples: "
            f"{over_limit[:10]}. Increase the model context/limit or shorten "
            "the fixed prompt without truncating situation evidence."
        )
    return {
        "minimum": min(lengths) if lengths else 0,
        "maximum": max(lengths) if lengths else 0,
        "mean": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def build_policy_model_and_peft(
    args: argparse.Namespace,
    stack: dict[str, Any],
    dtype: Any,
    policy_quantization: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    source = args.model_name_or_path
    is_adapter = args.policy_is_adapter or local_adapter_checkpoint(source)
    if is_adapter:
        if args.full_finetune:
            raise ValueError(
                "A PEFT adapter cannot be full-finetuned directly; merge it "
                "into its base model first or remove --full_finetune"
            )
        adapter_config = stack["PeftConfig"].from_pretrained(source)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if policy_quantization is not None:
            model_kwargs["quantization_config"] = policy_quantization
            model_kwargs["device_map"] = {
                "": int(os.environ.get("LOCAL_RANK", "0"))
            }
        base = stack["AutoModelForCausalLM"].from_pretrained(
            adapter_config.base_model_name_or_path,
            **model_kwargs,
        )
        if policy_quantization is not None:
            base = stack["prepare_model_for_kbit_training"](
                base,
                use_gradient_checkpointing=not args.no_gradient_checkpointing,
            )
        model, adapter_load_summary = load_peft_adapter_with_compat(
            base,
            source,
            stack["PeftModel"],
            stack["torch"],
            is_trainable=True,
        )
        return model, None, {
            "checkpoint": source,
            "is_existing_adapter": True,
            "base_model": adapter_config.base_model_name_or_path,
            "adapter_load": adapter_load_summary,
        }

    peft_config = None
    if not args.full_finetune:
        peft_config = stack["LoraConfig"](
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=parse_target_modules(args.lora_target_modules),
        )
    return source, peft_config, {
        "checkpoint": source,
        "is_existing_adapter": False,
        "full_finetune": args.full_finetune,
    }


def resolve_resume(value: str) -> str | bool | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"true", "yes", "last"}:
        return True
    return normalized
