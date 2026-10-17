#!/usr/bin/env python3
"""Train the appraisal-to-emotion intermediate stage."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from transformers import TrainingArguments


if __package__ in {None, ""}:
    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from appraisal_emotion_train.ae_format import tokenize_ae_example
    from appraisal_emotion_train.ae_trainer import (
        AETrainer,
        AEMultiObjectiveCollator,
        ProportionalObjectiveDataset,
        WarmupAdapterLoadConfig,
        load_trainable_warmup_adapter,
        tokenize_pair_record,
        tokenize_replay_example,
    )
    from train_grpo.spec import (
        CAREBENCH_REASONING_KEYS,
        build_policy_system_prompt,
        build_policy_user_prompt,
    )
else:
    from .ae_format import tokenize_ae_example
    from .ae_trainer import (
        AETrainer,
        AEMultiObjectiveCollator,
        ProportionalObjectiveDataset,
        WarmupAdapterLoadConfig,
        load_trainable_warmup_adapter,
        tokenize_pair_record,
        tokenize_replay_example,
    )
    from ..train_grpo.spec import (
        CAREBENCH_REASONING_KEYS,
        build_policy_system_prompt,
        build_policy_user_prompt,
    )


POLICY_DIMENSIONS = tuple(CAREBENCH_REASONING_KEYS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a CAREBench SFT Warmup adapter with conditional "
            "appraisal-to-emotion training."
        )
    )
    data = parser.add_argument_group("data")
    data.add_argument("--train_file", type=Path, required=True)
    data.add_argument("--pair_file", type=Path, required=True)
    data.add_argument("--eval_file", type=Path, required=True)
    data.add_argument(
        "--eval_pair_file",
        type=Path,
        default=None,
        help=(
            "Validation pair JSONL. Defaults to ae_pairs_val.jsonl beside "
            "--eval_file. Regenerate AE data if that file is missing."
        ),
    )
    data.add_argument(
        "--invalid_record_policy",
        choices=("error", "skip"),
        default="error",
    )
    data.add_argument("--max_length", type=int, default=3072)
    data.add_argument(
        "--preservation_data_fraction",
        type=float,
        default=0.1,
        help="Fraction of valid first-person train records used as Chain replay.",
    )
    data.add_argument("--preservation_max_samples", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--model_name_or_path", required=True)
    model.add_argument("--tokenizer_name_or_path", default="")
    model.add_argument("--chat_template_path", default="")
    model.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
    )
    model.add_argument("--load_in_4bit", action="store_true")
    model.add_argument("--no_gradient_checkpointing", action="store_true")
    model.add_argument("--trust_remote_code", action="store_true")
    model.add_argument("--enable_thinking", action="store_true")

    objective = parser.add_argument_group("objectives")
    objective.add_argument("--conditional_ratio", type=float, default=1.0)
    objective.add_argument("--pair_ratio", type=float, default=1.0)
    objective.add_argument("--preservation_ratio", type=float, default=0.1)
    objective.add_argument("--pair_loss_weight", type=float, default=1.0)
    objective.add_argument("--preservation_loss_weight", type=float, default=1.0)
    objective.add_argument("--contrastive_beta", type=float, default=1.0)
    objective.add_argument(
        "--epoch_size",
        type=int,
        default=0,
        help="Mixed objective examples per epoch; 0 derives it from train size.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--output_dir", required=True)
    training.add_argument("--num_train_epochs", type=float, default=1.0)
    training.add_argument("--max_steps", type=int, default=-1)
    training.add_argument("--learning_rate", type=float, default=1.0e-5)
    training.add_argument("--weight_decay", type=float, default=0.0)
    training.add_argument("--warmup_ratio", type=float, default=0.03)
    training.add_argument("--per_device_train_batch_size", type=int, default=1)
    training.add_argument("--per_device_eval_batch_size", type=int, default=1)
    training.add_argument("--gradient_accumulation_steps", type=int, default=8)
    training.add_argument("--max_grad_norm", type=float, default=1.0)
    training.add_argument("--logging_steps", type=int, default=5)
    training.add_argument("--eval_steps", type=int, default=50)
    training.add_argument(
        "--carebench_eval_max_samples",
        type=int,
        default=100,
        help="Maximum generated validation samples per eval; 0 uses all.",
    )
    training.add_argument(
        "--carebench_eval_max_new_tokens",
        type=int,
        default=192,
    )
    training.add_argument("--save_steps", type=int, default=50)
    training.add_argument("--save_total_limit", type=int, default=3)
    training.add_argument("--dataloader_num_workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--resume_from_checkpoint", default="")
    training.add_argument("--report_to", default="none")
    training.add_argument("--run_name", default="appraisal-emotion-ae")
    training.add_argument("--use_wandb", action="store_true")
    training.add_argument("--wandb_project", default="")
    training.add_argument("--wandb_entity", default="")
    training.add_argument("--wandb_group", default="")
    training.add_argument("--wandb_tags", default="")
    training.add_argument("--wandb_notes", default="")
    training.add_argument(
        "--wandb_mode",
        choices=("online", "offline"),
        default=None,
    )
    training.add_argument(
        "--wandb_log_model",
        choices=("false", "end", "checkpoint"),
        default=None,
    )
    training.add_argument("--wandb_run_id", default="")
    training.add_argument(
        "--wandb_resume",
        choices=("never", "allow", "must", "auto"),
        default=None,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.train_file, args.pair_file, args.eval_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.eval_pair_file is not None and not args.eval_pair_file.is_file():
        raise FileNotFoundError(args.eval_pair_file)
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if not 0.0 < args.preservation_data_fraction <= 1.0:
        raise ValueError("--preservation_data_fraction must be in (0, 1]")
    if args.preservation_max_samples < 0:
        raise ValueError("--preservation_max_samples cannot be negative")
    if args.epoch_size < 0:
        raise ValueError("--epoch_size cannot be negative")
    if args.carebench_eval_max_samples < 0:
        raise ValueError("--carebench_eval_max_samples cannot be negative")
    if args.carebench_eval_max_new_tokens <= 0:
        raise ValueError("--carebench_eval_max_new_tokens must be positive")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    if not records:
        raise ValueError(f"{path}: no JSONL records found")
    return records


def _map_records(
    records: Sequence[Mapping[str, Any]],
    builder: Callable[[Mapping[str, Any]], dict[str, Any]],
    *,
    split_name: str,
    invalid_record_policy: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            output.append(builder(record))
        except Exception as exc:  # noqa: BLE001
            identity = record.get("sample_id", record.get("pair_id", index))
            message = f"{split_name}/{identity}: {type(exc).__name__}: {exc}"
            if invalid_record_policy == "error":
                raise ValueError(message) from exc
            errors.append(message)
    if not output:
        raise ValueError(f"{split_name}: no valid records remain")
    return output, errors


def _select_preservation_records(
    train_records: Sequence[Mapping[str, Any]],
    *,
    fraction: float,
    maximum: int,
    seed: int,
) -> list[Mapping[str, Any]]:
    candidates = [
        record
        for record in train_records
        if record.get("annotator_type") == "first_person"
    ]
    if not candidates:
        raise ValueError("No first-person records are available for preservation replay")
    count = max(1, math.ceil(len(candidates) * fraction))
    if maximum:
        count = min(count, maximum)
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def _replay_builder(
    tokenizer: Any,
    record: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    situation = str(record["situation"])
    prompt_messages = [
        {
            "role": "system",
            "content": build_policy_system_prompt(POLICY_DIMENSIONS),
        },
        {
            "role": "user",
            "content": build_policy_user_prompt(situation, POLICY_DIMENSIONS),
        },
    ]
    response = json.dumps(
        {
            "appraisal_reasoning": record["appraisal_reasoning"],
            "emotion": record["emotion"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return tokenize_replay_example(
        tokenizer,
        prompt_messages=prompt_messages,
        assistant_response=response,
        sample_id=f"replay::{sample_id}",
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )


def _resolve_reporters(
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    reporters = [
        item.strip()
        for item in args.report_to.split(",")
        if item.strip() and item.strip().lower() != "none"
    ]
    if args.use_wandb and "wandb" not in reporters:
        reporters.append("wandb")
    if "wandb" not in reporters:
        return reporters, {"enabled": False}

    settings = {
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_RUN_GROUP": args.wandb_group,
        "WANDB_TAGS": args.wandb_tags,
        "WANDB_NOTES": args.wandb_notes,
        "WANDB_MODE": args.wandb_mode,
        "WANDB_LOG_MODEL": args.wandb_log_model,
        "WANDB_RUN_ID": args.wandb_run_id,
        "WANDB_RESUME": args.wandb_resume,
    }
    for name, value in settings.items():
        if value not in (None, ""):
            os.environ[name] = str(value)
    os.environ.setdefault("WANDB_JOB_TYPE", "appraisal-emotion-ae")

    run_id = os.environ.get("WANDB_RUN_ID", "")
    resume = os.environ.get("WANDB_RESUME", "")
    if resume == "must" and not run_id:
        raise ValueError("--wandb_resume must requires --wandb_run_id")

    summary = {
        "enabled": True,
        "project": os.environ.get("WANDB_PROJECT", ""),
        "entity": os.environ.get("WANDB_ENTITY", ""),
        "group": os.environ.get("WANDB_RUN_GROUP", ""),
        "tags": os.environ.get("WANDB_TAGS", ""),
        "notes": os.environ.get("WANDB_NOTES", ""),
        "mode": os.environ.get("WANDB_MODE", "online"),
        "log_model": os.environ.get("WANDB_LOG_MODEL", "false"),
        "run_id": run_id,
        "resume": resume,
        "run_name": args.run_name,
    }
    return reporters, summary


def _precision_flags(dtype: str) -> tuple[bool, bool]:
    if dtype == "bf16":
        return True, False
    if dtype == "fp16":
        return False, True
    if dtype == "fp32":
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return True, False
    return False, torch.cuda.is_available()


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    eval_pair_file = (
        args.eval_pair_file
        if args.eval_pair_file is not None
        else args.eval_file.with_name("ae_pairs_val.jsonl")
    )
    if not eval_pair_file.is_file():
        raise FileNotFoundError(
            f"Validation pair file not found: {eval_pair_file}. Re-run "
            "prepare_ae_data.py to emit ae_pairs_val.jsonl or pass "
            "--eval_pair_file explicitly."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_records = _read_jsonl(args.train_file)
    pair_records = _read_jsonl(args.pair_file)
    eval_records = _read_jsonl(args.eval_file)
    eval_pair_records = _read_jsonl(eval_pair_file)
    model, tokenizer, model_summary = load_trainable_warmup_adapter(
        WarmupAdapterLoadConfig(
            adapter_path=args.model_name_or_path,
            tokenizer_path=args.tokenizer_name_or_path,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            trust_remote_code=args.trust_remote_code,
            chat_template_path=args.chat_template_path,
        )
    )

    conditional_builder = lambda record: tokenize_ae_example(
        tokenizer,
        record,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )
    pair_builder = lambda record: tokenize_pair_record(
        tokenizer,
        record,
        max_length=args.max_length,
        enable_thinking=args.enable_thinking,
    )
    conditional_features, train_errors = _map_records(
        train_records,
        conditional_builder,
        split_name="train_conditional",
        invalid_record_policy=args.invalid_record_policy,
    )
    pair_features, pair_errors = _map_records(
        pair_records,
        pair_builder,
        split_name="train_pairs",
        invalid_record_policy=args.invalid_record_policy,
    )
    eval_pair_features, eval_pair_errors = _map_records(
        eval_pair_records,
        pair_builder,
        split_name="validation_pairs",
        invalid_record_policy=args.invalid_record_policy,
    )
    eval_features, eval_errors = _map_records(
        eval_records,
        lambda record: tokenize_ae_example(
            tokenizer,
            record,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
            include_text_metadata=True,
        ),
        split_name="validation",
        invalid_record_policy=args.invalid_record_policy,
    )
    eval_records_by_id = {
        str(record["sample_id"]): record for record in eval_records
    }
    carebench_eval_examples: list[dict[str, Any]] = []
    for feature in eval_features:
        sample_id = str(feature["sample_id"])
        reference = eval_records_by_id.get(sample_id)
        if reference is None:
            raise ValueError(f"Missing validation reference for {sample_id}")
        emotion_start = int(feature["emotion_start"])
        carebench_eval_examples.append(
            {
                "sample_id": sample_id,
                "prefix_input_ids": feature["input_ids"][:emotion_start],
                "assistant_prefix": feature["assistant_prefix"],
                "gold_emotion": reference["emotion"],
            }
        )
    random.Random(args.seed).shuffle(carebench_eval_examples)
    preservation_records = _select_preservation_records(
        train_records,
        fraction=args.preservation_data_fraction,
        maximum=args.preservation_max_samples,
        seed=args.seed,
    )
    preservation_features, preservation_errors = _map_records(
        preservation_records,
        lambda record: _replay_builder(tokenizer, record, args),
        split_name="preservation",
        invalid_record_policy=args.invalid_record_policy,
    )

    train_dataset = ProportionalObjectiveDataset(
        conditional=conditional_features,
        pairs=pair_features,
        preservation=preservation_features,
        conditional_ratio=args.conditional_ratio,
        pair_ratio=args.pair_ratio,
        preservation_ratio=args.preservation_ratio,
        epoch_size=args.epoch_size or None,
        seed=args.seed,
    )
    eval_dataset = ProportionalObjectiveDataset(
        conditional=eval_features,
        pairs=eval_pair_features,
        preservation=[],
        conditional_ratio=float(len(eval_features)),
        pair_ratio=float(len(eval_pair_features)),
        preservation_ratio=0.0,
        epoch_size=len(eval_features) + len(eval_pair_features),
        seed=args.seed,
    )
    bf16, fp16 = _precision_flags(args.dtype)
    reporters, wandb_summary = _resolve_reporters(args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        tf32=bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=reporters,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    trainer = AETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AEMultiObjectiveCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        pair_loss_weight=args.pair_loss_weight,
        preservation_loss_weight=args.preservation_loss_weight,
        contrastive_beta=args.contrastive_beta,
        carebench_eval_examples=carebench_eval_examples,
        carebench_eval_max_samples=args.carebench_eval_max_samples,
        carebench_eval_max_new_tokens=args.carebench_eval_max_new_tokens,
    )
    resume = args.resume_from_checkpoint.strip() or None
    train_result = trainer.train(resume_from_checkpoint=resume)
    eval_metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    output_dir = Path(args.output_dir)
    manifest = {
        "method": "appraisal_to_emotion_intermediate_training",
        "model": model_summary,
        "data": {
            "train_file": str(args.train_file),
            "pair_file": str(args.pair_file),
            "eval_file": str(args.eval_file),
            "eval_pair_file": str(eval_pair_file),
            "conditional_samples": len(conditional_features),
            "pair_samples": len(pair_features),
            "preservation_samples": len(preservation_features),
            "validation_samples": len(eval_features),
            "validation_pairs": len(eval_pair_features),
            "mixed_epoch_size": len(train_dataset),
            "mixed_objective_counts": train_dataset.objective_counts,
            "skipped": {
                "conditional": train_errors[:20],
                "pair": pair_errors[:20],
                "validation_pair": eval_pair_errors[:20],
                "preservation": preservation_errors[:20],
                "validation": eval_errors[:20],
            },
        },
        "loss": {
            "formula": "L_cond + lambda * L_pair + mu * L_preserve",
            "pair_loss_weight": args.pair_loss_weight,
            "preservation_loss_weight": args.preservation_loss_weight,
            "contrastive_beta": args.contrastive_beta,
            "pair_score": (
                "mean emotion label/intensity value-token log-likelihood; "
                "JSON structure and field-name tokens are masked"
            ),
            "preservation": "full-chain SFT replay",
            "lightweight_validation_metrics": [
                "valid_candidate_coverage",
                "positive/negative_example_f1",
                "positive/negative_label_exact",
                "positive/negative_intensity_rmse",
                "pair_loss",
                "pair_margin_a/b/mean",
                "pair_accuracy_a/b/both/mean",
            ],
        },
        "wandb": wandb_summary,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "arguments": vars(args),
    }
    _write_manifest(output_dir / "ae_training_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    data = manifest["data"]
    print(
        "[ae] complete "
        f"conditional={data['conditional_samples']} "
        f"pairs={data['pair_samples']} "
        f"preservation={data['preservation_samples']} "
        f"validation={data['validation_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
