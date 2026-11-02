"""Training validation, reproducibility artifacts, and GRPOTrainer assembly."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from .api_judge import APIJudge, JudgeCache, resolve_judge_settings
from .reward_components import (
    OPTIMIZATION_REWARD_FIELDS,
    REWARD_COMPONENT_FIELDS,
)
from .runtime import (
    apply_chat_template_override,
    build_policy_model_and_peft,
    choose_dtype,
    configure_wandb,
    import_training_stack,
    load_tokenizer,
    quantization_config,
    resolve_resume,
    validate_prompt_lengths,
)
from .spec import (
    APPRAISAL_CRITERIA,
    APPRAISAL_DEFINITIONS,
    APPRAISAL_DIMENSIONS,
    COHERENCE_CRITERION,
    DIMENSION_VALIDITY_GUIDANCE,
    RUBRIC_VERSION,
    TRANSITION_CRITERION,
    build_judge_system_prompt,
    build_policy_system_prompt,
    build_policy_user_prompt,
    policy_output_example,
    judge_output_schema,
)


def validate_training_arguments(args: argparse.Namespace) -> None:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required for training")
    if args.load_in_4bit and args.full_finetune:
        raise ValueError("--load_in_4bit requires LoRA; remove --full_finetune")
    positive_names = (
        "num_generations",
        "max_prompt_length",
        "max_completion_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "judge_max_tokens",
        "judge_max_concurrency",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_generations < 2:
        raise ValueError("--num_generations must be at least 2 for GRPO")
    if args.judge_max_retries < 0 or args.judge_parse_retries < 0:
        raise ValueError("Judge retry counts must be non-negative")
    if args.judge_timeout_seconds <= 0:
        raise ValueError("--judge_timeout_seconds must be positive")
    if not 0 < args.temperature:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    finite_names = (
        "judge_temperature",
        "invalid_component_reward",
        "appraisal_reward_weight",
        "grounding_reward_weight",
        "coherence_reward_weight",
        "transition_reward_weight",
        "outcome_reward_weight",
        "outcome_label_weight",
        "outcome_intensity_weight",
        "reward_normalization_epsilon",
    )
    for name in finite_names:
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name} must be finite")
    if (
        args.judge_provider == "glm"
        and not args.judge_omit_temperature
        and not 0 <= args.judge_temperature <= 1
    ):
        raise ValueError("--judge_temperature must be in [0, 1] for GLM")
    if args.judge_provider != "glm" and args.judge_thinking != "auto":
        raise ValueError(
            "--judge_thinking enabled/disabled is supported only with "
            "--judge_provider glm"
        )
    optimization_weights = [
        args.appraisal_reward_weight,
        args.grounding_reward_weight,
        args.coherence_reward_weight,
        args.transition_reward_weight,
        args.outcome_reward_weight,
    ]
    if any(weight < 0 for weight in optimization_weights):
        raise ValueError("Optimization reward weights must be non-negative")
    if sum(optimization_weights) <= 0:
        raise ValueError("At least one optimization reward weight must be positive")
    if args.outcome_label_weight < 0 or args.outcome_intensity_weight < 0:
        raise ValueError("Outcome label/intensity weights must be non-negative")
    if args.outcome_label_weight + args.outcome_intensity_weight <= 0:
        raise ValueError("Outcome label + intensity weights must be positive")
    if args.reward_normalization_epsilon <= 0:
        raise ValueError("--reward_normalization_epsilon must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    generation_batch = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    if generation_batch % args.num_generations != 0:
        raise ValueError(
            "The default TRL generation batch "
            "(WORLD_SIZE * per-device train batch * gradient accumulation = "
            f"{generation_batch}) must be divisible by "
            f"--num_generations={args.num_generations}"
        )
    if args.eval_file:
        eval_batch = world_size * args.per_device_eval_batch_size
        if eval_batch % args.num_generations != 0:
            raise ValueError(
                "Global eval batch "
                "(WORLD_SIZE * per-device eval batch = "
                f"{eval_batch}) must be divisible by "
                f"--num_generations={args.num_generations}"
            )


def write_rubric_artifact(
    output_dir: Path,
    args: argparse.Namespace,
    settings: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_dimensions = args.appraisal_dimensions
    artifact = {
        "rubric_version": RUBRIC_VERSION,
        "appraisal_dimensions": selected_dimensions,
        "dimensions": {
            dimension: APPRAISAL_DEFINITIONS[dimension]
            for dimension in selected_dimensions
        },
        "dimension_validity_guidance": {
            dimension: DIMENSION_VALIDITY_GUIDANCE[dimension]
            for dimension in selected_dimensions
        },
        "appraisal_criteria": APPRAISAL_CRITERIA,
        "coherence_criterion": COHERENCE_CRITERION,
        "transition_criterion": TRANSITION_CRITERION,
        "score_scale": {"minimum": 0, "maximum": 4},
        "aggregation": {
            "formula": (
                "0.35 * mean_d[grounding_d * mean(validity_d, fidelity_d)] "
                "+ 0.05 * coherence + 0.10 * transition + "
                "0.50 * (0.8 * label + 0.2 * intensity) at default weights"
            ),
            "persisted_reward_components": REWARD_COMPONENT_FIELDS,
            "optimization_reward_components": OPTIMIZATION_REWARD_FIELDS,
            "optimization_weights": {
                "appraisal_reward": args.appraisal_reward_weight,
                "grounding_reward": args.grounding_reward_weight,
                "coherence_reward": args.coherence_reward_weight,
                "transition_reward": args.transition_reward_weight,
                "outcome_reward": args.outcome_reward_weight,
            },
            "transition_reward": (
                "judge.transition.score / 4 by default; optional grounding gate"
            ),
            "appraisal_reward": (
                "mean across dimensions of grounding * mean(dimension-specific "
                "validity, experiencer fidelity)"
            ),
            "grounding_reward": "0.7 * grounding mean + 0.3 * mean of the lowest two dimension grounding scores",
            "gold_outcome_scores": {
                "positive_label_score": "diagnostic example-level set F1",
                "negative_label_score": "diagnostic example-level set F1",
                "outcome_label_reward": (
                    "example-level set F1 over the union of valence-tagged labels"
                ),
                "positive_intensity_score": "1 - abs(pred-gold)/6",
                "negative_intensity_score": "1 - abs(pred-gold)/6",
                "label_weight": args.outcome_label_weight,
                "intensity_weight": args.outcome_intensity_weight,
            },
            "process_gate": {
                "enabled": args.use_process_gate,
                "mode": args.process_gate_mode,
                "inputs": ["grounding_reward", "raw_transition_reward"],
                "target": "transition_reward only; disabled by default; gold outcome is not gated",
            },
            "normalization": (
                "combine raw optimization components by configured weights, "
                "then standardize the total once within each num_generations "
                "group; component diagnostics use the same total standard deviation"
            ),
            "format_hard_gate": {
                "invalid_policy_output": (
                    "invalid_component_reward assigned to every component; "
                    "API Judge is not called"
                ),
            },
        },
        "judge_provider": settings["provider"],
        "judge_model": settings["model"],
        "judge_base_url": settings["base_url"] or None,
        "judge_api_key_env": settings["api_key_env"] or None,
        "judge_response_format": {
            "requested": settings["requested_response_format"],
            "effective": settings["response_format"],
        },
        "judge_token_parameter": {
            "requested": settings["requested_token_parameter"],
            "effective": settings["token_parameter"],
        },
        "judge_thinking": {
            "requested": settings["requested_thinking"],
            "effective": settings["thinking"] or None,
        },
        "judge_reasoning_effort": {
            "requested": settings["requested_reasoning_effort"],
            "effective": settings["reasoning_effort"] or None,
        },
        "judge_rationale_mode": args.judge_rationale_mode,
        "policy_output_example": policy_output_example(selected_dimensions),
        "policy_system_prompt": build_policy_system_prompt(selected_dimensions),
        "policy_user_prompt_template": build_policy_user_prompt(
            "{situation}", selected_dimensions
        ),
        "system_prompt": build_judge_system_prompt(
            selected_dimensions,
            args.judge_rationale_mode,
        ),
        "output_schema": judge_output_schema(
            selected_dimensions,
            args.judge_rationale_mode,
        ),
    }
    with (output_dir / "api_judge_rubric.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_grpo_config(
    args: argparse.Namespace,
    stack: dict[str, Any],
    torch: Any,
    dtype: Any,
    use_bf16: bool,
    use_fp16: bool,
    reporters: list[str],
    policy_model: Any,
    has_eval: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if has_eval else "no",
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "loss_type": args.loss_type,
        # The reward callable normalizes the raw weighted total once per group.
        "scale_rewards": "none",
        "mask_truncated_completions": True,
        "remove_unused_columns": False,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "tf32": bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ),
        "report_to": reporters,
        "run_name": args.run_name or "policy-api-judge-grpo",
        "seed": args.seed,
        "data_seed": args.seed,
        "log_completions": args.log_completions,
        "chat_template_kwargs": {
            "enable_thinking": args.enable_thinking
        },
        "trust_remote_code": args.trust_remote_code,
        "model_init_kwargs": (
            {
                "dtype": dtype,
                "trust_remote_code": args.trust_remote_code,
            }
            if isinstance(policy_model, str)
            else None
        ),
    }
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps
    if args.deepspeed:
        kwargs["deepspeed"] = args.deepspeed
    return stack["GRPOConfig"](**kwargs)


def save_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    settings: dict[str, str],
    cache_path: Path | None,
    policy_summary: dict[str, Any],
    data_summary: dict[str, Any],
    wandb_summary: dict[str, Any],
    train_metrics: dict[str, Any],
) -> None:
    manifest = {
        "method": "first_person_configurable_appraisal_policy_grpo_api_judge",
        "rubric_version": RUBRIC_VERSION,
        "policy": policy_summary,
        "judge": {
            "provider": settings["provider"],
            "provider_section": settings["provider_section"],
            "model": settings["model"],
            "base_url": settings["base_url"] or None,
            "api_key_env": settings["api_key_env"] or None,
            "response_format": {
                "requested": settings["requested_response_format"],
                "effective": settings["response_format"],
            },
            "token_parameter": {
                "requested": settings["requested_token_parameter"],
                "effective": settings["token_parameter"],
            },
            "thinking": {
                "requested": settings["requested_thinking"],
                "effective": settings["thinking"] or None,
            },
            "reasoning_effort": {
                "requested": settings["requested_reasoning_effort"],
                "effective": settings["reasoning_effort"] or None,
            },
            "reference_mode": args.judge_reference_mode,
            "cache_path": str(cache_path) if cache_path is not None else None,
            "failure_policy": args.judge_failure_policy,
        },
        "reward": {
            "formula": (
                "0.35 * mean_d[grounding_d * mean(validity_d, fidelity_d)] "
                "+ 0.05 * coherence + 0.10 * transition + "
                "0.50 * (0.8 * label + 0.2 * intensity) at default weights"
            ),
            "supported_appraisal_dimensions": APPRAISAL_DIMENSIONS,
            "appraisal_dimensions": args.appraisal_dimensions,
            "appraisal_criteria": APPRAISAL_CRITERIA,
            "coherence_criterion": COHERENCE_CRITERION,
            "transition_criterion": TRANSITION_CRITERION,
            "persisted_components": REWARD_COMPONENT_FIELDS,
            "optimization_components": OPTIMIZATION_REWARD_FIELDS,
            "optimization_weights": {
                "appraisal_reward": args.appraisal_reward_weight,
                "grounding_reward": args.grounding_reward_weight,
                "coherence_reward": args.coherence_reward_weight,
                "transition_reward": args.transition_reward_weight,
                "outcome_reward": args.outcome_reward_weight,
            },
            "outcome_subreward_weights": {
                "labels": args.outcome_label_weight,
                "intensities": args.outcome_intensity_weight,
            },
            "outcome_label_reward": (
                "example-level set F1 over the union of valence-tagged labels; "
                "per-valence F1 values are diagnostics only"
            ),
            "process_gate": {
                "enabled": args.use_process_gate,
                "mode": args.process_gate_mode,
                "disabled_behavior": "gate=1.0",
                "target": "transition_reward; gold outcome is independent",
            },
            "invalid_component_reward": args.invalid_component_reward,
            "format_hard_gate": {
                "invalid_policy_output": (
                    "invalid_component_reward for every component; API Judge "
                    "is not called"
                ),
            },
            "transition_reward": (
                "judge.transition.score / 4 by default; optional grounding gate"
            ),
            "appraisal_reward": (
                "mean across dimensions of grounding * mean(dimension-specific "
                "validity, experiencer fidelity)"
            ),
            "grounding_reward": "0.7 * mean + 0.3 * mean of the lowest two dimension grounding scores",
            "outcome_reward": (
                "weighted_mean(gold_label_reward, gold_intensity_reward)"
            ),
            "normalization": {
                "requested": args.scale_rewards,
                "total_reward": (
                    "combine raw components, then normalize the weighted "
                    "total once; component diagnostics share the total standard deviation"
                ),
                "epsilon": args.reward_normalization_epsilon,
                "trl_scale_rewards": "none",
            },
            "scalar_training_signal": (
                "group-normalized raw weighted total; raw_total_reward and "
                "per-component reward_contributions are logged for auditing"
            ),
            "judge_failure_handling": (
                "return None in skip mode; raise in error mode"
            ),
            "lightweight_eval_metrics": [
                "appraisal_bleu",
                "appraisal_rouge_l",
                "positive/negative_example_f1",
                "positive/negative_label_exact",
                "positive/negative_intensity_rmse",
                "reference_coverage",
            ],
        },
        "data": data_summary,
        "wandb": wandb_summary,
        "arguments": vars(args).copy(),
        "train_metrics": train_metrics,
    }
    with (output_dir / "policy_api_judge_training_manifest.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_training(
    args: argparse.Namespace,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    data_summary: dict[str, Any],
) -> None:
    validate_training_arguments(args)
    settings = resolve_judge_settings(args)
    reporters, wandb_summary = configure_wandb(args)
    stack = import_training_stack(
        use_wandb=wandb_summary["enabled"],
        need_bitsandbytes=args.load_in_4bit,
    )
    torch = stack["torch"]
    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("--load_in_4bit requires a CUDA GPU")
    dtype, use_bf16, use_fp16 = choose_dtype(torch, args.dtype)

    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    policy_tokenizer = load_tokenizer(
        stack,
        tokenizer_source,
        args.trust_remote_code,
    )
    policy_tokenizer.padding_side = "left"
    apply_chat_template_override(
        policy_tokenizer,
        args.chat_template_path,
        stack,
        args.trust_remote_code,
    )
    if not getattr(policy_tokenizer, "chat_template", None):
        raise ValueError(
            "The policy tokenizer has no chat template; pass "
            "--chat_template_path"
        )
    data_summary["rendered_prompt_tokens"] = {
        "train": validate_prompt_lengths(
            policy_tokenizer,
            train_examples,
            "train",
            args.max_prompt_length,
            args.enable_thinking,
        ),
        "eval": validate_prompt_lengths(
            policy_tokenizer,
            eval_examples,
            "eval",
            args.max_prompt_length,
            args.enable_thinking,
        ),
        "accepted_maximum": args.max_prompt_length,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rubric_artifact(output_dir, args, settings)
    cache = None
    cache_path: Path | None = None
    if not args.no_judge_cache:
        cache_path = (
            Path(args.judge_cache_path)
            if args.judge_cache_path
            else output_dir / "judge_cache.sqlite3"
        )
        cache = JudgeCache(cache_path)
    judge = APIJudge(args, settings, stack["AsyncOpenAI"], cache)
    judge_reward = judge.reward_function()

    policy_quantization = quantization_config(
        stack,
        dtype,
        args.load_in_4bit,
    )
    policy_model, peft_config, policy_summary = build_policy_model_and_peft(
        args,
        stack,
        dtype,
        policy_quantization,
    )
    train_dataset = stack["Dataset"].from_list(train_examples)
    eval_dataset = (
        stack["Dataset"].from_list(eval_examples) if eval_examples else None
    )
    training_config = build_grpo_config(
        args,
        stack,
        torch,
        dtype,
        use_bf16,
        use_fp16,
        reporters,
        policy_model,
        has_eval=eval_dataset is not None,
    )

    trainer = stack["GRPOTrainer"](
        model=policy_model,
        reward_funcs=judge_reward,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=policy_tokenizer,
        peft_config=peft_config,
        quantization_config=(
            policy_quantization if isinstance(policy_model, str) else None
        ),
    )
    train_result = trainer.train(
        resume_from_checkpoint=resolve_resume(args.resume_from_checkpoint)
    )
    trainer.save_model(args.output_dir)
    trainer.save_state()
    policy_tokenizer.save_pretrained(args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    save_manifest(
        output_dir,
        args,
        settings,
        cache_path,
        policy_summary,
        data_summary,
        wandb_summary,
        train_result.metrics,
    )
    print(f"[done] saved API-Judge GRPO policy to {output_dir}")
