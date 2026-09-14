"""Command-line configuration for API-Judge GRPO training."""

from __future__ import annotations

import argparse

from .spec import APPRAISAL_DIMENSIONS, resolve_appraisal_dimensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO training for situation -> selected first-person appraisals -> "
            "CAREBench emotion, rewarded by an API LLM judge"
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help=(
            "Initial causal policy model or local PEFT adapter; required unless "
            "--preview_only or a schema-printing option is used"
        ),
    )
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument(
        "--eval_file",
        type=str,
        default="",
        help="Optional validation split; do not use the held-out test split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FirstPersonMethod/output/policy_api_judge_grpo",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--max_train_samples", type=int, default=None)
    data.add_argument("--max_eval_samples", type=int, default=None)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument(
        "--invalid_record_policy",
        choices=["error", "skip"],
        default="error",
    )
    data.add_argument("--allow_participant_overlap", action="store_true")
    data.add_argument(
        "--judge_reference_mode",
        choices=["none", "available", "required"],
        default="available",
        help=(
            "Whether the Judge receives hidden human CAREBench annotations; "
            "they are never included in the policy prompt"
        ),
    )
    data.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(APPRAISAL_DIMENSIONS),
        metavar="DIM1,DIM2,...",
        help=(
            "Comma-separated appraisal dimensions, or 'all'. Input order is "
            f"canonicalized to: {','.join(APPRAISAL_DIMENSIONS)}"
        ),
    )

    judge = parser.add_argument_group("API LLM Judge (OpenAI or GLM)")
    judge.add_argument(
        "--judge_provider",
        choices=["openai", "glm"],
        default="openai",
        help=(
            "API protocol profile. glm uses the official BigModel "
            "OpenAI-compatible endpoint and GLM-compatible request fields"
        ),
    )
    judge.add_argument("--judge_model", type=str, default="")
    judge.add_argument(
        "--judge_config",
        type=str,
        default="",
        help="Optional TOML with [providers.<section>] settings",
    )
    judge.add_argument(
        "--judge_provider_section",
        type=str,
        default="",
        help=(
            "TOML provider section; defaults to the value of "
            "--judge_provider"
        ),
    )
    judge.add_argument(
        "--judge_api_key_env",
        type=str,
        default="",
        help=(
            "Environment variable containing the API key. Defaults to "
            "OPENAI_API_KEY for OpenAI and ZAI_API_KEY (with legacy "
            "ZHIPUAI_API_KEY fallback) for GLM"
        ),
    )
    judge.add_argument(
        "--judge_base_url",
        type=str,
        default="",
        help=(
            "API base URL. GLM defaults to "
            "https://open.bigmodel.cn/api/paas/v4/"
        ),
    )
    judge.add_argument(
        "--judge_response_format",
        choices=["auto", "json_schema", "json_object", "text"],
        default="auto",
        help=(
            "auto uses strict json_schema for OpenAI and json_object for "
            "GLM; GLM automatically maps json_schema to json_object"
        ),
    )
    judge.add_argument(
        "--judge_token_parameter",
        choices=["auto", "max_completion_tokens", "max_tokens"],
        default="auto",
        help=(
            "auto uses max_completion_tokens for OpenAI and max_tokens for "
            "GLM"
        ),
    )
    judge.add_argument("--judge_max_tokens", type=int, default=2500)
    judge.add_argument("--judge_temperature", type=float, default=0.0)
    judge.add_argument("--judge_omit_temperature", action="store_true")
    judge.add_argument(
        "--judge_thinking",
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help=(
            "GLM Judge thinking mode. auto omits the provider-specific field; "
            "enabled/disabled sends thinking.type to the GLM API. This is "
            "independent of the policy model's --enable_thinking option"
        ),
    )
    judge.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    judge.add_argument("--judge_max_retries", type=int, default=3)
    judge.add_argument("--judge_parse_retries", type=int, default=1)
    judge.add_argument("--judge_max_concurrency", type=int, default=8)
    judge.add_argument(
        "--log_judge_raw_outputs",
        action="store_true",
        help=(
            "Print every raw Judge response as one [judge-raw] JSON line. "
            "Use only for diagnostics because responses may contain "
            "situation-derived text and substantially increase log volume"
        ),
    )
    judge.add_argument(
        "--judge_failure_policy",
        choices=["error", "skip"],
        default="skip",
        help=(
            "skip returns None so TRL excludes the candidate from all rewards; "
            "error raises for fail-fast training"
        ),
    )
    judge.add_argument("--judge_cache_path", type=str, default="")
    judge.add_argument("--no_judge_cache", action="store_true")

    reward = parser.add_argument_group("component rewards")
    reward.add_argument(
        "--invalid_component_reward",
        type=float,
        default=-1.0,
        help=(
            "Fixed value assigned to every reward component when Policy output "
            "fails the strict format hard gate; the API Judge is not called"
        ),
    )
    reward.add_argument("--appraisal_reward_weight", type=float, default=0.7)
    reward.add_argument("--coherence_reward_weight", type=float, default=0.1)
    reward.add_argument("--transition_reward_weight", type=float, default=0.1)
    reward.add_argument("--outcome_reward_weight", type=float, default=0.1)
    reward.add_argument("--outcome_label_weight", type=float, default=0.5)
    reward.add_argument("--outcome_intensity_weight", type=float, default=0.5)
    process_gate = reward.add_mutually_exclusive_group()
    process_gate.add_argument(
        "--use_process_gate",
        dest="use_process_gate",
        action="store_true",
        help="Enable process-gated outcome reward (default)",
    )
    process_gate.add_argument(
        "--no_process_gate",
        dest="use_process_gate",
        action="store_false",
        help="Disable the gate so outcome reward is not scaled by process quality",
    )
    parser.set_defaults(use_process_gate=True)
    reward.add_argument(
        "--process_gate_mode",
        choices=["min", "product", "geometric_mean"],
        default="min",
        help=(
            "Soft gate applied to gold outcome reward using appraisal and "
            "transition quality"
        ),
    )
    reward.add_argument(
        "--reward_normalization_epsilon",
        type=float,
        default=1.0e-4,
    )

    grpo = parser.add_argument_group("GRPO")
    grpo.add_argument("--num_train_epochs", type=float, default=1.0)
    grpo.add_argument("--max_steps", type=int, default=-1)
    grpo.add_argument("--learning_rate", type=float, default=1.0e-5)
    grpo.add_argument("--per_device_train_batch_size", type=int, default=1)
    grpo.add_argument("--per_device_eval_batch_size", type=int, default=4)
    grpo.add_argument("--gradient_accumulation_steps", type=int, default=8)
    grpo.add_argument("--warmup_ratio", type=float, default=0.03)
    grpo.add_argument("--weight_decay", type=float, default=0.0)
    grpo.add_argument("--num_generations", type=int, default=4)
    grpo.add_argument("--max_prompt_length", type=int, default=3072)
    grpo.add_argument("--max_completion_length", type=int, default=1536)
    grpo.add_argument("--temperature", type=float, default=0.8)
    grpo.add_argument("--top_p", type=float, default=0.95)
    grpo.add_argument("--beta", type=float, default=0.001)
    grpo.add_argument(
        "--loss_type",
        choices=["grpo", "dr_grpo", "dapo", "bnpo"],
        default="dapo",
    )
    grpo.add_argument(
        "--scale_rewards",
        choices=["group"],
        default="group",
        help=(
            "Process components and both outcome subrewards are normalized "
            "independently within the num_generations group; the process gate "
            "is applied afterward. TRL-level scaling is disabled to prevent a "
            "second normalization"
        ),
    )

    policy = parser.add_argument_group("policy model")
    policy.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    policy.add_argument("--full_finetune", action="store_true")
    policy.add_argument("--load_in_4bit", action="store_true")
    policy.add_argument("--policy_is_adapter", action="store_true")
    policy.add_argument("--lora_r", type=int, default=32)
    policy.add_argument("--lora_alpha", type=int, default=64)
    policy.add_argument("--lora_dropout", type=float, default=0.05)
    policy.add_argument("--lora_target_modules", type=str, default="all-linear")
    policy.add_argument("--no_gradient_checkpointing", action="store_true")
    policy.add_argument("--trust_remote_code", action="store_true")
    policy.add_argument("--tokenizer_name_or_path", type=str, default="")
    policy.add_argument("--chat_template_path", type=str, default="")
    policy.add_argument("--enable_thinking", action="store_true")

    logging = parser.add_argument_group("logging and checkpoints")
    logging.add_argument("--logging_steps", type=int, default=1)
    logging.add_argument("--eval_steps", type=int, default=25)
    logging.add_argument("--save_steps", type=int, default=25)
    logging.add_argument("--save_total_limit", type=int, default=2)
    logging.add_argument("--resume_from_checkpoint", type=str, default="")
    logging.add_argument("--deepspeed", type=str, default="")
    logging.add_argument("--report_to", type=str, default="none")
    logging.add_argument("--run_name", type=str, default="")
    logging.add_argument("--log_completions", action="store_true")
    logging.add_argument("--use_wandb", action="store_true")
    logging.add_argument("--wandb_project", type=str, default="")
    logging.add_argument("--wandb_entity", type=str, default="")
    logging.add_argument("--wandb_group", type=str, default="")
    logging.add_argument("--wandb_tags", type=str, default="")
    logging.add_argument("--wandb_notes", type=str, default="")
    logging.add_argument(
        "--wandb_mode",
        choices=["online", "offline"],
        default=None,
    )
    logging.add_argument(
        "--wandb_log_model",
        choices=["false", "end", "checkpoint"],
        default=None,
    )
    logging.add_argument("--wandb_run_id", type=str, default="")
    logging.add_argument(
        "--wandb_resume",
        choices=["never", "allow", "must", "auto"],
        default=None,
    )

    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--preview_samples", type=int, default=1)
    parser.add_argument("--print_output_schema", action="store_true")
    parser.add_argument("--print_judge_schema", action="store_true")
    return parser
