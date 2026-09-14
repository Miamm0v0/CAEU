#!/usr/bin/env python3
"""Train an appraisal-chain policy with a stepwise PRM and GRPO.

The policy generates the same structured JSON used by ``train_chain_sft.py``.
Each valid generation is deterministically converted into six PRM steps:

    relevance -> certainty -> congruence -> control -> accountability -> emotion

The PRM checkpoint produced by ``train_prm.py`` assigns a probability of
correctness at the final separator token of every step.  Those six scores are
aggregated into a trajectory reward and optimized with TRL's GRPOTrainer.

The sample-level rubric is deliberately not read here.  It is used upstream to
train the PRM; this script consumes only the frozen PRM and first-person events.

Inspect data and the exact policy/PRM representations without loading models::

    python FirstPersonMethod/scripts/train_policy_with_prm.py \
      --train_file train.json \
      --eval_file dev.json \
      --preview_only

Typical QLoRA run, initialized from a Chain-SFT adapter or merged checkpoint::

    accelerate launch FirstPersonMethod/scripts/train_policy_with_prm.py \
      --model_name_or_path FirstPersonMethod/output/chain_sft \
      --prm_model_name_or_path FirstPersonMethod/output/prm \
      --train_file train.json \
      --eval_file dev.json \
      --load_in_4bit \
      --prm_load_in_4bit \
      --use_wandb \
      --wandb_project first-person-appraisal-policy \
      --run_name qwen-chain-grpo-prm-v1 \
      --output_dir FirstPersonMethod/output/policy_prm_grpo

Important: PRM candidate steps used by ``train_prm.py`` should follow the same
canonical step text printed by ``--preview_only``.  The separator must also be
identical in PRM training and this script (newline by default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Sequence

from sft_common import (
    APPRAISAL_KEY_MAP,
    CHAIN_SYSTEM_PROMPT,
    CORE_APPRAISAL_ORDER,
    build_chain_example,
    build_chain_user_prompt,
    canonicalize_emotion_labels,
)
from train_prm import DEFAULT_PRM_PROMPT


CANONICAL_RATING_KEYS = [canonical for _, canonical in APPRAISAL_KEY_MAP]
EXPECTED_TOP_LEVEL_KEYS = {"appraisal_reasoning", "appraisal_ratings", "emotion"}
EXPECTED_EMOTION_KEYS = {
    "positive_intensity",
    "negative_intensity",
    "positive_labels",
    "negative_labels",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GRPO policy training with a frozen stepwise PRM for first-person "
            "event -> cognitive appraisal -> emotion chains"
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help=(
            "Initial causal policy model. A local Chain-SFT PEFT adapter is "
            "detected automatically; required unless --preview_only is used"
        ),
    )
    parser.add_argument(
        "--prm_model_name_or_path",
        type=str,
        default="",
        help="Frozen token-classification PRM checkpoint from train_prm.py",
    )
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument(
        "--eval_file",
        type=str,
        default="",
        help="Optional validation split. Never pass the held-out test split here",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="FirstPersonMethod/output/policy_prm_grpo",
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
    data.add_argument(
        "--allow_participant_overlap",
        action="store_true",
        help="Allow train/eval participant leakage only for a pipeline smoke test",
    )

    reward = parser.add_argument_group("PRM reward")
    reward.add_argument(
        "--step_separator",
        type=str,
        default=r"\n",
        help=r"Must match PRM training; literal \n, \t and \r are decoded",
    )
    reward.add_argument(
        "--reward_aggregation",
        choices=["geometric_mean", "mean", "minimum", "last"],
        default="geometric_mean",
    )
    reward.add_argument(
        "--step_weights",
        type=str,
        default="1,1,1,1,1,1",
        help=(
            "Six non-negative weights in relevance,certainty,congruence,control,"
            "accountability,emotion order"
        ),
    )
    reward.add_argument("--prm_reward_weight", type=float, default=1.0)
    reward.add_argument("--format_reward_weight", type=float, default=0.1)
    reward.add_argument(
        "--invalid_reward",
        type=float,
        default=0.0,
        help="PRM reward assigned to malformed or over-length generations",
    )
    reward.add_argument("--prm_batch_size", type=int, default=4)
    reward.add_argument("--prm_max_length", type=int, default=4096)
    reward.add_argument(
        "--prm_max_completion_length",
        type=int,
        default=3072,
        help="Must match PRM training; use -1 only if PRM training was unlimited",
    )
    reward.add_argument(
        "--prm_positive_label_id",
        type=int,
        default=-1,
        help="-1 infers the 'correct' label id from the PRM config",
    )
    reward.add_argument(
        "--prm_device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:N; auto uses the current process GPU",
    )
    reward.add_argument(
        "--prm_dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    reward.add_argument("--prm_load_in_4bit", action="store_true")
    reward.add_argument(
        "--prm_is_adapter",
        action="store_true",
        help="Treat a non-local/Hugging Face PRM id as a PEFT adapter",
    )

    grpo = parser.add_argument_group("GRPO")
    grpo.add_argument("--num_train_epochs", type=float, default=1.0)
    grpo.add_argument("--max_steps", type=int, default=-1)
    grpo.add_argument("--learning_rate", type=float, default=1.0e-5)
    grpo.add_argument("--per_device_train_batch_size", type=int, default=1)
    grpo.add_argument("--per_device_eval_batch_size", type=int, default=1)
    grpo.add_argument("--gradient_accumulation_steps", type=int, default=8)
    grpo.add_argument("--warmup_ratio", type=float, default=0.03)
    grpo.add_argument("--weight_decay", type=float, default=0.0)
    grpo.add_argument("--num_generations", type=int, default=4)
    grpo.add_argument(
        "--max_prompt_length",
        type=int,
        default=2048,
        help=(
            "Maximum accepted rendered prompt length. TRL 1.8 does not truncate "
            "GRPO prompts, so this script rejects longer samples instead"
        ),
    )
    grpo.add_argument("--max_completion_length", type=int, default=2048)
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
        choices=["group", "batch", "none"],
        default="group",
    )

    policy = parser.add_argument_group("policy model")
    policy.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
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
    policy.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Qwen3 thinking mode; disabled by default to preserve strict JSON",
    )

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
        "--wandb_mode", choices=["online", "offline"], default=None
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
    parser.add_argument(
        "--print_output_schema",
        action="store_true",
        help="Print the required policy JSON schema and exit",
    )
    return parser


def decode_separator(value: str) -> str:
    replacements = {r"\n": "\n", r"\t": "\t", r"\r": "\r"}
    decoded = value
    for escaped, actual in replacements.items():
        decoded = decoded.replace(escaped, actual)
    if not decoded:
        raise ValueError("--step_separator must not be empty")
    return decoded


def parse_step_weights(value: str) -> list[float]:
    try:
        weights = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--step_weights must contain six numbers") from exc
    if len(weights) != 6:
        raise ValueError("--step_weights must contain exactly six values")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("--step_weights values must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValueError("At least one --step_weights value must be positive")
    return weights


def output_schema() -> dict[str, Any]:
    return {
        "appraisal_reasoning": {
            dimension: f"concise first-person {dimension} reasoning"
            for dimension in CORE_APPRAISAL_ORDER
        },
        "appraisal_ratings": {key: 3 for key in CANONICAL_RATING_KEYS},
        "emotion": {
            "positive_intensity": 0,
            "negative_intensity": 0,
            "positive_labels": [],
            "negative_labels": [],
        },
    }


def load_records(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty JSON array or JSONL")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Every item in {path} must be an object")
    return records


def stable_limit(
    records: list[dict[str, Any]], limit: int | None, seed: int, namespace: str
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0 or limit >= len(records):
        return records

    def key(record: dict[str, Any]) -> str:
        identity = record.get("id", record.get("sample_id", "<missing>"))
        return hashlib.sha256(
            f"{namespace}:{seed}:{identity}".encode("utf-8")
        ).hexdigest()

    return sorted(records, key=key)[:limit]


def participant_id(record: dict[str, Any], sample_id: str) -> str:
    explicit = record.get("participant_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return sample_id.split("_", 1)[0]


def normalize_policy_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    sample_id = record.get("id", record.get("sample_id"))
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"item {index}: id must be a non-empty string")
    sample_id = sample_id.strip()
    situation = record.get("situation", record.get("event"))
    if not isinstance(situation, str) or not situation.strip():
        raise ValueError(f"{sample_id}: situation must be a non-empty string")
    perspective = record.get("perspective")
    if perspective is not None and perspective != "first_person":
        raise ValueError(f"{sample_id}: expected first_person perspective")
    return {
        "sample_id": sample_id,
        "participant_id": participant_id(record, sample_id),
        "situation": situation.strip(),
        "prompt": [
            {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
            {"role": "user", "content": build_chain_user_prompt(situation.strip())},
        ],
    }


def prepare_split(
    records: list[dict[str, Any]], split_name: str, invalid_policy: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            example = normalize_policy_record(record, index)
            if example["sample_id"] in seen:
                raise ValueError(
                    f"{split_name}: duplicate id {example['sample_id']}"
                )
            seen.add(example["sample_id"])
            examples.append(example)
        except (TypeError, ValueError) as exc:
            if invalid_policy == "error":
                raise
            errors.append(
                {
                    "item": str(index),
                    "id": str(record.get("id", record.get("sample_id", "<unknown>"))),
                    "error": str(exc),
                }
            )
    if not examples:
        raise ValueError(f"No valid {split_name} records remain")
    if errors:
        print(f"[data] split={split_name} skipped_invalid={len(errors)}")
        for error in errors[:10]:
            print(f"[data] skipped id={error['id']} error={error['error']}")
    return examples, errors


def reject_split_overlap(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    allow_participant_overlap: bool,
) -> None:
    if not evaluation:
        return
    sample_overlap = sorted(
        {item["sample_id"] for item in train}
        & {item["sample_id"] for item in evaluation}
    )
    if sample_overlap:
        raise ValueError(
            f"Train/eval sample overlap detected (first ids: {sample_overlap[:10]})"
        )
    participant_overlap = sorted(
        {item["participant_id"] for item in train}
        & {item["participant_id"] for item in evaluation}
    )
    if participant_overlap and not allow_participant_overlap:
        raise ValueError(
            "Train/eval participant overlap detected (first ids: {}). Re-split by "
            "participant; use --allow_participant_overlap only for a smoke test.".format(
                participant_overlap[:10]
            )
        )


def prepare_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_records = stable_limit(
        load_records(args.train_file), args.max_train_samples, args.seed, "train"
    )
    eval_records = (
        stable_limit(
            load_records(args.eval_file), args.max_eval_samples, args.seed, "eval"
        )
        if args.eval_file
        else []
    )
    train, train_errors = prepare_split(
        train_records, "train", args.invalid_record_policy
    )
    if eval_records:
        evaluation, eval_errors = prepare_split(
            eval_records, "eval", args.invalid_record_policy
        )
    else:
        evaluation, eval_errors = [], []
    reject_split_overlap(train, evaluation, args.allow_participant_overlap)
    summary = {
        "train_samples": len(train),
        "eval_samples": len(evaluation),
        "train_participants": len({item["participant_id"] for item in train}),
        "eval_participants": len(
            {item["participant_id"] for item in evaluation}
        ),
        "train_skipped_invalid": len(train_errors),
        "eval_skipped_invalid": len(eval_errors),
    }
    return train, evaluation, summary


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        return content if isinstance(content, str) else ""
    if isinstance(completion, list):
        parts = [
            item.get("content", "")
            for item in completion
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        if not parts and len(completion) == 1 and isinstance(completion[0], dict):
            parts = [completion[0].get("content", "")]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def require_exact_keys(value: Any, expected: set[str], location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys mismatch: missing={missing}, extra={extra}")
    return value


def require_bounded_int(value: Any, minimum: int, maximum: int, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{location} must be in [{minimum}, {maximum}]")
    return value


def parse_policy_chain(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty completion")
    try:
        payload = json.loads(
            stripped, object_pairs_hook=_object_without_duplicate_keys
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    require_exact_keys(payload, EXPECTED_TOP_LEVEL_KEYS, "root")

    reasoning = require_exact_keys(
        payload["appraisal_reasoning"], set(CORE_APPRAISAL_ORDER), "appraisal_reasoning"
    )
    for dimension in CORE_APPRAISAL_ORDER:
        value = reasoning[dimension]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"appraisal_reasoning.{dimension} must be non-empty")
        reasoning[dimension] = value.strip()

    ratings = require_exact_keys(
        payload["appraisal_ratings"], set(CANONICAL_RATING_KEYS), "appraisal_ratings"
    )
    for key in CANONICAL_RATING_KEYS:
        require_bounded_int(ratings[key], 1, 5, f"appraisal_ratings.{key}")

    emotion = require_exact_keys(
        payload["emotion"], EXPECTED_EMOTION_KEYS, "emotion"
    )
    require_bounded_int(
        emotion["positive_intensity"], 0, 6, "emotion.positive_intensity"
    )
    require_bounded_int(
        emotion["negative_intensity"], 0, 6, "emotion.negative_intensity"
    )
    emotion["positive_labels"] = canonicalize_emotion_labels(
        emotion["positive_labels"],
        "positive",
        field="emotion.positive_labels",
        reject_duplicates=False,
    )
    emotion["negative_labels"] = canonicalize_emotion_labels(
        emotion["negative_labels"],
        "negative",
        field="emotion.negative_labels",
        reject_duplicates=False,
    )
    return payload


def chain_to_prm_steps(chain: dict[str, Any]) -> list[str]:
    reasoning = chain["appraisal_reasoning"]
    ratings = chain["appraisal_ratings"]
    steps: list[str] = []
    for dimension in CORE_APPRAISAL_ORDER:
        dimension_ratings = {
            key: ratings[key]
            for key in CANONICAL_RATING_KEYS
            if key.startswith(dimension + ".")
        }
        content = {
            "reasoning": reasoning[dimension],
            "ratings": dimension_ratings,
        }
        steps.append(
            f"[{dimension}] "
            + json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        )
    steps.append(
        "[emotion] "
        + json.dumps(chain["emotion"], ensure_ascii=False, separators=(",", ":"))
    )
    return steps


def gold_steps_for_preview(record: dict[str, Any]) -> list[str] | None:
    try:
        example = build_chain_example(record)
        chain = parse_policy_chain(example["completion"][0]["content"])
        return chain_to_prm_steps(chain)
    except (KeyError, TypeError, ValueError):
        return None


def print_preview(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    raw_train_records: list[dict[str, Any]],
    summary: dict[str, Any],
    count: int,
) -> None:
    print("[preview] policy-with-PRM GRPO data")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raw_by_id = {
        str(record.get("id", record.get("sample_id", ""))): record
        for record in raw_train_records
    }
    for index, example in enumerate(train_examples[: max(0, count)]):
        print(f"\n===== TRAIN POLICY SAMPLE {index} =====")
        print(json.dumps(example, ensure_ascii=False, indent=2))
        raw = raw_by_id.get(example["sample_id"])
        steps = gold_steps_for_preview(raw) if raw is not None else None
        if steps:
            print("\n===== GOLD CHAIN AS CANONICAL PRM STEPS =====")
            print(json.dumps(steps, ensure_ascii=False, indent=2))
    if eval_examples and count > 0:
        print("\n===== EVAL POLICY SAMPLE 0 =====")
        print(json.dumps(eval_examples[0], ensure_ascii=False, indent=2))


def aggregate_step_scores(
    scores: list[float], method: str, weights: list[float]
) -> float:
    if len(scores) != 6:
        raise ValueError(f"Expected six PRM step scores, got {len(scores)}")
    if method == "minimum":
        return min(scores)
    if method == "last":
        return scores[-1]
    total_weight = sum(weights)
    if method == "mean":
        return sum(weight * score for weight, score in zip(weights, scores)) / total_weight
    if method == "geometric_mean":
        epsilon = 1.0e-8
        weighted_log = sum(
            weight * math.log(max(epsilon, score))
            for weight, score in zip(weights, scores)
        )
        return math.exp(weighted_log / total_weight)
    raise ValueError(f"Unknown reward aggregation: {method}")


def resolve_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested PRM device {device}, but CUDA is unavailable")
    return device


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


def local_adapter_checkpoint(path_value: str) -> bool:
    path = Path(path_value)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def first_real_device(model: Any, fallback: Any) -> Any:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return fallback


class ProcessReward:
    """Frozen PRM callable compatible with TRL custom reward functions."""

    __name__ = "process_reward"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch: Any,
        device: Any,
        separator: str,
        max_length: int,
        max_completion_length: int | None,
        batch_size: int,
        positive_label_id: int,
        aggregation: str,
        step_weights: list[float],
        invalid_reward: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.device = first_real_device(model, device)
        self.separator = separator
        self.max_length = max_length
        self.max_completion_length = max_completion_length
        self.batch_size = batch_size
        self.positive_label_id = positive_label_id
        self.aggregation = aggregation
        self.step_weights = step_weights
        self.invalid_reward = invalid_reward

    def encode_trajectory(
        self, situation: str, steps: list[str]
    ) -> tuple[list[int], list[int]] | None:
        prompt = DEFAULT_PRM_PROMPT.format(situation=situation)
        if not prompt.endswith(self.separator):
            prompt += self.separator
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if self.tokenizer.bos_token_id is not None:
            prompt_ids = [self.tokenizer.bos_token_id] + prompt_ids
        separator_ids = self.tokenizer.encode(
            self.separator, add_special_tokens=False
        )
        if not separator_ids:
            raise ValueError("PRM tokenizer maps --step_separator to zero tokens")

        input_ids = list(prompt_ids)
        completion_start = len(input_ids)
        score_positions: list[int] = []
        for step in steps:
            step_ids = self.tokenizer(step, add_special_tokens=False)["input_ids"]
            input_ids.extend(step_ids)
            input_ids.extend(separator_ids)
            score_positions.append(len(input_ids) - 1)
        if (
            self.max_completion_length is not None
            and len(input_ids) - completion_start > self.max_completion_length
        ):
            return None
        if len(input_ids) > self.max_length:
            return None
        return input_ids, score_positions

    def score_encoded(
        self, encoded: list[tuple[list[int], list[int]]]
    ) -> list[list[float]]:
        all_scores: list[list[float]] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("PRM tokenizer has neither pad token nor configured fallback")
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start : start + self.batch_size]
            max_size = max(len(input_ids) for input_ids, _ in chunk)
            padded_ids: list[list[int]] = []
            masks: list[list[int]] = []
            for input_ids, _ in chunk:
                padding = max_size - len(input_ids)
                padded_ids.append(input_ids + [pad_id] * padding)
                masks.append([1] * len(input_ids) + [0] * padding)
            input_tensor = self.torch.tensor(
                padded_ids, dtype=self.torch.long, device=self.device
            )
            mask_tensor = self.torch.tensor(
                masks, dtype=self.torch.long, device=self.device
            )
            with self.torch.inference_mode():
                logits = self.model(
                    input_ids=input_tensor, attention_mask=mask_tensor
                ).logits
                probabilities = self.torch.softmax(logits.float(), dim=-1)
            for row, (_, positions) in enumerate(chunk):
                scores = [
                    float(probabilities[row, position, self.positive_label_id].item())
                    for position in positions
                ]
                all_scores.append(scores)
        return all_scores

    def __call__(
        self,
        completions: list[Any],
        situation: list[str],
        log_extra: Callable[..., Any] | None = None,
        log_metric: Callable[..., Any] | None = None,
        **_: Any,
    ) -> list[float]:
        if len(completions) != len(situation):
            raise ValueError(
                "GRPO did not align completions with the situation dataset column"
            )
        rewards = [self.invalid_reward] * len(completions)
        valid_flags = [False] * len(completions)
        score_logs = ["[]"] * len(completions)
        valid_indices: list[int] = []
        encoded: list[tuple[list[int], list[int]]] = []
        for index, (completion, event) in enumerate(zip(completions, situation)):
            try:
                chain = parse_policy_chain(completion_text(completion))
                item = self.encode_trajectory(event, chain_to_prm_steps(chain))
                if item is None:
                    continue
                valid_indices.append(index)
                encoded.append(item)
                valid_flags[index] = True
            except (TypeError, ValueError):
                continue

        if encoded:
            for original_index, scores in zip(
                valid_indices, self.score_encoded(encoded)
            ):
                rewards[original_index] = aggregate_step_scores(
                    scores, self.aggregation, self.step_weights
                )
                score_logs[original_index] = json.dumps(
                    [round(score, 6) for score in scores], separators=(",", ":")
                )
        if log_extra is not None:
            log_extra("prm_valid", valid_flags)
            log_extra("prm_step_scores", score_logs)
        if log_metric is not None and rewards:
            log_metric("prm_valid_rate", sum(valid_flags) / len(valid_flags))
            log_metric("prm_trajectory_reward", sum(rewards) / len(rewards))
        return rewards


def format_reward(
    completions: list[Any],
    log_extra: Callable[..., Any] | None = None,
    log_metric: Callable[..., Any] | None = None,
    **_: Any,
) -> list[float]:
    """Strict schema reward; semantic correctness is left entirely to the PRM."""
    rewards: list[float] = []
    errors: list[str] = []
    for completion in completions:
        try:
            parse_policy_chain(completion_text(completion))
            rewards.append(1.0)
            errors.append("")
        except (TypeError, ValueError) as exc:
            rewards.append(0.0)
            errors.append(str(exc)[:200])
    if log_extra is not None:
        log_extra("format_error", errors)
    if log_metric is not None and rewards:
        log_metric("strict_format_rate", sum(rewards) / len(rewards))
    return rewards


def import_training_stack(use_wandb: bool, need_bitsandbytes: bool) -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftConfig,
            PeftModel,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForTokenClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Policy training requires torch, transformers, datasets, peft, and trl"
        ) from exc
    if need_bitsandbytes:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "4-bit loading requires bitsandbytes: pip install bitsandbytes"
            ) from exc
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("W&B logging requires: pip install wandb") from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftConfig": PeftConfig,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForTokenClassification": AutoModelForTokenClassification,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def parse_reporters(value: str) -> list[str]:
    if not value.strip() or value.strip().lower() == "none":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_wandb(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
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
    if requested and not any(item.lower() == "wandb" for item in reporters):
        reporters.append("wandb")
    if not requested:
        return reporters, {"enabled": False}

    variables = {
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "WANDB_RUN_GROUP": args.wandb_group,
        "WANDB_TAGS": args.wandb_tags,
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
    if args.wandb_resume in {"allow", "must"} and not os.environ.get("WANDB_RUN_ID"):
        raise ValueError(
            f"--wandb_resume {args.wandb_resume} requires --wandb_run_id"
        )
    os.environ.setdefault("WANDB_JOB_TYPE", "policy-grpo-with-prm")
    summary = {
        "enabled": True,
        "project": os.environ.get("WANDB_PROJECT") or None,
        "entity": os.environ.get("WANDB_ENTITY") or None,
        "group": os.environ.get("WANDB_RUN_GROUP") or None,
        "mode": os.environ.get("WANDB_MODE", "online"),
        "run_id": os.environ.get("WANDB_RUN_ID") or None,
        "resume": os.environ.get("WANDB_RESUME") or None,
    }
    return reporters, summary


def parse_target_modules(value: str) -> str | list[str]:
    if value.strip() == "all-linear":
        return "all-linear"
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must not be empty")
    return modules


def quantization_config(stack: dict[str, Any], dtype: Any, enabled: bool) -> Any:
    if not enabled:
        return None
    return stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def load_tokenizer(
    stack: dict[str, Any], source: str, trust_remote_code: bool
) -> Any:
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        source, trust_remote_code=trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {source} has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def infer_positive_label_id(model: Any, requested: int) -> int:
    num_labels = int(getattr(model.config, "num_labels", 0))
    if requested >= 0:
        if requested >= num_labels:
            raise ValueError(
                f"--prm_positive_label_id={requested} but PRM has {num_labels} labels"
            )
        return requested
    label2id = getattr(model.config, "label2id", {}) or {}
    normalized = {str(key).lower(): int(value) for key, value in label2id.items()}
    for name in ("correct", "positive", "pass", "label_1"):
        if name in normalized:
            return normalized[name]
    if num_labels == 2:
        return 1
    raise ValueError(
        "Cannot infer the PRM correct label; pass --prm_positive_label_id"
    )


def load_prm(
    args: argparse.Namespace,
    stack: dict[str, Any],
    torch: Any,
    device: Any,
) -> tuple[Any, Any, int, dict[str, Any]]:
    source = args.prm_model_name_or_path
    prm_dtype, _, _ = choose_dtype(torch, args.prm_dtype)
    quantization = quantization_config(
        stack, prm_dtype, args.prm_load_in_4bit
    )
    if args.prm_load_in_4bit and device.type != "cuda":
        raise ValueError("--prm_load_in_4bit requires a CUDA --prm_device")
    model_kwargs: dict[str, Any] = {
        "dtype": prm_dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
        model_kwargs["device_map"] = {"": device.index or 0}

    is_adapter = args.prm_is_adapter or local_adapter_checkpoint(source)
    if is_adapter:
        adapter_config = stack["PeftConfig"].from_pretrained(source)
        base_source = adapter_config.base_model_name_or_path
        base_model = stack["AutoModelForTokenClassification"].from_pretrained(
            base_source, num_labels=2, **model_kwargs
        )
        model = stack["PeftModel"].from_pretrained(
            base_model, source, is_trainable=False
        )
        tokenizer_source = source
    else:
        model = stack["AutoModelForTokenClassification"].from_pretrained(
            source, **model_kwargs
        )
        tokenizer_source = source
    if quantization is None:
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        tokenizer = load_tokenizer(
            stack, tokenizer_source, args.trust_remote_code
        )
    except OSError:
        if not is_adapter:
            raise
        tokenizer = load_tokenizer(stack, base_source, args.trust_remote_code)
    tokenizer.padding_side = "right"
    positive_label_id = infer_positive_label_id(model, args.prm_positive_label_id)
    summary = {
        "checkpoint": source,
        "is_adapter": is_adapter,
        "device": str(device),
        "dtype": str(prm_dtype),
        "load_in_4bit": args.prm_load_in_4bit,
        "positive_label_id": positive_label_id,
    }
    return model, tokenizer, positive_label_id, summary


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
    template_tokenizer = load_tokenizer(stack, template_path, trust_remote_code)
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
            f"{split_name} contains {len(over_limit)} rendered prompts longer than "
            f"--max_prompt_length={max_prompt_length}; first samples: "
            f"{over_limit[:10]}. Increase the limit/model context or shorten the "
            "prompt without silently truncating event evidence."
        )
    return {
        "minimum": min(lengths) if lengths else 0,
        "maximum": max(lengths) if lengths else 0,
        "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
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
                "A PEFT Chain-SFT adapter cannot be full-finetuned directly; merge "
                "it into the base model first or remove --full_finetune"
            )
        adapter_config = stack["PeftConfig"].from_pretrained(source)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if policy_quantization is not None:
            model_kwargs["quantization_config"] = policy_quantization
            model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
        base = stack["AutoModelForCausalLM"].from_pretrained(
            adapter_config.base_model_name_or_path, **model_kwargs
        )
        if policy_quantization is not None:
            base = stack["prepare_model_for_kbit_training"](
                base,
                use_gradient_checkpointing=not args.no_gradient_checkpointing,
            )
        model = stack["PeftModel"].from_pretrained(
            base, source, is_trainable=True
        )
        return model, None, {
            "checkpoint": source,
            "is_existing_adapter": True,
            "base_model": adapter_config.base_model_name_or_path,
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


def validate_training_arguments(args: argparse.Namespace) -> None:
    if not args.model_name_or_path:
        raise ValueError("--model_name_or_path is required for training")
    if not args.prm_model_name_or_path:
        raise ValueError("--prm_model_name_or_path is required for training")
    if args.load_in_4bit and args.full_finetune:
        raise ValueError("--load_in_4bit requires LoRA; remove --full_finetune")
    for name in (
        "num_generations",
        "max_prompt_length",
        "max_completion_length",
        "prm_batch_size",
        "prm_max_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.prm_max_completion_length == 0 or args.prm_max_completion_length < -1:
        raise ValueError(
            "--prm_max_completion_length must be positive or -1 for unlimited"
        )
    if not 0 < args.temperature:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    for name in ("prm_reward_weight", "format_reward_weight", "invalid_reward"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name} must be finite")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = (
        world_size
        * args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "GRPO effective batch size (WORLD_SIZE * per-device batch * gradient "
            f"accumulation = {effective_batch}) must be divisible by "
            f"--num_generations={args.num_generations}"
        )


def run_training(
    args: argparse.Namespace,
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    data_summary: dict[str, Any],
    step_weights: list[float],
) -> None:
    validate_training_arguments(args)
    reporters, wandb_summary = configure_wandb(args)
    stack = import_training_stack(
        use_wandb=wandb_summary["enabled"],
        need_bitsandbytes=args.load_in_4bit or args.prm_load_in_4bit,
    )
    torch = stack["torch"]
    if (args.load_in_4bit or args.prm_load_in_4bit) and not torch.cuda.is_available():
        raise RuntimeError("4-bit policy/PRM loading requires a CUDA GPU")
    dtype, use_bf16, use_fp16 = choose_dtype(torch, args.dtype)

    tokenizer_source = args.tokenizer_name_or_path or args.model_name_or_path
    policy_tokenizer = load_tokenizer(
        stack, tokenizer_source, args.trust_remote_code
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
            "The policy tokenizer has no chat template; pass --chat_template_path"
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

    prm_device = resolve_device(torch, args.prm_device)
    prm_model, prm_tokenizer, positive_label_id, prm_summary = load_prm(
        args, stack, torch, prm_device
    )
    process_reward = ProcessReward(
        model=prm_model,
        tokenizer=prm_tokenizer,
        torch=torch,
        device=prm_device,
        separator=args.step_separator,
        max_length=args.prm_max_length,
        max_completion_length=(
            None
            if args.prm_max_completion_length == -1
            else args.prm_max_completion_length
        ),
        batch_size=args.prm_batch_size,
        positive_label_id=positive_label_id,
        aggregation=args.reward_aggregation,
        step_weights=step_weights,
        invalid_reward=args.invalid_reward,
    )

    policy_quantization = quantization_config(
        stack, dtype, args.load_in_4bit
    )
    policy_model, peft_config, policy_summary = build_policy_model_and_peft(
        args, stack, dtype, policy_quantization
    )
    train_dataset = stack["Dataset"].from_list(train_examples)
    eval_dataset = (
        stack["Dataset"].from_list(eval_examples) if eval_examples else None
    )

    config_kwargs: dict[str, Any] = {
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
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "scale_rewards": args.scale_rewards,
        "mask_truncated_completions": True,
        "reward_weights": [args.prm_reward_weight, args.format_reward_weight],
        "remove_unused_columns": False,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "tf32": bool(
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        ),
        "report_to": reporters,
        "run_name": args.run_name or "policy-prm-grpo",
        "seed": args.seed,
        "data_seed": args.seed,
        "log_completions": args.log_completions,
        "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
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
    if eval_dataset is not None:
        config_kwargs["eval_steps"] = args.eval_steps
    if args.deepspeed:
        config_kwargs["deepspeed"] = args.deepspeed
    training_config = stack["GRPOConfig"](**config_kwargs)

    trainer = stack["GRPOTrainer"](
        model=policy_model,
        reward_funcs=[process_reward, format_reward],
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "first_person_policy_grpo_with_stepwise_prm",
        "policy": policy_summary,
        "prm": prm_summary,
        "reward": {
            "step_order": CORE_APPRAISAL_ORDER + ["emotion"],
            "step_separator": args.step_separator,
            "aggregation": args.reward_aggregation,
            "step_weights": step_weights,
            "prm_weight": args.prm_reward_weight,
            "format_weight": args.format_reward_weight,
            "invalid_reward": args.invalid_reward,
        },
        "data": data_summary,
        "wandb": wandb_summary,
        "arguments": vars(args),
        "train_metrics": train_result.metrics,
    }
    with (output_dir / "policy_prm_training_manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[done] saved PRM-optimized policy to {output_dir}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.print_output_schema:
        print(json.dumps(output_schema(), ensure_ascii=False, indent=2))
        return 0
    args.step_separator = decode_separator(args.step_separator)
    step_weights = parse_step_weights(args.step_weights)

    # Keep raw records only for rendering a gold canonical-step preview.  They
    # are never added to the RL dataset or exposed to the policy reward.
    raw_train_records = stable_limit(
        load_records(args.train_file), args.max_train_samples, args.seed, "train"
    )
    train_examples, eval_examples, summary = prepare_data(args)
    summary["reward_step_order"] = CORE_APPRAISAL_ORDER + ["emotion"]
    summary["step_weights"] = step_weights
    if args.preview_only:
        print_preview(
            train_examples,
            eval_examples,
            raw_train_records,
            summary,
            args.preview_samples,
        )
        return 0
    run_training(args, train_examples, eval_examples, summary, step_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
