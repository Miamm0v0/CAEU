#!/usr/bin/env python3
"""Shared data preparation and TRL training utilities for first-person SFT.

The module deliberately imports the heavy training stack only when training
starts.  Data validation and ``--preview_only`` therefore work in a lightweight
Python environment without PyTorch or Hugging Face packages installed.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


CORE_APPRAISAL_ORDER = [
    "relevance",
    "certainty",
    "congruence",
    "control",
    "accountability",
]

# The organized file uses compact keys and a 0-4 scale.  SFT targets use the
# original CAREBench dimension names and 1-5 scale so their predictions can be
# compared directly with the benchmark outputs.
APPRAISAL_KEY_MAP = [
    ("A1_relevance_general", "relevance.general"),
    ("A2_relevance_urgency", "relevance.urgency"),
    ("A3_relevance_goals", "relevance.goals"),
    ("A4_relevance_physical_wellbeing", "relevance.bodily motives"),
    ("A5_relevance_people", "relevance.social motives"),
    ("A6_relevance_identity", "relevance.identity motives"),
    ("B1_certainty_clarity", "certainty.construal"),
    ("B2_certainty_next", "certainty.outlook"),
    ("B3_certainty_expected", "certainty.predictability"),
    ("B4_certainty_novelty", "certainty.novelty"),
    ("C1_congruence_good", "congruence.general"),
    ("C2_congruence_improvement", "congruence.outlook"),
    ("C3_congruence_better_than_expected", "congruence.positive prediction error"),
    ("C4_congruence_worse_than_expected", "congruence.negative prediction error"),
    ("D1_control_general", "control.general"),
    ("D2_control_stay_or_leave", "control.select"),
    ("D3_control_others", "control.vicarious"),
    ("D4_control_effort", "control.effortful"),
    ("E1_accountability_self", "accountability.self"),
    ("E2_accountability_others", "accountability.other"),
    ("E3_accountability_intentional", "accountability.intentionality"),
    ("E4_accountability_fairness", "accountability.fairness"),
]

POSITIVE_LABEL_MAP = {
    "hopeful": "Hopeful, optimistic, encouraged",
    "grateful": "Grateful, appreciative, thankful",
    "glad": "Glad, happy, joyful",
    "love": "Love, closeness, trust",
    "amused": "Amused, entertained",
    "sympathy": "Sympathy, concern, compassion",
    "inspiration": "Inspiration, admiration, appreciation",
    "calm": "Calm, peaceful, relaxed",
    "proud": "Proud, confident, determined",
    "excited": "Excited, enthusiastic, elated",
}

NEGATIVE_LABEL_MAP = {
    "angry": "Angry, frustrated, annoyed",
    "worried": "Worried, nervous, fearful",
    "sad": "Sad, downhearted, unhappy",
    "disgust": "Disgust, distaste, revulsion",
    "despair": "Despair, hopelessness, sorrow",
    "sorrow": "Despair, hopelessness, sorrow",
    "ashamed": "Ashamed, humiliated, embarrassed",
    "lonely": "Lonely, isolated, disconnected from others",
    "panicked": "Panicked, alarmed, freaked out",
    "guilty": "Guilty, blameworthy, repentant",
    "confused": "Confused, disoriented, surprised",
}

POSITIVE_LABELS = list(POSITIVE_LABEL_MAP.values())
NEGATIVE_LABELS = []
for _label in NEGATIVE_LABEL_MAP.values():
    if _label not in NEGATIVE_LABELS:
        NEGATIVE_LABELS.append(_label)

DIRECT_SYSTEM_PROMPT = (
    "You infer the emotions experienced by the first-person author immediately "
    "after an event. Use only evidence from the event. Return exactly one valid "
    "JSON object without markdown or additional commentary."
)

CHAIN_SYSTEM_PROMPT = (
    "You reason about first-person emotion through a structured cognitive appraisal "
    "chain. Analyze relevance, certainty, congruence, control, and accountability "
    "before inferring emotion. Keep every claim grounded in the event and return "
    "exactly one valid JSON object without markdown or additional commentary."
)


def _label_lines(values):
    return "\n".join("- " + value for value in values)


def build_direct_user_prompt(situation):
    return """Imagine that you are the person who wrote the event below. Infer how
you felt immediately after it ended.

Event:
{situation}

Return a JSON object with exactly these fields:
- positive_intensity: integer from 0 to 6
- negative_intensity: integer from 0 to 6
- positive_labels: zero or more labels from the positive list
- negative_labels: zero or more labels from the negative list

Positive labels:
{positive_labels}

Negative labels:
{negative_labels}
""".format(
        situation=situation,
        positive_labels=_label_lines(POSITIVE_LABELS),
        negative_labels=_label_lines(NEGATIVE_LABELS),
    ).strip()


def build_chain_user_prompt(situation):
    rating_keys = [canonical for _, canonical in APPRAISAL_KEY_MAP]
    return """Imagine that you are the person who wrote the event below. Produce the
complete event-to-appraisal-to-emotion chain.

Event:
{situation}

Reason in this order: relevance, certainty, congruence, control, accountability,
then emotion. The five appraisal_reasoning values must be concise first-person
explanations grounded in the event. The 22 appraisal_ratings must use integers
from 1 to 5, where 1 means Strongly disagree and 5 means Strongly agree.

Use exactly these appraisal_ratings keys:
{rating_keys}

Emotion intensity must be an integer from 0 to 6. Emotion labels must come only
from these lists.

Positive labels:
{positive_labels}

Negative labels:
{negative_labels}

Return exactly this top-level structure:
{{
  "appraisal_reasoning": {{...}},
  "appraisal_ratings": {{...}},
  "emotion": {{
    "positive_intensity": 0,
    "negative_intensity": 0,
    "positive_labels": [],
    "negative_labels": []
  }}
}}
""".format(
        situation=situation,
        rating_keys="\n".join("- " + key for key in rating_keys),
        positive_labels=_label_lines(POSITIVE_LABELS),
        negative_labels=_label_lines(NEGATIVE_LABELS),
    ).strip()


def _require_nonempty_string(value, record_id, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{}: {} must be a non-empty string".format(record_id, field))
    return value.strip()


def _require_int(value, minimum, maximum, record_id, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{}: {} must be an integer".format(record_id, field))
    if value < minimum or value > maximum:
        raise ValueError(
            "{}: {}={} is outside [{}, {}]".format(
                record_id, field, value, minimum, maximum
            )
        )
    return value


def _canonicalize_labels(raw_labels, mapping, allowed, record_id, field):
    if not isinstance(raw_labels, list):
        raise ValueError("{}: {} must be an array".format(record_id, field))

    selected = set()
    for raw_label in raw_labels:
        if not isinstance(raw_label, str):
            raise ValueError("{}: {} contains a non-string label".format(record_id, field))
        label = raw_label.strip()
        if label in allowed:
            selected.add(label)
        elif label.lower() in mapping:
            selected.add(mapping[label.lower()])
        else:
            raise ValueError(
                "{}: unknown label {!r} in {}".format(record_id, raw_label, field)
            )
    return [label for label in allowed if label in selected]


def normalize_emotion(record):
    record_id = str(record.get("id", "<unknown>"))
    emotion = record.get("emotion")
    if not isinstance(emotion, dict):
        raise ValueError("{}: emotion must be an object".format(record_id))

    return {
        "positive_intensity": _require_int(
            emotion.get("positive_intensity"), 0, 6, record_id, "emotion.positive_intensity"
        ),
        "negative_intensity": _require_int(
            emotion.get("negative_intensity"), 0, 6, record_id, "emotion.negative_intensity"
        ),
        "positive_labels": _canonicalize_labels(
            emotion.get("positive_labels"),
            POSITIVE_LABEL_MAP,
            POSITIVE_LABELS,
            record_id,
            "emotion.positive_labels",
        ),
        "negative_labels": _canonicalize_labels(
            emotion.get("negative_labels"),
            NEGATIVE_LABEL_MAP,
            NEGATIVE_LABELS,
            record_id,
            "emotion.negative_labels",
        ),
    }


def normalize_reasoning(record):
    record_id = str(record.get("id", "<unknown>"))
    raw = record.get("appraisal_reasoning")
    if not isinstance(raw, dict):
        raise ValueError("{}: appraisal_reasoning must be an object".format(record_id))

    output = {}
    for dimension in CORE_APPRAISAL_ORDER:
        output[dimension] = _require_nonempty_string(
            raw.get(dimension), record_id, "appraisal_reasoning." + dimension
        )
    return output


def normalize_ratings(record):
    record_id = str(record.get("id", "<unknown>"))
    raw = record.get("appraisal_ratings")
    if not isinstance(raw, dict):
        raise ValueError("{}: appraisal_ratings must be an object".format(record_id))

    compact_keys_present = all(compact in raw for compact, _ in APPRAISAL_KEY_MAP)
    canonical_keys_present = all(canonical in raw for _, canonical in APPRAISAL_KEY_MAP)

    output = {}
    if compact_keys_present:
        for compact, canonical in APPRAISAL_KEY_MAP:
            compact_score = _require_int(
                raw.get(compact), 0, 4, record_id, "appraisal_ratings." + compact
            )
            output[canonical] = compact_score + 1
        return output

    if canonical_keys_present:
        for _, canonical in APPRAISAL_KEY_MAP:
            output[canonical] = _require_int(
                raw.get(canonical), 1, 5, record_id, "appraisal_ratings." + canonical
            )
        return output

    missing_compact = [key for key, _ in APPRAISAL_KEY_MAP if key not in raw]
    missing_canonical = [key for _, key in APPRAISAL_KEY_MAP if key not in raw]
    raise ValueError(
        "{}: appraisal_ratings matches neither organized nor CAREBench schema; "
        "missing organized keys={}, missing canonical keys={}".format(
            record_id, missing_compact, missing_canonical
        )
    )


def _base_record_fields(record):
    if not isinstance(record, dict):
        raise ValueError("Each record must be an object")
    record_id = _require_nonempty_string(record.get("id"), "<unknown>", "id")
    situation = _require_nonempty_string(record.get("situation"), record_id, "situation")
    perspective = record.get("perspective")
    if perspective is not None and perspective != "first_person":
        raise ValueError("{}: expected first_person perspective".format(record_id))
    return record_id, situation


def _json_completion(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_direct_example(record):
    record_id, situation = _base_record_fields(record)
    # Direct and Chain SFT must use the same sample set for a controlled
    # comparison. Validate the chain fields here, but never expose them in the
    # direct prompt or completion.
    normalize_reasoning(record)
    normalize_ratings(record)
    return {
        "sample_id": record_id,
        "prompt": [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": build_direct_user_prompt(situation)},
        ],
        "completion": [
            {"role": "assistant", "content": _json_completion(normalize_emotion(record))}
        ],
    }


def build_chain_example(record):
    record_id, situation = _base_record_fields(record)
    chain = {
        "appraisal_reasoning": normalize_reasoning(record),
        "appraisal_ratings": normalize_ratings(record),
        "emotion": normalize_emotion(record),
    }
    return {
        "sample_id": record_id,
        "prompt": [
            {"role": "system", "content": CHAIN_SYSTEM_PROMPT},
            {"role": "user", "content": build_chain_user_prompt(situation)},
        ],
        "completion": [{"role": "assistant", "content": _json_completion(chain)}],
    }


def load_records(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Training data file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("{} must contain a non-empty JSON array".format(path))

    seen = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError("{}: item {} must be an object".format(path, index))
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("{}: item {} has no valid id".format(path, index))
        if record_id in seen:
            raise ValueError("{}: duplicate id {}".format(path, record_id))
        seen.add(record_id)
    return payload


def _stable_key(record, seed, namespace):
    raw = "{}:{}:{}".format(namespace, seed, record["id"]).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _participant_id(record):
    explicit = record.get("participant_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    # Organized CAREBench ids start with the original participant id followed
    # by an underscore and collection timestamp/case suffix.
    return str(record["id"]).split("_", 1)[0]


def stable_train_eval_split(records, eval_ratio, seed):
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("--eval_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two records are required for an internal split")

    groups = {}
    for record in records:
        groups.setdefault(_participant_id(record), []).append(record)
    if len(groups) < 2:
        raise ValueError("At least two participants are required for an internal split")

    ordered_group_ids = sorted(
        groups,
        key=lambda group_id: hashlib.sha256(
            "split:{}:{}".format(seed, group_id).encode("utf-8")
        ).hexdigest(),
    )
    target_eval_size = max(1, int(round(len(records) * eval_ratio)))
    eval_group_ids = []
    eval_count = 0
    for group_id in ordered_group_ids[:-1]:
        if eval_count >= target_eval_size:
            break
        eval_group_ids.append(group_id)
        eval_count += len(groups[group_id])

    if not eval_group_ids:
        eval_group_ids = [ordered_group_ids[0]]
    eval_group_set = set(eval_group_ids)
    train = [record for record in records if _participant_id(record) not in eval_group_set]
    evaluation = [record for record in records if _participant_id(record) in eval_group_set]
    return train, evaluation


def _limit_records(records, limit, seed, namespace):
    if limit is None or limit <= 0 or limit >= len(records):
        return records
    ordered = sorted(records, key=lambda item: _stable_key(item, seed, namespace))
    return ordered[:limit]


def _build_examples(records, example_builder, split_name, invalid_record_policy):
    examples = []
    errors = []
    for record in records:
        try:
            examples.append(example_builder(record))
        except ValueError as exc:
            if invalid_record_policy == "error":
                raise
            errors.append(
                {
                    "sample_id": str(record.get("id", "<unknown>")),
                    "error": str(exc),
                }
            )

    if errors:
        print(
            "[data] split={} skipped_invalid={} policy={}".format(
                split_name, len(errors), invalid_record_policy
            )
        )
        for error in errors[:10]:
            print(
                "[data] skipped sample_id={} error={}".format(
                    error["sample_id"], error["error"]
                )
            )
        if len(errors) > 10:
            print("[data] ... {} additional invalid records".format(len(errors) - 10))

    if not examples and records:
        raise ValueError("No valid {} samples remain after validation".format(split_name))
    return examples, errors


def prepare_examples(args, example_builder):
    if args.eval_ratio < 0.0 or args.eval_ratio >= 1.0:
        raise ValueError("--eval_ratio must be in [0, 1)")
    train_records = load_records(args.train_file)
    if args.eval_file:
        eval_records = load_records(args.eval_file)
        train_ids = {item["id"] for item in train_records}
        overlap = sorted(train_ids.intersection(item["id"] for item in eval_records))
        if overlap:
            raise ValueError(
                "Train/eval overlap detected (first ids: {})".format(overlap[:10])
            )
        train_participants = {_participant_id(item) for item in train_records}
        participant_overlap = sorted(
            train_participants.intersection(_participant_id(item) for item in eval_records)
        )
        if participant_overlap:
            raise ValueError(
                "Train/eval participant overlap detected (first ids: {})".format(
                    participant_overlap[:10]
                )
            )
    elif args.eval_ratio > 0:
        train_records, eval_records = stable_train_eval_split(
            train_records, args.eval_ratio, args.seed
        )
    else:
        eval_records = []

    train_records = _limit_records(
        train_records, args.max_train_samples, args.seed, "train-limit"
    )
    eval_records = _limit_records(
        eval_records, args.max_eval_samples, args.seed, "eval-limit"
    )

    train_examples, train_errors = _build_examples(
        train_records,
        example_builder,
        split_name="train",
        invalid_record_policy=args.invalid_record_policy,
    )
    eval_examples, eval_errors = _build_examples(
        eval_records,
        example_builder,
        split_name="eval",
        invalid_record_policy=args.invalid_record_policy,
    )
    stats = {
        "train_records_before_validation": len(train_records),
        "eval_records_before_validation": len(eval_records),
        "train_skipped_invalid": len(train_errors),
        "eval_skipped_invalid": len(eval_errors),
        "train_error_preview": train_errors[:20],
        "eval_error_preview": eval_errors[:20],
    }
    return train_examples, eval_examples, stats


def print_preview(method_name, train_examples, eval_examples, count):
    print("[preview] method={}".format(method_name))
    print("[preview] train_samples={}".format(len(train_examples)))
    print("[preview] eval_samples={}".format(len(eval_examples)))
    for index, example in enumerate(train_examples[: max(0, count)]):
        print("\n===== TRAIN SAMPLE {}: {} =====".format(index, example["sample_id"]))
        print(json.dumps(example, ensure_ascii=False, indent=2))


def add_common_arguments(parser, default_output_dir, default_max_length):
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="",
        help="Hugging Face model id or local model directory (required for training)",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="first_person_organized.json",
        help="Organized first-person JSON array",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default="",
        help="Optional separate validation JSON array",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Stable validation split ratio when --eval_file is omitted; use 0 to disable",
    )
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--max_length", type=int, default=default_max_length)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Default: 2e-4 for LoRA/QLoRA, 2e-5 for full fine-tuning",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument("--packing", action="store_true")
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train all model parameters instead of LoRA adapters",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use NF4 QLoRA; requires CUDA and bitsandbytes",
    )
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help="'all-linear' or a comma-separated module-name list",
    )
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default="",
        help="Optional tokenizer/model id or Jinja chat-template path",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="'none' or comma-separated integrations such as wandb,tensorboard",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--dataset_num_proc", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--invalid_record_policy",
        type=str,
        choices=["skip", "error"],
        default="skip",
        help="Skip incomplete targets with a report, or fail immediately for data auditing",
    )
    parser.add_argument(
        "--preview_only",
        action="store_true",
        help="Validate and print transformed samples without importing training libraries",
    )
    parser.add_argument("--preview_samples", type=int, default=1)
    return parser


def _import_training_stack(load_in_4bit):
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "Training requires Python >= 3.10 for the current TRL/Transformers stack; "
            "the active interpreter is {}".format(sys.version.split()[0])
        )
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Missing training dependencies. Install PyTorch for your CUDA version, then "
            "install transformers, datasets, accelerate, trl, and peft. Original error: {}".format(
                exc
            )
        )

    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--load_in_4bit requires bitsandbytes: pip install bitsandbytes"
            ) from exc

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def _resolve_dtype(torch, requested):
    if requested == "bf16":
        return torch.bfloat16, True, False
    if requested == "fp16":
        return torch.float16, False, True
    if requested == "fp32":
        return torch.float32, False, False

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        return torch.float16, False, True
    return torch.float32, False, False


def _parse_report_to(value):
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "none":
        return "none"
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def _parse_target_modules(value):
    cleaned = value.strip()
    if cleaned == "all-linear":
        return cleaned
    modules = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not modules:
        raise ValueError("--lora_target_modules cannot be empty")
    return modules


def run_sft(args, method_name, example_builder):
    train_examples, eval_examples, preparation_stats = prepare_examples(
        args, example_builder
    )
    if args.preview_only:
        print_preview(
            method_name,
            train_examples,
            eval_examples,
            count=args.preview_samples,
        )
        return 0

    if not args.model_name_or_path.strip():
        raise ValueError("--model_name_or_path is required unless --preview_only is used")
    if args.load_in_4bit and args.full_finetune:
        raise ValueError("--load_in_4bit requires LoRA; remove --full_finetune")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")

    stack = _import_training_stack(args.load_in_4bit)
    torch = stack["torch"]
    dtype, use_bf16, use_fp16 = _resolve_dtype(torch, args.dtype)

    if args.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("--load_in_4bit requires a CUDA GPU")

    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None) and not args.chat_template_path:
        raise ValueError(
            "The tokenizer has no chat template. Supply --chat_template_path with a "
            "compatible tokenizer/model id or Jinja template."
        )

    train_dataset = stack["Dataset"].from_list(train_examples)
    eval_dataset = (
        stack["Dataset"].from_list(eval_examples) if eval_examples else None
    )

    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 2.0e-5 if args.full_finetune else 2.0e-4

    model_init_kwargs = {"dtype": dtype}
    config_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "learning_rate": learning_rate,
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
        "bf16": use_bf16,
        "fp16": use_fp16,
        "tf32": bool(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
        ),
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "max_length": args.max_length,
        "packing": args.packing,
        "completion_only_loss": True,
        "report_to": _parse_report_to(args.report_to),
        "run_name": args.run_name or method_name,
        "seed": args.seed,
        "data_seed": args.seed,
        "dataset_num_proc": args.dataset_num_proc,
        "optim": "adamw_torch",
        "model_init_kwargs": model_init_kwargs,
        "trust_remote_code": args.trust_remote_code,
    }
    if eval_dataset is not None:
        config_kwargs["eval_steps"] = args.eval_steps
    if args.chat_template_path:
        config_kwargs["chat_template_path"] = args.chat_template_path

    training_config = stack["SFTConfig"](**config_kwargs)

    peft_config = None
    if not args.full_finetune:
        peft_config = stack["LoraConfig"](
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_parse_target_modules(args.lora_target_modules),
        )

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    trainer = stack["SFTTrainer"](
        model=args.model_name_or_path,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quantization_config,
    )

    resume = args.resume_from_checkpoint.strip() or None
    train_result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method_name,
        "model_name_or_path": args.model_name_or_path,
        "train_samples": len(train_examples),
        "eval_samples": len(eval_examples),
        "data_preparation": preparation_stats,
        "completion_only_loss": True,
        "full_finetune": args.full_finetune,
        "load_in_4bit": args.load_in_4bit,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_metrics": train_result.metrics,
        "arguments": vars(args),
    }
    with (output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("[done] saved {} model to {}".format(method_name, output_dir))
    return 0


def build_parser(description, default_output_dir, default_max_length):
    parser = argparse.ArgumentParser(description=description)
    return add_common_arguments(parser, default_output_dir, default_max_length)
