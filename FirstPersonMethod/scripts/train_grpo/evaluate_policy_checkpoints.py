#!/usr/bin/env python3
"""Generate from and evaluate Policy warm-up checkpoints.

This evaluator uses the exact configurable Policy prompt and strict output
parser used by API-Judge GRPO.  It supports full Hugging Face model directories
and local PEFT/LoRA adapters, including the ``checkpoint-N`` directories saved
by ``train_policy_warmup_sft.py``.

Automatic text/emotion metrics are conditional on a valid Policy JSON output
and an available human reference.  Format coverage is reported separately.
When API judging is enabled, invalid Policy JSON receives the configured hard
gate reward, while API/response-parse failures are excluded rather than treated
as zero-quality candidates.

Example:

    python FirstPersonMethod/scripts/train_grpo/evaluate_policy_checkpoints.py \
      --checkpoint_root /path/to/policy_warmup_sft \
      --eval_file /path/to/eval.json \
      --output_dir /path/to/policy_warmup_sft/checkpoint_evaluation \
      --include_base_model \
      --base_model_name_or_path /path/to/Qwen3-4B-Instruct-2507 \
      --generation_batch_size 4 \
      --bertscore_device cpu

Add the following for the same rubric used by GRPO:

    --use_api_judge --judge_provider glm --judge_model glm-5.1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import compute_bleu_score, rouge_l_f1  # noqa: E402
from peft_adapter_compat import load_peft_adapter_with_compat  # noqa: E402
from sft_common import (  # noqa: E402
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
)
from train_grpo.api_judge import (  # noqa: E402
    APIJudge,
    JudgeCache,
    resolve_judge_settings,
)
from train_grpo.data import (  # noqa: E402
    load_records,
    prepare_split,
    stable_limit,
)
from train_grpo.parsing import parse_policy_output  # noqa: E402
from train_grpo.spec import (  # noqa: E402
    CAREBENCH_REASONING_KEYS,
    resolve_appraisal_dimensions,
)


DEFAULT_CAREBENCH_DIMENSIONS = list(CAREBENCH_REASONING_KEYS)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
TOKENIZER_MARKERS = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
)
MODEL_MARKERS = (
    "adapter_config.json",
    "config.json",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    source: str
    step: int | None
    checkpoint_type: str


def sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or "checkpoint"


def is_local_model_directory(path: Path) -> bool:
    return path.is_dir() and any((path / marker).is_file() for marker in MODEL_MARKERS)


def is_adapter_directory(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def infer_adapter_base(path: Path) -> str:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return ""
    config = read_json_object(config_path)
    value = config.get("base_model_name_or_path")
    return value.strip() if isinstance(value, str) else ""


def infer_base_model(
    checkpoint_root: Path | None,
    explicit_paths: Sequence[str],
) -> str:
    candidates: list[Path] = []
    if checkpoint_root is not None:
        candidates.append(checkpoint_root)
        candidates.extend(
            sorted(
                (
                    item
                    for item in checkpoint_root.glob("checkpoint-*")
                    if item.is_dir()
                ),
                key=lambda item: checkpoint_sort_key(item.name),
            )
        )
    candidates.extend(Path(value) for value in explicit_paths)
    for candidate in candidates:
        base = infer_adapter_base(candidate)
        if base:
            return base

    if checkpoint_root is not None:
        manifest_path = checkpoint_root / "training_manifest.json"
        if manifest_path.is_file():
            manifest = read_json_object(manifest_path)
            value = manifest.get("model_name_or_path")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def checkpoint_sort_key(name: str) -> tuple[int, int | str]:
    match = CHECKPOINT_PATTERN.fullmatch(name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def unique_checkpoint_specs(specs: Iterable[CheckpointSpec]) -> list[CheckpointSpec]:
    output: list[CheckpointSpec] = []
    seen_sources: set[str] = set()
    seen_labels: set[str] = set()
    for spec in specs:
        local = Path(spec.source)
        identity = str(local.resolve()) if local.exists() else spec.source
        if identity in seen_sources:
            continue
        label = sanitize_label(spec.label)
        if label in seen_labels:
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
            label = f"{label}-{digest}"
        seen_sources.add(identity)
        seen_labels.add(label)
        output.append(
            CheckpointSpec(
                label=label,
                source=spec.source,
                step=spec.step,
                checkpoint_type=spec.checkpoint_type,
            )
        )
    return output


def discover_checkpoint_specs(
    checkpoint_root_value: str,
    explicit_paths: Sequence[str],
    include_final: bool,
    include_base_model: bool,
    base_model_name_or_path: str,
) -> tuple[list[CheckpointSpec], str]:
    checkpoint_root = (
        Path(checkpoint_root_value).expanduser()
        if checkpoint_root_value.strip()
        else None
    )
    if checkpoint_root is not None and not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    specs: list[CheckpointSpec] = []
    resolved_base = base_model_name_or_path.strip()
    if include_base_model:
        if not resolved_base:
            resolved_base = infer_base_model(checkpoint_root, explicit_paths)
        if not resolved_base:
            raise ValueError(
                "--include_base_model needs --base_model_name_or_path, or a "
                "discoverable base_model_name_or_path in adapter_config.json"
            )
        specs.append(
            CheckpointSpec(
                label="base",
                source=resolved_base,
                step=0,
                checkpoint_type="base",
            )
        )

    if checkpoint_root is not None:
        checkpoint_dirs: list[tuple[int, Path]] = []
        for item in checkpoint_root.glob("checkpoint-*"):
            match = CHECKPOINT_PATTERN.fullmatch(item.name)
            if item.is_dir() and match and is_local_model_directory(item):
                checkpoint_dirs.append((int(match.group(1)), item))
        for step, item in sorted(checkpoint_dirs, key=lambda pair: pair[0]):
            specs.append(
                CheckpointSpec(
                    label=item.name,
                    source=str(item),
                    step=step,
                    checkpoint_type=(
                        "adapter" if is_adapter_directory(item) else "full_model"
                    ),
                )
            )
        if include_final:
            if is_local_model_directory(checkpoint_root):
                specs.append(
                    CheckpointSpec(
                        label="final",
                        source=str(checkpoint_root),
                        step=None,
                        checkpoint_type=(
                            "adapter"
                            if is_adapter_directory(checkpoint_root)
                            else "full_model"
                        ),
                    )
                )
            else:
                print(
                    "[checkpoints] warning: --include_final was requested but "
                    f"{checkpoint_root} has no adapter_config.json or config.json"
                )

    for path_value in explicit_paths:
        path = Path(path_value).expanduser()
        if not is_local_model_directory(path):
            raise FileNotFoundError(
                "Explicit checkpoint must be a local model/adapter directory: "
                f"{path}"
            )
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        specs.append(
            CheckpointSpec(
                label=path.name,
                source=str(path),
                step=int(match.group(1)) if match else None,
                checkpoint_type=(
                    "adapter" if is_adapter_directory(path) else "full_model"
                ),
            )
        )

    specs = unique_checkpoint_specs(specs)
    if not specs:
        raise ValueError(
            "No checkpoints found. Supply --checkpoint_root and/or --checkpoints."
        )
    return specs, resolved_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_artifact_signature(source: str) -> list[dict[str, Any]] | None:
    path = Path(source)
    if not path.is_dir():
        return None
    patterns = (
        "adapter_config.json",
        "adapter_model.*",
        "config.json",
        "model*.safetensors",
        "pytorch_model*.bin",
    )
    files: dict[str, Path] = {}
    for pattern in patterns:
        for item in path.glob(pattern):
            if item.is_file():
                files[item.name] = item
    return [
        {
            "name": name,
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
        }
        for name, item in sorted(files.items())
    ]


def stable_json_hash(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def normalized_rmse(differences: Sequence[float], value_range: float) -> float | None:
    if not differences:
        return None
    return math.sqrt(
        sum((difference / value_range) ** 2 for difference in differences)
        / len(differences)
    )


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def multilabel_metrics(
    pairs: Sequence[tuple[set[str], set[str]]],
    labels: Sequence[str],
) -> dict[str, Any]:
    if not pairs:
        return {
            "example_f1": None,
            "micro_f1": None,
            "macro_f1": None,
            "exact_match": None,
            "samples": 0,
            "per_label": {},
        }

    example_f1_values: list[float] = []
    exact_matches = 0
    micro_tp = micro_fp = micro_fn = 0
    per_label_counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    allowed = set(labels)
    for gold_raw, prediction_raw in pairs:
        gold = gold_raw & allowed
        prediction = prediction_raw & allowed
        tp = len(gold & prediction)
        fp = len(prediction - gold)
        fn = len(gold - prediction)
        if not gold and not prediction:
            example_f1 = 1.0
        else:
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            example_f1 = (
                safe_div(2.0 * precision * recall, precision + recall)
                if precision + recall
                else 0.0
            )
        example_f1_values.append(example_f1)
        exact_matches += int(gold == prediction)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        for label in labels:
            in_gold = label in gold
            in_prediction = label in prediction
            if in_gold and in_prediction:
                per_label_counts[label]["tp"] += 1
            elif in_prediction:
                per_label_counts[label]["fp"] += 1
            elif in_gold:
                per_label_counts[label]["fn"] += 1

    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        safe_div(
            2.0 * micro_precision * micro_recall,
            micro_precision + micro_recall,
        )
        if micro_precision + micro_recall
        else 0.0
    )
    per_label: dict[str, dict[str, float]] = {}
    label_f1_values: list[float] = []
    for label in labels:
        counts = per_label_counts[label]
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = (
            safe_div(2.0 * precision * recall, precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        label_f1_values.append(f1)
    return {
        "example_f1": mean_or_none(example_f1_values),
        "micro_f1": micro_f1,
        "macro_f1": mean_or_none(label_f1_values),
        "exact_match": exact_matches / len(pairs),
        "samples": len(pairs),
        "per_label": per_label,
    }


class BertScoreBackend:
    def __init__(
        self,
        enabled: bool,
        device: str,
        model_type: str,
        batch_size: int,
        rescale_with_baseline: bool,
    ) -> None:
        self.enabled = enabled
        self.device = device
        self.model_type = model_type.strip()
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline
        self.scorer: Any = None
        self.warning = ""

    def _load(self) -> None:
        if not self.enabled or self.scorer is not None or self.warning:
            return
        try:
            import torch
            from bert_score import BERTScorer

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs: dict[str, Any] = {
                "lang": "en",
                "device": device,
                "rescale_with_baseline": self.rescale_with_baseline,
            }
            if self.model_type:
                kwargs["model_type"] = self.model_type
            self.scorer = BERTScorer(**kwargs)
        except Exception as exc:
            self.warning = f"BERTScore unavailable: {exc}"

    def score(
        self,
        references: Sequence[str],
        predictions: Sequence[str],
    ) -> list[float] | None:
        if not references:
            return []
        self._load()
        if self.scorer is None:
            return None
        try:
            _, _, f1 = self.scorer.score(
                list(predictions),
                list(references),
                verbose=False,
                batch_size=self.batch_size,
            )
            return [float(value) for value in f1]
        except Exception as exc:
            self.warning = f"BERTScore failed: {exc}"
            self.scorer = None
            return None


def decode_reference(example: dict[str, Any]) -> dict[str, Any]:
    raw = example.get("reference_json", "{}")
    value = json.loads(raw) if isinstance(raw, str) and raw else {}
    return value if isinstance(value, dict) else {}


def evaluate_automatic_metrics(
    rows: Sequence[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
    appraisal_dimensions: Sequence[str],
    bertscore: BertScoreBackend,
) -> dict[str, Any]:
    reasoning_pairs: list[tuple[str, str, str]] = []
    positive_differences: list[float] = []
    negative_differences: list[float] = []
    positive_label_pairs: list[tuple[set[str], set[str]]] = []
    negative_label_pairs: list[tuple[set[str], set[str]]] = []
    emotion_reference_samples = 0

    for row in rows:
        candidate = row.get("parsed_output")
        if not isinstance(candidate, dict):
            continue
        sample_id = str(row.get("sample_id", ""))
        example = examples_by_id.get(sample_id)
        if example is None:
            continue
        reference = decode_reference(example)
        gold_reasoning = reference.get("legacy_human_appraisal_reasoning")
        if isinstance(gold_reasoning, dict):
            prediction_reasoning = candidate.get("appraisal_reasoning", {})
            for dimension in appraisal_dimensions:
                gold_text = gold_reasoning.get(dimension)
                prediction_text = prediction_reasoning.get(dimension)
                if (
                    isinstance(gold_text, str)
                    and gold_text.strip()
                    and isinstance(prediction_text, str)
                    and prediction_text.strip()
                ):
                    reasoning_pairs.append(
                        (
                            dimension,
                            gold_text.strip(),
                            prediction_text.strip(),
                        )
                    )

        gold_emotion = reference.get(
            "gold_emotion",
            reference.get("human_emotion"),
        )
        prediction_emotion = candidate.get("emotion")
        if not isinstance(gold_emotion, dict) or not isinstance(
            prediction_emotion,
            dict,
        ):
            continue
        emotion_reference_samples += 1
        positive_differences.append(
            float(
                prediction_emotion["positive_intensity"]
                - gold_emotion["positive_intensity"]
            )
        )
        negative_differences.append(
            float(
                prediction_emotion["negative_intensity"]
                - gold_emotion["negative_intensity"]
            )
        )
        positive_label_pairs.append(
            (
                set(gold_emotion["positive_labels"]),
                set(prediction_emotion["positive_labels"]),
            )
        )
        negative_label_pairs.append(
            (
                set(gold_emotion["negative_labels"]),
                set(prediction_emotion["negative_labels"]),
            )
        )

    bert_values = bertscore.score(
        [gold for _, gold, _ in reasoning_pairs],
        [prediction for _, _, prediction in reasoning_pairs],
    )
    pair_metrics: list[dict[str, Any]] = []
    for index, (dimension, gold, prediction) in enumerate(reasoning_pairs):
        pair_metrics.append(
            {
                "dimension": dimension,
                "bleu": compute_bleu_score(gold, prediction),
                "rouge_l": rouge_l_f1(gold, prediction),
                "bertscore": (bert_values[index] if bert_values is not None else None),
            }
        )

    def aggregate_reasoning(
        selected: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "bleu": mean_or_none([float(item["bleu"]) for item in selected]),
            "rouge_l": mean_or_none([float(item["rouge_l"]) for item in selected]),
            "bertscore": mean_or_none(
                [
                    float(item["bertscore"])
                    for item in selected
                    if item["bertscore"] is not None
                ]
            ),
            "pairs": len(selected),
        }

    reasoning_by_dimension = {
        dimension: aggregate_reasoning(
            [item for item in pair_metrics if item["dimension"] == dimension]
        )
        for dimension in appraisal_dimensions
    }
    warnings = [bertscore.warning] if bertscore.warning else []
    return {
        "metric_scope": (
            "Conditional on strict Policy-format validity and an available "
            "human reference; consult format/reference coverage."
        ),
        "appraisal_reasoning": {
            "overall": aggregate_reasoning(pair_metrics),
            "dimension": reasoning_by_dimension,
        },
        "appraisals": {
            "normalized_rmse": None,
            "reason": (
                "The Policy output contract contains natural-language appraisal "
                "reasoning but no appraisal ratings."
            ),
        },
        "positive_emotion": {
            "intensity_normalized_rmse": normalized_rmse(
                positive_differences,
                6.0,
            ),
            **multilabel_metrics(positive_label_pairs, POSITIVE_LABELS),
        },
        "negative_emotion": {
            "intensity_normalized_rmse": normalized_rmse(
                negative_differences,
                6.0,
            ),
            **multilabel_metrics(negative_label_pairs, NEGATIVE_LABELS),
        },
        "coverage": {
            "reasoning_reference_pairs": len(reasoning_pairs),
            "emotion_reference_samples": emotion_reference_samples,
        },
        "warnings": warnings,
    }


def import_inference_stack(load_in_4bit: bool) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise RuntimeError("Policy checkpoint evaluation requires Python >= 3.10")
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires torch, transformers, accelerate, and peft"
        ) from exc
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("--load_in_4bit requires bitsandbytes") from exc
    return {
        "torch": torch,
        "PeftConfig": PeftConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def resolve_dtype(torch: Any, requested: str, device: Any) -> Any:
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def model_input_device(model: Any, fallback: Any) -> Any:
    try:
        device = model.get_input_embeddings().weight.device
        if getattr(device, "type", "") != "meta":
            return device
    except (AttributeError, RuntimeError):
        pass
    for parameter in model.parameters():
        if getattr(parameter.device, "type", "") != "meta":
            return parameter.device
    return fallback


def has_tokenizer_files(path_value: str) -> bool:
    path = Path(path_value)
    return path.is_dir() and any(
        (path / marker).is_file() for marker in TOKENIZER_MARKERS
    )


def choose_tokenizer_source(
    args: argparse.Namespace,
    spec: CheckpointSpec,
    adapter_base: str,
) -> str:
    if args.tokenizer_name_or_path.strip():
        return args.tokenizer_name_or_path.strip()
    if has_tokenizer_files(spec.source):
        return spec.source
    if args.checkpoint_root and has_tokenizer_files(args.checkpoint_root):
        return args.checkpoint_root
    if adapter_base:
        return adapter_base
    return spec.source


def apply_chat_template_override(
    tokenizer: Any,
    path_value: str,
    stack: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if not path_value.strip():
        return
    path = Path(path_value)
    if path.is_file():
        tokenizer.chat_template = path.read_text(encoding="utf-8")
        return
    other = stack["AutoTokenizer"].from_pretrained(
        path_value,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if not getattr(other, "chat_template", None):
        raise ValueError(f"{path_value} tokenizer has no chat template")
    tokenizer.chat_template = other.chat_template


def load_checkpoint_model(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    stack: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    torch = stack["torch"]
    requested_device = (
        "cuda:0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for --device {args.device}")
    if args.load_in_4bit and device.type != "cuda":
        raise RuntimeError("--load_in_4bit requires CUDA")
    dtype = resolve_dtype(torch, args.dtype, device)

    source_path = Path(spec.source)
    adapter = is_adapter_directory(source_path)
    adapter_base = ""
    if adapter:
        peft_config = stack["PeftConfig"].from_pretrained(spec.source)
        adapter_base = (
            args.base_model_name_or_path.strip()
            or str(peft_config.base_model_name_or_path).strip()
        )
        if not adapter_base:
            raise ValueError(f"Cannot resolve base model for adapter {spec.source}")

    tokenizer_source = choose_tokenizer_source(args, spec, adapter_base)
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Tokenizer {tokenizer_source} has neither pad nor EOS token"
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    apply_chat_template_override(
        tokenizer,
        args.chat_template_path,
        stack,
        args,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template; pass --chat_template_path")

    quantization = None
    if args.load_in_4bit:
        quantization = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    elif quantization is not None:
        model_kwargs["device_map"] = {"": 0}

    model_source = adapter_base if adapter else spec.source
    model = stack["AutoModelForCausalLM"].from_pretrained(
        model_source,
        **model_kwargs,
    )
    adapter_load_summary: dict[str, Any] = {}
    if adapter:
        model, adapter_load_summary = load_peft_adapter_with_compat(
            model,
            spec.source,
            stack["PeftModel"],
            torch,
            is_trainable=False,
            local_files_only=args.local_files_only,
        )
    if not args.device_map and quantization is None:
        model.to(device)
    model.eval()
    input_device = model_input_device(model, device)
    metadata = {
        "adapter": adapter,
        "base_model": adapter_base or None,
        "tokenizer_source": tokenizer_source,
        "dtype": str(dtype),
        "input_device": str(input_device),
        "device_map": args.device_map or None,
        "load_in_4bit": args.load_in_4bit,
        "adapter_load": adapter_load_summary,
    }
    return model, tokenizer, input_device, metadata


def render_prompts(
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    enable_thinking: bool,
    max_prompt_length: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rendered: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for example in examples:
        sample_id = example["sample_id"]
        try:
            text = tokenizer.apply_chat_template(
                example["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            prompt_tokens = len(token_ids)
            if prompt_tokens > max_prompt_length:
                raise ValueError(
                    f"prompt has {prompt_tokens} tokens, exceeding "
                    f"--max_prompt_length={max_prompt_length}"
                )
            rendered.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                }
            )
        except Exception as exc:
            errors[sample_id] = str(exc)
    return rendered, errors


def generate_rendered_batch(
    items: Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    input_device: Any,
    args: argparse.Namespace,
    torch: Any,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        [item["text"] for item in items],
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    encoded = {key: value.to(input_device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    generation_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.inference_mode():
        generated = model.generate(**generation_kwargs)
    rows: list[dict[str, Any]] = []
    for item, sequence in zip(items, generated):
        continuation = sequence[input_width:]
        raw = tokenizer.decode(
            continuation,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        completion_tokens = len(
            tokenizer(
                raw,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
        )
        rows.append(
            {
                "sample_id": item["sample_id"],
                "raw_completion": raw,
                "prompt_tokens": item["prompt_tokens"],
                "completion_tokens": completion_tokens,
                "generation_error": "",
            }
        )
    return rows


def generate_checkpoint_predictions(
    spec: CheckpointSpec,
    examples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stack = import_inference_stack(args.load_in_4bit)
    torch = stack["torch"]
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model: Any = None
    tokenizer: Any = None
    try:
        model, tokenizer, input_device, model_metadata = load_checkpoint_model(
            spec,
            args,
            stack,
        )
        rendered, render_errors = render_prompts(
            examples,
            tokenizer,
            args.enable_thinking,
            args.max_prompt_length,
        )
        generated_by_id: dict[str, dict[str, Any]] = {}
        for start in range(0, len(rendered), args.generation_batch_size):
            batch = rendered[start : start + args.generation_batch_size]
            try:
                batch_rows = generate_rendered_batch(
                    batch,
                    model,
                    tokenizer,
                    input_device,
                    args,
                    torch,
                )
            except Exception as batch_exc:
                if len(batch) == 1:
                    batch_rows = [
                        {
                            "sample_id": batch[0]["sample_id"],
                            "raw_completion": "",
                            "prompt_tokens": batch[0]["prompt_tokens"],
                            "completion_tokens": 0,
                            "generation_error": str(batch_exc),
                        }
                    ]
                else:
                    print(
                        f"[generate] checkpoint={spec.label} batch failed; "
                        "retrying samples individually"
                    )
                    batch_rows = []
                    for item in batch:
                        try:
                            batch_rows.extend(
                                generate_rendered_batch(
                                    [item],
                                    model,
                                    tokenizer,
                                    input_device,
                                    args,
                                    torch,
                                )
                            )
                        except Exception as sample_exc:
                            batch_rows.append(
                                {
                                    "sample_id": item["sample_id"],
                                    "raw_completion": "",
                                    "prompt_tokens": item["prompt_tokens"],
                                    "completion_tokens": 0,
                                    "generation_error": str(sample_exc),
                                }
                            )
            for row in batch_rows:
                generated_by_id[row["sample_id"]] = row
            completed = min(start + len(batch), len(rendered))
            print(
                f"[generate] checkpoint={spec.label} "
                f"completed={completed}/{len(rendered)}",
                flush=True,
            )

        rows: list[dict[str, Any]] = []
        for example in examples:
            sample_id = example["sample_id"]
            if sample_id in render_errors:
                generated = {
                    "sample_id": sample_id,
                    "raw_completion": "",
                    "prompt_tokens": None,
                    "completion_tokens": 0,
                    "generation_error": render_errors[sample_id],
                }
            else:
                generated = generated_by_id[sample_id]
            raw = generated["raw_completion"]
            parsed: dict[str, Any] | None = None
            parse_error = ""
            if not generated["generation_error"]:
                try:
                    parsed = parse_policy_output(
                        raw,
                        args.appraisal_dimensions,
                    )
                except (TypeError, ValueError) as exc:
                    parse_error = str(exc)
            rows.append(
                {
                    **generated,
                    "parsed_output": parsed,
                    "parse_error": parse_error,
                }
            )
        return rows, model_metadata
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def prediction_manifest_payload(
    spec: CheckpointSpec,
    examples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    eval_file: Path,
) -> dict[str, Any]:
    return {
        "checkpoint": asdict(spec),
        "checkpoint_artifacts": checkpoint_artifact_signature(spec.source),
        "eval_file": str(eval_file.resolve()),
        "eval_file_sha256": sha256_file(eval_file),
        "sample_ids": [example["sample_id"] for example in examples],
        "appraisal_dimensions": args.appraisal_dimensions,
        "generation": {
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "enable_thinking": args.enable_thinking,
            "seed": args.seed,
            "dtype": args.dtype,
            "load_in_4bit": args.load_in_4bit,
            "tokenizer_name_or_path": args.tokenizer_name_or_path,
            "chat_template_path": args.chat_template_path,
        },
    }


def load_or_generate_predictions(
    spec: CheckpointSpec,
    examples: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    checkpoint_output: Path,
    eval_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_path = checkpoint_output / "predictions.jsonl"
    manifest_path = checkpoint_output / "generation_manifest.json"
    manifest = prediction_manifest_payload(spec, examples, args, eval_file)
    manifest["fingerprint"] = stable_json_hash(manifest)
    if args.reuse_predictions:
        if not predictions_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"--reuse_predictions needs {predictions_path} and {manifest_path}"
            )
        previous = read_json_object(manifest_path)
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                f"Prediction manifest mismatch for {spec.label}; regenerate "
                "without --reuse_predictions"
            )
        rows = load_jsonl(predictions_path)
        expected = [example["sample_id"] for example in examples]
        if [str(row.get("sample_id", "")) for row in rows] != expected:
            raise ValueError(
                f"{predictions_path} sample order/content does not match eval data"
            )
        print(f"[generate] checkpoint={spec.label} reused predictions")
        return rows, previous.get("model", {})

    rows, model_metadata = generate_checkpoint_predictions(spec, examples, args)
    manifest["model"] = model_metadata
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    write_jsonl(predictions_path, rows)
    write_json(manifest_path, manifest)
    return rows, model_metadata


def find_trainer_state(
    spec: CheckpointSpec,
    checkpoint_root_value: str,
) -> Path | None:
    candidates: list[Path] = []
    if checkpoint_root_value:
        candidates.append(Path(checkpoint_root_value) / "trainer_state.json")
    source_path = Path(spec.source)
    if source_path.is_dir():
        candidates.append(source_path / "trainer_state.json")
        candidates.append(source_path.parent / "trainer_state.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def checkpoint_eval_loss(
    spec: CheckpointSpec,
    checkpoint_root_value: str,
) -> float | None:
    state_path = find_trainer_state(spec, checkpoint_root_value)
    if state_path is None:
        return None
    state = read_json_object(state_path)
    history = state.get("log_history")
    if not isinstance(history, list):
        return None
    candidates: list[tuple[int, float]] = []
    for item in history:
        if not isinstance(item, dict) or "eval_loss" not in item:
            continue
        try:
            candidates.append((int(item.get("step", -1)), float(item["eval_loss"])))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None
    if spec.step is not None:
        exact = [loss for step, loss in candidates if step == spec.step]
        return exact[-1] if exact else None
    return sorted(candidates, key=lambda pair: pair[0])[-1][1]


async def judge_prediction_rows(
    rows: Sequence[dict[str, Any]],
    examples_by_id: dict[str, dict[str, Any]],
    judge: APIJudge,
) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        sample_id = str(row["sample_id"])
        example = examples_by_id[sample_id]
        tasks.append(
            judge.score_one(
                sample_id,
                example["situation"],
                row.get("raw_completion", ""),
                example["reference_json"],
            )
        )
    return await asyncio.gather(*tasks)


def mean_mapping(
    results: Sequence[dict[str, Any]],
    field: str,
    key: str,
) -> float | None:
    values = [
        float(result[field][key])
        for result in results
        if isinstance(result.get(field), dict) and key in result[field]
    ]
    return mean_or_none(values)


def aggregate_judge_results(
    results: Sequence[dict[str, Any]],
    appraisal_dimensions: Sequence[str],
) -> dict[str, Any]:
    valid_count = sum(bool(result.get("valid")) for result in results)
    invalid_count = sum(not bool(result.get("valid")) for result in results)
    failed_count = sum(bool(result.get("failed")) for result in results)
    judged = [
        result
        for result in results
        if result.get("valid")
        and not result.get("failed")
        and isinstance(result.get("reward_components"), dict)
    ]

    def reward_component_mean(
        rows: Sequence[dict[str, Any]], field: str
    ) -> float | None:
        return mean_or_none(
            [
                float(result["reward_components"][field])
                for result in rows
                if isinstance(result.get("reward_components"), dict)
            ]
        )

    def outcome_diagnostic_mean(field: str) -> float | None:
        return mean_or_none(
            [
                float(result["outcome_diagnostics"][field])
                for result in judged
                if field in result.get("outcome_diagnostics", {})
            ]
        )

    summary = {
        "requested": len(results),
        "format_valid": valid_count,
        "format_invalid_hard_gate": invalid_count,
        "judge_success": len(judged),
        "judge_api_or_parse_failures": failed_count,
        "judge_success_rate_among_valid": safe_div(len(judged), valid_count),
        "judge_coverage_total": safe_div(len(judged), len(results)),
        "reward_component_mean_with_hard_gate": {
            field: reward_component_mean(results, field)
            for field in (
                "appraisal_reward",
                "coherence_reward",
                "transition_reward",
                "outcome_label_reward",
                "outcome_intensity_reward",
                "outcome_reward",
            )
        },
        "reward_component_mean_valid": {
            field: reward_component_mean(judged, field)
            for field in (
                "appraisal_reward",
                "coherence_reward",
                "transition_reward",
                "outcome_label_reward",
                "outcome_intensity_reward",
                "outcome_reward",
            )
        },
        "outcome_atomic_score_mean": {
            field: outcome_diagnostic_mean(field)
            for field in (
                "positive_label_score",
                "negative_label_score",
                "positive_intensity_score",
                "negative_intensity_score",
                "process_gate",
                "process_gate_enabled",
            )
        },
        "dimension_score_mean": {
            dimension: mean_mapping(judged, "dimension_scores", dimension)
            for dimension in appraisal_dimensions
        },
        "appraisal_criterion_mean": {
            key: mean_mapping(judged, "appraisal_criterion_scores", key)
            for key in (
                "dimension_specific_validity",
                "situation_grounding",
                "experiencer_fidelity",
            )
        },
        "raw_appraisal_emotion_linkage_mean": mean_or_none(
            [float(result["raw_transition_reward"]) for result in judged]
        ),
        "cache_hit_rate_among_judged": safe_div(
            sum(bool(result.get("cache_hit")) for result in judged),
            len(judged),
        ),
        "failure_policy": (
            "API/response-parse failures are excluded; invalid Policy JSON uses "
            "the configured hard-gate reward."
        ),
    }
    return summary


def format_coverage(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    generation_success = sum(not row.get("generation_error") for row in rows)
    format_valid = sum(isinstance(row.get("parsed_output"), dict) for row in rows)
    prompt_tokens = [
        float(row["prompt_tokens"])
        for row in rows
        if isinstance(row.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        float(row["completion_tokens"])
        for row in rows
        if isinstance(row.get("completion_tokens"), (int, float))
    ]
    return {
        "eval_samples": total,
        "generation_success": generation_success,
        "generation_success_rate": safe_div(generation_success, total),
        "format_valid": format_valid,
        "format_valid_rate": safe_div(format_valid, total),
        "generation_errors": sum(bool(row.get("generation_error")) for row in rows),
        "parse_errors": sum(bool(row.get("parse_error")) for row in rows),
        "mean_prompt_tokens": mean_or_none(prompt_tokens),
        "mean_completion_tokens": mean_or_none(completion_tokens),
    }


def flatten_summary(
    spec: CheckpointSpec,
    eval_loss: float | None,
    coverage: dict[str, Any],
    automatic: dict[str, Any],
    judge_summary: dict[str, Any] | None,
    elapsed_seconds: float,
    appraisal_dimensions: Sequence[str],
) -> dict[str, Any]:
    reasoning = automatic["appraisal_reasoning"]["overall"]
    positive = automatic["positive_emotion"]
    negative = automatic["negative_emotion"]
    row: dict[str, Any] = {
        "checkpoint": spec.label,
        "checkpoint_step": spec.step,
        "checkpoint_type": spec.checkpoint_type,
        "checkpoint_path": spec.source,
        "eval_loss": eval_loss,
        **coverage,
        "reasoning_reference_pairs": automatic["coverage"]["reasoning_reference_pairs"],
        "emotion_reference_samples": automatic["coverage"]["emotion_reference_samples"],
        "appraisal_reasoning_bleu": reasoning["bleu"],
        "appraisal_reasoning_rouge_l": reasoning["rouge_l"],
        "appraisal_reasoning_bertscore": reasoning["bertscore"],
        "appraisals_normalized_rmse": None,
        "positive_intensity_normalized_rmse": positive["intensity_normalized_rmse"],
        "positive_example_f1": positive["example_f1"],
        "positive_micro_f1": positive["micro_f1"],
        "positive_macro_f1": positive["macro_f1"],
        "negative_intensity_normalized_rmse": negative["intensity_normalized_rmse"],
        "negative_example_f1": negative["example_f1"],
        "negative_micro_f1": negative["micro_f1"],
        "negative_macro_f1": negative["macro_f1"],
        "judge_enabled": judge_summary is not None,
        "elapsed_seconds": elapsed_seconds,
    }
    if judge_summary is not None:
        row.update(
            {
                "judge_success": judge_summary["judge_success"],
                "judge_success_rate_among_valid": judge_summary[
                    "judge_success_rate_among_valid"
                ],
                "judge_coverage_total": judge_summary["judge_coverage_total"],
                "judge_api_or_parse_failures": judge_summary[
                    "judge_api_or_parse_failures"
                ],
                "judge_invalid_hard_gate": judge_summary["format_invalid_hard_gate"],
                "judge_dimension_specific_validity": judge_summary[
                    "appraisal_criterion_mean"
                ]["dimension_specific_validity"],
                "judge_situation_grounding": judge_summary["appraisal_criterion_mean"][
                    "situation_grounding"
                ],
                "judge_experiencer_fidelity": judge_summary["appraisal_criterion_mean"][
                    "experiencer_fidelity"
                ],
                "judge_raw_appraisal_emotion_linkage": judge_summary[
                    "raw_appraisal_emotion_linkage_mean"
                ],
            }
        )
        for field, value in judge_summary[
            "reward_component_mean_valid"
        ].items():
            row[f"judge_{field}"] = value
        for field, value in judge_summary[
            "outcome_atomic_score_mean"
        ].items():
            row[f"judge_{field}"] = value
        for dimension in appraisal_dimensions:
            row[f"judge_{dimension}"] = judge_summary["dimension_score_mean"][dimension]
    return row


def add_judge_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("optional API LLM Judge")
    group.add_argument("--use_api_judge", action="store_true")
    group.add_argument(
        "--judge_provider",
        choices=["openai", "glm"],
        default="openai",
    )
    group.add_argument("--judge_model", type=str, default="")
    group.add_argument("--judge_config", type=str, default="")
    group.add_argument("--judge_provider_section", type=str, default="")
    group.add_argument("--judge_api_key_env", type=str, default="")
    group.add_argument("--judge_base_url", type=str, default="")
    group.add_argument(
        "--judge_response_format",
        choices=["auto", "json_schema", "json_object", "text"],
        default="auto",
    )
    group.add_argument(
        "--judge_token_parameter",
        choices=["auto", "max_completion_tokens", "max_tokens"],
        default="auto",
    )
    group.add_argument("--judge_max_tokens", type=int, default=2500)
    group.add_argument("--judge_temperature", type=float, default=0.0)
    group.add_argument("--judge_omit_temperature", action="store_true")
    group.add_argument(
        "--judge_thinking",
        choices=["auto", "enabled", "disabled"],
        default="auto",
        help=(
            "GLM Judge thinking mode; independent of policy "
            "--enable_thinking"
        ),
    )
    group.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    group.add_argument("--judge_max_retries", type=int, default=3)
    group.add_argument("--judge_parse_retries", type=int, default=1)
    group.add_argument("--judge_max_concurrency", type=int, default=8)
    group.add_argument("--log_judge_raw_outputs", action="store_true")
    group.add_argument(
        "--judge_failure_policy",
        choices=["error", "skip"],
        default="skip",
    )
    group.add_argument("--judge_cache_path", type=str, default="")
    group.add_argument("--no_judge_cache", action="store_true")
    group.add_argument("--invalid_component_reward", type=float, default=-1.0)
    group.add_argument("--outcome_label_weight", type=float, default=0.5)
    group.add_argument("--outcome_intensity_weight", type=float, default=0.5)
    process_gate = group.add_mutually_exclusive_group()
    process_gate.add_argument(
        "--use_process_gate",
        dest="use_process_gate",
        action="store_true",
    )
    process_gate.add_argument(
        "--no_process_gate",
        dest="use_process_gate",
        action="store_false",
    )
    parser.set_defaults(use_process_gate=True)
    group.add_argument(
        "--process_gate_mode",
        choices=["min", "product", "geometric_mean"],
        default="min",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Evaluate generation quality across Policy warm-up checkpoints")
    )
    checkpoint = parser.add_argument_group("checkpoints")
    checkpoint.add_argument(
        "--checkpoint_root",
        type=str,
        default="",
        help="Training output directory containing checkpoint-N directories",
    )
    checkpoint.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Optional explicit local full-model or PEFT checkpoint directories",
    )
    checkpoint.add_argument(
        "--include_final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate the final model/adapter at --checkpoint_root",
    )
    checkpoint.add_argument("--include_base_model", action="store_true")
    checkpoint.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="",
        help=(
            "Override an adapter's recorded base model and/or identify the base "
            "entry used with --include_base_model"
        ),
    )

    data = parser.add_argument_group("data and output")
    data.add_argument("--eval_file", type=str, required=True)
    data.add_argument("--output_dir", type=str, required=True)
    data.add_argument("--max_eval_samples", type=int, default=None)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument(
        "--invalid_record_policy",
        choices=["error", "skip"],
        default="error",
    )
    data.add_argument(
        "--judge_reference_mode",
        choices=["none", "available", "required"],
        default="available",
    )
    data.add_argument(
        "--appraisal_dimensions",
        type=resolve_appraisal_dimensions,
        default=list(DEFAULT_CAREBENCH_DIMENSIONS),
        metavar="DIM1,DIM2,...",
    )
    data.add_argument(
        "--reuse_predictions",
        action="store_true",
        help=(
            "Reuse predictions only when their saved data/generation fingerprint "
            "matches this invocation"
        ),
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument("--generation_batch_size", type=int, default=1)
    generation.add_argument("--max_prompt_length", type=int, default=3072)
    generation.add_argument("--max_new_tokens", type=int, default=1536)
    generation.add_argument("--do_sample", action="store_true")
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top_p", type=float, default=0.95)
    generation.add_argument("--enable_thinking", action="store_true")
    generation.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    generation.add_argument("--device", type=str, default="auto")
    generation.add_argument(
        "--device_map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
    )
    generation.add_argument("--load_in_4bit", action="store_true")
    generation.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    generation.add_argument("--tokenizer_name_or_path", type=str, default="")
    generation.add_argument("--chat_template_path", type=str, default="")
    generation.add_argument("--trust_remote_code", action="store_true")
    generation.add_argument("--local_files_only", action="store_true")

    metrics = parser.add_argument_group("automatic metrics")
    metrics.add_argument("--skip_bertscore", action="store_true")
    metrics.add_argument(
        "--bertscore_device",
        type=str,
        default="cpu",
        help="cpu (safe default), cuda, cuda:N, or auto",
    )
    metrics.add_argument("--bertscore_model_type", type=str, default="")
    metrics.add_argument("--bertscore_batch_size", type=int, default=16)
    metrics.add_argument(
        "--bertscore_rescale_with_baseline",
        action="store_true",
        help="Disabled by default to match FirstPersonMethod/scripts/evaluate.py",
    )
    add_judge_arguments(parser)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint_root and not args.checkpoints:
        raise ValueError("Supply --checkpoint_root and/or one or more --checkpoints")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation_batch_size must be positive")
    if args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Prompt and generation token limits must be positive")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be positive with --do_sample")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.bertscore_batch_size <= 0:
        raise ValueError("--bertscore_batch_size must be positive")
    if args.judge_max_concurrency <= 0:
        raise ValueError("--judge_max_concurrency must be positive")
    if args.outcome_label_weight + args.outcome_intensity_weight <= 0:
        raise ValueError("Outcome label + intensity weights must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    eval_file = Path(args.eval_file)
    if not eval_file.is_file():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, resolved_base = discover_checkpoint_specs(
        args.checkpoint_root,
        args.checkpoints,
        args.include_final,
        args.include_base_model,
        args.base_model_name_or_path,
    )
    if not args.base_model_name_or_path and resolved_base:
        args.base_model_name_or_path = resolved_base

    records = stable_limit(
        load_records(args.eval_file),
        args.max_eval_samples,
        args.seed,
        "policy-checkpoint-eval",
    )
    examples, data_errors = prepare_split(
        records,
        "checkpoint_eval",
        args.invalid_record_policy,
        args.judge_reference_mode,
        args.appraisal_dimensions,
    )
    examples_by_id = {example["sample_id"]: example for example in examples}
    if data_errors:
        write_jsonl(output_dir / "invalid_eval_records.jsonl", data_errors)

    bertscore = BertScoreBackend(
        enabled=not args.skip_bertscore,
        device=args.bertscore_device,
        model_type=args.bertscore_model_type,
        batch_size=args.bertscore_batch_size,
        rescale_with_baseline=args.bertscore_rescale_with_baseline,
    )

    judge: APIJudge | None = None
    judge_settings: dict[str, str] | None = None
    if args.use_api_judge:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError("--use_api_judge requires the openai package") from exc
        judge_settings = resolve_judge_settings(args)
        cache = None
        if not args.no_judge_cache:
            cache_path = (
                Path(args.judge_cache_path)
                if args.judge_cache_path
                else output_dir / "judge_cache.sqlite"
            )
            cache = JudgeCache(cache_path)
        judge = APIJudge(args, judge_settings, AsyncOpenAI, cache)

    print(
        f"[data] eval_samples={len(examples)} "
        f"dimensions={','.join(args.appraisal_dimensions)}"
    )
    print("[checkpoints] " + ", ".join(f"{spec.label}={spec.source}" for spec in specs))
    summary_rows: list[dict[str, Any]] = []
    detailed_results: list[dict[str, Any]] = []

    for index, spec in enumerate(specs, start=1):
        started = time.monotonic()
        checkpoint_output = output_dir / spec.label
        print(
            f"[checkpoint] {index}/{len(specs)} label={spec.label} "
            f"source={spec.source}",
            flush=True,
        )
        rows, model_metadata = load_or_generate_predictions(
            spec,
            examples,
            args,
            checkpoint_output,
            eval_file,
        )
        coverage = format_coverage(rows)
        automatic = evaluate_automatic_metrics(
            rows,
            examples_by_id,
            args.appraisal_dimensions,
            bertscore,
        )

        judge_results: list[dict[str, Any]] | None = None
        judge_summary: dict[str, Any] | None = None
        if judge is not None:
            print(
                f"[judge] checkpoint={spec.label} candidates={len(rows)}",
                flush=True,
            )
            judge_results = asyncio.run(
                judge_prediction_rows(rows, examples_by_id, judge)
            )
            judge_summary = aggregate_judge_results(
                judge_results,
                args.appraisal_dimensions,
            )
            write_jsonl(
                checkpoint_output / "judge_results.jsonl",
                [
                    {
                        "sample_id": row["sample_id"],
                        **result,
                    }
                    for row, result in zip(rows, judge_results)
                ],
            )

        eval_loss = checkpoint_eval_loss(spec, args.checkpoint_root)
        elapsed = time.monotonic() - started
        metrics = {
            "checkpoint": asdict(spec),
            "eval_loss": eval_loss,
            "model": model_metadata,
            "format": coverage,
            "automatic": automatic,
            "judge": judge_summary,
            "elapsed_seconds": elapsed,
        }
        write_json(checkpoint_output / "metrics.json", metrics)
        summary_row = flatten_summary(
            spec,
            eval_loss,
            coverage,
            automatic,
            judge_summary,
            elapsed,
            args.appraisal_dimensions,
        )
        summary_rows.append(summary_row)
        detailed_results.append(metrics)
        write_csv(output_dir / "summary.csv", summary_rows)
        write_json(output_dir / "summary.json", detailed_results)
        outcome_reward = (
            None
            if judge_summary is None
            else judge_summary["reward_component_mean_valid"]["outcome_reward"]
        )
        print(
            f"[result] checkpoint={spec.label} "
            f"format_valid_rate={coverage['format_valid_rate']:.4f} "
            f"gated_outcome_reward={outcome_reward}",
            flush=True,
        )

    run_manifest = {
        "script": str(SCRIPT_PATH),
        "eval_file": str(eval_file.resolve()),
        "eval_file_sha256": sha256_file(eval_file),
        "eval_samples": len(examples),
        "invalid_eval_records": len(data_errors),
        "appraisal_dimensions": args.appraisal_dimensions,
        "checkpoints": [asdict(spec) for spec in specs],
        "judge": (
            {
                key: value
                for key, value in (judge_settings or {}).items()
                if key != "api_key"
            }
            if judge_settings is not None
            else None
        ),
        "metric_scope": (
            "Automatic metrics are valid-output/reference conditional. API "
            "failures are excluded; invalid Policy JSON is hard-gated."
        ),
        "arguments": vars(args),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    print(f"[done] summary={output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
