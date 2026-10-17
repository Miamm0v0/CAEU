#!/usr/bin/env python3
"""Custom trainer and batching utilities for AE intermediate training."""

from __future__ import annotations

import math
import os
from quopri import encode
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import Trainer


if __package__ in {None, ""}:
    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from appraisal_emotion_train.ae_format import (
        IGNORE_INDEX,
        POLICY_DIMENSIONS,
        tokenize_ae_example,
    )
    from appraisal_emotion_train.ae_losses import (
        AELossOutput,
        PairContrastiveResult,
        conditional_emotion_sft_loss,
        paired_appraisal_contrastive_loss,
        replay_preservation_loss,
        sequence_mean_log_likelihood,
    )
    from peft_adapter_compat import load_peft_adapter_with_compat
    from train_grpo.lightweight_metrics import carebench_metric_summary
    from train_grpo.parsing import parse_policy_output
else:
    from .ae_format import IGNORE_INDEX, POLICY_DIMENSIONS, tokenize_ae_example
    from .ae_losses import (
        AELossOutput,
        PairContrastiveResult,
        conditional_emotion_sft_loss,
        paired_appraisal_contrastive_loss,
        replay_preservation_loss,
        sequence_mean_log_likelihood,
    )
    from ..peft_adapter_compat import load_peft_adapter_with_compat
    from ..train_grpo.lightweight_metrics import carebench_metric_summary
    from ..train_grpo.parsing import parse_policy_output


OBJECTIVES = ("conditional", "pair", "preservation")
PAIR_BRANCHES = (
    "ea_given_aa",
    "ea_given_ab",
    "eb_given_ab",
    "eb_given_aa",
)


@dataclass(frozen=True)
class WarmupAdapterLoadConfig:
    adapter_path: str
    tokenizer_path: str = ""
    dtype: str = "bf16"
    load_in_4bit: bool = False
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    chat_template_path: str = ""


def _resolve_dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if value != "auto":
        raise ValueError("dtype must be one of auto, bf16, fp16, fp32")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_chat_template(tokenizer: Any, value: str, trust_remote_code: bool) -> None:
    if not value:
        return
    path = Path(value)
    if path.is_file():
        tokenizer.chat_template = path.read_text(encoding="utf-8")
        return
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(
        value,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not getattr(source, "chat_template", None):
        raise ValueError(f"Chat-template source {value!r} has no chat template")
    tokenizer.chat_template = source.chat_template


def load_trainable_warmup_adapter(
    config: WarmupAdapterLoadConfig,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load an existing Warmup LoRA as trainable without creating a new LoRA."""
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    adapter_path = Path(config.adapter_path)
    if not adapter_path.is_dir() or not (
        adapter_path / "adapter_config.json"
    ).is_file():
        raise ValueError(
            "AE training requires a local SFT Warmup PEFT checkpoint containing "
            f"adapter_config.json; got {config.adapter_path!r}"
        )
    if config.load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit AE training requires CUDA")

    dtype = _resolve_dtype(config.dtype)
    adapter_config = PeftConfig.from_pretrained(config.adapter_path)
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.load_in_4bit:
        model_kwargs.update(
            {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
                "device_map": {"": int(os.environ.get("LOCAL_RANK", "0"))},
            }
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        adapter_config.base_model_name_or_path,
        **model_kwargs,
    )
    if config.load_in_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    model, adapter_summary = load_peft_adapter_with_compat(
        base_model,
        config.adapter_path,
        PeftModel,
        torch,
        is_trainable=True,
    )

    if config.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("The loaded Warmup adapter has no trainable parameters")
    unexpected = [
        name
        for name, _ in trainable
        if "lora_" not in name and "modules_to_save" not in name
    ]
    if unexpected:
        raise RuntimeError(
            "AE loading unexpectedly made base-model parameters trainable; "
            f"first names={unexpected[:10]}"
        )

    tokenizer_source = (
        config.tokenizer_path
        or (
            config.adapter_path
            if (adapter_path / "tokenizer_config.json").is_file()
            else adapter_config.base_model_name_or_path
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("AE strict prefix masking requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _load_chat_template(
        tokenizer,
        config.chat_template_path,
        config.trust_remote_code,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template")

    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return model, tokenizer, {
        "adapter_path": config.adapter_path,
        "base_model": adapter_config.base_model_name_or_path,
        "tokenizer": str(tokenizer_source),
        "continued_existing_adapter": True,
        "created_new_adapter": False,
        "adapter_load": adapter_summary,
        "dtype": str(dtype),
        "load_in_4bit": config.load_in_4bit,
        "gradient_checkpointing": config.gradient_checkpointing,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
    }


def _replace_emotion(
    conditioning_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
    sample_id: str,
) -> dict[str, Any]:
    result = dict(conditioning_record)
    result["sample_id"] = sample_id
    result["emotion"] = target_record["emotion"]
    return result


def tokenize_pair_record(
    tokenizer: Any,
    pair: Mapping[str, Any],
    *,
    max_length: int,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create four paths scored only on emotion label/intensity value tokens."""
    pair_id = pair.get("pair_id")
    left = pair.get("left")
    right = pair.get("right")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError("Pair record has no pair_id")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError(f"{pair_id}: left and right must be objects")
    if left.get("source_sample_id") != right.get("source_sample_id"):
        raise ValueError(f"{pair_id}: source_sample_id differs across pair")
    if left.get("situation") != right.get("situation"):
        raise ValueError(f"{pair_id}: situation differs across pair")

    records = {
        "ea_given_aa": left,
        "ea_given_ab": _replace_emotion(
            right,
            left,
            f"{pair_id}::Ea-given-Ab",
        ),
        "eb_given_ab": right,
        "eb_given_aa": _replace_emotion(
            left,
            right,
            f"{pair_id}::Eb-given-Aa",
        ),
    }
    branches = {
        name: tokenize_ae_example(
            tokenizer,
            record,
            max_length=max_length,
            enable_thinking=enable_thinking,
            system_prompt=system_prompt,
            chat_template_kwargs=chat_template_kwargs,
            loss_scope="emotion_values",
        )
        for name, record in records.items()
    }
    difference = pair.get("difference")
    raw_weight = (
        difference.get("contrast_weight", 1.0)
        if isinstance(difference, Mapping)
        else 1.0
    )
    pair_weight = float(raw_weight)
    if not math.isfinite(pair_weight) or pair_weight <= 0.0:
        raise ValueError(f"{pair_id}: contrast_weight must be finite and positive")
    return {
        "pair_id": pair_id,
        "source_sample_id": left["source_sample_id"],
        "branches": branches,
        "pair_weight": pair_weight,
    }


def tokenize_replay_example(
    tokenizer: Any,
    *,
    prompt_messages: Sequence[Mapping[str, str]],
    assistant_response: str,
    sample_id: str,
    max_length: int,
    enable_thinking: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokenize a full-chain SFT replay example with prompt-only masking."""
    if not assistant_response:
        raise ValueError(f"{sample_id}: assistant_response is empty")
    template_kwargs = {
        "enable_thinking": enable_thinking,
        **dict(chat_template_kwargs or {}),
    }
    rendered_prompt = tokenizer.apply_chat_template(
        list(prompt_messages),
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": assistant_response},
    ]
    rendered_full = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        **template_kwargs,
    )
    expected_start = rendered_prompt + assistant_response
    if not rendered_full.startswith(expected_start):
        raise ValueError(
            f"{sample_id}: chat template does not preserve the replay completion "
            "boundary exactly"
        )
    rendered_suffix = rendered_full[len(expected_start) :]

    def encode(text: str) -> list[int]:
        values = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=False,
        )["input_ids"]
        if values and isinstance(values[0], list):
            values = values[0]
        return list(values)

    # Tokenize the already-rendered prompt directly.
    # Keep prompt, target, and suffix boundaries explicit.
    prompt_ids = encode(rendered_prompt)
    target_ids = encode(assistant_response)
    suffix_ids = encode(rendered_suffix)

    if not prompt_ids:
        raise ValueError(f"{sample_id}: replay prompt has no tokens")

    if not target_ids:
        raise ValueError(f"{sample_id}: replay target has no tokens")

    input_ids = [*prompt_ids, *target_ids, *suffix_ids]

    # Verify that segmented tokenization preserves the full text.
    decoded_full = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if decoded_full != rendered_full:
        raise ValueError(
            f"{sample_id}: segmented replay tokenization does not "
            "reconstruct the complete chat-template output"
        )
    suffix_ids = encode(rendered_suffix)
    if not target_ids:
        raise ValueError(f"{sample_id}: replay target has no tokens")
    input_ids = [*prompt_ids, *target_ids, *suffix_ids]
    if len(input_ids) > max_length:
        raise ValueError(
            f"{sample_id}: replay length {len(input_ids)} exceeds "
            f"max_length={max_length}; replay examples are never truncated"
        )
    labels = [IGNORE_INDEX] * len(prompt_ids)
    labels.extend(target_ids)
    labels.extend([IGNORE_INDEX] * len(suffix_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sample_id": sample_id,
    }


class ProportionalObjectiveDataset(Dataset):
    """Deterministic epoch stream with configurable objective proportions."""

    def __init__(
        self,
        *,
        conditional: Sequence[Mapping[str, Any]],
        pairs: Sequence[Mapping[str, Any]],
        preservation: Sequence[Mapping[str, Any]],
        conditional_ratio: float = 1.0,
        pair_ratio: float = 0.5,
        preservation_ratio: float = 0.1,
        epoch_size: int | None = None,
        seed: int = 42,
    ) -> None:
        sources = {
            "conditional": list(conditional),
            "pair": list(pairs),
            "preservation": list(preservation),
        }
        ratios = {
            "conditional": float(conditional_ratio),
            "pair": float(pair_ratio),
            "preservation": float(preservation_ratio),
        }
        if ratios["conditional"] <= 0.0:
            raise ValueError("conditional_ratio must be positive")
        if not sources["conditional"]:
            raise ValueError("Conditional AE data cannot be empty")
        for objective in OBJECTIVES:
            ratio = ratios[objective]
            if not math.isfinite(ratio) or ratio < 0.0:
                raise ValueError(f"{objective}_ratio must be finite and non-negative")
            if ratio > 0.0 and not sources[objective]:
                raise ValueError(
                    f"{objective}_ratio is positive but no {objective} data was provided"
                )

        ratio_total = sum(ratios.values())
        if epoch_size is None:
            conditional_share = ratios["conditional"] / ratio_total
            epoch_size = math.ceil(len(sources["conditional"]) / conditional_share)
        if epoch_size <= 0:
            raise ValueError("epoch_size must be positive")

        exact = {
            objective: epoch_size * ratios[objective] / ratio_total
            for objective in OBJECTIVES
        }
        counts = {objective: int(math.floor(exact[objective])) for objective in OBJECTIVES}
        remaining = epoch_size - sum(counts.values())
        priority = sorted(
            OBJECTIVES,
            key=lambda objective: (exact[objective] - counts[objective], ratios[objective]),
            reverse=True,
        )
        for objective in priority[:remaining]:
            counts[objective] += 1
        for objective in OBJECTIVES:
            if ratios[objective] > 0.0 and counts[objective] == 0:
                raise ValueError(
                    f"epoch_size={epoch_size} is too small to include {objective}"
                )

        rng = random.Random(seed)
        units: list[dict[str, Any]] = []
        for objective in OBJECTIVES:
            source = sources[objective]
            order = list(range(len(source)))
            rng.shuffle(order)
            for position in range(counts[objective]):
                if position and position % len(order) == 0:
                    rng.shuffle(order)
                units.append(
                    {
                        "objective": objective,
                        "features": source[order[position % len(order)]],
                    }
                )
        rng.shuffle(units)
        self._units = units
        self.objective_counts = counts
        self.objective_ratios = ratios
        self.epoch_size = epoch_size

    def __len__(self) -> int:
        return len(self._units)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._units[index]


class AEMultiObjectiveCollator:
    """Right-pad mixed conditional, pair, and replay training units."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = IGNORE_INDEX):
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)

    def _pad(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
        if not features:
            raise ValueError("Cannot pad an empty feature list")
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            target = list(feature["labels"])
            if not len(ids) == len(mask) == len(target):
                raise ValueError("input_ids, attention_mask, and labels differ in length")
            padding = maximum - len(ids)
            input_ids.append(ids + [self.pad_token_id] * padding)
            attention_mask.append(mask + [0] * padding)
            labels.append(target + [self.label_pad_token_id] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        conditional: list[Mapping[str, Any]] = []
        preservation: list[Mapping[str, Any]] = []
        pairs: list[Mapping[str, Any]] = []
        for unit in units:
            objective = unit.get("objective")
            features = unit.get("features")
            if objective == "conditional":
                conditional.append(features)
            elif objective == "preservation":
                preservation.append(features)
            elif objective == "pair":
                pairs.append(features)
            else:
                raise ValueError(f"Unknown AE objective: {objective!r}")

        batch: dict[str, Any] = {
            "objective_counts": {
                "conditional": len(conditional),
                "pair": len(pairs),
                "preservation": len(preservation),
            }
        }
        if conditional:
            batch["conditional"] = self._pad(conditional)
        if preservation:
            batch["preservation"] = self._pad(preservation)
        if pairs:
            for pair in pairs:
                branches = pair.get("branches")
                if not isinstance(branches, Mapping) or any(
                    branch not in branches for branch in PAIR_BRANCHES
                ):
                    raise ValueError("Pair features do not contain all four branches")
            batch["pair"] = {
                branch: self._pad(
                    [pair["branches"][branch] for pair in pairs]
                )
                for branch in PAIR_BRANCHES
            }
            batch["pair"]["weights"] = torch.tensor(
                [float(pair.get("pair_weight", 1.0)) for pair in pairs],
                dtype=torch.float32,
            )
        return batch


def _forward_logits(model: Any, batch: Mapping[str, Tensor]) -> Tensor:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("Causal LM forward pass did not return tensor logits")
    return logits


def _teacher_forced_accuracy(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    predictions = logits[:, :-1, :].argmax(dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(IGNORE_INDEX)
    correct = predictions.eq(targets) & valid
    token_accuracy = correct.sum() / valid.sum().clamp_min(1)
    sequence_exact = (correct.sum(dim=1) == valid.sum(dim=1)).float().mean()
    return token_accuracy, sequence_exact


class AETrainer(Trainer):
    """Trainer implementing conditional, four-path pair, and replay losses."""

    def __init__(
        self,
        *args: Any,
        pair_loss_weight: float = 1.0,
        preservation_loss_weight: float = 1.0,
        contrastive_beta: float = 1.0,
        carebench_eval_examples: Sequence[Mapping[str, Any]] | None = None,
        carebench_eval_max_samples: int = 0,
        carebench_eval_max_new_tokens: int = 192,
        **kwargs: Any,
    ) -> None:
        if pair_loss_weight < 0.0 or preservation_loss_weight < 0.0:
            raise ValueError("AE loss weights must be non-negative")
        if contrastive_beta <= 0.0:
            raise ValueError("contrastive_beta must be positive")
        if carebench_eval_max_samples < 0:
            raise ValueError("carebench_eval_max_samples cannot be negative")
        if carebench_eval_max_new_tokens <= 0:
            raise ValueError("carebench_eval_max_new_tokens must be positive")
        self.pair_loss_weight = float(pair_loss_weight)
        self.preservation_loss_weight = float(preservation_loss_weight)
        self.contrastive_beta = float(contrastive_beta)
        self.carebench_eval_examples = list(carebench_eval_examples or ())
        self.carebench_eval_max_samples = int(carebench_eval_max_samples)
        self.carebench_eval_max_new_tokens = int(carebench_eval_max_new_tokens)
        self._ae_metric_totals: dict[str, float] = defaultdict(float)
        self._ae_metric_counts: dict[str, int] = defaultdict(int)
        self._ae_event_counts: dict[str, int] = defaultdict(int)
        super().__init__(*args, **kwargs)
        if self.args.remove_unused_columns:
            raise ValueError(
                "AETrainer requires remove_unused_columns=False because its "
                "collator passes nested multi-objective batches"
            )

    def _record_metric(
        self,
        scope: str,
        name: str,
        value: Tensor | float,
        *,
        weight: int = 1,
    ) -> None:
        if weight <= 0:
            raise ValueError("Metric aggregation weight must be positive")
        number = float(value.detach().float().item() if isinstance(value, Tensor) else value)
        key = f"{scope}:{name}"
        self._ae_metric_totals[key] += number * weight
        self._ae_metric_counts[key] += weight

    def _record_count(self, scope: str, name: str, value: int) -> None:
        self._ae_event_counts[f"{scope}:{name}"] += int(value)

    def _pair_forward(
        self,
        model: Any,
        pair_batch: Mapping[str, Any],
    ) -> PairContrastiveResult:
        scores: dict[str, Tensor] = {}
        for branch in PAIR_BRANCHES:
            branch_batch = pair_batch[branch]
            logits = _forward_logits(model, branch_batch)
            scores[branch] = sequence_mean_log_likelihood(
                logits,
                branch_batch["labels"],
            ).mean_log_likelihood
        return paired_appraisal_contrastive_loss(
            score_ea_given_aa=scores["ea_given_aa"],
            score_ea_given_ab=scores["ea_given_ab"],
            score_eb_given_ab=scores["eb_given_ab"],
            score_eb_given_aa=scores["eb_given_aa"],
            beta=self.contrastive_beta,
            pair_weights=pair_batch["weights"],
        )

    def compute_loss(
        self,
        model: Any,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Tensor | tuple[Tensor, Mapping[str, Tensor]]:
        del num_items_in_batch
        scope = "train" if model.training else "eval"
        counts = inputs.get("objective_counts", {})
        losses: list[Tensor] = []
        conditional_loss: Tensor | None = None
        preservation_loss: Tensor | None = None
        pair_result: PairContrastiveResult | None = None

        conditional_batch = inputs.get("conditional")
        if conditional_batch is not None:
            conditional_logits = _forward_logits(model, conditional_batch)
            conditional_loss = conditional_emotion_sft_loss(
                conditional_logits,
                conditional_batch["labels"],
            )
            losses.append(conditional_loss)
            if scope == "eval":
                token_accuracy, sequence_exact = _teacher_forced_accuracy(
                    conditional_logits,
                    conditional_batch["labels"],
                )
                self._record_metric(scope, "emotion_token_accuracy", token_accuracy)
                self._record_metric(scope, "emotion_sequence_exact", sequence_exact)

        pair_batch = inputs.get("pair")
        if pair_batch is not None:
            pair_result = self._pair_forward(model, pair_batch)
            losses.append(pair_result.loss)

        preservation_batch = inputs.get("preservation")
        if preservation_batch is not None:
            preservation_logits = _forward_logits(model, preservation_batch)
            preservation_loss = replay_preservation_loss(
                preservation_logits,
                preservation_batch["labels"],
            )
            losses.append(preservation_loss)

        if not losses:
            raise ValueError("AE batch contains no active objective")
        zero = losses[0].new_zeros(())
        conditional_component = conditional_loss if conditional_loss is not None else zero
        pair_component = pair_result.loss if pair_result is not None else zero
        preservation_component = (
            preservation_loss if preservation_loss is not None else zero
        )
        total_loss = (
            conditional_component
            + self.pair_loss_weight * pair_component
            + self.preservation_loss_weight * preservation_component
        )
        diagnostics: dict[str, Tensor] = {}
        if pair_result is not None:
            diagnostics = {
                "pair_margin_a": pair_result.margin_a.mean(),
                "pair_margin_b": pair_result.margin_b.mean(),
                "pair_margin_mean": torch.cat(
                    (pair_result.margin_a, pair_result.margin_b)
                ).mean(),
                "pair_accuracy_a": pair_result.accuracy_a.mean(),
                "pair_accuracy_b": pair_result.accuracy_b.mean(),
                "pair_accuracy_mean": torch.cat(
                    (pair_result.accuracy_a, pair_result.accuracy_b)
                ).mean(),
                "pair_accuracy_both": (
                    pair_result.margin_a.gt(0) & pair_result.margin_b.gt(0)
                )
                .float()
                .mean(),
            }
        output = AELossOutput(
            total_loss=total_loss,
            conditional_loss=conditional_component,
            pair_loss=pair_component,
            preservation_loss=preservation_component,
            diagnostics=diagnostics,
        )
        self._record_metric(scope, "loss_total", output.total_loss)
        if conditional_loss is not None:
            self._record_metric(scope, "loss_conditional", output.conditional_loss)
        if pair_result is not None:
            pair_count = int(pair_result.margin_a.numel())
            self._record_metric(
                scope,
                "loss_pair",
                output.pair_loss,
                weight=pair_count,
            )
            self._record_metric(
                scope,
                "pair_loss_unweighted",
                pair_result.per_pair_loss.mean(),
                weight=pair_count,
            )
            for name, value in output.diagnostics.items():
                self._record_metric(scope, name, value, weight=pair_count)
        if preservation_loss is not None:
            self._record_metric(scope, "loss_preservation", output.preservation_loss)
        for objective in OBJECTIVES:
            self._record_count(
                scope,
                f"{objective}_examples",
                int(counts.get(objective, 0)),
            )
        if conditional_loss is not None:
            self._record_metric(
                scope,
                "emotion_perplexity",
                torch.exp(conditional_loss.detach().float().clamp(max=20.0)),
            )
        result_outputs = {"loss": total_loss}
        return (total_loss, result_outputs) if return_outputs else total_loss

    def prediction_step(
        self,
        model: Any,
        inputs: Mapping[str, Any],
        prediction_loss_only: bool,
        ignore_keys: Any = None,
    ) -> tuple[Tensor, None, None]:
        del prediction_loss_only, ignore_keys
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _carebench_generation_metrics(self) -> dict[str, float]:
        examples = self.carebench_eval_examples
        if self.carebench_eval_max_samples:
            examples = examples[: self.carebench_eval_max_samples]
        if not examples:
            return {}
        tokenizer = self.processing_class
        if tokenizer is None:
            raise ValueError("CAREBench validation generation requires a tokenizer")

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        results: list[dict[str, Any]] = []
        try:
            for example in examples:
                prefix_ids = example.get("prefix_input_ids")
                assistant_prefix = example.get("assistant_prefix")
                gold_emotion = example.get("gold_emotion")
                if (
                    not isinstance(prefix_ids, list)
                    or not prefix_ids
                    or not isinstance(assistant_prefix, str)
                    or not isinstance(gold_emotion, Mapping)
                ):
                    raise ValueError("Invalid CAREBench AE validation example")
                input_ids = torch.tensor(
                    [prefix_ids],
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        max_new_tokens=self.carebench_eval_max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                continuation = tokenizer.decode(
                    generated[0, input_ids.shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                try:
                    candidate = parse_policy_output(
                        assistant_prefix + continuation,
                        POLICY_DIMENSIONS,
                    )
                    valid = True
                except Exception:  # noqa: BLE001
                    candidate = None
                    valid = False
                results.append(
                    {
                        "valid": valid,
                        "candidate": candidate,
                        "reference": {"gold_emotion": dict(gold_emotion)},
                    }
                )
        finally:
            if was_training:
                model.train()

        metrics = carebench_metric_summary(results)
        # Appraisal is supplied as the condition during AE validation.
        return {
            name: value
            for name, value in metrics.items()
            if "appraisal" not in name
        }

    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: Any = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.accelerator.wait_for_everyone()
        if self.is_world_process_zero() and self.carebench_eval_examples:
            carebench = self._carebench_generation_metrics()
            prefixed = {
                f"{metric_key_prefix}_{name}": value
                for name, value in carebench.items()
            }
            metrics.update(prefixed)
            self.log(prefixed)
        self.accelerator.wait_for_everyone()
        return metrics

    def _flush_ae_metrics(self, logs: dict[str, Any], scope: str) -> None:
        prefix = f"{scope}:"
        for key in list(self._ae_metric_totals):
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            count = self._ae_metric_counts.pop(key)
            logs[f"{scope}_ae_{name}"] = self._ae_metric_totals.pop(key) / count
        for key in list(self._ae_event_counts):
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            logs[f"{scope}_ae_{name}"] = self._ae_event_counts.pop(key)

    def log(self, logs: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        scope = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        self._flush_ae_metrics(logs, scope)
        super().log(logs, *args, **kwargs)


__all__ = [
    "AETrainer",
    "AEMultiObjectiveCollator",
    "OBJECTIVES",
    "PAIR_BRANCHES",
    "ProportionalObjectiveDataset",
    "WarmupAdapterLoadConfig",
    "load_trainable_warmup_adapter",
    "tokenize_pair_record",
    "tokenize_replay_example",
]
