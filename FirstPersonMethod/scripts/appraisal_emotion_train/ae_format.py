#!/usr/bin/env python3
"""Format appraisal-conditioned emotion examples for AE training.

The model receives the situation as the user message and a human appraisal as
an unfinished assistant JSON prefix. Only the following emotion JSON fragment
is supervised. This mirrors the prefix-continuation path used by
``counterfactual_prefix_transformers.py`` instead of moving the appraisal into
the user prompt.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    scripts_dir = Path(__file__).resolve().parent.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_grpo.parsing import parse_policy_output
    from train_grpo.spec import (
        CAREBENCH_REASONING_KEYS,
        build_policy_system_prompt,
    )
else:
    from train_grpo.parsing import parse_policy_output
    from train_grpo.spec import (
        CAREBENCH_REASONING_KEYS,
        build_policy_system_prompt,
    )


IGNORE_INDEX = -100
POLICY_DIMENSIONS = tuple(CAREBENCH_REASONING_KEYS)
EMOTION_KEYS = (
    "positive_intensity",
    "negative_intensity",
    "positive_labels",
    "negative_labels",
)


@dataclass(frozen=True)
class AETextExample:
    sample_id: str
    source_sample_id: str
    annotator_id: str
    annotator_type: str
    situation: str
    prompt_messages: list[dict[str, str]]
    assistant_prefix: str
    emotion_target: str
    full_assistant_response: str


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _ordered_reasoning(record: Mapping[str, Any]) -> dict[str, str]:
    raw = record.get("appraisal_reasoning")
    if not isinstance(raw, Mapping):
        raise ValueError("appraisal_reasoning must be an object")
    missing = [dimension for dimension in POLICY_DIMENSIONS if dimension not in raw]
    extra = sorted(set(raw) - set(POLICY_DIMENSIONS))
    if missing or extra:
        raise ValueError(
            "appraisal_reasoning keys mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return {
        dimension: _require_text(
            raw[dimension],
            f"appraisal_reasoning.{dimension}",
        )
        for dimension in POLICY_DIMENSIONS
    }


def _ordered_emotion(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("emotion")
    if not isinstance(raw, Mapping):
        raise ValueError("emotion must be an object")
    missing = [key for key in EMOTION_KEYS if key not in raw]
    extra = sorted(set(raw) - set(EMOTION_KEYS))
    if missing or extra:
        raise ValueError(f"emotion keys mismatch; missing={missing}, extra={extra}")
    return {key: raw[key] for key in EMOTION_KEYS}


def build_assistant_prefix(reasoning: Mapping[str, str]) -> str:
    """Build the same unfinished JSON prefix used for prefix intervention."""
    ordered = {
        dimension: _require_text(
            reasoning.get(dimension),
            f"appraisal_reasoning.{dimension}",
        )
        for dimension in POLICY_DIMENSIONS
    }
    complete = json.dumps(
        {"appraisal_reasoning": ordered},
        ensure_ascii=False,
        indent=2,
    )
    if not complete.endswith("\n}"):
        raise RuntimeError("Unexpected appraisal JSON serialization")
    return complete[:-2] + ","


def build_emotion_target(emotion: Mapping[str, Any]) -> str:
    """Serialize the emotion key/value plus the outer JSON closing brace."""
    complete = json.dumps(
        {"emotion": {key: emotion[key] for key in EMOTION_KEYS}},
        ensure_ascii=False,
        indent=2,
    )
    opening = "{\n  "
    if not complete.startswith(opening) or not complete.endswith("\n}"):
        raise RuntimeError("Unexpected emotion JSON serialization")
    return "\n  " + complete[len(opening) :]


def build_prompt_messages(
    situation: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Return a chain prompt whose user content is exactly the situation."""
    situation = _require_text(situation, "situation")
    system = (
        build_policy_system_prompt(POLICY_DIMENSIONS)
        if system_prompt is None
        else _require_text(system_prompt, "system_prompt")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": situation},
    ]


def format_ae_record(
    record: Mapping[str, Any],
    system_prompt: str | None = None,
) -> AETextExample:
    """Validate one prepared conditional record and construct its AE text."""
    if not isinstance(record, Mapping):
        raise ValueError("AE record must be an object")
    sample_id = _require_text(record.get("sample_id"), "sample_id")
    source_sample_id = _require_text(
        record.get("source_sample_id"),
        f"{sample_id}: source_sample_id",
    )
    annotator_id = _require_text(
        record.get("annotator_id"),
        f"{sample_id}: annotator_id",
    )
    annotator_type = _require_text(
        record.get("annotator_type"),
        f"{sample_id}: annotator_type",
    )
    situation = _require_text(record.get("situation"), f"{sample_id}: situation")
    reasoning = _ordered_reasoning(record)
    emotion = _ordered_emotion(record)

    # Validate the complete chain with the same parser used by policy training.
    complete_payload = {
        "appraisal_reasoning": reasoning,
        "emotion": emotion,
    }
    complete_json = json.dumps(complete_payload, ensure_ascii=False, indent=2)
    parsed = parse_policy_output(complete_json, POLICY_DIMENSIONS)
    if parsed["appraisal_reasoning"] != reasoning or parsed["emotion"] != emotion:
        raise ValueError(
            f"{sample_id}: AE input is not in canonical CAREBench form; run "
            "prepare_ae_data.py before formatting"
        )

    prefix = build_assistant_prefix(reasoning)
    target = build_emotion_target(emotion)
    full_response = prefix + target
    if json.loads(full_response) != complete_payload:
        raise RuntimeError(f"{sample_id}: prefix and emotion target do not rejoin")

    return AETextExample(
        sample_id=sample_id,
        source_sample_id=source_sample_id,
        annotator_id=annotator_id,
        annotator_type=annotator_type,
        situation=situation,
        prompt_messages=build_prompt_messages(situation, system_prompt),
        assistant_prefix=prefix,
        emotion_target=target,
        full_assistant_response=full_response,
    )


def format_ae_pair(
    pair: Mapping[str, Any],
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Format both valid annotation paths in one prepared contrastive pair."""
    if not isinstance(pair, Mapping):
        raise ValueError("AE pair must be an object")
    pair_id = _require_text(pair.get("pair_id"), "pair_id")
    left_raw = pair.get("left")
    right_raw = pair.get("right")
    if not isinstance(left_raw, Mapping) or not isinstance(right_raw, Mapping):
        raise ValueError(f"{pair_id}: left and right must be conditional records")
    left = format_ae_record(left_raw, system_prompt)
    right = format_ae_record(right_raw, system_prompt)
    if left.source_sample_id != right.source_sample_id:
        raise ValueError(f"{pair_id}: pair sides have different source_sample_id")
    if left.situation != right.situation:
        raise ValueError(f"{pair_id}: pair sides have different situations")
    if left.annotator_id == right.annotator_id:
        raise ValueError(f"{pair_id}: pair sides must use different annotators")
    return {
        "pair_id": pair_id,
        "source_sample_id": left.source_sample_id,
        "situation": left.situation,
        "left": asdict(left),
        "right": asdict(right),
        "difference": pair.get("difference", {}),
    }


def _flat_token_ids(value: Any, source: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in value
    ):
        raise TypeError(f"{source} did not return a flat token-id list")
    return value


def _encode_text(tokenizer: Any, text: str, source: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=False,
    )
    return _flat_token_ids(encoded, source)


def _render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    continue_final_message: bool,
    chat_template_kwargs: Mapping[str, Any],
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        **chat_template_kwargs,
    }
    if continue_final_message:
        kwargs["continue_final_message"] = True
    rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) must return text")
    return rendered


def _template_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    continue_final_message: bool,
    chat_template_kwargs: Mapping[str, Any],
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        **chat_template_kwargs,
    }
    if continue_final_message:
        kwargs["continue_final_message"] = True
    return _flat_token_ids(
        tokenizer.apply_chat_template(list(messages), **kwargs),
        "apply_chat_template",
    )


def validate_loss_mask(
    input_ids: Sequence[int],
    labels: Sequence[int],
    emotion_start: int,
    emotion_end: int,
    *,
    require_full_target: bool = True,
) -> None:
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have equal length")
    if not 0 <= emotion_start < emotion_end <= len(input_ids):
        raise ValueError("Invalid emotion token span")
    if any(label != IGNORE_INDEX for label in labels[:emotion_start]):
        raise ValueError("Situation or appraisal-prefix tokens are supervised")
    target_labels = labels[emotion_start:emotion_end]
    target_ids = input_ids[emotion_start:emotion_end]
    if require_full_target:
        if list(target_labels) != list(target_ids):
            raise ValueError("Emotion target labels do not match input_ids")
    else:
        if not any(label != IGNORE_INDEX for label in target_labels):
            raise ValueError("Emotion value mask has no supervised tokens")
        if any(
            label not in (IGNORE_INDEX, token_id)
            for label, token_id in zip(target_labels, target_ids)
        ):
            raise ValueError("Emotion value labels do not match input_ids")
    if any(label != IGNORE_INDEX for label in labels[emotion_end:]):
        raise ValueError("Tokens after the emotion target are supervised")


def _emotion_value_character_spans(
    emotion_target: str,
    emotion: Mapping[str, Any],
) -> list[tuple[int, int]]:
    """Locate intensity numbers and label text inside the serialized target."""
    spans: list[tuple[int, int]] = []
    for key in ("positive_intensity", "negative_intensity"):
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*(-?\d+)',
            emotion_target,
        )
        if match is None or int(match.group(1)) != int(emotion[key]):
            raise ValueError(f"Cannot locate serialized emotion value for {key}")
        spans.append(match.span(1))

    for key in ("positive_labels", "negative_labels"):
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*\[',
            emotion_target,
        )
        if match is None:
            raise ValueError(f"Cannot locate serialized emotion array for {key}")
        array_end = emotion_target.find("]", match.end())
        if array_end < 0:
            raise ValueError(f"Cannot locate closing bracket for {key}")
        cursor = match.end()
        for label in emotion[key]:
            literal = json.dumps(str(label), ensure_ascii=False)
            position = emotion_target.find(literal, cursor, array_end)
            if position < 0:
                raise ValueError(f"Cannot locate serialized label {label!r} in {key}")
            # CAREBench canonical labels do not require JSON escaping. Refuse an
            # ambiguous character mask instead of accidentally scoring syntax.
            if literal != f'"{label}"':
                raise ValueError(f"Escaped emotion label is unsupported: {label!r}")
            spans.append((position + 1, position + len(literal) - 1))
            cursor = position + len(literal)
    return spans


def _emotion_value_token_mask(
    tokenizer: Any,
    emotion_target: str,
    target_ids: Sequence[int],
    emotion: Mapping[str, Any],
) -> list[bool]:
    encoded = tokenizer(
        emotion_target,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    encoded_ids = _flat_token_ids(encoded, "emotion target with offsets")
    if encoded_ids != list(target_ids):
        raise ValueError("Emotion offset tokenization differs from target tokenization")
    offsets = encoded.get("offset_mapping")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if not isinstance(offsets, list) or len(offsets) != len(target_ids):
        raise ValueError("Tokenizer returned invalid emotion offset mapping")
    spans = _emotion_value_character_spans(emotion_target, emotion)
    mask: list[bool] = []
    for offset in offsets:
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or any(not isinstance(value, int) for value in offset)
        ):
            raise ValueError("Tokenizer returned a malformed target token offset")
        start, end = offset
        mask.append(
            end > start
            and any(start < span_end and end > span_start for span_start, span_end in spans)
        )
    if not any(mask):
        raise ValueError("Emotion value token mask is empty")
    return mask


def tokenize_ae_example(
    tokenizer: Any,
    record: Mapping[str, Any] | AETextExample,
    *,
    max_length: int | None = None,
    enable_thinking: bool = False,
    system_prompt: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    include_text_metadata: bool = False,
    loss_scope: str = "full_emotion_json",
) -> dict[str, Any]:
    """Tokenize one AE example with loss only on the emotion JSON target.

    The sequence is deliberately encoded in three segments: the exact
    ``continue_final_message`` prefix used at inference, the emotion target,
    and the chat-template assistant suffix. Segmenting prevents a tokenizer
    from merging the final appraisal-prefix whitespace into the first emotion
    token and moving the loss boundary.
    """
    example = (
        record if isinstance(record, AETextExample) else format_ae_record(record, system_prompt)
    )
    if loss_scope not in {"full_emotion_json", "emotion_values"}:
        raise ValueError(
            "loss_scope must be 'full_emotion_json' or 'emotion_values'"
        )
    template_kwargs = {
        "enable_thinking": enable_thinking,
        **dict(chat_template_kwargs or {}),
    }
    prefix_messages = [
        *example.prompt_messages,
        {"role": "assistant", "content": example.assistant_prefix},
    ]
    full_messages = [
        *example.prompt_messages,
        {"role": "assistant", "content": example.full_assistant_response},
    ]
    rendered_prefix = _render_chat(
        tokenizer,
        prefix_messages,
        continue_final_message=True,
        chat_template_kwargs=template_kwargs,
    )
    rendered_full = _render_chat(
        tokenizer,
        full_messages,
        continue_final_message=False,
        chat_template_kwargs=template_kwargs,
    )
    expected_start = rendered_prefix + example.emotion_target
    if not rendered_full.startswith(expected_start):
        raise ValueError(
            f"{example.sample_id}: chat template does not preserve the unfinished "
            "assistant prefix exactly; refusing to build an ambiguous loss mask"
        )
    rendered_suffix = rendered_full[len(expected_start) :]

    prefix_ids = _template_token_ids(
        tokenizer,
        prefix_messages,
        continue_final_message=True,
        chat_template_kwargs=template_kwargs,
    )
    rendered_prefix_ids = _encode_text(
        tokenizer,
        rendered_prefix,
        "rendered assistant prefix",
    )
    if prefix_ids != rendered_prefix_ids:
        raise ValueError(
            f"{example.sample_id}: tokenized chat prefix differs from its rendered "
            "text; tokenizer/chat-template pair is incompatible with strict AE masking"
        )
    target_ids = _encode_text(tokenizer, example.emotion_target, "emotion target")
    suffix_ids = _encode_text(tokenizer, rendered_suffix, "assistant suffix")
    if not target_ids:
        raise ValueError(f"{example.sample_id}: emotion target has no tokens")

    input_ids = [*prefix_ids, *target_ids, *suffix_ids]
    emotion_start = len(prefix_ids)
    emotion_end = emotion_start + len(target_ids)
    target_supervision = [True] * len(target_ids)
    if loss_scope == "emotion_values":
        complete_payload = json.loads(example.full_assistant_response)
        target_supervision = _emotion_value_token_mask(
            tokenizer,
            example.emotion_target,
            target_ids,
            complete_payload["emotion"],
        )
    labels = [IGNORE_INDEX] * emotion_start
    labels.extend(
        token_id if supervised else IGNORE_INDEX
        for token_id, supervised in zip(target_ids, target_supervision)
    )
    labels.extend([IGNORE_INDEX] * len(suffix_ids))
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if len(input_ids) > max_length:
            raise ValueError(
                f"{example.sample_id}: encoded length {len(input_ids)} exceeds "
                f"max_length={max_length}; AE examples are never truncated because "
                "that could invalidate the emotion loss boundary"
            )
    validate_loss_mask(
        input_ids,
        labels,
        emotion_start,
        emotion_end,
        require_full_target=loss_scope == "full_emotion_json",
    )

    result: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_mask": [0] * emotion_start
        + [int(supervised) for supervised in target_supervision]
        + [0] * len(suffix_ids),
        "loss_scope": loss_scope,
        "supervised_target_tokens": sum(target_supervision),
        "emotion_start": emotion_start,
        "emotion_end": emotion_end,
        "sample_id": example.sample_id,
        "source_sample_id": example.source_sample_id,
        "annotator_id": example.annotator_id,
        "annotator_type": example.annotator_type,
    }
    if include_text_metadata:
        result.update(
            {
                "situation": example.situation,
                "prompt_messages": example.prompt_messages,
                "assistant_prefix": example.assistant_prefix,
                "emotion_target": example.emotion_target,
                "full_assistant_response": example.full_assistant_response,
                "rendered_prefix": rendered_prefix,
                "rendered_suffix": rendered_suffix,
            }
        )
    return result


__all__ = [
    "AETextExample",
    "EMOTION_KEYS",
    "IGNORE_INDEX",
    "POLICY_DIMENSIONS",
    "build_assistant_prefix",
    "build_emotion_target",
    "build_prompt_messages",
    "format_ae_pair",
    "format_ae_record",
    "tokenize_ae_example",
    "validate_loss_mask",
]
