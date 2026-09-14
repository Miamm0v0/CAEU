#!/usr/bin/env python3
"""Single-call full-chain and core-appraisal-conditioned benchmark tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import baseline_vllm as baseline
import sft_common
from train_grpo.parsing import parse_policy_output
from train_grpo.spec import (
    APPRAISAL_DEFINITIONS,
    APPRAISAL_DIMENSIONS,
    policy_output_example,
)


FULL_CHAIN_TASK = "full-chain"
FULL_CHAIN_OUTPUT_TASKS = tuple(baseline.TASKS)
CHAIN_EMOTION_TASK = "chain-emotion"
CHAIN_EMOTION_OUTPUT_TASKS = (
    "positive-level",
    "negative-level",
    "positive-labels",
    "negative-labels",
)
CHAIN_CORE_APPRAISAL_SOURCES = ("separate", CHAIN_EMOTION_TASK)
CHAIN_ALL_TASK = "chain-all"
CHAIN_TASK_TO_OUTPUT_TASK = {
    "chain-appraisals": "appraisals",
    "chain-positive-level": "positive-level",
    "chain-negative-level": "negative-level",
    "chain-positive-labels": "positive-labels",
    "chain-negative-labels": "negative-labels",
    "chain-core-appraisals": "core-appraisals",
}
CHAIN_TASKS = tuple(CHAIN_TASK_TO_OUTPUT_TASK)
GRPO_SCHEMA_CHAIN_TASKS = (
    "chain-appraisals",
    "chain-core-appraisals",
)
CHAIN_ALL_EMOTION_MODES = ("separate", CHAIN_EMOTION_TASK)
CHAIN_ALL_WITH_JOINT_EMOTION_TASKS = (
    "chain-appraisals",
    CHAIN_EMOTION_TASK,
    "chain-core-appraisals",
)
DIRECT_APPRAISALS_TASK = "direct-appraisals"
DIRECT_EMOTION_TASK = "direct-emotion"
OFFICIAL_DIRECT_TASK = "official-direct"
OFFICIAL_DIRECT_TASKS = (DIRECT_APPRAISALS_TASK, DIRECT_EMOTION_TASK)
CORE_APPRAISAL_DIMENSIONS = tuple(sft_common.CORE_APPRAISAL_ORDER)
POLICY_APPRAISAL_DIMENSIONS = tuple(APPRAISAL_DIMENSIONS[:5])
EMOTION_LABEL_SPACE_CONFIG_KEY = "_emotion_label_space"
EMOTION_OUTPUT_MODE_CONFIG_KEY = "_emotion_output_mode"
EMOTION_OUTPUT_MODES = ("single-label", "multi-label")
EVALUATION_ONLY_EMOTION_RANKING_FIELD = "evaluation_only_emotion_ranking"
EMOTION_LABEL_SPACES = {
    "carebench": {
        "positive": dict(sft_common.POSITIVE_LABEL_DESCRIPTIONS),
        "negative": dict(sft_common.NEGATIVE_LABEL_DESCRIPTIONS),
    },
    "covidet": {
        "positive": {
            "anticipation": "anticipation",
            "joy": "joy",
            "trust": "trust",
        },
        "negative": {
            "anger": "anger",
            "disgust": "disgust",
            "fear": "fear",
            "sadness": "sadness",
        },
    },
    "crowd-envent": {
        "positive": {
            "joy": "joy",
            "pride": "pride",
            "relief": "relief",
            "trust": "trust",
        },
        "negative": {
            "anger": "anger",
            "boredom": "boredom",
            "disgust": "disgust",
            "fear": "fear",
            "guilt": "guilt",
            "sadness": "sadness",
            "shame": "shame",
            "surprise": "surprise",
        },
    },
}
NATIVE_EMOTION_LABELS = {
    "covidet": (
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "trust",
    ),
    "crowd-envent": (
        "anger",
        "boredom",
        "disgust",
        "fear",
        "guilt",
        "joy",
        "pride",
        "relief",
        "sadness",
        "shame",
        "surprise",
        "trust",
        "no-emotion",
    ),
}
POLICY_TO_CAREBENCH_CORE_DIMENSION = {
    "relevance": "relevance",
    "epistemic": "certainty",
    "goal_congruence": "congruence",
    "agency_accountability": "accountability",
    "control_coping_potential": "control",
}
CAREBENCH_TO_POLICY_CORE_DIMENSION = {
    carebench_dimension: policy_dimension
    for policy_dimension, carebench_dimension in (
        POLICY_TO_CAREBENCH_CORE_DIMENSION.items()
    )
}
if set(POLICY_TO_CAREBENCH_CORE_DIMENSION) != set(POLICY_APPRAISAL_DIMENSIONS):
    raise RuntimeError("Policy-to-CAREBench appraisal mapping is incomplete")
if set(POLICY_TO_CAREBENCH_CORE_DIMENSION.values()) != set(
    CORE_APPRAISAL_DIMENSIONS
):
    raise RuntimeError("Policy-to-CAREBench appraisal mapping has invalid targets")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from a model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty chain response")

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Response does not contain a valid JSON object")


def _require_dict(value: Any, field: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_int(value: Any, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{field}={value} is outside [{minimum}, {maximum}]")
    return value


def _reverse_score_map(
    prompt_cfg: Dict[str, Any], task: str, minimum: int, maximum: int
) -> Dict[int, str]:
    raw_map = prompt_cfg.get("label_maps", {}).get(task, {})
    if not isinstance(raw_map, dict):
        raise ValueError(f"Missing [label_maps.{task}] in prompt config")

    result: Dict[int, str] = {}
    for raw_label, raw_score in raw_map.items():
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            continue
        if minimum <= raw_score <= maximum:
            result[raw_score] = str(raw_label)
    expected = set(range(minimum, maximum + 1))
    if set(result) != expected:
        raise ValueError(
            f"[label_maps.{task}] must define every score from {minimum} to {maximum}"
        )
    return result


def set_emotion_label_space(
    prompt_cfg: Dict[str, Any], label_space: str
) -> None:
    if label_space not in EMOTION_LABEL_SPACES:
        raise ValueError(
            f"Unknown emotion label space {label_space!r}; "
            f"allowed={list(EMOTION_LABEL_SPACES)}"
        )
    prompt_cfg[EMOTION_LABEL_SPACE_CONFIG_KEY] = label_space


def emotion_label_space(prompt_cfg: Dict[str, Any]) -> str:
    value = prompt_cfg.get(EMOTION_LABEL_SPACE_CONFIG_KEY, "carebench")
    if value not in EMOTION_LABEL_SPACES:
        raise ValueError(f"Invalid configured emotion label space: {value!r}")
    return str(value)


def set_emotion_output_mode(
    prompt_cfg: Dict[str, Any], output_mode: str
) -> None:
    if output_mode not in EMOTION_OUTPUT_MODES:
        raise ValueError(
            f"Unknown emotion output mode {output_mode!r}; "
            f"allowed={list(EMOTION_OUTPUT_MODES)}"
        )
    prompt_cfg[EMOTION_OUTPUT_MODE_CONFIG_KEY] = output_mode


def emotion_output_mode(prompt_cfg: Dict[str, Any]) -> str:
    value = prompt_cfg.get(EMOTION_OUTPUT_MODE_CONFIG_KEY)
    if value is None:
        return (
            "multi-label"
            if emotion_label_space(prompt_cfg) == "carebench"
            else "single-label"
        )
    if value not in EMOTION_OUTPUT_MODES:
        raise ValueError(f"Invalid configured emotion output mode: {value!r}")
    return str(value)


def emotion_label_descriptions(
    prompt_cfg: Dict[str, Any], valence: str
) -> Dict[str, str]:
    if valence not in {"positive", "negative"}:
        raise ValueError("valence must be 'positive' or 'negative'")
    return dict(EMOTION_LABEL_SPACES[emotion_label_space(prompt_cfg)][valence])


def emotion_label_option_lines(
    prompt_cfg: Dict[str, Any], valence: str
) -> str:
    descriptions = emotion_label_descriptions(prompt_cfg, valence)
    return "\n".join(
        f"- {label}: {description}"
        for label, description in descriptions.items()
    )


def native_emotion_labels(prompt_cfg: Dict[str, Any]) -> tuple[str, ...]:
    label_space = emotion_label_space(prompt_cfg)
    if label_space not in NATIVE_EMOTION_LABELS:
        raise ValueError(
            f"Native emotion labels are unavailable for {label_space!r}"
        )
    return NATIVE_EMOTION_LABELS[label_space]


def native_emotion_output_schema(prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    label_space = emotion_label_space(prompt_cfg)
    if label_space == "covidet":
        return {"labels": ["joy"]}
    if label_space == "crowd-envent":
        return {"label": "joy", "intensity": 3}
    raise ValueError(
        f"Native emotion output is unavailable for {label_space!r}"
    )


def crowd_envent_appraisal_definitions(
    prompt_cfg: Dict[str, Any],
) -> Dict[str, str]:
    external = prompt_cfg.get("external", {})
    crowd_config = (
        external.get("crowd-envent", {}) if isinstance(external, dict) else {}
    )
    definitions = (
        crowd_config.get("appraisal_ratings", {})
        if isinstance(crowd_config, dict)
        else {}
    )
    if not isinstance(definitions, dict) or len(definitions) != 21:
        raise ValueError(
            "Prompt config must define all 21 "
            "[external.crowd-envent.appraisal_ratings] entries"
        )
    normalized: Dict[str, str] = {}
    for field, description in definitions.items():
        if not isinstance(field, str) or not field.strip():
            raise ValueError("crowd-enVent appraisal field names must be non-empty")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                f"crowd-enVent appraisal description for {field!r} must be non-empty"
            )
        normalized[field.strip()] = description.strip()
    return normalized


def _parse_native_emotion(
    raw_emotion: Any, prompt_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    raw_emotion = _require_dict(raw_emotion, "emotion")
    label_space = emotion_label_space(prompt_cfg)
    allowed = native_emotion_labels(prompt_cfg)
    if label_space == "covidet":
        _require_exact_keys(raw_emotion, ["labels"], "emotion")
        raw_labels = raw_emotion["labels"]
        if not isinstance(raw_labels, list):
            raise ValueError("emotion.labels must be an array")
        selected: set[str] = set()
        for raw_label in raw_labels:
            if not isinstance(raw_label, str):
                raise ValueError("emotion.labels contains a non-string label")
            label = raw_label.strip().casefold()
            if label not in allowed:
                raise ValueError(
                    f"unknown label {raw_label!r} in emotion.labels; "
                    f"allowed={list(allowed)}"
                )
            selected.add(label)
        return {"labels": [label for label in allowed if label in selected]}
    if label_space == "crowd-envent":
        _require_exact_keys(raw_emotion, ["label", "intensity"], "emotion")
        raw_label = raw_emotion["label"]
        if not isinstance(raw_label, str):
            raise ValueError("emotion.label must be a string")
        label = raw_label.strip().casefold()
        if label in {"none", "no emotion"}:
            label = "no-emotion"
        if label not in allowed:
            raise ValueError(
                f"unknown label {raw_label!r} in emotion.label; "
                f"allowed={list(allowed)}"
            )
        return {
            "label": label,
            "intensity": _require_int(
                raw_emotion["intensity"], 1, 5, "emotion.intensity"
            ),
        }
    raise ValueError(f"Unsupported native emotion label space: {label_space!r}")


def _parse_ranked_multilabel_emotion(
    raw_emotion: Any, prompt_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    raw_emotion = _require_dict(raw_emotion, "emotion")
    _require_exact_keys(raw_emotion, ["labels", "intensity"], "emotion")
    raw_labels = raw_emotion["labels"]
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("emotion.labels must contain at least one label")
    allowed = native_emotion_labels(prompt_cfg)
    labels: List[str] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, str):
            raise ValueError("emotion.labels contains a non-string label")
        label = raw_label.strip().casefold()
        if label in {"none", "no emotion"}:
            label = "no-emotion"
        if label not in allowed:
            raise ValueError(
                f"unknown label {raw_label!r} in emotion.labels; "
                f"allowed={list(allowed)}"
            )
        if label not in labels:
            labels.append(label)
    if "no-emotion" in labels and len(labels) != 1:
        raise ValueError("no-emotion cannot be combined with another emotion label")
    return {
        "labels": labels,
        "intensity": _require_int(
            raw_emotion["intensity"], 1, 5, "emotion.intensity"
        ),
    }


def _parse_valenced_multilabel_emotion(
    raw_emotion: Any, prompt_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    raw_emotion = _require_dict(raw_emotion, "emotion")
    _require_exact_keys(
        raw_emotion,
        [
            "positive_intensity",
            "negative_intensity",
            "positive_labels",
            "negative_labels",
        ],
        "emotion",
    )
    positive_intensity = _require_int(
        raw_emotion["positive_intensity"],
        0,
        6,
        "emotion.positive_intensity",
    )
    negative_intensity = _require_int(
        raw_emotion["negative_intensity"],
        0,
        6,
        "emotion.negative_intensity",
    )
    positive_labels = _normalize_emotion_labels(
        raw_emotion["positive_labels"],
        "positive",
        "emotion.positive_labels",
        prompt_cfg,
    )
    negative_labels = _normalize_emotion_labels(
        raw_emotion["negative_labels"],
        "negative",
        "emotion.negative_labels",
        prompt_cfg,
    )
    if positive_intensity == 0 and positive_labels:
        raise ValueError("positive_labels must be empty when positive_intensity is 0")
    if negative_intensity == 0 and negative_labels:
        raise ValueError("negative_labels must be empty when negative_intensity is 0")
    return {
        "positive_intensity": positive_intensity,
        "negative_intensity": negative_intensity,
        "positive_labels": positive_labels,
        "negative_labels": negative_labels,
    }


def _normalize_emotion_labels(
    raw_labels: Any,
    valence: str,
    field: str,
    prompt_cfg: Dict[str, Any],
) -> List[str]:
    if emotion_label_space(prompt_cfg) == "carebench":
        return sft_common.canonicalize_emotion_labels(
            raw_labels,
            valence,
            field=field,
            reject_duplicates=False,
        )
    if not isinstance(raw_labels, list):
        raise ValueError(f"{field} must be an array")
    descriptions = emotion_label_descriptions(prompt_cfg, valence)
    aliases: Dict[str, str] = {}
    for label, description in descriptions.items():
        aliases[label.casefold()] = label
        for alias in description.split(","):
            aliases[alias.strip().casefold()] = label
    selected: List[str] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, str):
            raise ValueError(f"{field} contains a non-string label")
        normalized = raw_label.strip().casefold()
        if normalized not in aliases:
            raise ValueError(f"unknown label {raw_label!r} in {field}")
        label = aliases[normalized]
        if label not in selected:
            selected.append(label)
    return selected


def _parse_chain_emotion_payload(
    raw_text: str, prompt_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    label_space = emotion_label_space(prompt_cfg)
    if label_space == "carebench":
        if emotion_output_mode(prompt_cfg) != "multi-label":
            raise ValueError("CAREBench chain-emotion requires multi-label output")
        return parse_policy_output(raw_text, POLICY_APPRAISAL_DIMENSIONS)

    payload = _extract_json_object(raw_text)
    output_mode = emotion_output_mode(prompt_cfg)
    expected_root_keys = ["appraisal_reasoning", "emotion"]
    if label_space == "crowd-envent" and output_mode == "multi-label":
        expected_root_keys.append(EVALUATION_ONLY_EMOTION_RANKING_FIELD)
    _require_exact_keys(payload, expected_root_keys, "root")
    reasoning, _ = _parse_chain_core_appraisals(
        payload,
        prompt_cfg,
        use_policy_schema=True,
    )
    if output_mode == "multi-label":
        emotion = _parse_valenced_multilabel_emotion(
            payload.get("emotion"), prompt_cfg
        )
    else:
        emotion = _parse_native_emotion(payload.get("emotion"), prompt_cfg)
    result = {"appraisal_reasoning": reasoning, "emotion": emotion}
    if label_space == "crowd-envent" and output_mode == "multi-label":
        result[EVALUATION_ONLY_EMOTION_RANKING_FIELD] = (
            _parse_evaluation_only_emotion_ranking(
                payload.get(EVALUATION_ONLY_EMOTION_RANKING_FIELD),
                emotion,
                prompt_cfg,
            )
        )
    return result


def _parse_evaluation_only_emotion_ranking(
    raw_ranking: Any,
    emotion: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
) -> List[str]:
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError(
            f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} must be a non-empty array"
        )
    allowed = set(native_emotion_labels(prompt_cfg))
    ranking: List[str] = []
    for raw_label in raw_ranking:
        if not isinstance(raw_label, str):
            raise ValueError(
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} contains a non-string label"
            )
        label = raw_label.strip().casefold()
        if label not in allowed:
            raise ValueError(
                f"unknown label {raw_label!r} in "
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD}; "
                f"allowed={list(native_emotion_labels(prompt_cfg))}"
            )
        if label in ranking:
            raise ValueError(
                f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} contains duplicate "
                f"label {label!r}"
            )
        ranking.append(label)

    native_candidates = set(emotion["positive_labels"]) | set(
        emotion["negative_labels"]
    )
    expected_candidates = native_candidates or {"no-emotion"}
    if set(ranking) != expected_candidates:
        raise ValueError(
            f"{EVALUATION_ONLY_EMOTION_RANKING_FIELD} must contain exactly the "
            "labels selected in positive_labels and negative_labels, or only "
            "no-emotion when both lists are empty"
        )
    return ranking


def _require_exact_keys(
    value: Dict[str, Any], expected: List[str], field: str
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise ValueError(f"{field} keys mismatch: missing={missing}, extra={extra}")


def _parse_chain_core_appraisals(
    payload: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    *,
    use_policy_schema: bool,
) -> tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    reasoning_field = "appraisal_reasoning" if use_policy_schema else "core_appraisals"
    reasoning_dimensions = (
        POLICY_APPRAISAL_DIMENSIONS
        if use_policy_schema
        else CORE_APPRAISAL_DIMENSIONS
    )
    raw_core = _require_dict(payload.get(reasoning_field), reasoning_field)
    _require_exact_keys(
        raw_core, list(reasoning_dimensions), reasoning_field
    )
    questions = prompt_cfg.get("core-appraisals", {}).get("questions", {})
    if not isinstance(questions, dict) or any(
        dimension not in questions for dimension in CORE_APPRAISAL_DIMENSIONS
    ):
        raise ValueError(
            "Prompt config must define all [core-appraisals.questions] entries"
        )

    core: Dict[str, str] = {}
    for dimension in reasoning_dimensions:
        answer = raw_core[dimension]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"{reasoning_field}.{dimension} must be a non-empty string"
            )
        core[dimension] = answer.strip()
    derived: Dict[str, Dict[str, str]] = {}
    for carebench_dimension in CORE_APPRAISAL_DIMENSIONS:
        source_dimension = (
            CAREBENCH_TO_POLICY_CORE_DIMENSION[carebench_dimension]
            if use_policy_schema
            else carebench_dimension
        )
        derived[carebench_dimension] = {
            "question": str(questions[carebench_dimension]),
            "answer": core[source_dimension],
        }
    return core, derived


def parse_chain_task_output(
    raw_text: str,
    prompt_cfg: Dict[str, Any],
    chain_task: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse one single-call chain task and derive its evaluator input."""
    if chain_task not in CHAIN_TASK_TO_OUTPUT_TASK:
        raise ValueError(f"Unsupported chain task: {chain_task}")

    payload = _extract_json_object(raw_text)
    target_task = CHAIN_TASK_TO_OUTPUT_TASK[chain_task]
    target_field = target_task.replace("-", "_")
    use_policy_schema = chain_task in GRPO_SCHEMA_CHAIN_TASKS
    reasoning_field = "appraisal_reasoning" if use_policy_schema else "core_appraisals"
    expected_root_keys = [reasoning_field]
    if target_task != "core-appraisals":
        expected_root_keys.append(target_field)
    _require_exact_keys(payload, expected_root_keys, "root")

    core, derived_core = _parse_chain_core_appraisals(
        payload,
        prompt_cfg,
        use_policy_schema=use_policy_schema,
    )
    trace: Dict[str, Any] = {reasoning_field: core}
    if target_task == "core-appraisals":
        return trace, derived_core

    target_value = payload[target_field]
    if target_task == "appraisals":
        raw_appraisals = _require_dict(target_value, "appraisals")
        rating_keys = [canonical for _, canonical in sft_common.APPRAISAL_KEY_MAP]
        _require_exact_keys(raw_appraisals, rating_keys, "appraisals")
        score_to_label = _reverse_score_map(prompt_cfg, "appraisals", 1, 5)
        appraisals: Dict[str, Dict[str, Any]] = {}
        for key in rating_keys:
            raw_rating = _require_dict(raw_appraisals[key], f"appraisals.{key}")
            _require_exact_keys(raw_rating, ["score", "label"], f"appraisals.{key}")
            score = _require_int(raw_rating["score"], 1, 5, f"appraisals.{key}.score")
            expected_label = score_to_label[score]
            if raw_rating["label"] != expected_label:
                raise ValueError(
                    f"appraisals.{key}.label must be {expected_label!r} for score {score}"
                )
            appraisals[key] = {"score": score, "label": expected_label}
        trace[target_field] = appraisals
        return trace, appraisals

    if target_task in {"positive-level", "negative-level"}:
        raw_level = _require_dict(target_value, target_field)
        _require_exact_keys(raw_level, ["score", "label"], target_field)
        score = _require_int(raw_level["score"], 0, 6, f"{target_field}.score")
        score_to_label = _reverse_score_map(prompt_cfg, target_task, 0, 6)
        expected_label = score_to_label[score]
        if raw_level["label"] != expected_label:
            raise ValueError(
                f"{target_field}.label must be {expected_label!r} for score {score}"
            )
        level = {"score": score, "label": expected_label}
        trace[target_field] = level
        return trace, level

    valence = "positive" if target_task == "positive-labels" else "negative"
    canonical_labels = _normalize_emotion_labels(
        target_value, valence, target_field, prompt_cfg
    )
    descriptions = emotion_label_descriptions(prompt_cfg, valence)
    trace[target_field] = canonical_labels
    return trace, {"labels": [descriptions[label] for label in canonical_labels]}


def parse_chain_emotion_output(
    raw_text: str,
    prompt_cfg: Dict[str, Any],
    *,
    include_core_appraisals: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Parse CAREBench or dataset-native appraisal-to-emotion output."""
    policy_output = _parse_chain_emotion_payload(raw_text, prompt_cfg)
    emotion = policy_output["emotion"]
    task_outputs: Dict[str, Dict[str, Any]] = {}
    if emotion_label_space(prompt_cfg) == "carebench":
        positive_score_to_label = _reverse_score_map(
            prompt_cfg, "positive-level", 0, 6
        )
        negative_score_to_label = _reverse_score_map(
            prompt_cfg, "negative-level", 0, 6
        )
        positive_descriptions = emotion_label_descriptions(prompt_cfg, "positive")
        negative_descriptions = emotion_label_descriptions(prompt_cfg, "negative")
        task_outputs.update(
            {
                "positive-level": {
                    "score": emotion["positive_intensity"],
                    "label": positive_score_to_label[emotion["positive_intensity"]],
                },
                "negative-level": {
                    "score": emotion["negative_intensity"],
                    "label": negative_score_to_label[emotion["negative_intensity"]],
                },
                "positive-labels": {
                    "labels": [
                        positive_descriptions[label]
                        for label in emotion["positive_labels"]
                    ]
                },
                "negative-labels": {
                    "labels": [
                        negative_descriptions[label]
                        for label in emotion["negative_labels"]
                    ]
                },
            }
        )
    if include_core_appraisals:
        questions = prompt_cfg.get("core-appraisals", {}).get("questions", {})
        if not isinstance(questions, dict) or any(
            dimension not in questions for dimension in CORE_APPRAISAL_DIMENSIONS
        ):
            raise ValueError(
                "Prompt config must define all [core-appraisals.questions] entries"
            )
        reasoning = policy_output["appraisal_reasoning"]
        task_outputs["core-appraisals"] = {
            carebench_dimension: {
                "question": str(questions[carebench_dimension]),
                "answer": reasoning[
                    CAREBENCH_TO_POLICY_CORE_DIMENSION[carebench_dimension]
                ],
            }
            for carebench_dimension in CORE_APPRAISAL_DIMENSIONS
        }
    return policy_output, task_outputs


def parse_direct_task_output(
    raw_text: str, prompt_cfg: Dict[str, Any], task: str
) -> Dict[str, Any]:
    """Parse one crowd-enVent text-only prediction branch."""
    if task not in OFFICIAL_DIRECT_TASKS:
        raise ValueError(f"Unsupported direct task: {task}")
    if emotion_label_space(prompt_cfg) != "crowd-envent":
        raise ValueError(
            f"{OFFICIAL_DIRECT_TASK} is only available for crowd-envent"
        )

    payload = _extract_json_object(raw_text)
    if task == DIRECT_APPRAISALS_TASK:
        _require_exact_keys(payload, ["appraisal_ratings"], "root")
        raw_ratings = _require_dict(
            payload.get("appraisal_ratings"), "appraisal_ratings"
        )
        definitions = crowd_envent_appraisal_definitions(prompt_cfg)
        _require_exact_keys(
            raw_ratings, list(definitions), "appraisal_ratings"
        )
        return {
            "appraisal_ratings": {
                field: _require_int(
                    raw_ratings[field], 1, 5, f"appraisal_ratings.{field}"
                )
                for field in definitions
            }
        }

    _require_exact_keys(payload, ["emotion"], "root")
    if emotion_output_mode(prompt_cfg) == "multi-label":
        emotion = _parse_ranked_multilabel_emotion(
            payload.get("emotion"), prompt_cfg
        )
    else:
        emotion = _parse_native_emotion(payload.get("emotion"), prompt_cfg)
    return {"emotion": emotion}


def parse_full_chain_output(
    raw_text: str, prompt_cfg: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Validate a GRPO-keyed full chain and derive CAREBench task outputs."""
    payload = _extract_json_object(raw_text)

    raw_reasoning = _require_dict(
        payload.get("appraisal_reasoning"), "appraisal_reasoning"
    )
    _require_exact_keys(
        raw_reasoning,
        list(POLICY_APPRAISAL_DIMENSIONS),
        "appraisal_reasoning",
    )
    reasoning: Dict[str, str] = {}
    for dimension in POLICY_APPRAISAL_DIMENSIONS:
        answer = raw_reasoning.get(dimension)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"appraisal_reasoning.{dimension} must be a non-empty string"
            )
        reasoning[dimension] = answer.strip()

    raw_ratings = _require_dict(payload.get("appraisal_ratings"), "appraisal_ratings")
    expected_rating_keys = [canonical for _, canonical in sft_common.APPRAISAL_KEY_MAP]
    ratings: Dict[str, int] = {}
    for key in expected_rating_keys:
        ratings[key] = _require_int(
            raw_ratings.get(key), 1, 5, f"appraisal_ratings.{key}"
        )

    raw_emotion = _require_dict(payload.get("emotion"), "emotion")
    emotion = {
        "positive_intensity": _require_int(
            raw_emotion.get("positive_intensity"), 0, 6, "emotion.positive_intensity"
        ),
        "negative_intensity": _require_int(
            raw_emotion.get("negative_intensity"), 0, 6, "emotion.negative_intensity"
        ),
        "positive_labels": _normalize_emotion_labels(
            raw_emotion.get("positive_labels"),
            "positive",
            "emotion.positive_labels",
            prompt_cfg,
        ),
        "negative_labels": _normalize_emotion_labels(
            raw_emotion.get("negative_labels"),
            "negative",
            "emotion.negative_labels",
            prompt_cfg,
        ),
    }

    appraisal_score_to_label = _reverse_score_map(prompt_cfg, "appraisals", 1, 5)
    positive_score_to_label = _reverse_score_map(prompt_cfg, "positive-level", 0, 6)
    negative_score_to_label = _reverse_score_map(prompt_cfg, "negative-level", 0, 6)
    questions = prompt_cfg.get("core-appraisals", {}).get("questions", {})
    if not isinstance(questions, dict):
        raise ValueError("Missing [core-appraisals.questions] in prompt config")

    full_chain = {
        "appraisal_reasoning": reasoning,
        "appraisal_ratings": ratings,
        "emotion": emotion,
    }
    task_outputs: Dict[str, Dict[str, Any]] = {
        "appraisals": {
            key: {"score": score, "label": appraisal_score_to_label[score]}
            for key, score in ratings.items()
        },
        "core-appraisals": {
            carebench_dimension: {
                "question": str(questions[carebench_dimension]),
                "answer": reasoning[
                    CAREBENCH_TO_POLICY_CORE_DIMENSION[carebench_dimension]
                ],
            }
            for carebench_dimension in CORE_APPRAISAL_DIMENSIONS
            if carebench_dimension in questions
        },
        "positive-level": {
            "score": emotion["positive_intensity"],
            "label": positive_score_to_label[emotion["positive_intensity"]],
        },
        "negative-level": {
            "score": emotion["negative_intensity"],
            "label": negative_score_to_label[emotion["negative_intensity"]],
        },
        "positive-labels": {
            "labels": [
                emotion_label_descriptions(prompt_cfg, "positive")[label]
                for label in emotion["positive_labels"]
            ]
        },
        "negative-labels": {
            "labels": [
                emotion_label_descriptions(prompt_cfg, "negative")[label]
                for label in emotion["negative_labels"]
            ]
        },
    }
    if len(task_outputs["core-appraisals"]) != len(CORE_APPRAISAL_DIMENSIONS):
        missing = [
            dimension
            for dimension in CORE_APPRAISAL_DIMENSIONS
            if dimension not in questions
        ]
        raise ValueError(f"Missing core appraisal questions: {missing}")
    return full_chain, task_outputs


def _prompt_template(
    prompt_cfg: Dict[str, Any], task: str
) -> tuple[str, str]:
    templates = prompt_cfg.get("templates", {})
    task_template = templates.get(task)
    if not isinstance(task_template, dict):
        raise ValueError(f"Prompt template missing for task: {task}")
    system_prompt = str(task_template.get("system", "")).strip()
    user_template = str(task_template.get("user", "")).strip()
    if not user_template:
        raise ValueError(f"Prompt content missing for task: {task}")
    return system_prompt, user_template


def _prompt_components(
    prompt_cfg: Dict[str, Any]
) -> tuple[List[str], Dict[str, str], Dict[str, str]]:
    rating_keys = [canonical for _, canonical in sft_common.APPRAISAL_KEY_MAP]
    appraisal_statements = prompt_cfg.get("appraisals", {}).get(
        "dimension_to_statement", {}
    )
    if not isinstance(appraisal_statements, dict) or any(
        key not in appraisal_statements for key in rating_keys
    ):
        raise ValueError(
            "Prompt config must define all [appraisals.dimension_to_statement] entries"
        )
    core_questions = prompt_cfg.get("core-appraisals", {}).get("questions", {})
    if not isinstance(core_questions, dict) or any(
        dimension not in core_questions for dimension in CORE_APPRAISAL_DIMENSIONS
    ):
        raise ValueError(
            "Prompt config must define all [core-appraisals.questions] entries"
        )
    return rating_keys, appraisal_statements, core_questions


def _format_prompt(
    prompt_cfg: Dict[str, Any],
    task: str,
    user_template: str,
    scenario: str,
    rating_keys: List[str],
    appraisal_statements: Dict[str, str],
    core_questions: Dict[str, str],
    output_schema: Dict[str, Any],
    *,
    use_policy_schema: bool = False,
) -> str:
    if use_policy_schema:
        core_question_lines = "\n".join(
            f"- {policy_dimension}: "
            f"{core_questions[carebench_dimension]}"
            for policy_dimension, carebench_dimension in (
                POLICY_TO_CAREBENCH_CORE_DIMENSION.items()
            )
        )
    else:
        core_question_lines = "\n".join(
            f"- {dimension}: {core_questions[dimension]}"
            for dimension in CORE_APPRAISAL_DIMENSIONS
        )
    try:
        return user_template.format(
            scenario=scenario,
            rating_keys="\n".join(f"- {key}" for key in rating_keys),
            appraisal_statements="\n".join(
                f"- {key}: {appraisal_statements[key]}" for key in rating_keys
            ),
            core_appraisal_questions=core_question_lines,
            positive_labels=emotion_label_option_lines(prompt_cfg, "positive"),
            negative_labels=emotion_label_option_lines(prompt_cfg, "negative"),
            output_schema=json.dumps(output_schema, ensure_ascii=False, indent=2),
        )
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in [templates.{task}]: {exc}") from exc


def build_full_chain_prompts(
    prompt_cfg: Dict[str, Any], scenario: str
) -> tuple[str, str]:
    """Build the joint prompt from ``[templates.full-chain]`` in TOML."""
    system_prompt, user_template = _prompt_template(prompt_cfg, FULL_CHAIN_TASK)
    rating_keys, appraisal_statements, core_questions = _prompt_components(prompt_cfg)
    output_schema = {
        "appraisal_reasoning": {
            dimension: "concise first-person explanation"
            for dimension in POLICY_APPRAISAL_DIMENSIONS
        },
        "appraisal_ratings": {key: 1 for key in rating_keys},
        "emotion": {
            "positive_intensity": 0,
            "negative_intensity": 0,
            "positive_labels": [],
            "negative_labels": [],
        },
    }
    user_prompt = _format_prompt(
        prompt_cfg,
        FULL_CHAIN_TASK,
        user_template,
        scenario,
        rating_keys,
        appraisal_statements,
        core_questions,
        output_schema,
        use_policy_schema=True,
    )
    return system_prompt, user_prompt


def process_full_chain(
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
) -> tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """Generate all six tasks with one TOML-configured model call."""
    system_prompt, user_prompt = build_full_chain_prompts(prompt_cfg, scenario)
    response = client.chat(system_prompt, user_prompt)
    try:
        full_chain, task_outputs = parse_full_chain_output(response.content, prompt_cfg)
    except (KeyError, TypeError, ValueError) as exc:
        return None, None, [
            {"error": f"Full-chain parse error: {exc}", "raw_output": response.content}
        ]
    return full_chain, task_outputs, []


def build_chain_task_prompts(
    prompt_cfg: Dict[str, Any], scenario: str, chain_task: str
) -> tuple[str, str]:
    """Build a prompt that generates core appraisals and one target in one call."""
    if chain_task not in CHAIN_TASK_TO_OUTPUT_TASK:
        raise ValueError(f"Unsupported chain task: {chain_task}")
    system_prompt, user_template = _prompt_template(prompt_cfg, chain_task)
    rating_keys, appraisal_statements, core_questions = _prompt_components(prompt_cfg)
    use_policy_schema = chain_task in GRPO_SCHEMA_CHAIN_TASKS
    reasoning_dimensions = (
        POLICY_APPRAISAL_DIMENSIONS
        if use_policy_schema
        else CORE_APPRAISAL_DIMENSIONS
    )
    core_schema = {
        dimension: "concise first-person answer"
        for dimension in reasoning_dimensions
    }
    reasoning_field = "appraisal_reasoning" if use_policy_schema else "core_appraisals"
    output_schema: Dict[str, Any] = {reasoning_field: core_schema}
    target_task = CHAIN_TASK_TO_OUTPUT_TASK[chain_task]
    if target_task == "appraisals":
        score_to_label = _reverse_score_map(prompt_cfg, "appraisals", 1, 5)
        output_schema["appraisals"] = {
            key: {"score": 1, "label": score_to_label[1]}
            for key in rating_keys
        }
    elif target_task in {"positive-level", "negative-level"}:
        field = target_task.replace("-", "_")
        score_to_label = _reverse_score_map(prompt_cfg, target_task, 0, 6)
        output_schema[field] = {"score": 0, "label": score_to_label[0]}
    elif target_task in {"positive-labels", "negative-labels"}:
        output_schema[target_task.replace("-", "_")] = []

    user_prompt = _format_prompt(
        prompt_cfg,
        chain_task,
        user_template,
        scenario,
        rating_keys,
        appraisal_statements,
        core_questions,
        output_schema,
        use_policy_schema=use_policy_schema,
    )
    return system_prompt, user_prompt


def build_chain_emotion_prompts(
    prompt_cfg: Dict[str, Any], scenario: str
) -> tuple[str, str]:
    """Build five appraisals followed by the selected dataset's emotion output."""
    label_space = emotion_label_space(prompt_cfg)
    output_mode = emotion_output_mode(prompt_cfg)
    use_valenced_multilabel = label_space == "carebench" or output_mode == "multi-label"
    if label_space == "carebench":
        template_name = CHAIN_EMOTION_TASK
    elif use_valenced_multilabel and label_space == "crowd-envent":
        template_name = f"{CHAIN_EMOTION_TASK}-multilabel-{label_space}"
    elif use_valenced_multilabel:
        template_name = CHAIN_EMOTION_TASK
    else:
        template_name = f"{CHAIN_EMOTION_TASK}-{label_space}"
    system_template, user_template = _prompt_template(
        prompt_cfg, template_name
    )
    if label_space == "carebench":
        output_schema = policy_output_example(POLICY_APPRAISAL_DIMENSIONS)
    elif use_valenced_multilabel:
        output_schema = {
            "appraisal_reasoning": {
                dimension: "concise first-person explanation"
                for dimension in POLICY_APPRAISAL_DIMENSIONS
            },
            "emotion": {
                "positive_intensity": 0,
                "negative_intensity": 0,
                "positive_labels": [],
                "negative_labels": [],
            },
            EVALUATION_ONLY_EMOTION_RANKING_FIELD: ["no-emotion"],
        }
    else:
        output_schema = {
            "appraisal_reasoning": {
                dimension: "concise first-person explanation"
                for dimension in POLICY_APPRAISAL_DIMENSIONS
            },
            "emotion": native_emotion_output_schema(prompt_cfg),
        }
    try:
        system_prompt = system_template.format(
            appraisal_dimension_names=", ".join(POLICY_APPRAISAL_DIMENSIONS)
        )
        user_prompt = user_template.format(
            scenario=scenario,
            appraisal_definition_lines="\n".join(
                f"- {dimension}: {APPRAISAL_DEFINITIONS[dimension]}"
                for dimension in POLICY_APPRAISAL_DIMENSIONS
            ),
            positive_labels=emotion_label_option_lines(prompt_cfg, "positive"),
            negative_labels=emotion_label_option_lines(prompt_cfg, "negative"),
            emotion_labels="\n".join(
                f"- {label}" for label in NATIVE_EMOTION_LABELS.get(label_space, ())
            ),
            output_schema=json.dumps(output_schema, ensure_ascii=False, indent=2),
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder in [templates.{CHAIN_EMOTION_TASK}]: {exc}"
        ) from exc
    return system_prompt, user_prompt


def build_direct_task_prompts(
    prompt_cfg: Dict[str, Any], scenario: str, task: str
) -> tuple[str, str]:
    """Build an official-style crowd-enVent T->A or T->E prompt."""
    if task not in OFFICIAL_DIRECT_TASKS:
        raise ValueError(f"Unsupported direct task: {task}")
    label_space = emotion_label_space(prompt_cfg)
    if label_space != "crowd-envent":
        raise ValueError(
            f"{OFFICIAL_DIRECT_TASK} is only available for crowd-envent"
        )

    output_mode = emotion_output_mode(prompt_cfg)
    template_name = (
        f"{task}-multilabel-{label_space}"
        if task == DIRECT_EMOTION_TASK and output_mode == "multi-label"
        else f"{task}-{label_space}"
    )
    system_prompt, user_template = _prompt_template(prompt_cfg, template_name)
    definitions = crowd_envent_appraisal_definitions(prompt_cfg)
    if task == DIRECT_APPRAISALS_TASK:
        output_schema: Dict[str, Any] = {
            "appraisal_ratings": {field: 3 for field in definitions}
        }
    elif output_mode == "multi-label":
        output_schema = {"emotion": {"labels": ["joy"], "intensity": 3}}
    else:
        output_schema = {"emotion": native_emotion_output_schema(prompt_cfg)}
    try:
        user_prompt = user_template.format(
            scenario=scenario,
            appraisal_rating_lines="\n".join(
                f"- {field}: {description}"
                for field, description in definitions.items()
            ),
            emotion_labels="\n".join(
                f"- {label}" for label in native_emotion_labels(prompt_cfg)
            ),
            output_schema=json.dumps(output_schema, ensure_ascii=False, indent=2),
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown placeholder in [templates.{template_name}]: {exc}"
        ) from exc
    return system_prompt, user_prompt


def process_direct_task(
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    task: str,
) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run one crowd-enVent text-only prediction branch."""
    system_prompt, user_prompt = build_direct_task_prompts(
        prompt_cfg, scenario, task
    )
    response = client.chat(system_prompt, user_prompt)
    try:
        output = parse_direct_task_output(response.content, prompt_cfg, task)
    except (KeyError, TypeError, ValueError) as exc:
        return None, [
            {"error": f"{task} parse error: {exc}", "raw_output": response.content}
        ]
    return output, []


def process_chain_emotion(
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    *,
    include_core_appraisals: bool = False,
) -> tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    """Generate five policy appraisals followed by one emotion output."""
    system_prompt, user_prompt = build_chain_emotion_prompts(prompt_cfg, scenario)
    response = client.chat(system_prompt, user_prompt)
    try:
        policy_output, task_outputs = parse_chain_emotion_output(
            response.content,
            prompt_cfg,
            include_core_appraisals=include_core_appraisals,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, None, [
            {
                "error": f"{CHAIN_EMOTION_TASK} parse error: {exc}",
                "raw_output": response.content,
            }
        ]
    return policy_output, task_outputs, []


def process_chain_task(
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    chain_task: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run exactly one generation for a core-appraisal-conditioned task."""
    system_prompt, user_prompt = build_chain_task_prompts(
        prompt_cfg, scenario, chain_task
    )
    response = client.chat(system_prompt, user_prompt)
    try:
        trace, derived_output = parse_chain_task_output(
            response.content, prompt_cfg, chain_task
        )
    except (KeyError, TypeError, ValueError) as exc:
        return None, None, [
            {"error": f"{chain_task} parse error: {exc}", "raw_output": response.content}
        ]
    return trace, derived_output, []


def resolve_selected_tasks(
    task_value: str,
    chain_all_emotion_mode: str = "separate",
    chain_core_appraisals_source: str = "separate",
) -> List[str]:
    """Expand aggregate task names while retaining the original task order."""
    normalized = task_value.strip().lower()
    normalized_emotion_mode = chain_all_emotion_mode.strip().lower()
    normalized_core_source = chain_core_appraisals_source.strip().lower()
    if normalized_emotion_mode not in CHAIN_ALL_EMOTION_MODES:
        raise ValueError(
            "Invalid chain-all emotion mode: {}. Valid values: {}".format(
                chain_all_emotion_mode,
                ", ".join(CHAIN_ALL_EMOTION_MODES),
            )
        )
    if normalized_core_source not in CHAIN_CORE_APPRAISAL_SOURCES:
        raise ValueError(
            "Invalid chain core-appraisals source: {}. Valid values: {}".format(
                chain_core_appraisals_source,
                ", ".join(CHAIN_CORE_APPRAISAL_SOURCES),
            )
        )
    if normalized == "all":
        return list(baseline.TASKS)
    if normalized == OFFICIAL_DIRECT_TASK:
        if normalized_core_source == CHAIN_EMOTION_TASK:
            raise ValueError(
                "--chain_core_appraisals_source chain-emotion is not used by "
                f"--task {OFFICIAL_DIRECT_TASK}"
            )
        return list(OFFICIAL_DIRECT_TASKS)
    if normalized == CHAIN_ALL_TASK:
        if normalized_emotion_mode == CHAIN_EMOTION_TASK:
            selected = list(CHAIN_ALL_WITH_JOINT_EMOTION_TASKS)
            if normalized_core_source == CHAIN_EMOTION_TASK:
                selected.remove("chain-core-appraisals")
            return selected
        if normalized_core_source == CHAIN_EMOTION_TASK:
            raise ValueError(
                "--chain_core_appraisals_source chain-emotion requires "
                "--chain_all_emotion_mode chain-emotion"
            )
        return list(CHAIN_TASKS)
    if normalized == FULL_CHAIN_TASK:
        return [FULL_CHAIN_TASK]
    if normalized == CHAIN_EMOTION_TASK:
        return [CHAIN_EMOTION_TASK]
    if normalized_core_source == CHAIN_EMOTION_TASK:
        raise ValueError(
            "--chain_core_appraisals_source chain-emotion is only valid with "
            "--task chain-all or --task chain-emotion"
        )
    if (
        normalized in baseline.TASKS
        or normalized in CHAIN_TASK_TO_OUTPUT_TASK
        or normalized in OFFICIAL_DIRECT_TASKS
    ):
        return [normalized]
    valid = [
        "all",
        CHAIN_ALL_TASK,
        CHAIN_EMOTION_TASK,
        OFFICIAL_DIRECT_TASK,
        FULL_CHAIN_TASK,
        *baseline.TASKS,
        *CHAIN_TASKS,
        *OFFICIAL_DIRECT_TASKS,
    ]
    raise ValueError(
        f"Invalid --task: {task_value}. Valid values: {', '.join(valid)}"
    )
