#!/usr/bin/env python3
"""Generate role-conditioned CHIARO predictions with direct or appraisal chains."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable


UTILS_DIR = Path(__file__).resolve().parent
REPO_ROOT = UTILS_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "FirstPersonMethod" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from baseline_transformers_backend import TransformersClient
from chiaro_common import (
    NEGATIVE_EMOTIONS,
    POSITIVE_EMOTIONS,
    extract_json_object,
    letter_for_emotion,
    load_chiaro_items,
    parse_appraisal_reasoning,
    parse_free_emotion,
    read_prediction_records,
    select_free_emotion,
    select_valence_constrained_emotion,
    write_prediction_records,
)


DEFAULT_EVAL_FILE = REPO_ROOT / "Chiaro-main" / "data" / "chiaro_test.json"
DEFAULT_PROMPT_FILE = UTILS_DIR / "prompts" / "chiaro_prompt.toml"
PROMPT_VERSION = "chiaro-caeu-0.3"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib
    with path.open("rb") as handle:
        return tomllib.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local model on CHIARO using a direct joint prompt or "
            "two role-conditioned appraisal chains per scene"
        )
    )
    parser.add_argument(
        "--generation_schema",
        choices=["direct", "chain"],
        default="chain",
        help=(
            "direct uses one joint two-agent call without appraisal; chain uses "
            "one five-dimensional appraisal-to-emotion call per agent."
        ),
    )
    parser.add_argument(
        "--emotion_mode",
        choices=["valence-constrained", "valence-free"],
        default="valence-constrained",
        help=(
            "For direct, valence-constrained exposes each role's five official "
            "options. For chain, both modes generate all 10 labels in CAREBench "
            "format; constrained selects from the official target valence, while "
            "free selects the higher-intensity valence."
        ),
    )
    parser.add_argument("--eval_file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        default="all",
        help="Optional split filter; chiaro_test.json already contains only test scenes.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--predictions_file", type=Path, required=True)
    parser.add_argument("--invalid_file", type=Path, default=None)
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--overwrite_predictions", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--base_model",
        "--base_model_name_or_path",
        dest="base_model",
        type=str,
        default="",
    )
    parser.add_argument("--model_is_adapter", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help=(
            "Defaults to 40 for direct constrained, 512 for direct free, and "
            "1536 for chain."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--enable_thinking", type=str2bool, default=False)
    parser.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--device_map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
    )
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load_in_4bit", action="store_true")
    quantization.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=str2bool, default=True)
    return parser


def resolve_max_tokens(args: argparse.Namespace) -> int:
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            raise ValueError("--max_tokens must be positive")
        return args.max_tokens
    if args.generation_schema == "chain":
        return 1536
    if args.emotion_mode == "valence-free":
        return 512
    return 40


def create_client(args: argparse.Namespace) -> TransformersClient:
    return TransformersClient(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        max_tokens=resolve_max_tokens(args),
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        enable_thinking=args.enable_thinking,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        revision=args.revision,
        seed=args.seed,
        base_model_name_or_path=args.base_model,
        model_is_adapter=args.model_is_adapter,
    )


def _template(prompt_cfg: dict[str, Any], name: str) -> tuple[str, str]:
    section = prompt_cfg.get("templates", {}).get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Prompt file is missing [templates.{name}]")
    system = section.get("system")
    user = section.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError(f"Prompt template {name} needs system and user strings")
    return system, user


def _format_options(options: dict[str, str]) -> str:
    return "\n".join(f"{letter}. {label}" for letter, label in sorted(options.items()))


def _appraisal_schema(emotion_schema: dict[str, Any]) -> str:
    return json.dumps(
        {
            "appraisal_reasoning": {
                "relevance": "first-person appraisal",
                "epistemic": "first-person appraisal",
                "goal_congruence": "first-person appraisal",
                "agency_accountability": "first-person appraisal",
                "control_coping_potential": "first-person appraisal",
            },
            "emotion": emotion_schema,
        },
        ensure_ascii=False,
        indent=2,
    )


def _free_emotion_schema() -> dict[str, Any]:
    return {
        "positive_intensity": 0,
        "negative_intensity": 0,
        "positive_labels": ["one or more positive CHIARO labels"],
        "negative_labels": ["one or more negative CHIARO labels"],
    }


def _parse_joint_letters(raw: str, item: dict[str, Any]) -> dict[str, Any]:
    match_a = re.search(r"AGENT\s*A[^A-Za-z]*([A-E])\b", raw, re.IGNORECASE)
    match_b = re.search(r"AGENT\s*B[^A-Za-z]*([A-E])\b", raw, re.IGNORECASE)
    if match_a is None or match_b is None:
        raise ValueError("Expected both 'AGENT A: <letter>' and 'AGENT B: <letter>'")
    letter_a = match_a.group(1).upper()
    letter_b = match_b.group(1).upper()
    if letter_a not in item["options_a"] or letter_b not in item["options_b"]:
        raise ValueError("Predicted letter is not present in the corresponding options")
    return {
        "letter_A": letter_a,
        "emotion_A": item["options_a"][letter_a],
        "letter_B": letter_b,
        "emotion_B": item["options_b"][letter_b],
    }


def _parse_direct_free(raw: str) -> dict[str, Any]:
    payload = extract_json_object(raw)
    if set(payload) != {"agent_a", "agent_b"}:
        raise ValueError("Direct free output must contain exactly agent_a and agent_b")
    parsed: dict[str, Any] = {}
    for source_key, slot in (("agent_a", "A"), ("agent_b", "B")):
        agent = payload[source_key]
        if not isinstance(agent, dict) or set(agent) != {"emotion"}:
            raise ValueError(f"{source_key} must contain exactly an emotion object")
        emotion = parse_free_emotion(agent["emotion"])
        # Selection is part of parse validity so an ambiguous intensity tie is
        # retried instead of escaping later and terminating the entire run.
        select_free_emotion(emotion)
        parsed[slot] = {"emotion": emotion}
    return parsed


def _parse_chain(
    raw: str,
    *,
    emotion_mode: str,
    allowed_emotions: tuple[str, ...],
) -> dict[str, Any]:
    payload = extract_json_object(raw)
    if set(payload) != {"appraisal_reasoning", "emotion"}:
        raise ValueError(
            "Chain output must contain exactly appraisal_reasoning and emotion"
        )
    appraisal = parse_appraisal_reasoning(payload["appraisal_reasoning"])
    emotion = parse_free_emotion(payload["emotion"])
    if emotion_mode == "valence-constrained":
        select_valence_constrained_emotion(emotion, allowed_emotions)
    else:
        select_free_emotion(emotion)
    return {"appraisal_reasoning": appraisal, "emotion": emotion}


class CallFailure(RuntimeError):
    def __init__(self, attempts: list[dict[str, Any]]) -> None:
        self.attempts = attempts
        last_error = attempts[-1]["error"] if attempts else "unknown generation error"
        super().__init__(last_error)


def _call_with_retries(
    client: TransformersClient,
    system_prompt: str,
    user_prompt: str,
    parser: Callable[[str], dict[str, Any]],
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 1):
        raw_text: str | None = None
        response_metadata: dict[str, Any] | None = None
        try:
            response = client.chat(system_prompt, user_prompt)
            raw_text = response.content
            response_metadata = response.raw
            parsed = parser(raw_text)
            return parsed, {
                "raw_output": raw_text,
                "response_metadata": response_metadata,
                "attempt": attempt,
            }
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output": raw_text,
                    "response_metadata": response_metadata,
                }
            )
    raise CallFailure(attempts)


def _base_record(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mcq_id_a": item["id"],
        "sentence": item["sentence"],
        "agent_a_role": item["agent_a_role"],
        "agent_b_role": item["agent_b_role"],
        "options_a": item["options_a"],
        "options_b": item["options_b"],
        "gold_A": item.get("generation_emotion_a"),
        "gold_B": item.get("generation_emotion_b"),
        "model": args.model,
        "version": item.get("version"),
        "split": item.get("split"),
        "generation_schema": args.generation_schema,
        "emotion_mode": args.emotion_mode,
        "prompt_version": PROMPT_VERSION,
    }


def _generate_direct(
    item: dict[str, Any],
    args: argparse.Namespace,
    prompt_cfg: dict[str, Any],
    client: TransformersClient,
) -> dict[str, Any]:
    record = _base_record(item, args)
    if args.emotion_mode == "valence-constrained":
        system, user_template = _template(prompt_cfg, "direct_constrained")
        user = user_template.format(
            sentence=item["sentence"],
            agent_a_role=item["agent_a_role"],
            agent_b_role=item["agent_b_role"],
            options_a=_format_options(item["options_a"]),
            options_b=_format_options(item["options_b"]),
            letters_a=", ".join(sorted(item["options_a"])),
            letters_b=", ".join(sorted(item["options_b"])),
        )
        parsed, trace = _call_with_retries(
            client,
            system,
            user,
            lambda raw: _parse_joint_letters(raw, item),
            args.max_retries,
        )
        record.update(
            {
                "llm_letter_A": parsed["letter_A"],
                "llm_emotion_A": parsed["emotion_A"],
                "llm_letter_B": parsed["letter_B"],
                "llm_emotion_B": parsed["emotion_B"],
                "raw": trace["raw_output"],
                "generation_trace": trace,
            }
        )
        return record

    system, user_template = _template(prompt_cfg, "direct_free")
    output_schema = json.dumps(
        {
            "agent_a": {"emotion": _free_emotion_schema()},
            "agent_b": {"emotion": _free_emotion_schema()},
        },
        ensure_ascii=False,
        indent=2,
    )
    user = user_template.format(
        sentence=item["sentence"],
        agent_a_role=item["agent_a_role"],
        agent_b_role=item["agent_b_role"],
        positive_labels=", ".join(POSITIVE_EMOTIONS),
        negative_labels=", ".join(NEGATIVE_EMOTIONS),
        output_schema=output_schema,
    )
    parsed, trace = _call_with_retries(
        client, system, user, _parse_direct_free, args.max_retries
    )
    emotion_a = parsed["A"]["emotion"]
    emotion_b = parsed["B"]["emotion"]
    label_a, selection_a = select_free_emotion(emotion_a)
    label_b, selection_b = select_free_emotion(emotion_b)
    record.update(
        {
            "llm_letter_A": letter_for_emotion(
                item["options_a"], label_a
            ),
            "llm_emotion_A": label_a,
            "llm_letter_B": letter_for_emotion(
                item["options_b"], label_b
            ),
            "llm_emotion_B": label_b,
            "emotion_selection_A": selection_a,
            "emotion_selection_B": selection_b,
            "agent_a_output": parsed["A"],
            "agent_b_output": parsed["B"],
            "raw": trace["raw_output"],
            "generation_trace": trace,
        }
    )
    return record


def _chain_prompt(
    item: dict[str, Any],
    slot: str,
    args: argparse.Namespace,
    prompt_cfg: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    lower = slot.lower()
    allowed = tuple(item[f"options_{lower}"].values())
    appraisal_cfg = prompt_cfg["appraisal"]
    if args.emotion_mode == "valence-constrained":
        system, user_template = _template(prompt_cfg, "chain_constrained")
        emotion_schema: dict[str, Any] = _free_emotion_schema()
    else:
        system, user_template = _template(prompt_cfg, "chain_free")
        emotion_schema = _free_emotion_schema()
    common = {
        "appraisal_dimension_names": appraisal_cfg["dimension_names"],
        "sentence": item["sentence"],
        "target_role": item[f"agent_{lower}_role"],
        "appraisal_definition_lines": appraisal_cfg["definition_lines"].strip(),
        "allowed_emotions": ", ".join(allowed),
        "positive_labels": ", ".join(POSITIVE_EMOTIONS),
        "negative_labels": ", ".join(NEGATIVE_EMOTIONS),
        "output_schema": _appraisal_schema(emotion_schema),
    }
    return system.format(**common), user_template.format(**common), allowed


def _generate_chain(
    item: dict[str, Any],
    args: argparse.Namespace,
    prompt_cfg: dict[str, Any],
    client: TransformersClient,
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for slot in ("A", "B"):
        system, user, allowed = _chain_prompt(item, slot, args, prompt_cfg)
        parsed, trace = _call_with_retries(
            client,
            system,
            user,
            lambda raw, allowed=allowed: _parse_chain(
                raw,
                emotion_mode=args.emotion_mode,
                allowed_emotions=allowed,
            ),
            args.max_retries,
        )
        outputs[slot] = parsed
        traces[slot] = trace

    emotion_a = outputs["A"]["emotion"]
    emotion_b = outputs["B"]["emotion"]
    if args.emotion_mode == "valence-constrained":
        label_a, selection_a = select_valence_constrained_emotion(
            emotion_a, tuple(item["options_a"].values())
        )
        label_b, selection_b = select_valence_constrained_emotion(
            emotion_b, tuple(item["options_b"].values())
        )
    else:
        label_a, selection_a = select_free_emotion(emotion_a)
        label_b, selection_b = select_free_emotion(emotion_b)
    record = _base_record(item, args)
    record.update(
        {
            "llm_letter_A": letter_for_emotion(item["options_a"], label_a),
            "llm_emotion_A": label_a,
            "llm_letter_B": letter_for_emotion(item["options_b"], label_b),
            "llm_emotion_B": label_b,
            "agent_a_output": outputs["A"],
            "agent_b_output": outputs["B"],
            "raw": {
                "agent_a": traces["A"]["raw_output"],
                "agent_b": traces["B"]["raw_output"],
            },
            "generation_trace": traces,
        }
    )
    record["emotion_selection_A"] = selection_a
    record["emotion_selection_B"] = selection_b
    return record


def _load_existing(
    path: Path,
    selected_ids: set[str],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    if args.overwrite_predictions:
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for record in read_prediction_records(path):
        item_id = str(record.get("mcq_id_a", ""))
        if item_id not in selected_ids:
            continue
        if record.get("generation_schema") != args.generation_schema:
            raise ValueError(
                f"Existing prediction {item_id} uses another generation_schema; "
                "pass --overwrite_predictions or choose another output file"
            )
        if record.get("emotion_mode") != args.emotion_mode:
            raise ValueError(
                f"Existing prediction {item_id} uses another emotion_mode; "
                "pass --overwrite_predictions or choose another output file"
            )
        if record.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"Existing prediction {item_id} uses another prompt_version; "
                "pass --overwrite_predictions or choose another output file"
            )
        existing[item_id] = record
    return existing


def _write_invalid(path: Path, invalid_by_id: dict[str, dict[str, Any]]) -> None:
    rows = sorted(invalid_by_id.values(), key=lambda row: str(row.get("mcq_id_a", "")))
    write_prediction_records(path, rows)


def execute(args: argparse.Namespace) -> int:
    if args.max_retries <= 0:
        raise ValueError("--max_retries must be positive")
    if args.save_every <= 0:
        raise ValueError("--save_every must be positive")
    if not args.eval_file.is_file():
        raise FileNotFoundError(f"CHIARO evaluation file not found: {args.eval_file}")
    if not args.prompt_path.is_file():
        raise FileNotFoundError(f"CHIARO prompt file not found: {args.prompt_path}")

    items = load_chiaro_items(
        args.eval_file, split=args.split, max_samples=args.max_samples
    )
    if not items:
        print("No CHIARO scenes selected.")
        return 0
    selected_ids = {str(item["id"]) for item in items}
    prediction_by_id = _load_existing(args.predictions_file, selected_ids, args)
    invalid_file = args.invalid_file or args.predictions_file.with_name(
        args.predictions_file.stem + ".invalid.jsonl"
    )
    invalid_by_id = (
        {}
        if args.overwrite_predictions
        else {
            str(row.get("mcq_id_a", "")): row
            for row in read_prediction_records(invalid_file)
            if str(row.get("mcq_id_a", "")) in selected_ids
        }
    )
    pending = [item for item in items if str(item["id"]) not in prediction_by_id]
    calls_per_scene = 1 if args.generation_schema == "direct" else 2
    print(
        f"[generate] scenes={len(items)} cached={len(items) - len(pending)} "
        f"pending={len(pending)} schema={args.generation_schema} "
        f"emotion_mode={args.emotion_mode} calls_per_scene={calls_per_scene} "
        f"max_tokens={resolve_max_tokens(args)}"
    )
    if not pending:
        return 0

    prompt_cfg = load_toml(args.prompt_path)
    client = create_client(args)
    order = {str(item["id"]): index for index, item in enumerate(items)}

    try:
        from tqdm import tqdm

        iterator = tqdm(pending, desc="Processing CHIARO scenes", unit="scene")
    except ImportError:
        iterator = pending

    valid_new = 0
    for number, item in enumerate(iterator, 1):
        item_id = str(item["id"])
        try:
            if args.generation_schema == "direct":
                record = _generate_direct(item, args, prompt_cfg, client)
            else:
                record = _generate_chain(item, args, prompt_cfg, client)
            prediction_by_id[item_id] = record
            invalid_by_id.pop(item_id, None)
            valid_new += 1
            if args.verbose:
                print(
                    f"[valid] {item_id}: A={record['llm_emotion_A']} "
                    f"B={record['llm_emotion_B']}"
                )
        except CallFailure as exc:
            invalid_by_id[item_id] = {
                "mcq_id_a": item_id,
                "sentence": item["sentence"],
                "generation_schema": args.generation_schema,
                "emotion_mode": args.emotion_mode,
                "attempts": exc.attempts,
            }
            if args.verbose:
                print(f"[invalid] {item_id}: {exc}")

        if number % args.save_every == 0 or number == len(pending):
            records = sorted(
                prediction_by_id.values(),
                key=lambda row: order.get(str(row.get("mcq_id_a", "")), len(order)),
            )
            write_prediction_records(args.predictions_file, records)
            _write_invalid(invalid_file, invalid_by_id)

    print(
        f"[summary] selected={len(items)} valid={len(prediction_by_id)} "
        f"new_valid={valid_new} invalid={len(invalid_by_id)} "
        f"predictions={args.predictions_file}"
    )
    return 0


def main() -> int:
    return execute(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
