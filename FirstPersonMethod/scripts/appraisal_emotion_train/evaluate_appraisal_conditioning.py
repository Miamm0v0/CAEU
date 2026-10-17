#!/usr/bin/env python3
"""Compare Warmup v3 and AE v1 under two appraisal conditions.

Each model is evaluated on the same CAREBench records with deterministic
decoding:

1. situation + human appraisal -> emotion
2. situation + model-generated appraisal -> emotion

The model-generated condition is the normal autoregressive chain. Its emotion
is already generated after, and therefore conditioned on, the appraisal in the
same response. The human condition replaces that prefix with the human
appraisal and asks the model to continue only the emotion portion.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "FirstPersonMethod" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import baseline_transformers_chain as chain  # noqa: E402
import baseline_vllm as baseline  # noqa: E402
from baseline_transformers_backend import TransformersClient  # noqa: E402
from appraisal_emotion_train.ae_format import (  # noqa: E402
    POLICY_DIMENSIONS,
    build_assistant_prefix,
    format_ae_record,
)
from train_grpo.lightweight_metrics import carebench_metric_summary  # noqa: E402


MODEL_NAMES = ("warmup_v3", "ae_v1")
CONDITIONS = ("human_appraisal", "model_generated_appraisal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed Warmup-v3 and AE-v1 models with human versus "
            "model-generated appraisal prefixes."
        )
    )
    parser.add_argument(
        "--eval_file",
        type=Path,
        required=True,
        help="Prepared AE JSONL containing situation, human appraisal, and emotion.",
    )
    parser.add_argument("--warmup_model", required=True)
    parser.add_argument("--ae_model", required=True)
    parser.add_argument(
        "--base_model",
        default="",
        help="Optional shared base-model override for both PEFT adapters.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional shared tokenizer path; otherwise inferred per model.",
    )
    parser.add_argument(
        "--models_are_adapters",
        action="store_true",
        help="Force both model paths to be loaded as PEFT adapters.",
    )
    parser.add_argument(
        "--prompt_path",
        type=Path,
        default=Path("FirstPersonMethod/scripts/prompts/baseline_prompt.toml"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--annotator_type",
        choices=("first_person", "all"),
        default="first_person",
        help="By default evaluate only the first-person human annotation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum records after filtering; 0 evaluates all records.",
    )
    parser.add_argument("--full_chain_max_new_tokens", type=int, default=1536)
    parser.add_argument("--emotion_max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device_map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default=None,
    )
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load_in_4bit", action="store_true")
    quantization.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--revision", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.is_file():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.is_file() else path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no JSONL records found")
    return records


def _select_records(
    records: Iterable[Mapping[str, Any]],
    annotator_type: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    selected = [
        dict(record)
        for record in records
        if annotator_type == "all"
        or record.get("annotator_type") == annotator_type
    ]
    selected.sort(key=lambda row: str(row.get("sample_id", "")))
    if max_samples:
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(
            f"No records remain after --annotator_type {annotator_type}"
        )
    return selected


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _build_client(
    args: argparse.Namespace,
    model_path: str,
) -> TransformersClient:
    return TransformersClient(
        model_name_or_path=model_path,
        tokenizer_name_or_path=args.tokenizer,
        max_tokens=args.full_chain_max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        do_sample=False,
        enable_thinking=False,
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
        model_is_adapter=args.models_are_adapters,
    )


def _release_client(client: TransformersClient) -> None:
    torch = client.torch
    del client.model
    del client
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _base_row(
    *,
    model_name: str,
    model_path: str,
    condition: str,
    record: Mapping[str, Any],
    human_appraisal: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "model_path": model_path,
        "condition": condition,
        "sample_id": record["sample_id"],
        "source_sample_id": record["source_sample_id"],
        "annotator_id": record["annotator_id"],
        "annotator_type": record["annotator_type"],
        "situation": record["situation"],
        "human_appraisal": dict(human_appraisal),
        "gold_emotion": dict(record["emotion"]),
    }


def _run_human_condition(
    *,
    client: TransformersClient,
    prompt_cfg: Mapping[str, Any],
    model_name: str,
    model_path: str,
    record: Mapping[str, Any],
    human_appraisal: Mapping[str, str],
    assistant_prefix: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    row = _base_row(
        model_name=model_name,
        model_path=model_path,
        condition="human_appraisal",
        record=record,
        human_appraisal=human_appraisal,
    )
    try:
        system_prompt, user_prompt = chain.build_chain_emotion_prompts(
            dict(prompt_cfg), str(record["situation"])
        )
        client.max_tokens = max_new_tokens
        response = client.continue_chat(
            system_prompt,
            user_prompt,
            assistant_prefix,
        )
        full_output = assistant_prefix + response.content
        candidate, _ = chain.parse_chain_emotion_output(
            full_output,
            dict(prompt_cfg),
        )
        row.update(
            {
                "valid": True,
                "conditioning_appraisal": dict(human_appraisal),
                "predicted_emotion": candidate["emotion"],
                "assistant_prefix": assistant_prefix,
                "continuation": response.content,
                "full_output": full_output,
                "generation": response.raw,
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain failures in coverage.
        row.update(
            {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def _run_model_condition(
    *,
    client: TransformersClient,
    prompt_cfg: Mapping[str, Any],
    model_name: str,
    model_path: str,
    record: Mapping[str, Any],
    human_appraisal: Mapping[str, str],
    max_new_tokens: int,
) -> dict[str, Any]:
    row = _base_row(
        model_name=model_name,
        model_path=model_path,
        condition="model_generated_appraisal",
        record=record,
        human_appraisal=human_appraisal,
    )
    try:
        system_prompt, user_prompt = chain.build_chain_emotion_prompts(
            dict(prompt_cfg), str(record["situation"])
        )
        client.max_tokens = max_new_tokens
        response = client.chat(system_prompt, user_prompt)
        candidate, _ = chain.parse_chain_emotion_output(
            response.content,
            dict(prompt_cfg),
        )
        row.update(
            {
                "valid": True,
                "conditioning_appraisal": candidate["appraisal_reasoning"],
                "predicted_emotion": candidate["emotion"],
                "full_output": response.content,
                "generation": response.raw,
            }
        )
    except Exception as exc:  # noqa: BLE001 - retain failures in coverage.
        row.update(
            {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def _emotion_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        candidate = None
        if row.get("valid"):
            candidate = {
                "appraisal_reasoning": row.get("conditioning_appraisal"),
                "emotion": row.get("predicted_emotion"),
            }
        metric_rows.append(
            {
                "valid": bool(row.get("valid")),
                "candidate": candidate,
                "reference": {"gold_emotion": row.get("gold_emotion")},
            }
        )
    metrics = carebench_metric_summary(metric_rows)
    return {
        name: value
        for name, value in metrics.items()
        if "appraisal" not in name
    }


def _metric_delta(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> dict[str, float]:
    return {
        name: float(left[name] - right[name])
        for name in sorted(set(left) & set(right))
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    eval_file = _resolve_path(args.eval_file)
    prompt_path = _resolve_path(args.prompt_path)
    if not eval_file.is_file():
        raise FileNotFoundError(eval_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(prompt_path)
    if args.max_samples < 0:
        raise ValueError("--max_samples cannot be negative")
    if args.full_chain_max_new_tokens <= 0 or args.emotion_max_new_tokens <= 0:
        raise ValueError("generation token limits must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and (predictions_path.exists() or summary_path.exists()):
        raise FileExistsError(
            f"Output exists in {args.output_dir}; pass --overwrite to replace it"
        )

    records = _select_records(
        _read_jsonl(eval_file),
        args.annotator_type,
        args.max_samples,
    )
    formatted: list[tuple[dict[str, Any], dict[str, str], str]] = []
    for record in records:
        example = format_ae_record(record)
        complete = json.loads(example.full_assistant_response)
        human_appraisal = complete["appraisal_reasoning"]
        formatted.append((record, human_appraisal, example.assistant_prefix))

    prompt_cfg = baseline.load_toml(prompt_path)
    model_paths = {
        "warmup_v3": args.warmup_model,
        "ae_v1": args.ae_model,
    }
    rows: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        for model_name in MODEL_NAMES:
            model_path = model_paths[model_name]
            client = _build_client(args, model_path)
            try:
                for record, human_appraisal, assistant_prefix in formatted:
                    generated_row = _run_model_condition(
                        client=client,
                        prompt_cfg=prompt_cfg,
                        model_name=model_name,
                        model_path=model_path,
                        record=record,
                        human_appraisal=human_appraisal,
                        max_new_tokens=args.full_chain_max_new_tokens,
                    )
                    human_row = _run_human_condition(
                        client=client,
                        prompt_cfg=prompt_cfg,
                        model_name=model_name,
                        model_path=model_path,
                        record=record,
                        human_appraisal=human_appraisal,
                        assistant_prefix=assistant_prefix,
                        max_new_tokens=args.emotion_max_new_tokens,
                    )
                    for row in (generated_row, human_row):
                        rows.append(row)
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
            finally:
                _release_client(client)

    arms: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        for condition in CONDITIONS:
            arm_rows = [
                row
                for row in rows
                if row["model"] == model_name and row["condition"] == condition
            ]
            key = f"{model_name}/{condition}"
            arms[key] = {
                "samples": len(arm_rows),
                "valid": sum(bool(row.get("valid")) for row in arm_rows),
                "metrics": _emotion_metrics(arm_rows),
            }

    summary: dict[str, Any] = {
        "protocol": "warmup-v3-vs-ae-v1-appraisal-conditioning-v1",
        "eval_file": str(eval_file),
        "prompt_path": str(prompt_path),
        "sample_count": len(formatted),
        "annotator_type": args.annotator_type,
        "models": model_paths,
        "conditions": list(CONDITIONS),
        "decoding": {
            "do_sample": False,
            "enable_thinking": False,
            "full_chain_max_new_tokens": args.full_chain_max_new_tokens,
            "emotion_max_new_tokens": args.emotion_max_new_tokens,
        },
        "arms": arms,
        "human_minus_model_generated": {
            model_name: _metric_delta(
                arms[f"{model_name}/human_appraisal"]["metrics"],
                arms[f"{model_name}/model_generated_appraisal"]["metrics"],
            )
            for model_name in MODEL_NAMES
        },
        "ae_minus_warmup": {
            condition: _metric_delta(
                arms[f"ae_v1/{condition}"]["metrics"],
                arms[f"warmup_v3/{condition}"]["metrics"],
            )
            for condition in CONDITIONS
        },
    }
    _write_json_atomic(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(
        "[appraisal-conditioning] complete "
        f"samples={summary['sample_count']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
