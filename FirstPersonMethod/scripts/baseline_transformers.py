#!/usr/bin/env python3
"""Run the CAREBench first-person baseline with local Transformers inference.

This is the non-vLLM counterpart of ``baseline_vllm.py``. It retains the same
benchmark prompts, parsers and output layout while using a local Transformers
generation backend. PEFT/LoRA loading lives in
``baseline_transformers_backend.py``; single-call chain tasks live in
``baseline_transformers_chain.py``.

Examples::

    python FirstPersonMethod/scripts/baseline_transformers.py \
      --model /path/to/Qwen3-8B \
      --task positive-level \
      --single_file baseline_data/smoke/sample.json \
      --target_folder FirstPersonMethod/output/transformers_smoke \
      --dtype bf16 --enable_thinking false --do_sample false

    python FirstPersonMethod/scripts/baseline_transformers.py \
      --model /path/to/policy_adapter \
      --base_model /path/to/base_model \
      --task chain-all \
      --source_folder baseline_data/test_raw \
      --target_folder FirstPersonMethod/output/policy_test \
      --dtype bf16 --enable_thinking false --do_sample false

``--task all`` uses the six original task flows. Each ``chain-*`` task makes
one generation call per event, first producing five core appraisals and then
one target. ``chain-all`` can run the four emotion targets separately or
replace them with one joint ``chain-emotion`` call. ``full-chain`` keeps the
older one-call trajectory that produces all six targets together.
For CAREBench, ``chain-emotion`` uses the exact GRPO policy schema and derives
all four emotion benchmark tasks from one generation. For external datasets,
it keeps the same appraisal chain but emits the dataset-native emotion schema. With
``--chain_core_appraisals_source chain-emotion``, the same trajectory also
supplies the mapped CAREBench core-appraisals output instead of making a
separate ``chain-core-appraisals`` call.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import baseline_vllm as baseline
from baseline_transformers_backend import (
    TransformersClient,
    choose_tokenizer_source,
    local_path_has_tokenizer,
)
from baseline_transformers_chain import (
    CHAIN_ALL_TASK,
    CHAIN_ALL_EMOTION_MODES,
    CHAIN_CORE_APPRAISAL_SOURCES,
    CHAIN_EMOTION_OUTPUT_TASKS,
    CHAIN_EMOTION_TASK,
    CHAIN_TASKS,
    CHAIN_TASK_TO_OUTPUT_TASK,
    CORE_APPRAISAL_DIMENSIONS,
    DIRECT_APPRAISALS_TASK,
    DIRECT_EMOTION_TASK,
    EMOTION_LABEL_SPACES,
    FULL_CHAIN_OUTPUT_TASKS,
    FULL_CHAIN_TASK,
    OFFICIAL_DIRECT_TASK,
    build_chain_task_prompts,
    build_chain_emotion_prompts,
    build_direct_task_prompts,
    build_full_chain_prompts,
    parse_chain_task_output,
    parse_chain_emotion_output,
    parse_direct_task_output,
    parse_full_chain_output,
    process_chain_task,
    process_chain_emotion,
    process_direct_task,
    process_full_chain,
    resolve_selected_tasks,
    set_emotion_label_space,
    set_emotion_output_mode,
)


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    scenario: str
    source: str


def build_argument_parser(*, model_required: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-person CAREBench baseline via local Transformers"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help=(
            "One benchmark task, 'all' for six separate query flows, or "
            "'full-chain' for one generation that produces all six tasks; "
            "use chain-<task> for one core-appraisal-conditioned target or "
            "'chain-all' for all six conditioned outputs; 'chain-emotion' "
            "generates the GRPO policy schema and all four emotion targets; "
            f"'{OFFICIAL_DIRECT_TASK}' runs separate crowd-enVent text-to-ratings "
            "and text-to-emotion branches"
        ),
    )
    parser.add_argument(
        "--source_folder", type=str, default=None, help="Folder of raw sample JSON files"
    )
    parser.add_argument(
        "--dataset_format",
        choices=list(EMOTION_LABEL_SPACES),
        default="carebench",
        help=(
            "Input/label adapter. carebench reads per-sample JSON files; "
            "covidet and crowd-envent read samples[].situation from --dataset_file."
        ),
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Aggregated CovidET or crowd-enVent evaluation JSON.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Use only the first N samples from an aggregated dataset file.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Sampling seed used by crowd-enVent when --max_samples is set.",
    )
    parser.add_argument(
        "--chain_all_emotion_mode",
        choices=list(CHAIN_ALL_EMOTION_MODES),
        default="separate",
        help=(
            "Only affects --task chain-all: 'separate' runs four independent "
            "emotion chain calls; 'chain-emotion' replaces them with one joint call"
        ),
    )
    parser.add_argument(
        "--chain_core_appraisals_source",
        choices=list(CHAIN_CORE_APPRAISAL_SOURCES),
        default="separate",
        help=(
            "Source used for the CAREBench core-appraisals metric. 'separate' "
            "runs chain-core-appraisals (default); 'chain-emotion' maps the "
            "appraisal_reasoning generated inside chain-emotion and avoids the "
            "extra core generation call"
        ),
    )
    parser.add_argument(
        "--single_file", type=str, default=None, help="One raw sample JSON for a smoke test"
    )
    parser.add_argument("--target_folder", type=str, default=None)
    parser.add_argument("--verbose", type=baseline.str2bool, default=True)
    parser.add_argument(
        "--model",
        type=str,
        required=model_required,
        help="Hugging Face model id, full-model directory, or PEFT adapter directory",
    )
    parser.add_argument(
        "--base_model",
        "--base_model_name_or_path",
        dest="base_model",
        type=str,
        default="",
        help=(
            "Optional base model override for a PEFT/LoRA --model. Local "
            "adapters otherwise use base_model_name_or_path from "
            "adapter_config.json."
        ),
    )
    parser.add_argument(
        "--model_is_adapter",
        action="store_true",
        help=(
            "Force --model to load as a PEFT adapter. Local directories with "
            "adapter_config.json are detected automatically."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help=(
            "Optional tokenizer id/path; defaults to a tokenizer saved with "
            "--model, then to the PEFT base model"
        ),
    )
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--do_sample",
        type=baseline.str2bool,
        default=True,
        help="Sampling matches the vLLM baseline; false gives deterministic greedy decoding",
    )
    parser.add_argument(
        "--enable_thinking",
        type=baseline.str2bool,
        default=False,
        help="Qwen3 thinking mode; disabled by default for label parsing",
    )
    parser.add_argument(
        "--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or cuda:N; ignored for dispatched device maps",
    )
    parser.add_argument(
        "--device_map",
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        default=None,
        help="Accelerate device map for multi-GPU/offload; omit to use --device",
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
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="File workers; model.generate calls are serialized for thread safety",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="FirstPersonMethod/scripts/prompts/baseline_prompt.toml",
    )
    return parser


def _empty_sample_result(selected_tasks: List[str]) -> Dict[str, Any]:
    return {
        "processed": {task: 0 for task in selected_tasks},
        "skipped": {task: 0 for task in selected_tasks},
        "ambiguous": {task: 0 for task in selected_tasks},
        "invalid": {task: 0 for task in selected_tasks},
        "invalid_rows": {task: [] for task in selected_tasks},
        "last_verbose": None,
    }


def _record_invalid(
    sample_result: Dict[str, Any],
    task: str,
    sample_id: str,
    input_path: Union[Path, str],
    issues: List[Dict[str, Any]],
) -> None:
    issue_rows = [
        {
            "sample_id": sample_id,
            "input_file": str(input_path),
            "task": task,
            **issue,
        }
        for issue in issues
    ]
    if not issue_rows:
        issue_rows = [
            {
                "sample_id": sample_id,
                "input_file": str(input_path),
                "task": task,
                "error": "Unknown invalid sample",
            }
        ]
    sample_result["invalid"][task] += 1
    sample_result["invalid_rows"][task].extend(issue_rows)


def _run_task(
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    scenario: str,
    task: str,
    chain_core_appraisals_source: str = "separate",
) -> tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
    bool,
]:
    output: Optional[Dict[str, Any]] = None
    derived_outputs: Optional[Dict[str, Dict[str, Any]]] = None
    derived_output: Optional[Dict[str, Any]] = None
    issues: List[Dict[str, Any]] = []
    is_ambiguous = False
    try:
        if task == FULL_CHAIN_TASK:
            output, derived_outputs, issues = process_full_chain(
                client, prompt_cfg, scenario
            )
        elif task == CHAIN_EMOTION_TASK:
            output, derived_outputs, issues = process_chain_emotion(
                client,
                prompt_cfg,
                scenario,
                include_core_appraisals=(
                    chain_core_appraisals_source == CHAIN_EMOTION_TASK
                ),
            )
        elif task in CHAIN_TASK_TO_OUTPUT_TASK:
            output, derived_output, issues = process_chain_task(
                client, prompt_cfg, scenario, task
            )
        elif task in {DIRECT_APPRAISALS_TASK, DIRECT_EMOTION_TASK}:
            output, issues = process_direct_task(
                client, prompt_cfg, scenario, task
            )
        elif task == "appraisals":
            output, issues = baseline.process_appraisals(client, prompt_cfg, scenario)
        elif task == "core-appraisals":
            output, issues = baseline.process_core_appraisals(
                client, prompt_cfg, scenario
            )
        elif task in {"positive-level", "negative-level"}:
            output, issues = baseline.process_level_task(
                client, prompt_cfg, scenario, task
            )
        elif task in {"positive-labels", "negative-labels"}:
            output, issues, is_ambiguous = baseline.process_labels_task(
                client, prompt_cfg, scenario, task
            )
        else:
            issues = [{"error": f"Unsupported task: {task}"}]
    except Exception as exc:
        issues = [{"error": f"Task execution error: {exc}"}]
    return output, derived_outputs, derived_output, issues, is_ambiguous


def _write_task_outputs(
    model_base_dir: Path,
    sample_id: str,
    task: str,
    output: Dict[str, Any],
    derived_outputs: Optional[Dict[str, Dict[str, Any]]],
    derived_output: Optional[Dict[str, Any]],
    chain_core_appraisals_source: str = "separate",
) -> Optional[str]:
    if task in {FULL_CHAIN_TASK, CHAIN_EMOTION_TASK}:
        if derived_outputs is None:
            return f"{task} output has no derived task outputs"
        if task == FULL_CHAIN_TASK:
            output_tasks = FULL_CHAIN_OUTPUT_TASKS
        else:
            output_tasks = CHAIN_EMOTION_OUTPUT_TASKS
            if chain_core_appraisals_source == CHAIN_EMOTION_TASK:
                output_tasks = (*output_tasks, "core-appraisals")
        for derived_task in output_tasks:
            if derived_task not in derived_outputs:
                continue
            baseline.write_json(
                model_base_dir / derived_task / f"{sample_id}.json",
                derived_outputs[derived_task],
            )
    elif task in CHAIN_TASK_TO_OUTPUT_TASK:
        if derived_output is None:
            return f"{task} output has no derived task output"
        target_task = CHAIN_TASK_TO_OUTPUT_TASK[task]
        baseline.write_json(
            model_base_dir / target_task / f"{sample_id}.json",
            derived_output,
        )

    # The trajectory is written last and acts as the resume marker.
    baseline.write_json(model_base_dir / task / f"{sample_id}.json", output)
    return None


def _refresh_core_appraisals_from_trajectory(
    trajectory_path: Path,
    model_base_dir: Path,
    sample_id: str,
    task: str,
    prompt_cfg: Dict[str, Any],
) -> None:
    """Rebuild the selected CAREBench core output when resuming a run."""
    raw_text = trajectory_path.read_text(encoding="utf-8")
    if task == CHAIN_EMOTION_TASK:
        _, derived_outputs = parse_chain_emotion_output(
            raw_text,
            prompt_cfg,
            include_core_appraisals=True,
        )
        core_output = derived_outputs["core-appraisals"]
    elif task == "chain-core-appraisals":
        _, core_output = parse_chain_task_output(
            raw_text,
            prompt_cfg,
            "chain-core-appraisals",
        )
    else:
        raise ValueError(f"Cannot restore core appraisals from task {task}")
    baseline.write_json(
        model_base_dir / "core-appraisals" / f"{sample_id}.json",
        core_output,
    )


def _process_one_sample(
    input_path: Union[Path, DatasetSample],
    selected_tasks: List[str],
    model_base_dir: Path,
    client: baseline.BaseLLMClient,
    prompt_cfg: Dict[str, Any],
    chain_core_appraisals_source: str = "separate",
) -> Dict[str, Any]:
    sample_result = _empty_sample_result(selected_tasks)
    if isinstance(input_path, DatasetSample):
        sample_id = input_path.sample_id
        scenario = input_path.scenario
        input_source: Union[Path, str] = input_path.source
    else:
        sample_id = input_path.stem
        input_source = input_path
        try:
            sample_id, scenario = baseline.parse_sample(input_path)
        except Exception as exc:
            for task in selected_tasks:
                _record_invalid(
                    sample_result,
                    task,
                    sample_id,
                    input_source,
                    [{"error": f"Input parse error: {exc}"}],
                )
            return sample_result

    for task in selected_tasks:
        output_file = model_base_dir / task / f"{sample_id}.json"
        if output_file.exists():
            selected_core_source_task = (
                CHAIN_EMOTION_TASK
                if chain_core_appraisals_source == CHAIN_EMOTION_TASK
                else "chain-core-appraisals"
            )
            if task == selected_core_source_task:
                try:
                    _refresh_core_appraisals_from_trajectory(
                        output_file,
                        model_base_dir,
                        sample_id,
                        task,
                        prompt_cfg,
                    )
                except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                    # A stale trajectory may use the pre-GRPO appraisal schema.
                    # Regenerate this task below instead of silently retaining a
                    # core-appraisals file from the wrong source.
                    pass
                else:
                    sample_result["skipped"][task] += 1
                    continue
            else:
                sample_result["skipped"][task] += 1
                continue

        output, derived_outputs, derived_output, issues, is_ambiguous = _run_task(
            client,
            prompt_cfg,
            scenario,
            task,
            chain_core_appraisals_source,
        )
        if issues or output is None:
            _record_invalid(sample_result, task, sample_id, input_source, issues)
            continue

        write_error = _write_task_outputs(
            model_base_dir,
            sample_id,
            task,
            output,
            derived_outputs,
            derived_output,
            chain_core_appraisals_source,
        )
        if write_error:
            _record_invalid(
                sample_result,
                task,
                sample_id,
                input_source,
                [{"error": write_error}],
            )
            continue

        sample_result["processed"][task] += 1
        if task in {"positive-labels", "negative-labels"} and is_ambiguous:
            sample_result["ambiguous"][task] += 1
        sample_result["last_verbose"] = {
            "sample_id": sample_id,
            "text": scenario,
            "output": output,
            "task": task,
        }
    return sample_result


def load_dataset_samples(
    dataset_file: Path,
    max_samples: Optional[int] = None,
    *,
    dataset_format: str = "covidet",
    sample_seed: int = 42,
) -> List[DatasetSample]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    with dataset_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{dataset_file} must contain a non-empty samples array")

    if max_samples is None or max_samples >= len(samples):
        indexed_samples = list(enumerate(samples))
    elif dataset_format == "crowd-envent":
        selected_indices = random.Random(sample_seed).sample(
            range(len(samples)), max_samples
        )
        indexed_samples = [(index, samples[index]) for index in selected_indices]
    else:
        indexed_samples = list(enumerate(samples[:max_samples]))
    result: List[DatasetSample] = []
    seen: set[str] = set()
    for index, sample in indexed_samples:
        if not isinstance(sample, dict):
            raise ValueError(f"{dataset_file}: samples[{index}] must be an object")
        sample_id = sample.get("id")
        scenario = sample.get("situation")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{dataset_file}: samples[{index}] has no id")
        sample_id = sample_id.strip()
        if sample_id in seen:
            raise ValueError(f"{dataset_file}: duplicate sample id {sample_id!r}")
        if not isinstance(scenario, str) or not scenario.strip():
            raise ValueError(f"{dataset_file}: {sample_id} has no situation")
        seen.add(sample_id)
        result.append(
            DatasetSample(
                sample_id=sample_id,
                scenario=scenario.strip(),
                source=f"{dataset_file}#samples[{index}]",
            )
        )
    return result


def resolve_input_samples(
    args: argparse.Namespace,
) -> List[Union[Path, DatasetSample]]:
    if args.dataset_format == "carebench":
        if args.dataset_file:
            raise ValueError(
                "--dataset_file requires --dataset_format covidet or crowd-envent"
            )
        if args.max_samples is not None:
            raise ValueError("--max_samples is only valid with --dataset_file")
        if args.source_folder and args.single_file:
            print("[warning] --source_folder is set; --single_file will be ignored.")
        return baseline.list_input_files(
            args.source_folder, None if args.source_folder else args.single_file
        )

    if not args.dataset_file:
        raise ValueError(
            f"--dataset_file is required for --dataset_format {args.dataset_format}"
        )
    if args.source_folder or args.single_file:
        raise ValueError(
            "--source_folder/--single_file cannot be combined with --dataset_file"
        )
    return load_dataset_samples(
        Path(args.dataset_file),
        args.max_samples,
        dataset_format=args.dataset_format,
        sample_seed=args.sample_seed,
    )


def execute(args: argparse.Namespace) -> int:
    if args.num_threads <= 0:
        raise ValueError("--num_threads must be >= 1")
    if baseline.tqdm is None:
        raise RuntimeError("Missing dependency 'tqdm'. Install with: pip install tqdm")
    if args.num_threads > 1:
        print(
            "[warning] local model.generate calls are serialized; --num_threads > 1 "
            "only overlaps JSON parsing and file writes, not GPU generation."
        )

    selected_tasks = resolve_selected_tasks(
        args.task,
        args.chain_all_emotion_mode,
        args.chain_core_appraisals_source,
    )
    input_samples = resolve_input_samples(args)
    if not input_samples:
        print("No input files found.")
        return 0

    prompt_cfg = baseline.load_toml(Path(args.prompt_path))
    set_emotion_label_space(prompt_cfg, args.dataset_format)
    if hasattr(args, "emotion_output_mode"):
        set_emotion_output_mode(prompt_cfg, args.emotion_output_mode)
    client = TransformersClient(
        model_name_or_path=args.model,
        tokenizer_name_or_path=args.tokenizer,
        max_tokens=args.max_tokens,
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

    if args.target_folder:
        model_base_dir = Path(args.target_folder)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        model_base_dir = (
            repo_root
            / "output"
            / "first_person"
            / "baseline_transformers"
            / baseline.sanitize_model_name(args.model)
        )
    baseline.ensure_dir(model_base_dir)

    processed_count = {task: 0 for task in selected_tasks}
    skipped_count = {task: 0 for task in selected_tasks}
    ambiguous_count = {task: 0 for task in selected_tasks}
    invalid_count = {task: 0 for task in selected_tasks}
    invalid_rows_by_task: Dict[str, List[Dict[str, Any]]] = {
        task: [] for task in selected_tasks
    }
    last_verbose: Optional[Dict[str, Any]] = None

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(
                _process_one_sample,
                sample,
                selected_tasks,
                model_base_dir,
                client,
                prompt_cfg,
                args.chain_core_appraisals_source,
            )
            for sample in input_samples
        ]
        with baseline.tqdm(
            total=len(futures), desc="Processing samples", unit="sample"
        ) as progress:
            for future in as_completed(futures):
                result = future.result()
                for task in selected_tasks:
                    processed_count[task] += int(result["processed"][task])
                    skipped_count[task] += int(result["skipped"][task])
                    ambiguous_count[task] += int(result["ambiguous"][task])
                    invalid_count[task] += int(result["invalid"][task])
                    invalid_rows_by_task[task].extend(result["invalid_rows"][task])
                if result["last_verbose"] is not None:
                    last_verbose = result["last_verbose"]
                progress.update(1)

    for task in selected_tasks:
        baseline.append_jsonl(
            model_base_dir / f"invalid_samples_{task}.jsonl",
            invalid_rows_by_task[task],
        )
        print(
            "[summary] task={} processed={} skipped={} ambiguous={} invalid={}".format(
                task,
                processed_count[task],
                skipped_count[task],
                ambiguous_count[task],
                invalid_count[task],
            )
        )

    if args.verbose and last_verbose:
        print("[verbose] last task:", last_verbose["task"])
        print("[verbose] last sample_id:", last_verbose["sample_id"])
        print("[verbose] last text:", last_verbose["text"])
        print(
            "[verbose] last output:",
            json.dumps(last_verbose["output"], ensure_ascii=False),
        )
    return 0


def main() -> int:
    return execute(build_argument_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
