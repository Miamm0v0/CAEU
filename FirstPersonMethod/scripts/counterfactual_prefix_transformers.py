#!/usr/bin/env python3
"""Run three-way CAREBench appraisal-prefix continuations with Transformers.

Protocol
--------
1. A normal ``baseline_transformers.py --task chain-emotion`` run produces one
   model-authored ``appraisal_reasoning`` profile for each native situation.
2. This runner rebuilds the original baseline prompt and supplies that profile
   as an unfinished assistant JSON prefix, stopping before the target field.
3. Two complete gold appraisal profiles are loaded: the original first-person
   profile and one real third-person annotator profile. Selected dimensions,
   or all five by default, replace the corresponding model-authored text.
4. The model continues independently from the model-authored, first-person-
   intervened, and third-person-intervened prefixes to produce CAREBench
   ratings, emotion, or both.

The prompt always contains the native situation only. Human appraisal text is
introduced exclusively on the assistant-output side of the intervention.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from counterfactual_transformers_common import (
    add_transformers_arguments,
    build_transformers_client,
    default_model_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIRST_PERSON_SCRIPTS = REPO_ROOT / "FirstPersonMethod" / "scripts"
if str(FIRST_PERSON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIRST_PERSON_SCRIPTS))

import baseline_transformers_chain as chain  # noqa: E402
import baseline_vllm as baseline  # noqa: E402
from train_grpo.parsing import parse_policy_output  # noqa: E402


PROTOCOL = "carebench-selectable-appraisal-prefix-intervention-v3"
OUTCOMES = ("ratings", "emotion")
SINGLE_OUTPUT_BRANCHES = ("origin", "first-person-appraisal")
THIRD_PERSON_BRANCH = "third-person-appraisal"
EMOTION_TASKS = tuple(chain.CHAIN_EMOTION_OUTPUT_TASKS)
POLICY_DIMENSIONS = tuple(chain.POLICY_APPRAISAL_DIMENSIONS)
CAREBENCH_TO_POLICY = dict(chain.CAREBENCH_TO_POLICY_CORE_DIMENSION)

RATING_TO_CORE: dict[str, str] = {}
for _core_dimension, _rating_dimensions in {
    "relevance": (
        "relevance.general",
        "relevance.urgency",
        "relevance.goals",
        "relevance.bodily motives",
        "relevance.social motives",
        "relevance.identity motives",
    ),
    "certainty": (
        "certainty.construal",
        "certainty.outlook",
        "certainty.predictability",
        "certainty.novelty",
    ),
    "congruence": (
        "congruence.general",
        "congruence.outlook",
        "congruence.positive prediction error",
        "congruence.negative prediction error",
    ),
    "control": (
        "control.general",
        "control.select",
        "control.vicarious",
        "control.effortful",
    ),
    "accountability": (
        "accountability.self",
        "accountability.other",
        "accountability.intentionality",
        "accountability.fairness",
    ),
}.items():
    for _rating_dimension in _rating_dimensions:
        RATING_TO_CORE[_rating_dimension] = _core_dimension


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_counterfactual_items(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty JSON list")
    if any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"{path}: every counterfactual item must be an object")
    return payload


def extract_native_situation(record: dict[str, Any], location: str) -> str:
    direct = record.get("situation")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    story_collection = record.get("story_collection")
    scenario = (
        story_collection.get("final_scenario")
        if isinstance(story_collection, dict)
        else None
    )
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError(f"{location}: missing native situation")
    scenario = scenario.strip()
    if "\n\nAppraisal profile:" in scenario or scenario.startswith(
        "Original situation:"
    ):
        raise ValueError(
            f"{location}: final_scenario contains a legacy appended appraisal "
            "profile; rebuild data with --original_context situation-only"
        )
    return scenario


def source_sample_id(item: dict[str, Any], fallback: str) -> str:
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("source_sample_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def intervention_core_dimension(item: dict[str, Any], folder: str) -> str:
    metadata = item.get("metadata")
    candidates: list[Any] = []
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("counterfactual_core_dimension"),
                metadata.get("counterfactual_dimension"),
            ]
        )
    candidates.append(folder)

    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized = candidate.strip()
        if normalized in CAREBENCH_TO_POLICY:
            return normalized
        mapped = RATING_TO_CORE.get(normalized)
        if mapped:
            return mapped
    raise ValueError(
        f"Cannot map intervention dimension {folder!r} to a CAREBench core appraisal"
    )


def profile_candidates(record: dict[str, Any], source: str) -> list[Any]:
    if source == "third-person":
        return [record.get("third_person_appraisal_reasoning")]
    if source != "first-person":
        raise ValueError(f"Unknown appraisal profile source: {source}")

    profiles: list[Any] = [
        record.get("first_person_appraisal_reasoning"),
        record.get("appraisal_reasoning"),
    ]
    cognitive_questions = record.get("cognitive_questions")
    if isinstance(cognitive_questions, dict):
        profiles.append(cognitive_questions.get("summary_answers"))
    return profiles


def extract_complete_profile(
    record: dict[str, Any], source: str, location: str
) -> dict[str, str]:
    for profile in profile_candidates(record, source):
        if not isinstance(profile, dict):
            continue
        normalized: dict[str, str] = {}
        for carebench_dimension, policy_dimension in CAREBENCH_TO_POLICY.items():
            value = profile.get(policy_dimension, profile.get(carebench_dimension))
            if isinstance(value, str) and value.strip():
                normalized[policy_dimension] = value.strip()
        if len(normalized) == len(POLICY_DIMENSIONS):
            return {
                dimension: normalized[dimension] for dimension in POLICY_DIMENSIONS
            }

    if source == "third-person":
        remedy = (
            "Rebuild the counterfactual data with the current "
            "build_appraisal_counterfactual_data.py."
        )
    else:
        remedy = "Use a first-person input file containing all five appraisals."
    raise ValueError(
        f"{location}: missing complete {source} five-dimensional appraisal "
        f"profile. {remedy}"
    )


def resolve_baseline_trajectory(
    baseline_root: Path, sample_stem: str
) -> Path:
    candidates = [
        baseline_root / "chain-emotion" / f"{sample_stem}.json",
        baseline_root / f"{sample_stem}.json"
        if baseline_root.name == "chain-emotion"
        else baseline_root / "__not_a_candidate__",
    ]
    if baseline_root.exists() and baseline_root.is_dir():
        candidates.extend(
            child / "chain-emotion" / f"{sample_stem}.json"
            for child in sorted(baseline_root.iterdir())
            if child.is_dir() and child.name.startswith("run_")
        )
    matches = [path for path in candidates if path.is_file()]
    unique_matches = list(dict.fromkeys(path.resolve() for path in matches))
    if not unique_matches:
        raise FileNotFoundError(
            f"No chain-emotion baseline trajectory for {sample_stem!r} under "
            f"{baseline_root}"
        )
    if len(unique_matches) > 1:
        raise ValueError(
            "Multiple baseline trajectories found for {}: {}. Point "
            "--baseline_root at one concrete run.".format(
                sample_stem, ", ".join(str(path) for path in unique_matches)
            )
        )
    return unique_matches[0]


def load_baseline_reasoning(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: baseline trajectory must be an object")
    parsed = parse_policy_output(
        json.dumps(payload, ensure_ascii=False), POLICY_DIMENSIONS
    )
    return {
        dimension: parsed["appraisal_reasoning"][dimension]
        for dimension in POLICY_DIMENSIONS
    }


def build_assistant_prefix(reasoning: dict[str, str]) -> str:
    ordered_reasoning: dict[str, str] = {}
    for dimension in POLICY_DIMENSIONS:
        value = reasoning.get(dimension)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing appraisal prefix text for {dimension}")
        ordered_reasoning[dimension] = value.strip()

    complete = json.dumps(
        {"appraisal_reasoning": ordered_reasoning},
        ensure_ascii=False,
        indent=2,
    )
    if not complete.endswith("\n}"):
        raise RuntimeError("Unexpected JSON serialization while building prefix")
    # The target key itself is intentionally not included: generation starts
    # exactly where either "appraisals" or "emotion" would begin.
    return complete[:-2] + ",\n  "


def changed_dimensions(
    original: dict[str, str], replacement: dict[str, str]
) -> list[str]:
    return [
        dimension
        for dimension in POLICY_DIMENSIONS
        if original[dimension] != replacement[dimension]
    ]


def normalize_replacement_dimensions(values: Iterable[str]) -> tuple[str, ...]:
    requested = [value.strip() for value in values if value.strip()]
    if not requested:
        raise ValueError("--replace_dimensions requires at least one value")
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError(
                "--replace_dimensions all cannot be combined with named dimensions"
            )
        return POLICY_DIMENSIONS

    selected = {CAREBENCH_TO_POLICY[value] for value in requested}
    return tuple(
        dimension for dimension in POLICY_DIMENSIONS if dimension in selected
    )


def replace_appraisal_dimensions(
    original: dict[str, str],
    source: dict[str, str],
    dimensions: Iterable[str],
) -> dict[str, str]:
    selected = set(dimensions)
    return {
        dimension: (
            source[dimension] if dimension in selected else original[dimension]
        )
        for dimension in POLICY_DIMENSIONS
    }


def build_prompts(
    prompt_cfg: dict[str, Any], situation: str, outcome: str
) -> tuple[str, str]:
    if outcome == "emotion":
        return chain.build_chain_emotion_prompts(prompt_cfg, situation)
    if outcome == "ratings":
        return chain.build_chain_task_prompts(
            prompt_cfg, situation, "chain-appraisals"
        )
    raise ValueError(f"Unsupported outcome: {outcome}")


def parse_continuation(
    prefix: str,
    continuation: str,
    prompt_cfg: dict[str, Any],
    outcome: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_output = prefix + continuation
    if outcome == "emotion":
        trace, derived = chain.parse_chain_emotion_output(full_output, prompt_cfg)
        return trace, derived
    trace, derived = chain.parse_chain_task_output(
        full_output, prompt_cfg, "chain-appraisals"
    )
    return trace, derived


def outcome_paths(
    model_dir: Path,
    branch: str,
    outcome: str,
    sample_name: str,
    dimension: str | None = None,
) -> list[Path]:
    if branch in SINGLE_OUTPUT_BRANCHES:
        if outcome == "ratings":
            return [model_dir / branch / "appraisals" / sample_name]
        return [model_dir / branch / task / sample_name for task in EMOTION_TASKS]

    if not dimension:
        raise ValueError(f"{branch} output paths require a selection dimension")
    if outcome == "ratings":
        return [
            model_dir
            / branch
            / "appraisals"
            / dimension
            / sample_name
        ]
    return [
        model_dir / branch / task / dimension / sample_name
        for task in EMOTION_TASKS
    ]


def trajectory_path(
    model_dir: Path,
    branch: str,
    outcome: str,
    sample_name: str,
    dimension: str | None = None,
) -> Path:
    path = model_dir / "trajectories" / branch / outcome
    if dimension:
        path /= dimension
    return path / sample_name


def outputs_complete(paths: Iterable[Path], audit_path: Path) -> bool:
    return audit_path.is_file() and all(path.is_file() for path in paths)


def write_single_outputs(
    model_dir: Path,
    branch: str,
    sample_name: str,
    outcome: str,
    derived: dict[str, Any],
) -> None:
    if branch not in SINGLE_OUTPUT_BRANCHES:
        raise ValueError(f"Not a single-output branch: {branch}")
    if outcome == "ratings":
        write_json(model_dir / branch / "appraisals" / sample_name, derived)
        return
    for task in EMOTION_TASKS:
        write_json(model_dir / branch / task / sample_name, derived[task])


def write_third_person_outputs(
    model_dir: Path,
    sample_name: str,
    folder_dimension: str,
    outcome: str,
    rows: list[dict[str, Any]],
) -> None:
    if outcome == "ratings":
        write_json(
            model_dir
            / THIRD_PERSON_BRANCH
            / "appraisals"
            / folder_dimension
            / sample_name,
            rows,
        )
        return
    for task in EMOTION_TASKS:
        write_json(
            model_dir
            / THIRD_PERSON_BRANCH
            / task
            / folder_dimension
            / sample_name,
            [row[task] for row in rows],
        )


def build_parser(default_outcome: str = "all") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continue model-authored, first-person-gold, and "
            "third-person-gold CAREBench appraisal prefixes via Transformers"
        )
    )
    add_transformers_arguments(
        parser,
        str2bool=baseline.str2bool,
        max_tokens_default=2048,
    )
    parser.set_defaults(do_sample=False)
    parser.add_argument(
        "--outcome",
        choices=("all", *OUTCOMES),
        default=default_outcome,
        help="Continue the appraisal prefix to ratings, emotion, or both.",
    )
    parser.add_argument(
        "--first_person_root",
        required=True,
        help=(
            "Folder of original per-sample files containing native situations "
            "and complete first-person appraisal profiles."
        ),
    )
    parser.add_argument(
        "--source_root",
        required=True,
        help=(
            "Counterfactual data root with one subfolder per selection dimension. "
            "Each item must contain a complete third_person_appraisal_reasoning profile."
        ),
    )
    parser.add_argument(
        "--baseline_root",
        required=True,
        help="One baseline run root containing chain-emotion/<sample>.json.",
    )
    parser.add_argument(
        "--target_root",
        default=None,
        help=(
            "Output run directory. Explicit paths are used directly; otherwise "
            "CAREBench/output/first_person/counterfactual_prefix_transformers/<model>."
        ),
    )
    parser.add_argument(
        "--dimension",
        default="all",
        help="One counterfactual selection folder or all.",
    )
    parser.add_argument(
        "--replace_dimensions",
        "--replace-dimensions",
        nargs="+",
        choices=("all", *CAREBENCH_TO_POLICY),
        default=["all"],
        help=(
            "Appraisal dimensions replaced in both human-profile branches. "
            "Use 'all' (default), or one or more of: {}. Unselected "
            "dimensions retain the model-authored appraisal text."
        ).format(", ".join(CAREBENCH_TO_POLICY)),
    )
    parser.add_argument(
        "--prompt_path",
        default="FirstPersonMethod/scripts/prompts/baseline_prompt.toml",
        help="The same baseline prompt TOML used for normal generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs already produced by this protocol.",
    )
    parser.add_argument(
        "--verbose",
        type=baseline.str2bool,
        default=True,
    )
    return parser


def resolve_prompt_path(value: str) -> Path:
    path = Path(value)
    if path.is_file() or path.is_absolute():
        return path
    candidate = REPO_ROOT / path
    return candidate if candidate.is_file() else path


def selected_outcomes(value: str) -> tuple[str, ...]:
    return OUTCOMES if value == "all" else (value,)


def main(default_outcome: str = "all") -> int:
    args = build_parser(default_outcome).parse_args()
    if args.do_sample:
        raise ValueError(
            "Prefix intervention requires --do_sample false so paired branches "
            "differ only in the intervened appraisal text"
        )

    first_person_root = Path(args.first_person_root)
    source_root = Path(args.source_root)
    baseline_root = Path(args.baseline_root)
    prompt_path = resolve_prompt_path(args.prompt_path)
    for path, name in (
        (first_person_root, "first_person_root"),
        (source_root, "source_root"),
        (baseline_root, "baseline_root"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} not found: {path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"prompt_path not found: {prompt_path}")

    if args.dimension.strip().lower() == "all":
        dimensions = [path.name for path in sorted(source_root.iterdir()) if path.is_dir()]
    else:
        dimensions = [args.dimension.strip()]
    if not dimensions:
        raise ValueError("No intervention dimensions selected")
    replacement_dimensions = normalize_replacement_dimensions(
        args.replace_dimensions
    )

    jobs: list[tuple[str, Path]] = []
    for dimension in dimensions:
        directory = source_root / dimension
        if not directory.is_dir():
            print(f"[warning] dimension folder not found: {directory}")
            continue
        jobs.extend((dimension, path) for path in sorted(directory.glob("*.json")))
    if not jobs:
        print("No counterfactual input files found.")
        return 0

    model_dir = default_model_dir(
        script_file=__file__,
        target_root=args.target_root,
        model=args.model,
        family="counterfactual_prefix_transformers",
        sanitize_model_name=baseline.sanitize_model_name,
        target_root_is_model_dir=True,
    )
    manifest_path = model_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = load_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL:
            raise ValueError(
                f"Existing target_root is not a {PROTOCOL} run: {model_dir}; "
                "choose a new directory or pass --overwrite"
            )
        if manifest.get("replacement_dimensions") != list(
            replacement_dimensions
        ):
            raise ValueError(
                "Existing target_root uses different --replace_dimensions; "
                "choose a new directory or pass --overwrite"
            )
    elif model_dir.exists() and any(model_dir.iterdir()) and not args.overwrite:
        raise ValueError(
            f"Non-empty target_root has no compatible manifest: {model_dir}; "
            "choose a new directory or pass --overwrite"
        )
    model_dir.mkdir(parents=True, exist_ok=True)

    prompt_cfg = baseline.load_toml(prompt_path)
    outcomes = selected_outcomes(args.outcome)
    run_manifest = {
        "protocol": PROTOCOL,
        "status": "running",
        "model": args.model,
        "base_model": args.base_model or None,
        "first_person_root": str(first_person_root),
        "source_root": str(source_root),
        "baseline_root": str(baseline_root),
        "target_root": str(model_dir),
        "prompt_path": str(prompt_path),
        "outcomes": list(outcomes),
        "selection_dimensions": dimensions,
        "replacement_dimensions": list(replacement_dimensions),
        "branches": [*SINGLE_OUTPUT_BRANCHES, THIRD_PERSON_BRANCH],
        "appraisal_replacement_scope": (
            "all_five_dimensions"
            if replacement_dimensions == POLICY_DIMENSIONS
            else "selected_dimensions"
        ),
        "do_sample": args.do_sample,
        "enable_thinking": args.enable_thinking,
    }
    write_json(manifest_path, run_manifest)
    client = build_transformers_client(args)
    completed_single_branches: set[tuple[str, str, str]] = set()
    third_person_generation_cache: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    invalid_rows: list[dict[str, Any]] = []
    counts = {
        "input_files": len(jobs),
        "origin_continuations": 0,
        "first_person_appraisal_continuations": 0,
        "third_person_appraisal_continuations": 0,
        "reused_third_person_appraisal_continuations": 0,
        "skipped_origin": 0,
        "skipped_first_person_appraisal": 0,
        "skipped_third_person_appraisal_files": 0,
        "invalid_items": 0,
    }

    iterator: Iterable[tuple[str, Path]] = jobs
    if baseline.tqdm is not None:
        iterator = baseline.tqdm(jobs, desc="Prefix interventions", unit="file")

    last_audit: dict[str, Any] | None = None
    for folder_dimension, counterfactual_file in iterator:
        sample_name = counterfactual_file.name
        sample_stem = counterfactual_file.stem
        first_person_file = first_person_root / sample_name
        if not first_person_file.is_file():
            invalid_rows.append(
                {
                    "dimension": folder_dimension,
                    "input_file": str(counterfactual_file),
                    "error": f"Missing first-person file: {first_person_file}",
                }
            )
            counts["invalid_items"] += 1
            continue

        try:
            first_person = load_json(first_person_file)
            if not isinstance(first_person, dict):
                raise ValueError("first-person file must contain one object")
            situation = extract_native_situation(first_person, str(first_person_file))
            first_person_source_reasoning = extract_complete_profile(
                first_person, "first-person", str(first_person_file)
            )
            baseline_path = resolve_baseline_trajectory(baseline_root, sample_stem)
            original_reasoning = load_baseline_reasoning(baseline_path)
            original_prefix = build_assistant_prefix(original_reasoning)
            first_person_reasoning = replace_appraisal_dimensions(
                original_reasoning,
                first_person_source_reasoning,
                replacement_dimensions,
            )
            first_person_prefix = build_assistant_prefix(first_person_reasoning)
            items = parse_counterfactual_items(counterfactual_file)
        except Exception as exc:  # noqa: BLE001 - record and continue other files.
            invalid_rows.append(
                {
                    "dimension": folder_dimension,
                    "input_file": str(counterfactual_file),
                    "error": str(exc),
                }
            )
            counts["invalid_items"] += 1
            continue

        for outcome in outcomes:
            system_prompt, user_prompt = build_prompts(
                prompt_cfg, situation, outcome
            )
            single_profiles = (
                (
                    "origin",
                    "model-authored",
                    original_reasoning,
                    original_reasoning,
                    original_prefix,
                ),
                (
                    "first-person-appraisal",
                    "first-person-gold",
                    first_person_source_reasoning,
                    first_person_reasoning,
                    first_person_prefix,
                ),
            )
            for (
                branch,
                profile_source,
                source_reasoning,
                reasoning,
                prefix,
            ) in single_profiles:
                completed_key = (branch, sample_stem, outcome)
                if completed_key in completed_single_branches:
                    continue
                paths = outcome_paths(
                    model_dir, branch, outcome, sample_name
                )
                audit_path = trajectory_path(
                    model_dir, branch, outcome, sample_name
                )
                skip_key = (
                    "skipped_origin"
                    if branch == "origin"
                    else "skipped_first_person_appraisal"
                )
                count_key = (
                    "origin_continuations"
                    if branch == "origin"
                    else "first_person_appraisal_continuations"
                )
                if not args.overwrite and outputs_complete(paths, audit_path):
                    counts[skip_key] += 1
                    completed_single_branches.add(completed_key)
                    continue
                try:
                    response = client.continue_chat(
                        system_prompt, user_prompt, prefix
                    )
                    trace, derived = parse_continuation(
                        prefix,
                        response.content,
                        prompt_cfg,
                        outcome,
                    )
                    write_single_outputs(
                        model_dir, branch, sample_name, outcome, derived
                    )
                    audit = {
                        "protocol": PROTOCOL,
                        "branch": branch,
                        "profile_source": profile_source,
                        "sample_id": sample_stem,
                        "source_sample_id": first_person.get("id", sample_stem),
                        "participant_id": first_person.get("participant_id", ""),
                        "outcome": outcome,
                        "situation": situation,
                        "source_appraisal_reasoning": source_reasoning,
                        "appraisal_reasoning": reasoning,
                        "replaced_dimensions": (
                            []
                            if branch == "origin"
                            else list(replacement_dimensions)
                        ),
                        "textually_changed_dimensions_from_origin": (
                            []
                            if branch == "origin"
                            else changed_dimensions(original_reasoning, reasoning)
                        ),
                        "assistant_prefix": prefix,
                        "continuation": response.content,
                        "full_output": prefix + response.content,
                        "parsed_output": trace,
                        "generation": response.raw,
                    }
                    if branch == "origin":
                        audit["baseline_trajectory"] = str(baseline_path)
                    write_json(audit_path, audit)
                    counts[count_key] += 1
                    last_audit = audit
                except Exception as exc:  # noqa: BLE001
                    invalid_rows.append(
                        {
                            "dimension": folder_dimension,
                            "sample_id": sample_stem,
                            "branch": branch,
                            "outcome": outcome,
                            "error": str(exc),
                        }
                    )
                    counts["invalid_items"] += 1
                completed_single_branches.add(completed_key)

            third_paths = outcome_paths(
                model_dir,
                THIRD_PERSON_BRANCH,
                outcome,
                sample_name,
                folder_dimension,
            )
            third_audit_path = trajectory_path(
                model_dir,
                THIRD_PERSON_BRANCH,
                outcome,
                sample_name,
                folder_dimension,
            )
            if not args.overwrite and outputs_complete(
                third_paths, third_audit_path
            ):
                counts["skipped_third_person_appraisal_files"] += 1
                continue

            result_rows: list[dict[str, Any]] = []
            audit_rows: list[dict[str, Any]] = []
            for index, item in enumerate(items):
                participant_id = str(item.get("participant_id", "")).strip()
                try:
                    item_situation = extract_native_situation(
                        item, f"{counterfactual_file}[{index}]"
                    )
                    if item_situation != situation:
                        raise ValueError(
                            "Counterfactual and first-person native situations differ"
                        )
                    carebench_dimension = intervention_core_dimension(
                        item, folder_dimension
                    )
                    third_person_source_reasoning = extract_complete_profile(
                        item,
                        "third-person",
                        f"{counterfactual_file}[{index}]",
                    )
                    third_person_reasoning = replace_appraisal_dimensions(
                        original_reasoning,
                        third_person_source_reasoning,
                        replacement_dimensions,
                    )
                    third_person_prefix = build_assistant_prefix(
                        third_person_reasoning
                    )
                    cache_key = (
                        sample_stem,
                        participant_id,
                        outcome,
                        json.dumps(
                            third_person_reasoning,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                    cached = third_person_generation_cache.get(cache_key)
                    if cached is None:
                        response = client.continue_chat(
                            system_prompt, user_prompt, third_person_prefix
                        )
                        trace, derived = parse_continuation(
                            third_person_prefix,
                            response.content,
                            prompt_cfg,
                            outcome,
                        )
                        cached = {
                            "content": response.content,
                            "generation": response.raw,
                            "trace": trace,
                            "derived": derived,
                        }
                        third_person_generation_cache[cache_key] = cached
                        counts["third_person_appraisal_continuations"] += 1
                    else:
                        trace = cached["trace"]
                        derived = cached["derived"]
                        counts[
                            "reused_third_person_appraisal_continuations"
                        ] += 1

                    common_row = {
                        "participant_id": participant_id,
                        "scenario": situation,
                    }
                    if outcome == "emotion":
                        result_rows.append(
                            {
                                task: {**common_row, **derived[task]}
                                for task in EMOTION_TASKS
                            }
                        )
                    else:
                        rating_row = {
                            **common_row,
                            "appraisals": derived,
                            "target_appraisals": derived,
                        }
                        if folder_dimension in derived:
                            rating_row.update(derived[folder_dimension])
                        result_rows.append(rating_row)

                    audit = {
                        "protocol": PROTOCOL,
                        "branch": THIRD_PERSON_BRANCH,
                        "profile_source": "third-person-gold",
                        "sample_id": sample_stem,
                        "source_sample_id": source_sample_id(item, sample_stem),
                        "participant_id": participant_id,
                        "outcome": outcome,
                        "selection_dimension": carebench_dimension,
                        "situation": situation,
                        "original_appraisal_reasoning": original_reasoning,
                        "third_person_appraisal_reasoning": (
                            third_person_source_reasoning
                        ),
                        "intervened_appraisal_reasoning": third_person_reasoning,
                        "replaced_dimensions": list(replacement_dimensions),
                        "textually_changed_dimensions_from_origin": (
                            changed_dimensions(
                                original_reasoning, third_person_reasoning
                            )
                        ),
                        "original_assistant_prefix": original_prefix,
                        "replacement_assistant_prefix": third_person_prefix,
                        "continuation": cached["content"],
                        "full_output": third_person_prefix + cached["content"],
                        "parsed_output": trace,
                        "generation": cached["generation"],
                        "counterfactual_metadata": item.get("metadata", {}),
                    }
                    audit_rows.append(audit)
                    last_audit = audit
                except Exception as exc:  # noqa: BLE001
                    counts["invalid_items"] += 1
                    invalid_rows.append(
                        {
                            "dimension": folder_dimension,
                            "input_file": str(counterfactual_file),
                            "index": index,
                            "participant_id": participant_id,
                            "branch": THIRD_PERSON_BRANCH,
                            "outcome": outcome,
                            "error": str(exc),
                        }
                    )
                    if outcome == "emotion":
                        result_rows.append(
                            {
                                task: {
                                    "participant_id": participant_id,
                                    "scenario": situation,
                                    **(
                                        {"score": None, "label": ""}
                                        if task in {
                                            "positive-level",
                                            "negative-level",
                                        }
                                        else {"labels": []}
                                    ),
                                }
                                for task in EMOTION_TASKS
                            }
                        )
                    else:
                        result_rows.append(
                            {
                                "participant_id": participant_id,
                                "scenario": situation,
                                "appraisals": {},
                                "target_appraisals": {},
                            }
                        )

            write_third_person_outputs(
                model_dir,
                sample_name,
                folder_dimension,
                outcome,
                result_rows,
            )
            write_json(third_audit_path, audit_rows)

    write_json(model_dir / "invalid_samples.json", invalid_rows)
    manifest = {
        **run_manifest,
        "status": "complete",
        "counts": counts,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.verbose and last_audit:
        print(
            "[verbose] last branch={} participant={} outcome={}".format(
                last_audit["branch"],
                last_audit.get("participant_id", ""),
                last_audit["outcome"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
