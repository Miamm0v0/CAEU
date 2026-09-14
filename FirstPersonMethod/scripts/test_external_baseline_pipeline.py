from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
COVIDET_UTILS = REPO_ROOT / "covidet_utils"
CROWD_UTILS = REPO_ROOT / "crowd-envent_utils"
for import_path in (SCRIPTS_DIR, COVIDET_UTILS, CROWD_UTILS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import baseline_transformers as runner  # noqa: E402
import baseline_transformers_chain as chain  # noqa: E402
import convert_baseline_covidet as covidet_converter  # noqa: E402
import convert_baseline_crowd_envent as crowd_converter  # noqa: E402
import evaluate_policy_covidet as covidet_evaluator  # noqa: E402
import evaluate_policy_crowd_envent as crowd_evaluator  # noqa: E402
import sft_common  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _reasoning() -> dict[str, str]:
    return {
        dimension: f"My {dimension} appraisal."
        for dimension in chain.POLICY_APPRAISAL_DIMENSIONS
    }


def _appraisal_trajectory() -> dict[str, object]:
    return {
        "appraisal_reasoning": _reasoning(),
        "appraisals": {
            key: {"score": 3, "label": "Neither agree nor disagree"}
            for _, key in sft_common.APPRAISAL_KEY_MAP
        },
    }


class ExternalBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt_cfg = runner.baseline.load_toml(
            SCRIPTS_DIR / "prompts" / "baseline_prompt.toml"
        )

    def test_covidet_uses_baseline_prompt_with_native_labels(self) -> None:
        chain.set_emotion_label_space(self.prompt_cfg, "covidet")
        _, prompt = chain.build_chain_emotion_prompts(
            self.prompt_cfg, "I received unexpected news."
        )
        self.assertIn("- anticipation: anticipation", prompt)
        self.assertIn("- sadness: sadness", prompt)
        self.assertNotIn("hopeful: Hopeful", prompt)

        payload = {
            "appraisal_reasoning": _reasoning(),
            "emotion": {
                "positive_intensity": 4,
                "negative_intensity": 2,
                "positive_labels": ["joy", "trust"],
                "negative_labels": ["fear"],
            },
        }
        parsed, outputs = chain.parse_chain_emotion_output(
            json.dumps(payload), self.prompt_cfg
        )
        self.assertEqual(parsed["emotion"]["positive_labels"], ["joy", "trust"])
        self.assertEqual(outputs["negative-labels"]["labels"], ["fear"])

    def test_aggregated_dataset_input_reads_situation_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_file = Path(directory) / "test.json"
            _write_json(
                dataset_file,
                {"samples": [{"id": "sample-1", "situation": "My situation."}]},
            )
            samples = runner.load_dataset_samples(dataset_file)
        self.assertEqual(samples[0].sample_id, "sample-1")
        self.assertEqual(samples[0].scenario, "My situation.")

    def test_external_dataset_still_runs_exactly_two_baseline_chains(self) -> None:
        chain.set_emotion_label_space(self.prompt_cfg, "covidet")
        responses = [
            _appraisal_trajectory(),
            {
                "appraisal_reasoning": _reasoning(),
                "emotion": {
                    "positive_intensity": 4,
                    "negative_intensity": 2,
                    "positive_labels": ["joy"],
                    "negative_labels": ["fear"],
                },
            },
        ]

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def chat(self, system_prompt: str, user_prompt: str):
                del system_prompt, user_prompt
                payload = responses[self.calls]
                self.calls += 1
                return chain.baseline.ChatResponse(
                    content=json.dumps(payload), raw={"call": self.calls}
                )

        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            result = runner._process_one_sample(
                runner.DatasetSample("sample-1", "My situation.", "memory"),
                ["chain-appraisals", "chain-emotion"],
                output_root,
                client,
                self.prompt_cfg,
                "chain-emotion",
            )
            self.assertTrue((output_root / "chain-appraisals/sample-1.json").is_file())
            self.assertTrue((output_root / "chain-emotion/sample-1.json").is_file())
        self.assertEqual(client.calls, 2)
        self.assertEqual(result["invalid"], {"chain-appraisals": 0, "chain-emotion": 0})

    def test_covidet_converter_writes_evaluator_compatible_record(self) -> None:
        eval_file = REPO_ROOT / "FirstPersonMethod/data/covidet_appraisals/test.json"
        sample = covidet_evaluator.load_evaluation_file(eval_file)["samples"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / "baseline" / "chain-appraisals" / f"{sample['id']}.json",
                _appraisal_trajectory(),
            )
            _write_json(
                root / "baseline" / "chain-emotion" / f"{sample['id']}.json",
                {
                    "appraisal_reasoning": _reasoning(),
                    "emotion": {
                        "positive_intensity": 4,
                        "negative_intensity": 3,
                        "positive_labels": ["joy"],
                        "negative_labels": ["fear"],
                    },
                },
            )
            predictions = root / "predictions.jsonl"
            valid, invalid = covidet_converter.convert_outputs(
                eval_file, root / "baseline", predictions, max_samples=1
            )
            indexed = covidet_evaluator.prediction_index(
                covidet_evaluator.load_jsonl(predictions)
            )
        self.assertEqual((valid, invalid), (1, 0))
        self.assertEqual(indexed[sample["id"]]["emotion"]["labels"], ["fear", "joy"])

    def test_crowd_converter_selects_larger_baseline_intensity(self) -> None:
        eval_file = REPO_ROOT / "FirstPersonMethod/data/crowd_envent/test.json"
        all_samples = crowd_evaluator.load_evaluation_file(eval_file)["samples"]
        sample = crowd_evaluator.select_samples(all_samples, 1, 42)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_json(
                root / "baseline" / "chain-appraisals" / f"{sample['id']}.json",
                _appraisal_trajectory(),
            )
            _write_json(
                root / "baseline" / "chain-emotion" / f"{sample['id']}.json",
                {
                    "appraisal_reasoning": _reasoning(),
                    "emotion": {
                        "positive_intensity": 2,
                        "negative_intensity": 6,
                        "positive_labels": ["joy"],
                        "negative_labels": ["sadness"],
                    },
                },
            )
            predictions = root / "predictions.jsonl"
            valid, invalid = crowd_converter.convert_outputs(
                eval_file, root / "baseline", predictions, max_samples=1
            )
            indexed = crowd_evaluator.prediction_index(
                crowd_evaluator.load_jsonl(predictions)
            )
        self.assertEqual((valid, invalid), (1, 0))
        self.assertEqual(
            indexed[sample["id"]]["emotion"],
            {"label": "sadness", "intensity": 5},
        )


if __name__ == "__main__":
    unittest.main()
