from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skill_candidate import build_candidate_record, save_candidate_manifest
from evaluation.skill_dev_evaluation import evaluate_dataset_split, evaluate_dev, write_dev_report


def _sample(sample_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "task_family": "demo",
        "task_snapshot": {"goal": "完成演示任务"},
        "trajectory": [{"role": "user", "content": "完成演示任务"}],
    }


class SkillDevEvaluationTests(unittest.TestCase):
    def _candidate(self, root: Path) -> Path:
        skill = root / "candidate" / "SKILL.md"
        skill.parent.mkdir()
        skill.write_text(
            '---\nname: demo\ndescription: "Do demo workflow"\n---\n\n'
            "# Demo\n\n## Steps\n1. Do the verified steps.\n",
            encoding="utf-8",
        )
        record = build_candidate_record(
            family="demo",
            source_samples=[_sample("seed-1")],
            skill_id="demo",
            version=2,
            skill_path=skill,
        )
        candidate_manifest = root / "candidate.json"
        save_candidate_manifest(candidate_manifest, record)
        return candidate_manifest

    def test_runs_three_variant_dev_comparison_and_writes_gate_compatible_report(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            candidate_manifest = self._candidate(root)
            active = root / "active" / "SKILL.md"
            active.parent.mkdir()
            active.write_text('---\nname: active\ndescription: "Active workflow"\n---\n\n## Steps\n1. Work.\n', encoding="utf-8")
            samples = [_sample("seed-1"), _sample("dev-1"), _sample("holdout-1")]
            manifest = {
                "schema_version": "1.0",
                "families": [{"task_family": "demo", "splits": {
                    "seed": ["seed-1"], "dev": ["dev-1"], "holdout": ["holdout-1"], "regression": [],
                }}],
            }
            calls: list[tuple[str, str, Path | None]] = []

            async def executor(sample: dict, variant: str, skill_path: Path | None) -> dict:
                calls.append((sample["sample_id"], variant, skill_path))
                return {"passed": variant != "without_skill", "tokens": 10, "duration_ms": 5}

            report = asyncio.run(evaluate_dev(
                candidate_manifest=candidate_manifest,
                dataset_manifest=manifest,
                samples=samples,
                executor=executor,
                active_skill_path=active,
            ))
            self.assertEqual([row["system"] for row in report["systems"]], [
                "without_skill", "active_skill", "candidate_skill",
            ])
            self.assertEqual(report["systems"][0]["pass_rate"], 0.0)
            self.assertEqual(report["systems"][2]["pass_rate"], 1.0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][2], None)
            self.assertEqual(calls[2][2].name, "SKILL.md")
            report_path = root / "reports" / "dev.json"
            write_dev_report(report_path, report)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["stage"], "dev")

    def test_executor_error_is_recorded_as_external_failure(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            candidate_manifest = self._candidate(root)
            samples = [_sample("seed-1"), _sample("dev-1")]
            manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
                "seed": ["seed-1"], "dev": ["dev-1"], "holdout": [], "regression": [],
            }}]}

            def executor(*_args: object) -> bool:
                raise TimeoutError("model timed out")

            report = asyncio.run(evaluate_dev(
                candidate_manifest=candidate_manifest,
                dataset_manifest=manifest,
                samples=samples,
                executor=executor,
            ))
            candidate = next(row for row in report["systems"] if row["system"] == "candidate_skill")
            self.assertEqual(candidate["external_failures"], 1)
            self.assertEqual(candidate["evaluated_cases"], 0)

    def test_regression_reuses_the_same_comparison_contract(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            candidate_manifest = self._candidate(root)
            samples = [_sample("seed-1"), _sample("regression-1")]
            manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
                "seed": ["seed-1"], "dev": [], "holdout": [], "regression": ["regression-1"],
            }}]}

            report = asyncio.run(evaluate_dataset_split(
                candidate_manifest=candidate_manifest,
                dataset_manifest=manifest,
                samples=samples,
                executor=lambda *_: True,
                split="regression",
            ))

            self.assertEqual(report["stage"], "regression")
            self.assertEqual(report["regression_sample_ids"], ["regression-1"])
            self.assertEqual(report["systems"][-1]["pass_rate"], 1.0)

    def test_candidate_source_outside_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            candidate_manifest = self._candidate(root)
            data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            data["source_sample_ids"] = ["dev-1"]
            candidate_manifest.write_text(json.dumps(data), encoding="utf-8")
            samples = [_sample("seed-1"), _sample("dev-1")]
            manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
                "seed": ["seed-1"], "dev": ["dev-1"], "holdout": [], "regression": [],
            }}]}
            with self.assertRaisesRegex(ValueError, "must belong to Seed"):
                asyncio.run(evaluate_dev(
                    candidate_manifest=candidate_manifest,
                    dataset_manifest=manifest,
                    samples=samples,
                    executor=lambda *_: True,
                ))


if __name__ == "__main__":
    unittest.main()
