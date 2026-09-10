from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skill_candidate import build_candidate_record, save_candidate_manifest
from evaluation.skill_evolution_dev_editor import run_dev


class SkillEvolutionDevEditorTests(unittest.TestCase):
    def test_skilllearnbench_uses_only_dev_instances_and_official_summary(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            dataset = root / "datasets" / "v1"
            dataset.mkdir(parents=True)
            manifest = {
                "schema_version": "1.0", "source": "skilllearnbench",
                "families": [{"task_family": "demo", "splits": {
                    "seed": ["demo-1"], "dev": ["demo-2"],
                    "holdout": ["demo-3"], "regression": [],
                }}],
            }
            (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            samples = [
                {"sample_id": f"demo-{index}", "task_snapshot": {"instruction": "do"}, "trajectory": []}
                for index in range(1, 4)
            ]
            (dataset / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8",
            )
            skill = root / "candidates" / "v1" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text(
                '---\nname: demo\ndescription: "Demo workflow"\n---\n\n## Steps\n1. Work carefully.\n',
                encoding="utf-8",
            )
            record = build_candidate_record(
                family="demo", source_samples=[{"sample_id": "demo-1"}],
                skill_id="demo-skill", version=1, skill_path=skill,
            )
            candidate_manifest = skill.parent / "candidate.json"
            save_candidate_manifest(candidate_manifest, record)
            benchmark_manifest = root / "benchmark.json"
            benchmark_manifest.write_text(json.dumps({
                "benchmark_root": str(root / "benchmark"),
                "families": [{"family": "demo", "evaluation_instances": [
                    {"instance_id": "demo-2"}, {"instance_id": "demo-3"},
                ]}],
            }), encoding="utf-8")
            task = {
                "task_family": "demo", "agent_id": "main",
                "data_source": "skilllearnbench",
                "artifacts": {"benchmark_manifest": str(benchmark_manifest)},
            }
            summary = {
                "systems": [
                    {"system": "without_skill", "pass_rate": 0.0},
                    {"system": "candidate_skill", "pass_rate": 1.0},
                ],
                "cases": [{"sample_id": "demo/demo-2", "variant": "candidate_skill", "passed": True}],
            }

            def fake_evaluate(_manifest, **kwargs):
                self.assertEqual(kwargs["instance_ids"], ["demo-2"])
                self.assertTrue((kwargs["skill_root"] / "demo" / "SKILL.md").is_file())
                return 1

            with patch("evaluation.skilllearnbench_runner.evaluate", side_effect=fake_evaluate), patch(
                "evaluation.skilllearnbench_runner.summarize_trials", return_value=summary,
            ):
                report = asyncio.run(run_dev(
                    root, task, candidate_manifest=candidate_manifest,
                ))

            self.assertEqual(report["evaluator"], "skilllearnbench_official_verifier")
            self.assertEqual(report["dev_sample_ids"], ["demo-2"])


if __name__ == "__main__":
    unittest.main()
