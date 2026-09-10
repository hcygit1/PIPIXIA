from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.skill_evolution_runner import gate_report, run_dev_evaluation


class SkillEvolutionRunnerTests(unittest.TestCase):
    def test_gate_report_accepts_three_variant_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({"systems": [
                {"system": "without_skill", "pass_rate": 0.2, "avg_tokens": 100},
                {"system": "active_skill", "pass_rate": 0.5, "avg_tokens": 110},
                {"system": "candidate_skill", "pass_rate": 0.7, "avg_tokens": 120},
            ]}), encoding="utf-8")
            result = gate_report(path, static_check_passed=True, regression_candidate_passed=True)
            self.assertEqual(result["status"], "accepted")

    def test_summarized_external_failure_blocks_release(self) -> None:
        from evaluation.skilllearnbench_runner import summarize_trials

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = (
                ("no_skill", {"passed": False}),
                ("candidate", {"passed": True}),
                ("candidate", {"status": "error", "error": "request timed out"}),
            )
            for index, (skill_config, values) in enumerate(rows):
                trial = root / skill_config / f"trial-{index}"
                trial.mkdir(parents=True)
                (trial / "result.json").write_text(json.dumps({
                    "skill_config": skill_config,
                    **values,
                }), encoding="utf-8")

            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(summarize_trials(root)),
                encoding="utf-8",
            )
            result = gate_report(
                report_path,
                static_check_passed=True,
                regression_candidate_passed=True,
            )

        self.assertEqual(result["status"], "external_failure")
        self.assertEqual(result["classification"], "external_failure")
        self.assertEqual(result["next_action"], "repair_environment_and_rerun")

    def test_gate_report_routes_candidate_failure_to_skill_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({"systems": [
                {"system": "without_skill", "pass_rate": 0.8},
                {"system": "candidate_skill", "pass_rate": 0.4},
            ], "cases": [
                {"sample_id": "dev-2", "variant": "without_skill", "passed": True},
                {"sample_id": "dev-2", "variant": "candidate_skill", "passed": False},
            ]}), encoding="utf-8")
            result = gate_report(path, static_check_passed=True, regression_candidate_passed=True)
        self.assertEqual(result["classification"], "skill_defect")
        self.assertEqual(result["next_action"], "create_local_candidate_revision")
        self.assertEqual(result["evidence_sample_ids"], ["dev-2"])

    def test_dev_entry_validates_candidate_runs_adapter_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            skill = root / "candidate" / "SKILL.md"
            skill.parent.mkdir()
            skill.write_text(
                '---\nname: demo\ndescription: "Do demo workflow"\n---\n\n'
                "## Steps\n1. Work.\n",
                encoding="utf-8",
            )
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({
                "candidate_id": "candidate-1", "skill_id": "demo", "version": 2,
                "task_family": "demo", "source_sample_ids": ["seed-1"],
                "skill_path": str(skill), "status": "ready_for_dev",
                "checks": {"passed": True, "reasons": [], "warnings": []},
            }), encoding="utf-8")
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps({"families": [{"family": "demo"}]}), encoding="utf-8")
            trials = root / "trials"
            report_path = root / "reports" / "dev.json"
            summary = {"systems": [
                {"system": "without_skill", "pass_rate": 0.0},
                {"system": "candidate_skill", "pass_rate": 1.0},
            ]}
            with patch("evaluation.skilllearnbench_runner.evaluate", return_value=0) as evaluate_mock, \
                patch("evaluation.skilllearnbench_runner.summarize_trials", return_value=summary) as summary_mock:
                report = run_dev_evaluation(
                    benchmark_manifest_path=benchmark,
                    candidate_manifest_path=candidate,
                    family="demo",
                    report_path=report_path,
                    trials_dir=trials,
                    dry_run=True,
                )
            evaluate_mock.assert_called_once()
            summary_mock.assert_called_once_with(trials.resolve())
            self.assertEqual(report["stage"], "dev")
            self.assertEqual(report["systems"], summary["systems"])
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
