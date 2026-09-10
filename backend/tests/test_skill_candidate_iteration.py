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
from evaluation.skill_candidate_iteration import (
    IterationPolicy,
    create_local_revision,
    iterate_skill_candidate,
    promote_failures_to_regression,
)
from evaluation.skill_release_gate import ReleaseDecision


def _sample(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "task_snapshot": {"goal": "完成演示任务"},
        "trajectory": [{"role": "user", "content": "完成演示任务"}],
    }


def _candidate(root: Path, *, iteration: int = 0) -> Path:
    skill = root / f"candidate-{iteration}" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text('---\nname: demo\ndescription: "Demo workflow"\n---\n\n## Steps\n1. Work.\n', encoding="utf-8")
    record = build_candidate_record(
        family="demo", source_samples=[_sample("seed-1")], skill_id="demo",
        version=iteration + 1, skill_path=skill,
    )
    path = root / f"candidate-{iteration}.json"
    save_candidate_manifest(path, record)
    return path


class SkillCandidateIterationTests(unittest.TestCase):
    def test_failures_move_from_dev_to_regression(self) -> None:
        manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
            "seed": ["seed-1"], "dev": ["dev-1", "dev-2"], "holdout": ["holdout-1"], "regression": [],
        }}]}
        updated = promote_failures_to_regression(manifest, family="demo", sample_ids=["dev-2"])
        splits = updated["families"][0]["splits"]
        self.assertEqual(splits["dev"], ["dev-1"])
        self.assertEqual(splits["regression"], ["dev-2"])
        self.assertEqual(manifest["families"][0]["splits"]["dev"], ["dev-1", "dev-2"])

    def test_create_local_revision_records_parent_and_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            parent = _candidate(root)
            report = {
                "task_family": "demo",
                "systems": [{"system": "without_skill", "external_failures": 0}, {"system": "candidate_skill", "external_failures": 0}],
                "cases": [
                    {"sample_id": "dev-1", "variant": "without_skill", "passed": True},
                    {"sample_id": "dev-1", "variant": "candidate_skill", "passed": False},
                ],
            }

            async def llm(_prompt: str) -> str:
                return '---\nname: demo\ndescription: "Improved demo workflow"\n---\n\n## Steps\n1. Work safely.\n'

            child_path = root / "child.json"
            child = asyncio.run(create_local_revision(
                parent_manifest=parent, report=report, samples=[_sample("dev-1")],
                evolution_root=root / "skill_evolution", output_manifest=child_path, llm_call=llm,
            ))
            self.assertTrue(child.checks.passed)
            self.assertEqual(child.parent_candidate_id, json.loads(parent.read_text())["candidate_id"])
            self.assertEqual(child.failure_sample_ids, ("dev-1",))
            self.assertEqual(child.iteration, 1)
            self.assertTrue(child_path.is_file())
            self.assertTrue((root / "skill_evolution" / "demo" / "candidates" / "v1" / "candidate.json").is_file())

    def test_iteration_limit_abandons_the_evolution_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            parent = _candidate(root)
            report = {
                "task_family": "demo",
                "systems": [{"system": "without_skill", "external_failures": 0}, {"system": "candidate_skill", "external_failures": 0, "pass_rate": 0.1}],
                "cases": [
                    {"sample_id": "dev-1", "variant": "without_skill", "passed": True},
                    {"sample_id": "dev-1", "variant": "candidate_skill", "passed": False},
                ],
            }
            manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
                "seed": ["seed-1"], "dev": ["dev-1"], "holdout": [], "regression": [],
            }}]}
            saved: list[dict] = []
            async def evaluate(_candidate: Path, updated: dict) -> tuple[dict, ReleaseDecision]:
                return report, ReleaseDecision("rejected", ("candidate failed",))
            async def llm(_prompt: str) -> str:
                return '---\nname: demo\ndescription: "Improved demo workflow"\n---\n\n## Steps\n1. Work safely.\n'

            result = asyncio.run(iterate_skill_candidate(
                candidate_manifest=parent, dataset_manifest=manifest, samples=[_sample("dev-1")],
                initial_report=report, initial_decision=ReleaseDecision("rejected", ("candidate failed",)),
                evolution_root=root / "skill_evolution", manifest_writer=saved.append,
                evaluate_candidate=evaluate, llm_call=llm,
                policy=IterationPolicy(max_iterations=1, no_improvement_limit=1),
            ))
            self.assertEqual(result.status, "abandoned")
            self.assertEqual(
                result.attribution.next_action,
                "retain_artifacts_and_wait_for_more_samples",
            )
            self.assertGreaterEqual(len(saved), 1)
            state = json.loads((root / "skill_evolution" / "demo" / "iteration-state.json").read_text())
            self.assertEqual(state["status"], "abandoned")

    def test_static_repair_exhaustion_abandons_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            parent = _candidate(root)
            report = {
                "task_family": "demo",
                "systems": [
                    {"system": "without_skill", "external_failures": 0},
                    {"system": "candidate_skill", "external_failures": 0, "pass_rate": 0.1},
                ],
                "cases": [
                    {"sample_id": "dev-1", "variant": "without_skill", "passed": True},
                    {"sample_id": "dev-1", "variant": "candidate_skill", "passed": False},
                ],
            }
            manifest = {"schema_version": "1.0", "families": [{"task_family": "demo", "splits": {
                "seed": ["seed-1"], "dev": ["dev-1"], "holdout": [], "regression": [],
            }}]}
            llm_calls = 0

            def llm(_prompt: str) -> str:
                nonlocal llm_calls
                llm_calls += 1
                return "still invalid"

            def evaluate(_candidate: Path, _manifest: dict) -> tuple[dict, ReleaseDecision]:
                raise AssertionError("static-invalid Candidate must not enter evaluation")

            result = asyncio.run(iterate_skill_candidate(
                candidate_manifest=parent,
                dataset_manifest=manifest,
                samples=[_sample("dev-1")],
                initial_report=report,
                initial_decision=ReleaseDecision("rejected", ("candidate failed",)),
                evolution_root=root / "skill_evolution",
                manifest_writer=lambda _value: None,
                evaluate_candidate=evaluate,
                llm_call=llm,
            ))

            self.assertEqual(result.status, "abandoned")
            self.assertEqual(result.attribution.classification, "static_repair_exhausted")
            self.assertEqual(llm_calls, 3)  # one local revision plus two format repairs


if __name__ == "__main__":
    unittest.main()
