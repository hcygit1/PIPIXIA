from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skill_evolution_workflow import build_workflow


class SkillEvolutionWorkflowTests(unittest.TestCase):
    def test_stages_are_persistent_and_transitions_are_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            workflow = build_workflow(Path(root))
            task = workflow.create(agent_id="main", task_family="demo")
            self.assertEqual(task["data_source"], "langfuse")
            task = workflow.transition(
                task["id"], agent_id="main", stage="data_exported",
                artifacts={"samples": "samples.jsonl"},
            )
            self.assertEqual(task["stage"], "data_exported")
            self.assertEqual(task["artifacts"]["samples"], "samples.jsonl")
            with self.assertRaises(ValueError):
                workflow.transition(task["id"], agent_id="main", stage="published")
            reloaded = build_workflow(Path(root)).get(task["id"])
            self.assertEqual(reloaded["stage"], "data_exported")
            self.assertEqual(len(reloaded["history"]), 2)

    def test_execution_result_survives_refresh_without_advancing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            workflow = build_workflow(Path(root))
            task = workflow.create(agent_id="main", task_family="demo")
            workflow.record_execution(
                task["id"], agent_id="main", operation="export_traces", status="running",
            )
            workflow.record_execution(
                task["id"], agent_id="main", operation="export_traces", status="failed",
                error="Langfuse unavailable",
            )
            reloaded = build_workflow(Path(root)).get(task["id"])
            self.assertEqual(reloaded["stage"], "created")
            self.assertEqual(reloaded["execution"]["status"], "failed")
            self.assertEqual(reloaded["execution"]["error"], "Langfuse unavailable")

    def test_persists_skilllearnbench_source(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            workflow = build_workflow(Path(root))
            task = workflow.create(
                agent_id="main", task_family="demo",
                data_source="skilllearnbench",
            )
            self.assertEqual(
                build_workflow(Path(root)).get(task["id"])["data_source"],
                "skilllearnbench",
            )


if __name__ == "__main__":
    unittest.main()
