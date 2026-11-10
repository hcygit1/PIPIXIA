from __future__ import annotations

import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class SubagentRunModelTests(unittest.TestCase):
    def test_registry_state_imports_without_registry_cycle(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "import subagents.subagent_registry_state; "
                    "assert 'subagents.subagent_registry' not in sys.modules; "
                    "assert 'subagents.subagent_run_store' not in sys.modules"
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registry_reexports_domain_model(self) -> None:
        from subagents.subagent_registry import (
            SubagentCapacityError as RegistryCapacityError,
            SubagentRunRecord as RegistryRunRecord,
        )
        from subagents.subagent_run_model import (
            SubagentCapacityError,
            SubagentRunRecord,
        )

        self.assertIs(RegistryCapacityError, SubagentCapacityError)
        self.assertIs(RegistryRunRecord, SubagentRunRecord)

    def test_record_fields_defaults_and_persistence_round_trip(self) -> None:
        from subagents.subagent_registry_state import (
            _dict_to_record,
            _record_to_dict,
        )
        from subagents.subagent_run_model import SubagentRunRecord

        self.assertEqual(
            [item.name for item in fields(SubagentRunRecord)],
            [
                "run_id",
                "child_session_key",
                "requester_session_key",
                "requester_agent_id",
                "target_agent_id",
                "task",
                "label",
                "model",
                "parent_task_id",
                "cleanup",
                "spawn_depth",
                "created_at",
                "started_at",
                "ended_at",
                "outcome",
                "result_summary",
                "asyncio_task",
                "archive_at_ms",
                "announce_retry_count",
                "last_announce_retry_at",
                "state",
                "terminal_reason",
                "result_delivery_state",
                "delivery_work_id",
            ],
        )
        defaults = SubagentRunRecord("r", "c", "p", "a", "t", "task")
        self.assertEqual(defaults.cleanup, "keep")
        self.assertEqual(defaults.spawn_depth, 0)
        self.assertIsNone(defaults.asyncio_task)
        self.assertEqual(defaults.state, "running")
        self.assertEqual(defaults.result_delivery_state, "pending")

        runtime_task = object()
        record = SubagentRunRecord(
            run_id="run-1",
            child_session_key="child",
            requester_session_key="requester",
            requester_agent_id="main",
            target_agent_id="worker",
            task="work",
            label="label",
            model="model",
            cleanup="delete",
            spawn_depth=2,
            created_at=1.0,
            started_at=2.0,
            ended_at=3.0,
            outcome="completed",
            result_summary="result",
            asyncio_task=runtime_task,
            archive_at_ms=4.0,
            announce_retry_count=1,
            last_announce_retry_at=5.0,
            state="done",
            terminal_reason="finished",
            result_delivery_state="delivered",
            delivery_work_id="work-1",
        )

        serialized = _record_to_dict(record)
        restored = _dict_to_record(serialized)

        self.assertNotIn("asyncio_task", serialized)
        self.assertIsNone(restored.asyncio_task)
        for field in fields(SubagentRunRecord):
            if field.name != "asyncio_task":
                self.assertEqual(
                    getattr(restored, field.name),
                    getattr(record, field.name),
                )


if __name__ == "__main__":
    unittest.main()
