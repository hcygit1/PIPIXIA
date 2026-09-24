from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from evaluation.langfuse_exporter import (
    export_langfuse_traces,
    export_traces,
    fetch_langfuse_traces,
    fetch_langfuse_sessions,
    trace_to_sample,
)


class LangfuseExporterTests(unittest.TestCase):
    def test_trace_maps_to_existing_sample_contract_and_redacts_secrets(self) -> None:
        sample = trace_to_sample({
            "id": "trace-1",
            "input": "整理 api_key=secret-value 的任务",
            "output": {"ok": True},
            "metadata": {
                "task_result": "success",
                "skill_id": "skill-a",
                "skill_versions": {"整理": "3"},
                "skill_names": ["整理"],
            },
        })

        self.assertTrue(sample["sample_id"].startswith("lf-"))
        self.assertEqual(sample["category"], "success")
        self.assertEqual(sample["skill_version"], "3")
        self.assertIn("[REDACTED]", sample["task_snapshot"])

    def test_export_traces_deduplicates_by_trace_id(self) -> None:
        trace = {"id": "trace-1", "metadata": {"task_result": "failed"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            count = export_traces([trace, trace], path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "bad_case")

    def test_datetime_is_serialized_as_iso_8601(self) -> None:
        created_at = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
        sample = trace_to_sample({"id": "trace-time", "timestamp": created_at})
        self.assertEqual(sample["created_at"], "2026-09-07T12:00:00+00:00")

    def test_fetch_supports_nested_api_and_passes_filters(self) -> None:
        class TraceApi:
            def __init__(self) -> None:
                self.calls = []

            def list(self, **kwargs):
                self.calls.append(kwargs)
                return {"data": [{"id": "trace-1", "metadata": {"task_result": "success"}}]}

        class Client:
            def __init__(self) -> None:
                self.api = type("Api", (), {"trace": TraceApi()})()

        client = Client()
        traces = fetch_langfuse_traces(client, limit=10, session_id="session-1")
        self.assertEqual(len(traces), 1)
        self.assertEqual(client.api.trace.calls[0]["session_id"], "session-1")

    def test_fetch_sessions_uses_native_sessions_api(self) -> None:
        class SessionsApi:
            def list(self, **kwargs):
                return {"data": [{"id": "session-1", "created_at": "2026-09-24T00:00:00Z", "trace_count": 3}]}

        client = type("Client", (), {"api": type("Api", (), {"sessions": SessionsApi()})()})()
        self.assertEqual(fetch_langfuse_sessions(client), [{"session_id": "session-1", "created_at": "2026-09-24T00:00:00Z", "trace_count": 3}])

    def test_export_langfuse_traces_writes_fetched_rows(self) -> None:
        class TraceApi:
            def list(self, **kwargs):
                return [{"id": "trace-1", "metadata": {"task_result": "success"}}]

        client = type("Client", (), {"api": type("Api", (), {"trace": TraceApi()})()})()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            count = export_langfuse_traces(client, path)
            rows = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)

    def test_memory_scope_replaces_full_history_with_task_chunks(self) -> None:
        task_chunks = [
            SimpleNamespace(id="c1", turn_id="run-1", role="user", content="问题", task_id="task-1", skill_id=None),
            SimpleNamespace(id="c2", turn_id="run-1", role="assistant", content="回答", task_id="task-1", skill_id="skill-1"),
        ]
        store = SimpleNamespace(
            get_chunks_by_turn=lambda session_id, turn_id: task_chunks,
            get_chunks_by_task=lambda task_id: task_chunks,
            get_task=lambda task_id: SimpleNamespace(title="任务", summary="摘要", status="completed"),
        )
        sample = trace_to_sample({
            "id": "trace-1",
            "input": {"messages": ["旧会话"]},
            "metadata": {
                "langfuse_session_id": "session-1",
                "pipixia_run_id": "run-1",
            },
        }, memory_store=store)

        self.assertEqual(sample["task_id"], "task-1")
        self.assertEqual(sample["skill_id"], "skill-1")
        self.assertEqual(len(sample["trajectory"]), 2)
        self.assertEqual(sample["task_snapshot"]["title"], "任务")



if __name__ == "__main__":
    unittest.main()
