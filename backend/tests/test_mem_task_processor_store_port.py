from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, get_type_hints


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _task_processor_store_port() -> type[Any]:
    try:
        from mem.task_processor_store import MemTaskProcessorStore
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.task_processor_store should own the task processor storage port"
        ) from exc
    return MemTaskProcessorStore


class MemTaskProcessorStorePortTests(unittest.TestCase):
    def test_importing_task_processor_does_not_load_store_or_sqlite_vec(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.task_processor; "
                    "blocked = sorted(name for name in sys.modules "
                    "if name in {'mem.store', 'sqlite_vec'}); "
                    "assert not blocked, blocked"
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_task_processor_store_port_matches_consumed_store_operations(
        self,
    ) -> None:
        MemTaskProcessorStore = _task_processor_store_port()
        expected = {
            "assign_chunks_to_task",
            "create_boundary_review",
            "finalize_task",
            "get_active_task_by_session",
            "get_all_active_tasks",
            "get_boundary_review",
            "get_chunks_by_task",
            "get_chunks_by_turn",
            "get_pending_boundary_review",
            "get_task",
            "get_unassigned_chunks",
            "insert_task",
            "orphan_chunk",
            "resolve_boundary_review",
            "update_task",
            "upsert_task_embedding",
        }
        operations = {
            name
            for name, value in vars(MemTaskProcessorStore).items()
            if not name.startswith("_") and callable(value)
        }

        self.assertTrue(getattr(MemTaskProcessorStore, "_is_protocol", False))
        self.assertEqual(operations, expected)

        from mem.store import MemStore

        for name in expected:
            self.assertEqual(
                str(inspect.signature(getattr(MemTaskProcessorStore, name))),
                str(inspect.signature(getattr(MemStore, name))),
                name,
            )

    def test_task_processor_depends_on_port_without_constructor_changes(
        self,
    ) -> None:
        processor_tree = ast.parse(
            (BACKEND_DIR / "mem" / "task_processor.py").read_text(
                encoding="utf-8"
            )
        )
        imports = {
            node.module: {alias.name for alias in node.names}
            for node in ast.walk(processor_tree)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("mem.store", imports)
        self.assertIn(
            "MemTaskProcessorStore",
            imports.get("mem.task_processor_store", set()),
        )

        from mem.task_processor import MemTaskProcessor

        signature = inspect.signature(MemTaskProcessor.__init__)
        self.assertEqual(
            list(signature.parameters),
            [
                "self",
                "store",
                "embedder",
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "idle_timeout_ms",
                "on_task_completed",
            ],
        )
        self.assertEqual(
            signature.parameters["store"].annotation,
            "MemTaskProcessorStore",
        )

    def test_factory_store_annotation_resolves_to_port(self) -> None:
        MemTaskProcessorStore = _task_processor_store_port()
        from mem.task_processor import MemTaskProcessor

        hints = get_type_hints(MemTaskProcessor.from_config)

        self.assertIs(hints["store"], MemTaskProcessorStore)


if __name__ == "__main__":
    unittest.main()
