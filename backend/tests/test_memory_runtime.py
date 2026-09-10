from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


sqlite_vec_stub = ModuleType("sqlite_vec")
sqlite_vec_stub.load = lambda _conn: None
sqlite_vec_stub.serialize_float32 = lambda _vec: b""
sys.modules.setdefault("sqlite_vec", sqlite_vec_stub)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import AgentManager
from runtime.memory_runtime import MemoryRuntime


class MemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def test_agent_manager_exposes_runtime_owned_memory_collections(self) -> None:
        manager = AgentManager()

        self.assertIs(manager.mem_stores, manager._memory_runtime.stores)
        self.assertIs(manager.mem_embedders, manager._memory_runtime.embedders)
        self.assertIs(manager.mem_workers, manager._memory_runtime.workers)
        self.assertIs(manager.mem_recalls, manager._memory_runtime.recalls)

        replacement_stores = {"main": object()}
        replacement_embedders = {"main": object()}
        replacement_workers = {"main": object()}
        replacement_recalls = {"main": object()}
        manager.mem_stores = replacement_stores
        manager.mem_embedders = replacement_embedders
        manager.mem_workers = replacement_workers
        manager.mem_recalls = replacement_recalls

        self.assertIs(manager.mem_stores, replacement_stores)
        self.assertIs(manager.mem_embedders, replacement_embedders)
        self.assertIs(manager.mem_workers, replacement_workers)
        self.assertIs(manager.mem_recalls, replacement_recalls)
        self.assertIs(manager._memory_runtime.stores, replacement_stores)
        self.assertIs(manager._memory_runtime.embedders, replacement_embedders)
        self.assertIs(manager._memory_runtime.workers, replacement_workers)
        self.assertIs(manager._memory_runtime.recalls, replacement_recalls)

    async def test_initialize_agent_wires_memory_components(self) -> None:
        runtime = MemoryRuntime()
        store = object()
        embedder = object()
        task_processor = SimpleNamespace(on_chunks_ingested=AsyncMock())
        worker = SimpleNamespace(enqueue=AsyncMock())
        recall = object()
        mem_config = {
            "enabled": True,
            "storage": {"db_path": "/tmp/pipixia-memory/memory.db"},
            "embedding": {"dimensions": 8},
            "skill_evolution": {"enabled": True},
        }

        with (
            patch("runtime.memory_runtime.resolve_mem_config", return_value=mem_config),
            patch("mem.store.MemStore", return_value=store) as store_factory,
            patch("mem.embedder.MemEmbedder.from_config", return_value=embedder),
            patch(
                "mem.task_processor.MemTaskProcessor.from_config",
                return_value=task_processor,
            ) as task_processor_factory,
            patch(
                "mem.worker.MemWorker.from_config",
                return_value=worker,
            ) as worker_factory,
            patch("mem.recall.MemRecall.from_config", return_value=recall),
        ):
            runtime.initialize_agent("main")

        store_factory.assert_called_once_with(
            db_path="/tmp/pipixia-memory/memory.db",
            dimensions=8,
        )
        self.assertIs(runtime.stores["main"], store)
        self.assertIs(runtime.embedders["main"], embedder)
        self.assertIs(runtime.workers["main"], worker)
        self.assertIs(runtime.recalls["main"], recall)

        self.assertNotIn("on_task_completed", task_processor_factory.call_args.kwargs)

        on_chunks_ingested = worker_factory.call_args.kwargs[
            "on_chunks_ingested"
        ]
        await on_chunks_ingested("s1", True)
        task_processor.on_chunks_ingested.assert_awaited_once_with(
            "s1",
            True,
            owner="main",
        )

    def test_initialize_agent_does_not_enable_online_skill_evolution_by_default(self) -> None:
        runtime = MemoryRuntime()
        mem_config = {
            "enabled": True,
            "storage": {"db_path": "/tmp/pipixia-memory/memory.db"},
            "embedding": {"dimensions": 8},
        }
        task_processor = SimpleNamespace(on_chunks_ingested=AsyncMock())
        with (
            patch("runtime.memory_runtime.resolve_mem_config", return_value=mem_config),
            patch("mem.store.MemStore", return_value=object()),
            patch("mem.embedder.MemEmbedder.from_config", return_value=object()),
            patch("mem.task_processor.MemTaskProcessor.from_config", return_value=task_processor) as processor_factory,
            patch("mem.worker.MemWorker.from_config", return_value=object()),
            patch("mem.recall.MemRecall.from_config", return_value=object()),
        ):
            runtime.initialize_agent("main")

        processor_factory.assert_called_once()
        self.assertNotIn("on_task_completed", processor_factory.call_args.kwargs)

    def test_initialize_failure_closes_temporary_store_without_publishing(
        self,
    ) -> None:
        runtime = MemoryRuntime()
        store = SimpleNamespace(close=Mock())
        mem_config = {
            "enabled": True,
            "storage": {"db_path": "/tmp/pipixia-memory/memory.db"},
            "embedding": {"dimensions": 8},
        }

        with (
            patch("runtime.memory_runtime.resolve_mem_config", return_value=mem_config),
            patch("mem.store.MemStore", return_value=store),
            patch(
                "mem.embedder.MemEmbedder.from_config",
                side_effect=RuntimeError("embedder failed"),
            ),
        ):
            runtime.initialize_agent("main")

        store.close.assert_called_once_with()
        self.assertEqual(runtime.stores, {})
        self.assertEqual(runtime.embedders, {})
        self.assertEqual(runtime.workers, {})
        self.assertEqual(runtime.recalls, {})

    def test_reinitialize_replaces_components_and_closes_previous_store(
        self,
    ) -> None:
        runtime = MemoryRuntime()
        old_store = SimpleNamespace(close=Mock())
        new_store = SimpleNamespace(close=Mock())
        runtime.stores["main"] = old_store
        runtime.embedders["main"] = object()
        runtime.workers["main"] = object()
        runtime.recalls["main"] = object()
        mem_config = {
            "enabled": True,
            "storage": {"db_path": "/tmp/pipixia-memory/memory.db"},
            "embedding": {"dimensions": 8},
        }
        new_embedder = object()
        new_worker = object()
        new_recall = object()

        with (
            patch("runtime.memory_runtime.resolve_mem_config", return_value=mem_config),
            patch("mem.store.MemStore", return_value=new_store),
            patch(
                "mem.embedder.MemEmbedder.from_config",
                return_value=new_embedder,
            ),
            patch(
                "mem.task_processor.MemTaskProcessor.from_config",
                return_value=SimpleNamespace(on_chunks_ingested=AsyncMock()),
            ),
            patch(
                "mem.worker.MemWorker.from_config",
                return_value=new_worker,
            ),
            patch(
                "mem.recall.MemRecall.from_config",
                return_value=new_recall,
            ),
        ):
            runtime.initialize_agent("main")

        old_store.close.assert_called_once_with()
        new_store.close.assert_not_called()
        self.assertIs(runtime.stores["main"], new_store)
        self.assertIs(runtime.embedders["main"], new_embedder)
        self.assertIs(runtime.workers["main"], new_worker)
        self.assertIs(runtime.recalls["main"], new_recall)

    def test_disabled_memory_removes_existing_agent_components(self) -> None:
        runtime = MemoryRuntime()
        store = SimpleNamespace(close=Mock())
        runtime.stores["main"] = store
        runtime.embedders["main"] = object()
        runtime.workers["main"] = object()
        runtime.recalls["main"] = object()

        with patch(
            "runtime.memory_runtime.resolve_mem_config",
            return_value={"enabled": False},
        ):
            runtime.initialize_agent("main")

        store.close.assert_called_once_with()
        self.assertNotIn("main", runtime.stores)
        self.assertNotIn("main", runtime.embedders)
        self.assertNotIn("main", runtime.workers)
        self.assertNotIn("main", runtime.recalls)

    async def test_ingest_turn_enqueues_user_and_assistant_with_shared_turn_id(
        self,
    ) -> None:
        runtime = MemoryRuntime()
        worker = SimpleNamespace(enqueue=AsyncMock())
        runtime.workers["main"] = worker

        await runtime.ingest_turn(
            "main", "s1", " user ", " assistant ", turn_id="run-1"
        )

        batch = worker.enqueue.await_args.args[0]
        self.assertEqual([item.role for item in batch], ["user", "assistant"])
        self.assertEqual([item.content for item in batch], ["user", "assistant"])
        self.assertEqual({item.turn_id for item in batch}, {"run-1"})
        self.assertEqual({item.owner for item in batch}, {"main"})
        self.assertEqual({item.task_id for item in batch}, {None})
        self.assertEqual({item.source_type for item in batch}, {"main_agent"})
        worker.enqueue.assert_awaited_once_with(batch, session_end=False)

    async def test_ingest_turn_binds_subagent_messages_to_parent_task(self) -> None:
        runtime = MemoryRuntime()
        worker = SimpleNamespace(enqueue=AsyncMock())
        runtime.workers["worker"] = worker

        await runtime.ingest_turn(
            "worker",
            "subagent-session",
            "inspect",
            "done",
            turn_id="run-1",
            parent_task_id="task-main",
        )

        batch = worker.enqueue.await_args.args[0]
        self.assertEqual({item.task_id for item in batch}, {"task-main"})
        self.assertEqual({item.parent_task_id for item in batch}, {"task-main"})
        self.assertEqual({item.source_type for item in batch}, {"subagent"})

    async def test_ingest_messages_skips_system_and_empty_content(self) -> None:
        runtime = MemoryRuntime()
        worker = SimpleNamespace(enqueue=AsyncMock())
        runtime.workers["main"] = worker

        await runtime.ingest_messages(
            "main",
            "s1",
            [
                {"role": "system", "content": "internal"},
                {"role": "user", "content": "  "},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
            session_end=True,
        )

        batch = worker.enqueue.await_args.args[0]
        self.assertEqual([item.role for item in batch], ["user", "assistant"])
        self.assertEqual([item.content for item in batch], ["hello", "world"])
        worker.enqueue.assert_awaited_once_with(batch, session_end=True)

    def test_close_releases_stores_and_clears_all_components(self) -> None:
        runtime = MemoryRuntime()
        first_store = SimpleNamespace(close=Mock(side_effect=RuntimeError("failed")))
        second_store = SimpleNamespace(close=Mock())
        runtime.stores.update({"main": first_store, "writer": second_store})
        runtime.embedders["main"] = object()
        runtime.workers["main"] = object()
        runtime.recalls["main"] = object()

        runtime.close()

        first_store.close.assert_called_once_with()
        second_store.close.assert_called_once_with()
        self.assertEqual(runtime.stores, {})
        self.assertEqual(runtime.embedders, {})
        self.assertEqual(runtime.workers, {})
        self.assertEqual(runtime.recalls, {})


if __name__ == "__main__":
    unittest.main()
