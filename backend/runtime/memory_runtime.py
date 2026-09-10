"""Memory component lifecycle and ingestion for Agent runtime."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from config import resolve_mem_config


logger = logging.getLogger("runtime.agent")


class MemoryRuntime:
    def __init__(self) -> None:
        self.stores: dict[str, Any] = {}
        self.embedders: dict[str, Any] = {}
        self.workers: dict[str, Any] = {}
        self.recalls: dict[str, Any] = {}
        self.task_processors: dict[str, Any] = {}

    @staticmethod
    def _close_store(agent_id: str, store: Any) -> None:
        try:
            store.close()
        except Exception as e:
            logger.warning("Failed to close mem store for %s: %s", agent_id, e)

    def _remove_agent(self, agent_id: str) -> None:
        store = self.stores.pop(agent_id, None)
        self.embedders.pop(agent_id, None)
        self.workers.pop(agent_id, None)
        self.recalls.pop(agent_id, None)
        self.task_processors.pop(agent_id, None)
        if store is not None:
            self._close_store(agent_id, store)

    @staticmethod
    def _infer_parent_task_id(agent_id: str, session_id: str) -> str | None:
        """Resolve a child session's inherited task for legacy/batch paths."""
        try:
            from sessions.session_identity import session_key_from_session_id
            from subagents.subagent_registry import registry

            child_key = session_key_from_session_id(agent_id, session_id)
            for record in registry.list_runs():
                if record.child_session_key == child_key:
                    return record.parent_task_id
        except Exception as exc:
            logger.debug("Unable to infer parent task for %s/%s: %s", agent_id, session_id, exc)
        return None

    def initialize_agent(self, agent_id: str) -> None:
        try:
            mem_cfg = resolve_mem_config()
        except Exception as e:
            logger.error("Failed to initialize mem system for %s: %s", agent_id, e)
            return

        if not mem_cfg.get("enabled", True):
            self._remove_agent(agent_id)
            logger.info("Mem system disabled for %s", agent_id)
            return

        store: Any | None = None
        try:
            from mem.embedder import MemEmbedder
            from mem.recall import MemRecall
            from mem.store import MemStore
            from mem.task_processor import MemTaskProcessor
            from mem.worker import MemWorker

            store = MemStore(
                db_path=mem_cfg["storage"]["db_path"],
                dimensions=mem_cfg.get("embedding", {}).get("dimensions", 1536),
            )

            embedder = MemEmbedder.from_config(mem_cfg.get("embedding", {}))

            task_processor = MemTaskProcessor.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
            )

            async def _on_chunks_ingested(
                session_key: str,
                session_end: bool,
            ) -> None:
                await task_processor.on_chunks_ingested(
                    session_key,
                    session_end,
                    owner=agent_id,
                )

            worker = MemWorker.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                on_chunks_ingested=_on_chunks_ingested,
            )

            recall = MemRecall.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                agent_id=agent_id,
            )
        except Exception as e:
            if store is not None:
                self._close_store(agent_id, store)
            logger.error("Failed to initialize mem system for %s: %s", agent_id, e)
            return

        previous_store = self.stores.get(agent_id)
        self.stores[agent_id] = store
        self.embedders[agent_id] = embedder
        self.workers[agent_id] = worker
        self.recalls[agent_id] = recall
        self.task_processors[agent_id] = task_processor
        if previous_store is not None and previous_store is not store:
            self._close_store(agent_id, previous_store)
        logger.info("Mem system initialized for agent %s", agent_id)

    async def ingest_turn(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
        turn_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        worker = self.workers.get(agent_id)
        if not worker:
            return
        try:
            from mem.worker import IngestMessage

            parent_task_id = parent_task_id or self._infer_parent_task_id(
                agent_id,
                session_id,
            )
            memory_turn_id = turn_id or str(uuid.uuid4())
            batch: list[IngestMessage] = []
            if user_content.strip():
                batch.append(
                    IngestMessage(
                        role="user",
                        content=user_content.strip(),
                        session_key=session_id,
                        turn_id=memory_turn_id,
                        owner=agent_id,
                        task_id=parent_task_id,
                        parent_task_id=parent_task_id,
                        source_type="subagent" if parent_task_id else "main_agent",
                    )
                )
            if assistant_content.strip():
                batch.append(
                    IngestMessage(
                        role="assistant",
                        content=assistant_content.strip(),
                        session_key=session_id,
                        turn_id=memory_turn_id,
                        owner=agent_id,
                        task_id=parent_task_id,
                        parent_task_id=parent_task_id,
                        source_type="subagent" if parent_task_id else "main_agent",
                    )
                )
            if batch:
                await worker.enqueue(batch, session_end=False)
        except Exception as e:
            logger.warning("incremental_ingest failed for %s: %s", agent_id, e)

    async def ingest_messages(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        session_end: bool = False,
        parent_task_id: str | None = None,
    ) -> None:
        worker = self.workers.get(agent_id)
        if not worker or not messages:
            return
        try:
            from mem.worker import IngestMessage

            parent_task_id = parent_task_id or self._infer_parent_task_id(
                agent_id,
                session_id,
            )
            batch: list[IngestMessage] = []
            for message in messages:
                content = message.get("content", "").strip()
                if not content:
                    continue
                role = message.get("role", "user")
                if role == "system":
                    continue
                batch.append(
                    IngestMessage(
                        role=role,
                        content=content,
                        session_key=session_id,
                        turn_id=str(uuid.uuid4()),
                        owner=agent_id,
                        task_id=parent_task_id,
                        parent_task_id=parent_task_id,
                        source_type="subagent" if parent_task_id else "main_agent",
                    )
                )
            if batch:
                await worker.enqueue(batch, session_end=session_end)
        except Exception as e:
            logger.warning("batch_ingest failed for %s: %s", agent_id, e)

    def get_active_task_id(self, agent_id: str, session_id: str) -> str | None:
        store = self.stores.get(agent_id)
        if not store:
            return None
        task = store.get_active_task_by_session(session_id, owner=agent_id)
        return task.id if task else None

    def get_or_create_active_task_id(
        self, agent_id: str, session_id: str,
    ) -> str | None:
        """Return the main-session task, creating it before a child is spawned."""
        store = self.stores.get(agent_id)
        if not store:
            return None
        task = store.get_active_task_by_session(session_id, owner=agent_id)
        if task:
            return task.id

        from mem.models import Task

        task = Task(
            id=str(uuid.uuid4()),
            session_key=session_id,
            owner=agent_id,
            title="",
            summary="",
            status="active",
        )
        store.insert_task(task)
        logger.info(
            "Created parent task=%s before child spawn session=%s",
            task.id,
            session_id,
        )
        return task.id

    def close(self) -> None:
        for agent_id, store in list(self.stores.items()):
            self._close_store(agent_id, store)

        self.stores.clear()
        self.embedders.clear()
        self.workers.clear()
        self.recalls.clear()
        self.task_processors.clear()
