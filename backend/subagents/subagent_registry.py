"""子 Agent 注册表"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from sessions.session_requester_resolver import session_requester_resolver
from subagents.subagent_relationships import SubagentRelationshipService
from subagents.subagent_run_archive import SubagentRunArchiveService
from subagents.subagent_run_lifecycle import SubagentRunLifecycleService
from subagents.subagent_run_model import (
    SubagentCapacityError,
    SubagentRunRecord,
)
from subagents.subagent_run_queries import SubagentRunQueryService
from subagents.subagent_run_state import SubagentRunStateService
from subagents.subagent_run_store import SubagentRunStore

def _resolve_archive_after_ms() -> float | None:
    """从 config 读取 archive_after_minutes，返回毫秒数"""
    try:
        from config import get_config
        cfg = get_config()
        minutes = cfg.get("agents", {}).get("defaults", {}).get("subagents", {}).get("archive_after_minutes", 60)
        if not isinstance(minutes, (int, float)) or minutes <= 0:
            return None
        return max(1, int(minutes)) * 60_000
    except Exception:
        return 60 * 60_000  # 默认 60 分钟


class SubagentRegistry:
    def __init__(self, store: SubagentRunStore | None = None):
        self._store = store or SubagentRunStore()
        self._state = SubagentRunStateService(
            store=self._store,
            persist=lambda: self._persist_to_disk(),
        )
        self._queries = SubagentRunQueryService(
            store=self._store,
            snapshot=self._snapshot_record,
        )
        self._relationships = SubagentRelationshipService(
            self._queries.list_runs
        )
        self._archive = SubagentRunArchiveService(
            store=self._store,
            state=self._state,
            persist=lambda: self._persist_to_disk(),
        )
        self._lifecycle = SubagentRunLifecycleService(
            store=self._store,
            state=self._state,
            relationships=self._relationships,
            persist=lambda: self._persist_to_disk(),
            record_factory=SubagentRunRecord,
            snapshot=self._snapshot_record,
            resolve_archive_after_ms=_resolve_archive_after_ms,
            capacity_error=SubagentCapacityError,
        )
        self._restore_from_disk()

    @property
    def _runs(self) -> dict[str, SubagentRunRecord]:
        return self._store.records

    @property
    def _lock(self):
        return self._store.lock

    def _restore_from_disk(self) -> None:
        self._store.restore()

    def _persist_to_disk(self) -> None:
        self._store.persist()

    @staticmethod
    def _snapshot_record(
        record: SubagentRunRecord,
    ) -> SubagentRunRecord:
        return replace(record, asyncio_task=None)

    def register_run(
        self,
        run_id: str,
        child_session_key: str,
        requester_session_key: str,
        requester_agent_id: str,
        target_agent_id: str,
        task: str,
        label: str | None = None,
        model: str | None = None,
        parent_task_id: str | None = None,
        cleanup: str = "keep",
        spawn_depth: int = 0,
        max_active_for_requester: int | None = None,
    ) -> SubagentRunRecord:
        return self._lifecycle.register_run(
            run_id=run_id,
            child_session_key=child_session_key,
            requester_session_key=requester_session_key,
            requester_agent_id=requester_agent_id,
            target_agent_id=target_agent_id,
            task=task,
            label=label,
            model=model,
            parent_task_id=parent_task_id,
            cleanup=cleanup,
            spawn_depth=spawn_depth,
            max_active_for_requester=max_active_for_requester,
        )

    def set_task(self, run_id: str, task: Any) -> bool:
        return self._lifecycle.set_task(run_id, task)

    def mark_started(self, run_id: str) -> None:
        self._state.mark_started(run_id)

    def mark_completed(
        self,
        run_id: str,
        result_summary: str = "",
        outcome: str = "completed",
        terminal_reason: str | None = None,
    ) -> None:
        self._state.mark_completed(
            run_id,
            result_summary=result_summary,
            outcome=outcome,
            terminal_reason=terminal_reason,
        )

    def mark_terminated(self, run_id: str, reason: str = "killed") -> None:
        self._state.mark_terminated(run_id, reason)

    def kill(self, run_id: str, cascade: bool = True) -> bool:
        """终止 run，cascade=True 时递归终止其子 runs"""
        return self._lifecycle.kill(run_id, cascade=cascade)

    def list_runs_for_requester(
        self, requester_key: str, include_recent_minutes: int = 30
    ) -> list[SubagentRunRecord]:
        return self._queries.list_runs_for_requester(
            requester_key,
            include_recent_minutes,
        )

    def count_active_for_requester(self, requester_key: str) -> int:
        return self._queries.count_active_for_requester(
            requester_key
        )

    def get_run(self, run_id: str) -> SubagentRunRecord | None:
        return self._queries.get_run(run_id)

    def list_runs(self) -> list[SubagentRunRecord]:
        """Return a snapshot of all run records."""
        return self._queries.list_runs()

    def list_run_entries(
        self,
    ) -> list[tuple[str, SubagentRunRecord]]:
        """Return canonical registry keys with their records."""
        return self._queries.list_run_entries()

    def remove_run(self, run_id: str) -> bool:
        """Remove one run and persist the registry change."""
        return self._lifecycle.remove_run(run_id)

    def mark_announce_retry(self, run_id: str) -> bool:
        """标记 announce 重试，返回是否可继续重试（未超限且未过期）"""
        return self._state.mark_announce_retry(run_id)

    def mark_result_delivery_delivered(self, run_id: str) -> None:
        self._state.mark_result_delivery_delivered(run_id)

    def mark_result_delivery_dropped(self, run_id: str) -> None:
        self._state.mark_result_delivery_dropped(run_id)

    def set_result_delivery_state(self, run_id: str, new_state: str) -> None:
        self._state.set_result_delivery_state(run_id, new_state)

    def set_delivery_work_id(self, run_id: str, work_id: str | None) -> None:
        self._state.set_delivery_work_id(run_id, work_id)

    def get_requester_depth(self, requester_session_key: str) -> int:
        return self._relationships.get_requester_depth(requester_session_key)

    @staticmethod
    def session_key_from_child_session_key(child_session_key: str) -> str:
        """返回 child 会话可直接作为 requester 使用的 canonical key。"""
        return SubagentRelationshipService.session_key_from_child_session_key(
            child_session_key
        )

    def list_descendant_runs(
        self, root_session_key: str, include_recent_minutes: int = 60
    ) -> list[SubagentRunRecord]:
        return self._relationships.list_descendant_runs(
            root_session_key,
            include_recent_minutes,
        )

    def resolve_requester_for_child_session(
        self, child_session_key: str
    ) -> tuple[str, str] | None:
        return self._relationships.resolve_requester_for_child_session(
            child_session_key
        )

    def count_active_descendant_runs(self, root_session_key: str) -> int:
        return self._relationships.count_active_descendant_runs(root_session_key)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        return self._lifecycle.cleanup_old(max_age_hours)

    def replace_active_run_for_steer(
        self,
        previous_run_id: str,
        next_run_id: str,
        task: str,
    ) -> SubagentRunRecord | None:
        """原子接管活跃 run，并在锁外取消旧任务。"""
        return self._lifecycle.replace_active_run_for_steer(
            previous_run_id,
            next_run_id,
            task,
        )

    def sweep_expired(
        self,
        on_expire: Callable[[SubagentRunRecord], None] | None = None,
    ) -> int:
        """删除 archive_at_ms 已到期的 run，并归档会话。

        每 60 秒由 subagent_archive 调用。on_expire 负责归档/删除会话文件。
        """
        return self._archive.sweep_expired(on_expire=on_expire)


registry = SubagentRegistry()
session_requester_resolver.bind(
    registry.resolve_requester_for_child_session
)
