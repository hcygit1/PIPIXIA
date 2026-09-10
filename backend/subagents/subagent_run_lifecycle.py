"""Collection-level lifecycle operations for subagent runs."""

from __future__ import annotations

import time
from typing import Any, Callable


class SubagentRunLifecycleService:
    """Coordinate run collection changes without owning state transitions."""

    def __init__(
        self,
        *,
        store: Any,
        state: Any,
        relationships: Any,
        persist: Callable[[], None],
        record_factory: Callable[..., Any],
        snapshot: Callable[[Any], Any],
        resolve_archive_after_ms: Callable[[], float | None],
        capacity_error: Callable[[str], Exception],
        now: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._state = state
        self._relationships = relationships
        self._persist = persist
        self._record_factory = record_factory
        self._snapshot = snapshot
        self._resolve_archive_after_ms = resolve_archive_after_ms
        self._capacity_error = capacity_error
        self._now = now if now is not None else time.time

    def register_run(
        self,
        *,
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
    ) -> Any:
        now = self._now()
        archive_after_ms = self._resolve_archive_after_ms()
        archive_at_ms = (
            now * 1000 + archive_after_ms
            if archive_after_ms
            else None
        )
        record = self._record_factory(
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
            archive_at_ms=archive_at_ms,
        )
        with self._store.locked_records() as runs:
            if max_active_for_requester is not None:
                active = sum(
                    1
                    for current in runs.values()
                    if current.requester_session_key == requester_session_key
                    and current.ended_at is None
                )
                if active >= max_active_for_requester:
                    raise self._capacity_error(
                        "active sub-agent capacity reached"
                    )
            runs[run_id] = record
            self._persist()
        return self._snapshot(record)

    def set_task(self, run_id: str, task: Any) -> bool:
        with self._store.locked_records() as runs:
            record = runs.get(run_id)
            if record is not None and record.ended_at is None:
                record.asyncio_task = task
                return True

        try:
            if hasattr(task, "cancel"):
                task.cancel()
        except Exception:
            pass
        return False

    def remove_run(self, run_id: str) -> bool:
        with self._store.locked_records() as runs:
            removed = runs.pop(run_id, None)
        if removed is None:
            return False
        self._persist()
        return True

    def kill(self, run_id: str, cascade: bool = True) -> bool:
        """Terminate a run tree and cancel task handles after releasing the lock."""
        tasks_to_cancel: list[Any] = []
        with self._store.locked_records() as runs:
            root = runs.get(run_id)
            if root is None or root.ended_at is not None:
                return False

            pending = [run_id]
            visited: set[str] = set()
            while pending:
                current_id = pending.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)
                record = runs.get(current_id)
                if record is None or record.ended_at is not None:
                    continue

                if cascade:
                    child_key = (
                        self._relationships.session_key_from_child_session_key(
                            record.child_session_key
                        )
                    )
                    pending.extend(
                        child.run_id
                        for child in runs.values()
                        if child.requester_session_key == child_key
                        and child.ended_at is None
                    )

                self._state.terminate_record(record, "killed")
                if record.asyncio_task is not None:
                    tasks_to_cancel.append(record.asyncio_task)

            self._persist()

        for task in tasks_to_cancel:
            try:
                if hasattr(task, "cancel"):
                    task.cancel()
            except Exception:
                pass
        return True

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        cutoff = self._now() - max_age_hours * 3600
        with self._store.locked_records() as runs:
            to_remove = [
                run_id
                for run_id, record in runs.items()
                if record.ended_at is not None
                and record.ended_at < cutoff
            ]
            for run_id in to_remove:
                runs.pop(run_id, None)
        if to_remove:
            self._persist()
        return len(to_remove)

    def replace_active_run_for_steer(
        self,
        previous_run_id: str,
        next_run_id: str,
        task: str,
    ) -> Any | None:
        """Replace an active run atomically and cancel its task outside the lock."""
        old_task: Any = None
        now = self._now()
        archive_after_ms = self._resolve_archive_after_ms()
        archive_at_ms = (
            now * 1000 + archive_after_ms
            if archive_after_ms
            else None
        )
        with self._store.locked_records() as runs:
            previous = runs.get(previous_run_id)
            if previous is None or previous.ended_at is not None:
                return None
            record = self._record_factory(
                run_id=next_run_id,
                child_session_key=previous.child_session_key,
                requester_session_key=previous.requester_session_key,
                requester_agent_id=previous.requester_agent_id,
                target_agent_id=previous.target_agent_id,
                task=task,
                label=previous.label,
                model=previous.model,
                parent_task_id=previous.parent_task_id,
                cleanup=previous.cleanup,
                spawn_depth=previous.spawn_depth,
                created_at=now,
                started_at=now,
                archive_at_ms=archive_at_ms,
            )
            old_task = previous.asyncio_task
            runs.pop(previous_run_id, None)
            runs[next_run_id] = record
            self._persist()

        try:
            if old_task is not None and hasattr(old_task, "cancel"):
                old_task.cancel()
        except Exception:
            pass
        return self._snapshot(record)
