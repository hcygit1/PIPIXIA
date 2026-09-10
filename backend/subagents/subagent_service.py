"""Public compatibility facade for subagent application operations."""

from __future__ import annotations

import uuid
from typing import Any, Callable

from sessions.session_identity import session_key_from_session_id
from subagents.subagent_execution import (
    RunnerFactory,
    SubagentExecutionService,
)
from subagents.subagent_run_model import SubagentRunRecord
from subagents.subagent_run_operations import SubagentRunOperations
from subagents.subagent_scope import SubagentScopeResolver
from subagents.subagent_service_models import (
    KillResult,
    SpawnResult,
    SteerResult,
    SubagentListResult,
    SubagentServiceError,
)


class SubagentService:
    MAX_STEER_MESSAGE_CHARS = (
        SubagentExecutionService.MAX_STEER_MESSAGE_CHARS
    )
    MAX_RECENT_MINUTES = SubagentScopeResolver.MAX_RECENT_MINUTES

    def __init__(
        self,
        *,
        registry: Any = None,
        session_manager: Any = None,
        resolve_agent_config: Callable[[str], dict[str, Any]] | None = None,
        get_config: Callable[[], dict[str, Any]] | None = None,
        runner_factory: RunnerFactory | None = None,
        id_factory: Callable[[], str] | None = None,
        scope: Any = None,
        run_operations: Any = None,
        execution: Any = None,
    ) -> None:
        if registry is None:
            from subagents.subagent_registry import registry
        if session_manager is None:
            from sessions.session_manager import session_manager
        if resolve_agent_config is None:
            from config import resolve_agent_config
        if get_config is None:
            from config import get_config

        id_factory = id_factory or (lambda: uuid.uuid4().hex[:12])
        self._registry = registry
        self._scope = (
            scope
            if scope is not None
            else SubagentScopeResolver(
                session_manager=session_manager,
                get_config=get_config,
            )
        )
        self._run_operations = (
            run_operations
            if run_operations is not None
            else SubagentRunOperations(
                registry=registry,
                scope=self._scope,
            )
        )
        self._execution = (
            execution
            if execution is not None
            else SubagentExecutionService(
                registry=registry,
                session_manager=session_manager,
                resolve_agent_config=resolve_agent_config,
                runner_factory=runner_factory,
                id_factory=id_factory,
                scope=self._scope,
            )
        )

    def spawn(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str,
        task: str,
        target_agent_id: str = "",
        label: str | None = None,
        model: str | None = None,
        parent_task_id: str | None = None,
    ) -> SpawnResult:
        spawn_kwargs = dict(
            requester_agent_id=requester_agent_id,
            requester_session_id=requester_session_id,
            task=task,
            target_agent_id=target_agent_id,
            label=label,
            model=model,
        )
        if parent_task_id:
            spawn_kwargs["parent_task_id"] = parent_task_id
        return self._execution.spawn(**spawn_kwargs)

    def list_runs(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str | None,
        recent_minutes: int | None = None,
        recursive: bool = False,
    ) -> SubagentListResult:
        return self._run_operations.list_runs(
            requester_agent_id=requester_agent_id,
            requester_session_id=requester_session_id,
            recent_minutes=recent_minutes,
            recursive=recursive,
        )

    def kill(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str,
        target: str,
    ) -> KillResult:
        return self._run_operations.kill(
            requester_agent_id=requester_agent_id,
            requester_session_id=requester_session_id,
            target=target,
        )

    def steer(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str | None,
        run_id: str,
        message: str,
    ) -> SteerResult:
        return self._execution.steer(
            requester_agent_id=requester_agent_id,
            requester_session_id=requester_session_id,
            run_id=run_id,
            message=message,
        )

    def child_requester_key(
        self,
        record: SubagentRunRecord,
    ) -> str:
        return self._registry.session_key_from_child_session_key(
            record.child_session_key
        )

    def count_active_for_requester(self, requester_key: str) -> int:
        return self._registry.count_active_for_requester(requester_key)

    def parent_task_id_for_session(
        self, agent_id: str, session_id: str,
    ) -> str | None:
        """Resolve the root task inherited by a child-agent session."""
        session_key = session_key_from_session_id(agent_id, session_id)
        for record in self._registry.list_runs():
            if record.child_session_key == session_key:
                return record.parent_task_id
        return None
