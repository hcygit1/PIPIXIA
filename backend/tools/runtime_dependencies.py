"""Runtime capabilities consumed by session, memory, and status tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ToolRuntimeDependencies:
    get_session_manager: Callable[[], Any]
    get_memory_recall: Callable[[str], Any | None]
    get_memory_store: Callable[[str], Any | None]
    count_active_for_requester: Callable[[str], int]
    get_or_create_memory_task_id: Callable[[str, str], str | None] | None = None


def _get_global_session_manager() -> Any:
    from sessions.session_manager import session_manager

    return session_manager


def _get_global_memory_recall(agent_id: str) -> Any | None:
    from runtime.agent import agent_manager

    return agent_manager.mem_recalls.get(agent_id)


def _get_global_memory_store(agent_id: str) -> Any | None:
    from runtime.agent import agent_manager

    return agent_manager.mem_stores.get(agent_id)


def _count_global_active_for_requester(requester_key: str) -> int:
    from subagents.subagent_registry import registry

    return registry.count_active_for_requester(requester_key)


def _get_or_create_global_memory_task_id(
    agent_id: str, session_id: str,
) -> str | None:
    from runtime.agent import agent_manager

    return agent_manager._memory_runtime.get_or_create_active_task_id(
        agent_id,
        session_id,
    )


def default_tool_runtime_dependencies() -> ToolRuntimeDependencies:
    """Keep legacy direct factory calls dynamic and patch-compatible."""
    return ToolRuntimeDependencies(
        get_session_manager=_get_global_session_manager,
        get_memory_recall=_get_global_memory_recall,
        get_memory_store=_get_global_memory_store,
        count_active_for_requester=_count_global_active_for_requester,
        get_or_create_memory_task_id=(
            _get_or_create_global_memory_task_id
        ),
    )
