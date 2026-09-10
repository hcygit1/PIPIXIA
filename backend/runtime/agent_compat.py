"""Compatibility properties retained on AgentManager."""

from __future__ import annotations

import asyncio
from typing import Any

from runtime.agent_state import AgentState
from runtime.turn_context import PromptCacheEntry, SessionContextCacheEntry


class AgentManagerCompatibilityMixin:
    """Expose legacy manager attributes through their owning components."""

    @property
    def data_dir(self) -> str:
        return self._lifecycle.data_dir

    @data_dir.setter
    def data_dir(self, value: str) -> None:
        self._lifecycle.data_dir = value

    @property
    def _initialized(self) -> bool:
        return self._lifecycle.initialized

    @_initialized.setter
    def _initialized(self, value: bool) -> None:
        self._lifecycle.initialized = value

    @property
    def _states(self) -> dict[str, AgentState]:
        return self._state_runtime.states

    @property
    def _state_save_tasks(self) -> dict[str, asyncio.Task]:
        return self._state_runtime.save_tasks

    @property
    def _prompt_cache(self) -> dict[tuple[Any, ...], PromptCacheEntry]:
        return self._turn_context.prompt_cache

    @_prompt_cache.setter
    def _prompt_cache(
        self,
        value: dict[tuple[Any, ...], PromptCacheEntry],
    ) -> None:
        self._turn_context.prompt_cache = value

    @property
    def _session_context_cache(
        self,
    ) -> dict[tuple[str, str], SessionContextCacheEntry]:
        return self._turn_context.session_context_cache

    @_session_context_cache.setter
    def _session_context_cache(
        self,
        value: dict[tuple[str, str], SessionContextCacheEntry],
    ) -> None:
        self._turn_context.session_context_cache = value

    @property
    def mem_stores(self) -> dict[str, Any]:
        return self._memory_runtime.stores

    @mem_stores.setter
    def mem_stores(self, value: dict[str, Any]) -> None:
        self._memory_runtime.stores = value

    @property
    def mem_embedders(self) -> dict[str, Any]:
        return self._memory_runtime.embedders

    @mem_embedders.setter
    def mem_embedders(self, value: dict[str, Any]) -> None:
        self._memory_runtime.embedders = value

    @property
    def mem_workers(self) -> dict[str, Any]:
        return self._memory_runtime.workers

    @mem_workers.setter
    def mem_workers(self, value: dict[str, Any]) -> None:
        self._memory_runtime.workers = value

    @property
    def mem_recalls(self) -> dict[str, Any]:
        return self._memory_runtime.recalls

    @mem_recalls.setter
    def mem_recalls(self, value: dict[str, Any]) -> None:
        self._memory_runtime.recalls = value

    @property
    def mem_task_processors(self) -> dict[str, Any]:
        return self._memory_runtime.task_processors

    @mem_task_processors.setter
    def mem_task_processors(self, value: dict[str, Any]) -> None:
        self._memory_runtime.task_processors = value

    def collect_tools(
        self,
        agent_id: str,
        session_id: str = "",
    ) -> list[Any]:
        return self._tool_registry.collect_tools(agent_id, session_id)
