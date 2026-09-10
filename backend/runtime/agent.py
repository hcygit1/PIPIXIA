"""Agent 引擎核心 — AgentManager, AgentState, 生命周期, 自动压缩, 命令处理"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import (
    DATA_DIR,
    get_heartbeat_config,
    resolve_agent_config,
    resolve_agent_workspace,
    resolve_agent_dir,
    list_agents,
)
from runtime.prompt_builder import prompt_builder
from sessions.session_manager import session_manager
from infra.run_tracker import run_tracker
from infra.audit_log import audit_logger
from infra.token_counter import (
    count_messages_tokens,
    count_tokens,
    detect_compaction_level,
)
from sessions.session_pruning import prune_messages
from runtime.command_parser import parse_command, execute_command
from llm.model_selection import (
    get_model_display_name,
    resolve_agent_model,
    resolve_fallback_candidates,
    run_with_fallback_stream,
)
from llm.llm_factory import create_llm, llm_cache
from llm.models_config import models_config
from runtime.agent_state import AgentState
from runtime.agent_compat import AgentManagerCompatibilityMixin
from runtime.agent_environment_compat import (
    AgentManagerEnvironmentCompatibilityMixin,
)
from runtime.agent_session_compat import (
    AgentManagerSessionCompatibilityMixin,
)
from runtime.agent_turn_compat import (
    AgentManagerTurnPreparationCompatibilityMixin,
)
from runtime.agent_runtime_assembly import AgentRuntimeAssembler
from runtime.agent_runtime_bindings import AgentRuntimeBindings
from runtime.agent_turn_preparation import AgentTurnPreparationAdapter
from runtime.agent_state_runtime import AgentStateRuntime
from runtime.agent_lifecycle import AgentLifecycle
from runtime.memory_runtime import MemoryRuntime
from runtime.model_runtime import ModelRuntime
from runtime.session_commands import SessionCommands
from runtime.session_compactor import SessionCompactor
from runtime.session_lifecycle import SessionLifecycle
from runtime.tool_registry import ToolRegistry
from subagents.subagent_runner import SubagentRunner
from subagents.subagent_service import SubagentService
from runtime.turn_recovery import TurnRecovery
from runtime.turn_executor import (
    TurnExecutor,
    should_persist_input_message as _should_persist_input_message,
)
from runtime.turn_context import (
    PromptCacheEntry,
    SessionContextCacheEntry,
    TurnContext,
)
from runtime.turn_preparation import TurnPreparation
from runtime.turn_service import (
    BARE_SESSION_RESET_PROMPT,
    TurnService,
    TurnServicePorts,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 生命周期钩子
# ---------------------------------------------------------------------------

@dataclass
class LifecycleHooks:
    """显式生命周期钩子，用于审计、确认、记录等扩展"""

    async def on_before_tool_call(
        self, agent_id: str, run_id: str, tool_name: str, tool_input: dict[str, Any]
    ) -> None:
        """工具调用前（可在此拦截/确认）"""
        pass

    async def on_after_tool_call(
        self, agent_id: str, run_id: str, tool_name: str, tool_input: Any, tool_output: str
    ) -> None:
        """工具调用后（审计、记录）"""
        pass


from infra.event_bus import event_bus


# ---------------------------------------------------------------------------
# AgentManager — 核心引擎
# ---------------------------------------------------------------------------

class AgentManager(
    AgentManagerEnvironmentCompatibilityMixin,
    AgentManagerSessionCompatibilityMixin,
    AgentManagerTurnPreparationCompatibilityMixin,
    AgentManagerCompatibilityMixin,
):

    def __init__(
        self,
        session_manager: Any | None = None,
        runtime_bindings: AgentRuntimeBindings | None = None,
    ):
        self.lifecycle_hooks: LifecycleHooks | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._session_manager_override = session_manager
        bindings = (
            runtime_bindings
            if runtime_bindings is not None
            else AgentRuntimeBindings.from_module_symbols(globals())
        )
        AgentRuntimeAssembler(
            self,
            bindings,
            get_session_manager=self._get_session_manager,
        ).build().install_on(self)

    def _get_session_manager(self) -> Any:
        if self._session_manager_override is not None:
            return self._session_manager_override
        return globals()["session_manager"]

    def _init_mem_system(self, agent_id: str) -> None:
        self._memory_runtime.initialize_agent(agent_id)

    async def initialize(self, data_dir: str) -> None:
        await self._lifecycle.initialize(data_dir)

    def get_llm(self, agent_id: str = "main"):
        """获取指定 Agent 的 LLM 实例（per-agent 动态创建，按 Provider 配置路由）"""
        return self._model_runtime.get_llm(agent_id)

    def get_current_model_ref(self, agent_id: str = "main"):
        """获取 Agent 当前使用的 ModelRef"""
        return self._model_runtime.resolve_current(agent_id)

    def switch_model(self, agent_id: str, model_raw: str) -> str:
        """运行时切换 Agent 模型，返回新模型描述"""
        return self._model_runtime.switch(
            agent_id,
            model_raw,
        )

    def get_model_override(
        self,
        agent_id: str,
    ):
        return self._model_runtime.get_override(agent_id)

    def restore_model_override(
        self,
        agent_id: str,
        override,
    ) -> None:
        self._model_runtime.restore_override(
            agent_id,
            override,
        )

    def clear_model_overrides(
        self,
        agent_id: str | None = None,
    ) -> None:
        self._model_runtime.clear(agent_id)

    def get_state(self, agent_id: str) -> AgentState:
        return self._state_runtime.get_state(agent_id)

    async def wait_for_pending_tasks(self, timeout: float = 30.0) -> None:
        """等待所有后台任务完成，用于应用关闭前确保数据不丢失"""
        await self._lifecycle.wait_for_pending_tasks(
            pending_tasks=self._pending_tasks,
            timeout=timeout,
            save_all_states=self._save_all_states,
        )

    async def close(self, timeout: float = 30.0) -> None:
        """停止后台任务、关闭持久化资源，并将管理器恢复为未初始化状态。"""
        try:
            await self._lifecycle.close(
                pending_tasks=self._pending_tasks,
                timeout=timeout,
                save_all_states=self._save_all_states,
            )
        finally:
            self.lifecycle_hooks = None

    async def _save_all_states(self) -> None:
        """保存所有 Agent 状态到磁盘"""
        await self._state_runtime.save_all_states()

    # ------------------------------------------------------------------
    # 核心流式方法
    # ------------------------------------------------------------------

    async def astream(
        self,
        message: str,
        session_id: str,
        agent_id: str = "main",
        prompt_mode: str = "full",
        persist_input_role: str = "user",
        parent_task_id: str | None = None,
        extra_system_prompt: str | None = None,
        evaluation_mode: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for event in self._turn_service.stream(
            message,
            session_id,
            agent_id=agent_id,
            prompt_mode=prompt_mode,
            persist_input_role=persist_input_role,
            parent_task_id=parent_task_id,
            extra_system_prompt=extra_system_prompt,
            evaluation_mode=evaluation_mode,
        ):
            yield event

    async def compress_session(
        self, session_id: str, agent_id: str, level: str = "sliding",
    ) -> dict[str, Any]:
        return await self._session_lifecycle.compress_session(
            session_id,
            agent_id,
            level=level,
            generate_summary=self._generate_structured_summary,
            batch_ingest_messages=self._batch_ingest_messages,
            pending_tasks=self._pending_tasks,
        )

    # ------------------------------------------------------------------
    # Agent 注册
    # ------------------------------------------------------------------

    async def register_agent(self, agent_id: str) -> None:
        await self._lifecycle.register_agent(agent_id)


agent_manager = AgentManager()
