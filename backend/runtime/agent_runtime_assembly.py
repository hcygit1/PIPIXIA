"""Construction of the Agent runtime component graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from runtime.agent_lifecycle import AgentLifecycle
from runtime.agent_state_runtime import AgentStateRuntime
from runtime.memory_runtime import MemoryRuntime
from runtime.model_runtime import ModelRuntime
from runtime.session_commands import SessionCommands
from runtime.session_compactor import SessionCompactor
from runtime.session_lifecycle import SessionLifecycle
from runtime.tool_registry import ToolRegistry
from runtime.turn_context import TurnContext
from runtime.turn_executor import TurnExecutor
from runtime.turn_preparation import TurnPreparation
from runtime.turn_recovery import TurnRecovery
from runtime.turn_service import TurnService, TurnServicePorts
from runtime.agent_turn_preparation import AgentTurnPreparationAdapter
from runtime.agent_runtime_bindings import AgentRuntimeBindings
from runtime.command_parser import CommandExecutionDependencies
from subagents.subagent_runner import SubagentRunner
from subagents.subagent_service import SubagentService
from tools.runtime_dependencies import ToolRuntimeDependencies


@dataclass
class AgentRuntimeComponents:
    """The runtime objects owned by an AgentManager instance."""

    state_runtime: Any
    memory_runtime: Any
    model_runtime: Any
    subagent_service: Any
    tool_registry: Any
    turn_context: Any
    turn_preparation: Any
    turn_preparation_adapter: Any
    session_compactor: Any
    session_commands: Any
    turn_recovery: Any
    turn_executor: Any
    turn_service: Any
    command_dependencies: Any
    session_lifecycle: Any
    tool_name_cache: Any
    lifecycle: Any
    pending_tasks: set[asyncio.Task]

    def install_on(self, manager: Any) -> None:
        """Install assembled components without changing public compatibility fields."""
        manager._state_runtime = self.state_runtime
        manager._memory_runtime = self.memory_runtime
        manager._model_runtime = self.model_runtime
        manager.subagent_service = self.subagent_service
        manager._tool_registry = self.tool_registry
        manager._turn_context = self.turn_context
        manager._turn_preparation = self.turn_preparation
        manager._turn_preparation_adapter = self.turn_preparation_adapter
        manager._session_compactor = self.session_compactor
        manager._session_commands = self.session_commands
        manager._turn_recovery = self.turn_recovery
        manager._turn_executor = self.turn_executor
        manager._turn_service = self.turn_service
        manager._command_dependencies = self.command_dependencies
        manager._session_lifecycle = self.session_lifecycle
        manager._tool_name_cache = self.tool_name_cache
        manager._lifecycle = self.lifecycle
        manager._pending_tasks = self.pending_tasks


class AgentRuntimeAssembler:
    """Build the Agent runtime graph from one manager and its module bindings."""

    def __init__(
        self,
        manager: Any,
        bindings: AgentRuntimeBindings,
        get_session_manager: Callable[[], Any] | None = None,
    ) -> None:
        self._manager = manager
        self._bindings = bindings
        self._session_manager_provider = (
            get_session_manager
            or self._bindings.get_session_manager
        )

    def _get_session_manager(self) -> Any:
        return self._session_manager_provider()

    def build(self) -> AgentRuntimeComponents:
        manager = self._manager
        pending_tasks = manager._pending_tasks

        state_runtime = AgentStateRuntime(
            resolve_persist_config=manager._get_state_persist_config,
            resolve_state_path=manager._get_state_path,
            resolve_think_level=manager._resolve_think_level,
            is_initialized=lambda: manager._initialized,
        )
        memory_runtime = MemoryRuntime()
        model_runtime = ModelRuntime(
            resolve_configured_model=self._bindings.resolve_agent_model,
            resolve_configured_candidates=(
                self._bindings.resolve_fallback_candidates
            ),
            find_model=self._bindings.find_model_by_id,
            get_model=self._bindings.get_model,
            invalidate_llm=self._bindings.invalidate_llm,
            get_or_create_llm=self._bindings.get_or_create_llm,
            get_display_name=self._bindings.get_model_display_name,
        )
        subagent_service = SubagentService(
            runner_factory=lambda requester_agent_id: SubagentRunner(
                astream=manager.astream,
                requester_agent_id=requester_agent_id,
            )
        )
        tool_registry = ToolRegistry(
            subagent_service,
            runtime_dependencies=ToolRuntimeDependencies(
                get_session_manager=self._get_session_manager,
                get_memory_recall=lambda agent_id: memory_runtime.recalls.get(
                    agent_id
                ),
                get_memory_store=lambda agent_id: memory_runtime.stores.get(
                    agent_id
                ),
                count_active_for_requester=(
                    subagent_service.count_active_for_requester
                ),
                get_or_create_memory_task_id=(
                    memory_runtime.get_or_create_active_task_id
                ),
            ),
        )
        turn_context = TurnContext()
        turn_preparation = TurnPreparation(turn_context)
        turn_preparation_adapter = AgentTurnPreparationAdapter(
            preparation=turn_preparation,
            tool_registry=tool_registry,
            get_data_dir=lambda: manager.data_dir,
            get_memory_store=lambda agent_id: manager.mem_stores.get(agent_id),
            resolve_workspace=self._bindings.resolve_agent_workspace,
            resolve_agent_dir=self._bindings.resolve_agent_dir,
            resolve_agent_config=self._bindings.resolve_agent_config,
            get_heartbeat_config=self._bindings.get_heartbeat_config,
            get_current_model=lambda agent_id: manager.get_current_model_ref(
                agent_id
            ),
            build_prompt=self._bindings.build_system_prompt_with_report,
            load_history=lambda session_id, agent_id: self._get_session_manager().load_session_for_agent(
                session_id,
                agent_id,
            ),
            format_summary=self._bindings.format_session_summary,
            prune_history=self._bindings.prune_messages,
            count_tokens=self._bindings.count_tokens,
            count_messages_tokens=self._bindings.count_messages_tokens,
        )
        session_compactor = SessionCompactor(
            resolve_agent_config=self._bindings.resolve_agent_config,
            load_session=lambda session_id, agent_id: self._get_session_manager().load_session(
                session_id,
                agent_id,
            ),
            compress_history=lambda session_id, agent_id, count: self._get_session_manager().compress_history(
                session_id,
                agent_id,
                count,
            ),
            get_llm=lambda agent_id: manager.get_llm(agent_id),
            log_compress=manager._log_compress,
        )
        session_commands = SessionCommands(
            load_session=lambda session_id, agent_id: self._get_session_manager().load_session(
                session_id,
                agent_id,
            ),
            reset_session=lambda session_id, agent_id: self._get_session_manager().reset_session(
                session_id,
                agent_id,
            ),
            resolve_agent_config=self._bindings.resolve_agent_config,
            emit_event=manager._emit_runtime_event,
            audit_log=manager._audit_runtime_event,
        )
        turn_recovery = TurnRecovery(
            reset_session=lambda session_id, agent_id: self._get_session_manager().reset_session(
                session_id,
                agent_id,
            ),
            compress_session=manager._compress_for_recovery,
            audit_log=manager._audit_runtime_event,
            sleep=lambda seconds: asyncio.sleep(seconds),
        )
        turn_executor = TurnExecutor(
            create_llm=self._bindings.create_llm,
            build_messages=lambda history, message: manager._build_messages(
                history,
                message,
            ),
            get_lifecycle_hooks=lambda: manager.lifecycle_hooks,
            get_run_tracker=self._bindings.get_run_tracker,
            get_audit_logger=self._bindings.get_audit_logger,
            save_message=lambda *args, **kwargs: self._get_session_manager().save_message(
                *args,
                **kwargs,
            ),
            write_skills_snapshot=manager._write_skills_snapshot,
            emit_event=manager._emit_runtime_event,
            count_tokens=self._bindings.count_tokens,
            incremental_ingest=manager._ingest_completed_turn,
            get_pending_tasks=lambda: pending_tasks,
            maybe_auto_compact=manager._run_auto_compaction,
        )
        command_dependencies = CommandExecutionDependencies(
            session_manager_provider=self._get_session_manager,
        )

        def execute_command(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("dependencies", command_dependencies)
            return self._bindings.execute_command(*args, **kwargs)

        turn_service = TurnService(
            TurnServicePorts(
                get_state=lambda agent_id: manager.get_state(agent_id),
                parse_command=self._bindings.parse_command,
                execute_command=execute_command,
                switch_model=lambda agent_id, model: manager.switch_model(
                    agent_id,
                    model,
                ),
                get_current_model=lambda agent_id: manager.get_current_model_ref(
                    agent_id
                ),
                get_model_override=lambda agent_id: manager.get_model_override(
                    agent_id
                ),
                handle_reset=lambda *args, **kwargs: manager._handle_reset(
                    *args,
                    **kwargs,
                ),
                handle_reset_noflush=lambda *args, **kwargs: manager._handle_reset_noflush(
                    *args,
                    **kwargs,
                ),
                handle_compact=lambda *args, **kwargs: manager._handle_compact(
                    *args,
                    **kwargs,
                ),
                write_skills_snapshot=lambda agent_id: manager._write_skills_snapshot(
                    agent_id
                ),
                has_bootstrap=lambda agent_id: manager._has_bootstrap(agent_id),
                resolve_workspace=self._bindings.resolve_agent_workspace,
                get_locale=lambda: manager._get_locale(),
                get_tool_names=lambda agent_id: manager._get_or_build_tool_names(
                    agent_id
                ),
                build_prompt=lambda **kwargs: manager._get_or_build_prompt(
                    **kwargs
                ),
                get_session_context=lambda **kwargs: manager._get_or_build_session_context(
                    **kwargs
                ),
                build_tools=lambda agent_id, session_id: manager._build_tools(
                    agent_id,
                    session_id,
                ),
                resolve_budget=lambda agent_id: manager._resolve_context_budget(
                    agent_id
                ),
                resolve_agent_config=self._bindings.resolve_agent_config,
                resolve_candidates=lambda agent_id: model_runtime.resolve_candidates(
                    agent_id
                ),
                execute_turn=lambda request: manager._turn_executor.execute(
                    request
                ),
                run_fallback_stream=self._bindings.run_with_fallback_stream,
                recover_turn=lambda **kwargs: manager._turn_recovery.run(
                    **kwargs
                ),
            )
        )
        session_lifecycle = SessionLifecycle(
            compactor=session_compactor,
            resolve_agent_config=self._bindings.resolve_agent_config,
            load_session=lambda session_id, agent_id: self._get_session_manager().load_session(
                session_id,
                agent_id,
            ),
            detect_compaction_level=self._bindings.detect_compaction_level,
            audit_log=manager._audit_runtime_event,
            emit_event=manager._emit_runtime_event,
            get_store=lambda agent_id: manager.mem_stores.get(agent_id),
            get_state=lambda agent_id: manager.get_state(agent_id),
            batch_ingest_messages=manager._batch_ingest_messages,
            pending_tasks=pending_tasks,
        )
        tool_name_cache = tool_registry.name_cache
        lifecycle = AgentLifecycle(
            state_runtime=state_runtime,
            memory_runtime=memory_runtime,
            model_runtime=model_runtime,
            list_agents=self._bindings.list_agents,
            ensure_workspace=manager._ensure_agent_workspace,
            prompt_cache=turn_context.prompt_cache,
            session_context_cache=turn_context.session_context_cache,
            tool_name_cache=tool_name_cache,
        )
        return AgentRuntimeComponents(
            state_runtime=state_runtime,
            memory_runtime=memory_runtime,
            model_runtime=model_runtime,
            subagent_service=subagent_service,
            tool_registry=tool_registry,
            turn_context=turn_context,
            turn_preparation=turn_preparation,
            turn_preparation_adapter=turn_preparation_adapter,
            session_compactor=session_compactor,
            session_commands=session_commands,
            turn_recovery=turn_recovery,
            turn_executor=turn_executor,
            turn_service=turn_service,
            command_dependencies=command_dependencies,
            session_lifecycle=session_lifecycle,
            tool_name_cache=tool_name_cache,
            lifecycle=lifecycle,
            pending_tasks=pending_tasks,
        )
