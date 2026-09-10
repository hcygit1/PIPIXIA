"""Agent/Session tools: agents_list, sessions_list,
sessions_history, sessions_send, sessions_spawn, subagents"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from config import (
    list_agents,
    resolve_agent_config,
)
from tools.runtime_dependencies import (
    ToolRuntimeDependencies,
    default_tool_runtime_dependencies,
)

ANNOUNCE_ACQUIRE_TIMEOUT_SEC = 10
ANNOUNCE_RUN_TIMEOUT_SEC = 30


def _resolve_session_manager(provider: Any) -> Any:
    get_session_manager = provider
    if get_session_manager is None:
        get_session_manager = (
            default_tool_runtime_dependencies().get_session_manager
        )
    return get_session_manager()


# ---------------------------------------------------------------------------
# agents_list
# ---------------------------------------------------------------------------

class AgentsListTool(BaseTool):
    name: str = "agents_list"
    description: str = (
        "列出当前 Agent 允许协作的 Agent（ID、名称、描述）。"
        "结果根据配置中的 subagents.allow_agents 过滤。"
    )
    current_agent_id: str = "main"

    def _run(self, **kwargs) -> str:
        """Only list agents allowed for collaboration to prevent arbitrary agent calls.

        Rules:
        - Always include self (current_agent_id)
        - If subagents.allow_agents contains \"*\", show all configured agents
        - Otherwise, only show agents in the allow_agents list
        """
        agents = list_agents()
        if not agents:
            return "No agents configured."

        from config import resolve_agent_config

        requester_id = self.current_agent_id or "main"
        cfg = resolve_agent_config(requester_id) or {}
        subagents_cfg = cfg.get("subagents") or {}
        allow = subagents_cfg.get("allow_agents") or []

        # 归一化允许列表
        allow_any = "*" in allow
        allow_set = {a for a in allow if a and a != "*"}

        visible: list[dict[str, Any]] = []
        for a in agents:
            aid = a.get("id")
            if not aid:
                continue
            if aid == requester_id:
                visible.append(a)
                continue
            if allow_any or aid in allow_set:
                visible.append(a)

        if not visible:
            return "The current agent is not configured to collaborate with any other agents."

        lines = []
        for a in visible:
            lines.append(f"- {a['id']}: {a.get('name', 'Unnamed')} — {a.get('description', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_list
# ---------------------------------------------------------------------------

class SessionsListInput(BaseModel):
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    spawned_by: str = Field(
        default="",
        description="按此 session_key 过滤由该会话创建的子会话（可选）",
    )


class SessionsListTool(BaseTool):
    name: str = "sessions_list"
    description: str = "列出指定 Agent 的所有会话。spawned_by 可过滤由特定会话创建的子 Agent 会话。"
    args_schema: type[BaseModel] = SessionsListInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _get_session_manager: Any = None

    def _run(self, agent_id: str = "", spawned_by: str = "") -> str:
        target_id = agent_id or self.current_agent_id
        session_manager = _resolve_session_manager(
            self._get_session_manager
        )

        spawned_by_key: str | None = None
        if spawned_by and (spawned_by or "").strip():
            sk = (spawned_by or "").strip()
            if sk.startswith("agent:") and ":" in sk[6:]:
                spawned_by_key = sk
            else:
                spawned_by_key = session_manager.session_key_from_session_id(
                    target_id, sk
                )
        sessions = session_manager.list_sessions(target_id, spawned_by_session_key=spawned_by_key)
        if not sessions:
            return f"No sessions found for Agent '{target_id}'."
        lines = []
        for s in sessions:
            title = s.get("title", "No Title")
            sid = s.get("session_id", "?")
            msg_count = s.get("message_count", 0)
            lines.append(f"- {sid}: {title} ({msg_count} messages)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_history
# ---------------------------------------------------------------------------

class SessionsHistoryInput(BaseModel):
    session_id: str = Field(default="", description="会话 ID。为空则使用当前会话（来自 sessions_list）")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    limit: int = Field(default=20, description="最多获取的消息条数")


class SessionsHistoryTool(BaseTool):
    name: str = "sessions_history"
    description: str = "获取指定会话的消息历史。session_id 为空则用当前会话；否则使用 sessions_list 中的 session_id。"
    args_schema: type[BaseModel] = SessionsHistoryInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _get_session_manager: Any = None

    def _run(self, session_id: str = "", agent_id: str = "", limit: int = 20) -> str:
        target_id = agent_id or self.current_agent_id
        session_manager = _resolve_session_manager(
            self._get_session_manager
        )
        effective_sid = (session_id or "").strip() or self.current_session_id
        if not effective_sid:
            effective_sid = session_manager.resolve_main_session_id(target_id)
        data = session_manager.load_session(effective_sid, target_id)
        if data is None:
            return f"Session '{effective_sid}' does not exist."

        messages = data.get("messages", [])[-limit:]
        if not messages:
            return "Session has no messages."

        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# sessions_send
# ---------------------------------------------------------------------------

class SessionsSendInput(BaseModel):
    session_id: str = Field(default="", description="目标会话 ID。为空则使用当前会话（来自 sessions_list）")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    message: str = Field(description="要发送的消息")


class SessionsSendTool(BaseTool):
    name: str = "sessions_send"
    description: str = "向另一会话发送消息。session_id 为空则用当前会话；否则使用 sessions_list 中的 session_id。"
    args_schema: type[BaseModel] = SessionsSendInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _get_session_manager: Any = None

    def _run(self, session_id: str = "", message: str = "", agent_id: str = "") -> str:
        target_id = agent_id or self.current_agent_id
        session_manager = _resolve_session_manager(
            self._get_session_manager
        )
        effective_sid = (session_id or "").strip() or self.current_session_id
        if not effective_sid:
            effective_sid = session_manager.resolve_main_session_id(target_id)
        session_manager.save_message(effective_sid, target_id, "user", message)
        return f"Message sent to agent:{target_id}:{effective_sid}"


# ---------------------------------------------------------------------------
# sessions_spawn
# ---------------------------------------------------------------------------

class SessionsSpawnInput(BaseModel):
    task: str = Field(description="子 Agent 要执行的任务描述")
    agent_id: str = Field(default="", description="目标 Agent ID（默认当前 Agent）")
    label: str | None = Field(default=None, description="子 Agent 标签（可选）")
    model: str | None = Field(default=None, description="模型覆盖（可选）")


class SessionsSpawnTool(BaseTool):
    name: str = "sessions_spawn"
    description: str = (
        "在后台启动一个独立子 Agent 并行执行任务，不阻塞当前对话。"
        "子 Agent 拥有独立上下文和工具，完成后自动将结果通知你。"
        "适用场景：耗时操作（网络搜索、大量文件处理）、可并行的独立子任务、"
        "需要独立上下文的调研工作。可同时 spawn 多个子 Agent 并行执行。"
        "不适用：简单问答、需要与用户多轮交互的任务。"
    )
    args_schema: type[BaseModel] = SessionsSpawnInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    parent_task_id: str | None = None
    _get_or_create_memory_task_id: Any = None
    _subagent_service: Any = None

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(
        self,
        task: str,
        agent_id: str = "",
        label: str | None = None,
        model: str | None = None,
    ) -> str:
        target_id = agent_id or self.current_agent_id
        if self._subagent_service is None:
            return "Failed to start sub-agent: service unavailable"
        from subagents.subagent_service import (
            SubagentServiceError,
        )

        try:
            parent_task_id = self.parent_task_id
            if not parent_task_id:
                resolver = getattr(
                    self._subagent_service,
                    "parent_task_id_for_session",
                    None,
                )
                resolved = (
                    resolver(
                        self.current_agent_id,
                        self.current_session_id,
                    )
                    if callable(resolver)
                    else None
                )
                if isinstance(resolved, str) and resolved.strip():
                    parent_task_id = resolved.strip()
            if (
                not parent_task_id
                and self._get_or_create_memory_task_id is not None
            ):
                parent_task_id = self._get_or_create_memory_task_id(
                    self.current_agent_id,
                    self.current_session_id,
                )
            spawn_kwargs = dict(
                requester_agent_id=self.current_agent_id,
                requester_session_id=self.current_session_id,
                task=task,
                target_agent_id=target_id,
                label=label,
                model=model,
            )
            if parent_task_id:
                spawn_kwargs["parent_task_id"] = parent_task_id
            result = self._subagent_service.spawn(**spawn_kwargs)
        except SubagentServiceError as exc:
            if exc.code == "target_forbidden":
                return (
                    "Error: Current agent is not allowed to "
                    f"spawn tasks for '{target_id}'. Please "
                    "explicitly add this agent to agents.list[]"
                    ".subagents.allow_agents in the configuration."
                )
            if exc.code == "depth_limit":
                return f"Error: {exc}, cannot spawn more sub-agents."
            if exc.code == "children_limit":
                return f"Error: {exc}"
            return str(exc)

        return (
            f"Sub-agent spawned:\n"
            f"  run_id: {result.record.run_id}\n"
            f"  session_key: {result.record.child_session_key}\n"
            f"  Task: {task}"
        )

# ---------------------------------------------------------------------------
# subagents
# ---------------------------------------------------------------------------

class SubagentsInput(BaseModel):
    action: Literal["list", "kill", "steer"] = Field(description="操作类型")
    target: str | None = Field(default=None, description="目标 run_id（kill/steer 必填，或填 'all'）")
    message: str | None = Field(default=None, description="新指令（steer 时必填）")
    recent_minutes: int | None = Field(
        default=None,
        description="仅列出最近 N 分钟内完成的子 Agent（默认 30）",
    )


class SubagentsTool(BaseTool):
    name: str = "subagents"
    description: str = (
        "管理子 Agent。操作：list（列出所有子 Agent）、"
        "kill（终止子 Agent，target='all' 表示全部）、"
        "steer（向子 Agent 发送新指令并立即执行，中断当前任务）。"
    )
    args_schema: type[BaseModel] = SubagentsInput
    current_agent_id: str = "main"
    current_session_id: str = ""
    _subagent_service: Any = None

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use _arun for async execution")

    async def _arun(
        self,
        action: str,
        target: str | None = None,
        message: str | None = None,
        recent_minutes: int | None = None,
    ) -> str:
        if self._subagent_service is None:
            return "Error: Sub-agent service unavailable"
        from subagents.subagent_service import (
            SubagentServiceError,
        )

        if action == "list":
            result = self._subagent_service.list_runs(
                requester_agent_id=self.current_agent_id,
                requester_session_id=self.current_session_id,
                recent_minutes=recent_minutes,
            )
            if not result.records:
                return "No sub-agents found."
            lines = []
            for r in result.records:
                status = "Running" if r.ended_at is None else f"Completed({r.outcome})"
                elapsed = ""
                if r.started_at and not r.ended_at:
                    import time
                    elapsed = f" {int(time.time() - r.started_at)}s"
                lines.append(
                    f"- [{r.run_id}] {r.label or 'No Label'} | "
                    f"agent:{r.target_agent_id} | {status}{elapsed}\n"
                    f"  Task: {r.task[:100]}"
                )
            return "\n".join(lines)

        if action == "kill":
            if not target:
                return "Error: kill action requires target parameter (run_id or 'all')"
            try:
                result = self._subagent_service.kill(
                    requester_agent_id=self.current_agent_id,
                    requester_session_id=self.current_session_id,
                    target=target,
                )
            except SubagentServiceError as exc:
                if exc.code in ("out_of_scope", "not_found"):
                    return f"Error: Sub-agent run_id={target} not found"
                if exc.code == "already_ended":
                    return f"Error: Sub-agent {target} has already ended."
                return f"Error: {exc}"
            if target in ("all", "*"):
                return f"Terminated {result.killed} sub-agent(s)."
            return f"Terminated sub-agent: {target}"

        if action == "steer":
            if not target or not message:
                return "Error: steer action requires target (run_id) and message parameters"
            if len(message) > 4000:
                return (
                    "Error: steer message is too long "
                    f"({len(message)} chars, limit 4000)."
                )
            try:
                result = self._subagent_service.steer(
                    requester_agent_id=self.current_agent_id,
                    requester_session_id=self.current_session_id,
                    run_id=target,
                    message=message,
                )
            except SubagentServiceError as exc:
                if exc.code in ("out_of_scope", "not_found"):
                    return f"Error: Sub-agent run_id={target} not found"
                if exc.code == "already_ended":
                    return (
                        f"Sub-agent {target} has already ended, "
                        "no need to steer."
                    )
                if exc.code == "self_steer":
                    return "Error: Sub-agent cannot steer itself."
                if exc.code == "replace_failed":
                    return "Steer failed: unable to replace run."
                return str(exc)
            return (
                f"Steered sub-agent [{result.label}]: new "
                "instruction sent and execution started "
                f"(run_id={result.record.run_id})."
            )

        return f"Unknown action: {action}"


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def get_agent_tools(
    agent_id: str,
    subagent_service: Any = None,
    session_id: str = "",
    *,
    runtime_dependencies: ToolRuntimeDependencies | None = None,
) -> list[BaseTool]:
    dependencies = (
        runtime_dependencies or default_tool_runtime_dependencies()
    )
    # 注入 agentSessionKey/current_session_id，工具从上下文获取当前会话
    effective_session_id = session_id or ""
    if not effective_session_id:
        session_manager = dependencies.get_session_manager()
        effective_session_id = session_manager.resolve_main_session_id(agent_id)
    spawn_tool = SessionsSpawnTool(
        current_agent_id=agent_id,
        current_session_id=effective_session_id,
    )
    spawn_tool._subagent_service = subagent_service
    spawn_tool._get_or_create_memory_task_id = (
        dependencies.get_or_create_memory_task_id
    )
    subagents_tool = SubagentsTool(current_agent_id=agent_id, current_session_id=effective_session_id)
    subagents_tool._subagent_service = subagent_service
    sessions_list_tool = SessionsListTool(current_agent_id=agent_id)
    sessions_list_tool._get_session_manager = dependencies.get_session_manager
    sessions_history_tool = SessionsHistoryTool(
        current_agent_id=agent_id,
        current_session_id=effective_session_id,
    )
    sessions_history_tool._get_session_manager = (
        dependencies.get_session_manager
    )
    sessions_send_tool = SessionsSendTool(
        current_agent_id=agent_id,
        current_session_id=effective_session_id,
    )
    sessions_send_tool._get_session_manager = dependencies.get_session_manager
    return [
        AgentsListTool(current_agent_id=agent_id),
        sessions_list_tool,
        sessions_history_tool,
        sessions_send_tool,
        spawn_tool,
        subagents_tool,
    ]
