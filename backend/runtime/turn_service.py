"""Top-level orchestration for one Agent turn."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable

from runtime.turn_models import TurnExecutionRequest
from tools.skills_scanner import scan_skills_detailed


logger = logging.getLogger(__name__)

BARE_SESSION_RESET_PROMPT = (
    "A new session was started via /new or /reset. "
    "Greet the user in your configured persona "
    "(IDENTITY.md is already in your system prompt). "
    "Be yourself - use your defined voice, mannerisms, and mood. "
    "Keep it to 1-3 sentences and ask what they want to do. "
    "If the runtime model differs from default_model in the system "
    "prompt, mention the default model. "
    "Do not mention internal files, tools, memory status, or reasoning."
)


@dataclass(frozen=True)
class TurnServicePorts:
    get_state: Callable[[str], Any]
    parse_command: Callable[[str], Any]
    execute_command: Callable[..., Awaitable[dict[str, Any]]]
    switch_model: Callable[[str, str], str]
    get_current_model: Callable[[str], Any]
    get_model_override: Callable[[str], Any]
    handle_reset: Callable[..., AsyncGenerator[dict[str, Any], None]]
    handle_reset_noflush: Callable[
        ...,
        AsyncGenerator[dict[str, Any], None],
    ]
    handle_compact: Callable[
        ...,
        AsyncGenerator[dict[str, Any], None],
    ]
    write_skills_snapshot: Callable[[str], None]
    has_bootstrap: Callable[[str], bool]
    resolve_workspace: Callable[[str], Any]
    get_locale: Callable[[], str]
    get_tool_names: Callable[[str], tuple[str, ...]]
    build_prompt: Callable[..., tuple[str, Any, int]]
    get_session_context: Callable[..., Any]
    build_tools: Callable[[str, str], list[Any]]
    resolve_budget: Callable[[str], Any]
    resolve_agent_config: Callable[[str], dict[str, Any]]
    resolve_candidates: Callable[[str], list[Any]]
    execute_turn: Callable[
        [TurnExecutionRequest],
        AsyncGenerator[dict[str, Any], None],
    ]
    run_fallback_stream: Callable[
        ...,
        AsyncGenerator[dict[str, Any], None],
    ]
    recover_turn: Callable[
        ...,
        AsyncGenerator[dict[str, Any], None],
    ]


class TurnService:
    def __init__(self, ports: TurnServicePorts) -> None:
        self._ports = ports

    async def stream(
        self,
        message: str,
        session_id: str,
        *,
        agent_id: str = "main",
        prompt_mode: str = "full",
        persist_input_role: str = "user",
        parent_task_id: str | None = None,
        extra_system_prompt: str | None = None,
        evaluation_mode: bool = False,
    ) -> AsyncGenerator[dict[str, Any], None]:
        ports = self._ports
        state = ports.get_state(agent_id)

        parsed = ports.parse_command(message)
        if parsed:
            result = await ports.execute_command(
                parsed,
                agent_id,
                session_id,
                state,
                switch_model=ports.switch_model,
                get_current_model=ports.get_current_model,
            )
            if result.get("handled"):
                action = result.get("action", "")
                if action == "reset":
                    async for event in ports.handle_reset(
                        session_id,
                        agent_id,
                        model_override=result.get(
                            "model_override"
                        ),
                    ):
                        yield event
                    persist_input_role = ""
                    message = BARE_SESSION_RESET_PROMPT
                elif action == "reset_noflush":
                    async for event in (
                        ports.handle_reset_noflush(
                            session_id,
                            agent_id,
                        )
                    ):
                        yield event
                    persist_input_role = ""
                    message = BARE_SESSION_RESET_PROMPT
                elif action == "compact":
                    async for event in ports.handle_compact(
                        session_id,
                        agent_id,
                    ):
                        yield event
                    return
                else:
                    response = result["response"]
                    yield {
                        "type": "command_response",
                        "response": response,
                    }
                    yield {
                        "type": "done",
                        "content": response,
                        "session_id": session_id,
                    }
                    return

        ports.write_skills_snapshot(agent_id)

        extra_prompt = ""
        if ports.has_bootstrap(agent_id):
            try:
                ports.resolve_workspace(agent_id)
                extra_prompt = (
                    "\n\n## 首次运行引导\n\n"
                    "检测到 BOOTSTRAP.md，请先读取并执行其中的"
                    "引导步骤。完成后删除该文件。\n"
                )
            except Exception:
                pass

        model_override = ports.get_model_override(agent_id)
        if model_override is not None:
            extra_prompt += (
                "\n\n## 当前运行模型\n\n"
                f"当前运行模型：`{model_override}`。"
            )
        if extra_system_prompt:
            extra_prompt += f"\n\n{extra_system_prompt}"

        available_tool_names = list(
            ports.get_tool_names(agent_id)
        )
        system_prompt, prompt_report, prompt_tokens = (
            ports.build_prompt(
                agent_id=agent_id,
                prompt_mode=prompt_mode,
                available_tool_names=available_tool_names,
                extra_system_prompt=extra_prompt or None,
                locale=ports.get_locale(),
            )
        )
        logger.info(prompt_report.summary())

        context = ports.get_session_context(
            agent_id=agent_id,
            session_id=session_id,
        )
        history = context.pruned_history
        tools = [] if evaluation_mode else ports.build_tools(agent_id, session_id)
        budget = ports.resolve_budget(agent_id)
        recursion_limit = ports.resolve_agent_config(
            agent_id
        ).get("recursion_limit", 50)
        candidates = ports.resolve_candidates(agent_id)
        try:
            enabled_skills = [
                skill
                for skill in scan_skills_detailed(agent_id)
                if skill.get("enabled", True)
            ]
            observability_metadata = {
                "skill_names": [str(skill["name"]) for skill in enabled_skills],
                "skill_versions": {
                    str(skill["name"]): str(skill.get("version", "1.0"))
                    for skill in enabled_skills
                },
                "skill_count": len(enabled_skills),
            }
        except Exception as error:
            logger.warning("Unable to collect Skill metadata: %s", error)
            observability_metadata = {}
        if parent_task_id:
            observability_metadata.update({
                "task_id": parent_task_id,
                "parent_task_id": parent_task_id,
                "source_type": "subagent",
            })
        else:
            observability_metadata["source_type"] = "main_agent"

        async def run_for_model(
            provider: str,
            model: str,
        ):
            request = TurnExecutionRequest(
                agent_id=agent_id,
                session_id=session_id,
                state=state,
                provider=provider,
                model=model,
                message=message,
                persist_input_role=persist_input_role,
                parent_task_id=parent_task_id,
                system_prompt=system_prompt,
                tools=tools,
                history=history,
                recursion_limit=recursion_limit,
                prompt_tokens=prompt_tokens,
                summary_tokens=context.summary_tokens,
                history_tokens=context.history_tokens,
                active_tokens=budget.active_tokens,
                observability_metadata=observability_metadata,
                evaluation_mode=evaluation_mode,
            )
            async for event in ports.execute_turn(request):
                yield event

        async def run_fallback_stream():
            async for event in ports.run_fallback_stream(
                candidates,
                run_for_model,
                agent_id,
            ):
                yield event

        async for event in ports.recover_turn(
            agent_id=agent_id,
            session_id=session_id,
            state=state,
            stream=run_fallback_stream,
        ):
            yield event
