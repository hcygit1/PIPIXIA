"""子 Agent 单次运行执行器。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sessions.session_identity import session_key_from_session_id


class SubagentRunner:
    _FAILURE_HINTS = (
        "failure",
        "error",
        "exception",
        "timeout",
        "not found",
        "return none",
        "no results",
        "cannot",
        "failed",
        "无结果",
        "无法",
        "失败",
    )

    def __init__(
        self,
        astream: Any,
        requester_agent_id: str,
        *,
        registry: Any = None,
        session_manager: Any = None,
        event_bus: Any = None,
        delivery: Any = None,
    ) -> None:
        if registry is None:
            from subagents.subagent_registry import registry
        if session_manager is None:
            from sessions.session_manager import session_manager
        if event_bus is None:
            from infra.event_bus import event_bus
        if delivery is None:
            from subagents.subagent_delivery import (
                subagent_announce_delivery as delivery,
            )

        self._astream = astream
        self._requester_agent_id = requester_agent_id
        self._registry = registry
        self._session_manager = session_manager
        self._event_bus = event_bus
        self._delivery = delivery

    def start(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_id: str,
        task: str,
        requester_key: str,
        parent_task_id: str | None = None,
        run_timeout_seconds: float = 0,
    ) -> asyncio.Task[None]:
        async_task = asyncio.create_task(
            self.run(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                task=task,
                requester_key=requester_key,
                parent_task_id=parent_task_id,
                run_timeout_seconds=run_timeout_seconds,
            )
        )
        self._registry.set_task(run_id, async_task)
        return async_task

    def _looks_like_failure_output(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        return any(hint in normalized for hint in self._FAILURE_HINTS)

    def collect_latest_output(
        self,
        *,
        session_id: str,
        agent_id: str,
        streamed_text: str,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """优先使用流式文本；为空时回读会话和工具输出。"""
        if (streamed_text or "").strip():
            result = streamed_text.strip()
            return result, self._looks_like_failure_output(result)

        data = self._session_manager.load_session(
            session_id,
            agent_id,
        ) or {}
        messages = data.get("messages", []) if isinstance(data, dict) else []
        latest_assistant: dict[str, Any] | None = None
        for message in reversed(messages):
            if message.get("role") == "assistant":
                latest_assistant = message
                break

        content = (
            (latest_assistant or {}).get("content", "")
            if latest_assistant
            else ""
        )
        stored_tool_calls = (
            (latest_assistant or {}).get("tool_calls", [])
            if latest_assistant
            else []
        )
        merged_tool_calls = [*tool_calls, *stored_tool_calls]

        snippets: list[str] = []
        failure_count = 0
        for tool_call in merged_tool_calls:
            tool = str(tool_call.get("tool", "")).strip() or "tool"
            output = str(tool_call.get("output", "")).strip()
            if not output:
                continue
            if self._looks_like_failure_output(output):
                failure_count += 1
            snippets.append(f"[{tool}] {output[:280]}")

        parts: list[str] = []
        if (content or "").strip():
            parts.append(content.strip())
        if snippets:
            parts.append(
                "Summary of tool outputs:\n"
                + "\n".join(snippets[:4])
            )

        result = "\n\n".join(parts).strip()
        all_failed = (
            failure_count > 0
            and failure_count >= max(1, len(snippets))
        )
        return result, all_failed

    async def run(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_id: str,
        task: str,
        requester_key: str,
        parent_task_id: str | None = None,
        run_timeout_seconds: float = 0,
    ) -> None:
        from infra.event_bus import Events

        result_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        child_session_key = session_key_from_session_id(
            agent_id,
            session_id,
        )
        started_at: float | None = None

        try:
            self._registry.mark_started(run_id)
            active_record = self._registry.get_run(run_id)
            if active_record is None or active_record.ended_at is not None:
                return
            started_at = active_record.started_at or time.time()
            self._event_bus.emit(
                self._requester_agent_id,
                Events.subagent_start(
                    run_id=run_id,
                    agent_id=agent_id,
                    task=task[:200],
                ),
            )

            async def stream_child() -> None:
                last_progress_emit = 0.0
                chars_received = 0
                async for event in self._astream(
                    message=task,
                    session_id=session_id,
                    agent_id=agent_id,
                    prompt_mode="minimal",
                    parent_task_id=parent_task_id,
                ):
                    event_type = event.get("type")
                    if event_type == "token":
                        token = event.get("content", "") or ""
                        result_parts.append(token)
                        chars_received += len(token)
                        now = time.time()
                        if now - last_progress_emit >= 1.0:
                            last_progress_emit = now
                            self._event_bus.emit(
                                self._requester_agent_id,
                                Events.subagent_progress(
                                    run_id=run_id,
                                    chars=chars_received,
                                    elapsed_s=int(
                                        now - (started_at or now)
                                    ),
                                ),
                            )
                    elif event_type == "tool_start":
                        self._event_bus.emit(
                            self._requester_agent_id,
                            Events.subagent_tool(
                                run_id=run_id,
                                tool=event.get("tool", ""),
                            ),
                        )
                    elif event_type == "tool_end":
                        output = event.get("output", "") or ""
                        tool_calls.append({
                            "tool": event.get("tool", ""),
                            "output": output,
                        })
                        self._event_bus.emit(
                            self._requester_agent_id,
                            Events.subagent_tool_end(
                                run_id=run_id,
                                tool=event.get("tool", ""),
                                output_preview=str(output)[:160],
                            ),
                        )

            if run_timeout_seconds > 0:
                await asyncio.wait_for(
                    stream_child(),
                    timeout=run_timeout_seconds,
                )
            else:
                await stream_child()

            result, all_failed = self.collect_latest_output(
                session_id=session_id,
                agent_id=agent_id,
                streamed_text="".join(result_parts).strip(),
                tool_calls=tool_calls,
            )
            outcome_key = (
                "completed-empty"
                if not result
                else "completed-with-errors"
                if all_failed
                else "completed"
            )
            self._registry.mark_completed(
                run_id,
                result,
                outcome=outcome_key,
                terminal_reason=(
                    "all-tools-failed" if all_failed else None
                ),
            )
            completed = self._registry.get_run(run_id)
            if completed is None or completed.state != "succeeded":
                return

            self._event_bus.emit(
                self._requester_agent_id,
                Events.subagent_done(
                    run_id=run_id,
                    result=result[:300],
                ),
            )
            announce_outcome = "completed successfully"
            if outcome_key == "completed-empty":
                announce_outcome = "completed with empty output"
            elif outcome_key == "completed-with-errors":
                announce_outcome = "completed with tool errors"
            await self._delivery.deliver_to_requester(
                requester_key=requester_key,
                child_session_key=child_session_key,
                run_id=run_id,
                task=task,
                result=result,
                outcome=announce_outcome,
                label=completed.label,
                started_at=started_at,
                ended_at=completed.ended_at,
            )
        except asyncio.TimeoutError:
            result, _ = self.collect_latest_output(
                session_id=session_id,
                agent_id=agent_id,
                streamed_text="".join(result_parts).strip(),
                tool_calls=tool_calls,
            )
            fallback_result = result or (
                "Sub-agent execution timed out "
                f"({run_timeout_seconds}s)"
            )
            self._registry.mark_terminated(run_id, "timeout")
            timed_out = self._registry.get_run(run_id)
            if timed_out is None or timed_out.state != "timed_out":
                return
            self._event_bus.emit(
                self._requester_agent_id,
                Events.subagent_error(
                    run_id=run_id,
                    error=(
                        "timeout after "
                        f"{run_timeout_seconds}s"
                    ),
                ),
            )
            await self._delivery.deliver_to_requester(
                requester_key=requester_key,
                child_session_key=child_session_key,
                run_id=run_id,
                task=task,
                result=fallback_result,
                outcome="timed out",
                label=timed_out.label,
                started_at=started_at,
                ended_at=timed_out.ended_at,
            )
        except asyncio.CancelledError:
            self._registry.mark_terminated(run_id, "killed")
            cancelled = self._registry.get_run(run_id)
            if cancelled is not None and cancelled.state == "cancelled":
                self._event_bus.emit(
                    self._requester_agent_id,
                    Events.subagent_killed(run_id=run_id),
                )
        except Exception as exc:
            self._registry.mark_terminated(
                run_id,
                f"error: {exc}",
            )
            failed = self._registry.get_run(run_id)
            if failed is None or failed.state != "failed":
                return
            self._event_bus.emit(
                self._requester_agent_id,
                Events.subagent_error(
                    run_id=run_id,
                    error=str(exc)[:200],
                ),
            )
