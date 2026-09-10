"""Persist a completed turn and run post-response maintenance."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnCompletionResult:
    done_content: str
    usage_info: dict[str, Any]
    context_utilization: float


class TurnCompletionService:
    """Own completion persistence without owning SSE yield order."""

    def __init__(
        self,
        *,
        save_message: Callable[..., Any],
        write_skills_snapshot: Callable[[str], None],
        count_tokens: Callable[[str], int],
        parse_text_tool_calls: Callable[[str], list[Any]],
        strip_tool_call_patterns: Callable[[str], str],
        should_persist_input_message: Callable[[str], bool],
        create_task: Callable[[Awaitable[None]], Any],
        incremental_ingest: Callable[..., Awaitable[None]],
        get_pending_tasks: Callable[[], set[Any]],
        maybe_auto_compact: Callable[..., Awaitable[None]],
        log: Any = logger,
    ) -> None:
        self._save_message = save_message
        self._write_skills_snapshot = write_skills_snapshot
        self._count_tokens = count_tokens
        self._parse_text_tool_calls = parse_text_tool_calls
        self._strip_tool_call_patterns = strip_tool_call_patterns
        self._should_persist_input_message = should_persist_input_message
        self._create_task = create_task
        self._incremental_ingest = incremental_ingest
        self._get_pending_tasks = get_pending_tasks
        self._maybe_auto_compact = maybe_auto_compact
        self._logger = log

    def finalize(
        self,
        *,
        request: Any,
        turn: Any,
        model_ref: str,
        full_response: str,
        tool_calls_log: list[dict[str, Any]],
        run_tracker: Any,
        audit_logger: Any,
    ) -> TurnCompletionResult:
        done_content = (
            self._strip_tool_call_patterns(full_response)
            if self._parse_text_tool_calls(full_response)
            else full_response
        )
        turn_tokens = (
            self._count_tokens(request.message)
            + self._count_tokens(full_response)
        )
        for tool_call in tool_calls_log:
            turn_tokens += self._count_tokens(
                str(tool_call.get("output", ""))
            )
        total_context = (
            request.prompt_tokens
            + request.history_tokens
            + turn_tokens
        )
        context_utilization = (
            round(total_context / request.active_tokens, 3)
            if request.active_tokens
            else 0
        )

        evaluation_mode = bool(getattr(request, "evaluation_mode", False))
        if not evaluation_mode and self._should_persist_input_message(request.persist_input_role):
            self._save_message(
                request.session_id,
                request.agent_id,
                request.persist_input_role,
                request.message,
            )
        content_to_save = (
            self._strip_tool_call_patterns(full_response)
            if self._parse_text_tool_calls(full_response)
            else full_response
        )
        if not evaluation_mode:
            self._save_message(
                request.session_id,
                request.agent_id,
                "assistant",
                content_to_save,
                tool_calls=tool_calls_log if tool_calls_log else None,
            )
            self._write_skills_snapshot(request.agent_id)
        completed = run_tracker.complete_turn(turn.run_id)

        if completed:
            try:
                if not evaluation_mode:
                    request.state.record_turn(
                        completed.input_tokens,
                        completed.output_tokens,
                    )
                audit_logger.log_turn_end(
                    request.agent_id,
                    turn.run_id,
                    request.session_id,
                    tokens={
                        "input": completed.input_tokens,
                        "output": completed.output_tokens,
                    },
                    tool_calls=len(tool_calls_log),
                    duration_ms=completed.duration_ms,
                )
            except Exception:
                self._logger.exception(
                    "Failed to record completed turn %s",
                    turn.run_id,
                )

        usage_info: dict[str, Any] = {}
        if completed:
            usage_info = {
                "input_tokens": completed.input_tokens,
                "output_tokens": completed.output_tokens,
                "total_tokens": completed.total_tokens,
                "duration_ms": completed.duration_ms,
                "model": model_ref,
            }
        return TurnCompletionResult(
            done_content=done_content,
            usage_info=usage_info,
            context_utilization=context_utilization,
        )

    async def run_follow_up(
        self,
        *,
        request: Any,
        turn: Any,
        done_content: str,
    ) -> None:
        if bool(getattr(request, "evaluation_mode", False)):
            return
        try:
            ingested_user_content = (
                request.message
                if request.persist_input_role == "user"
                else ""
            )
            pending_tasks = self._get_pending_tasks()
            ingest_kwargs: dict[str, Any] = {}
            parent_task_id = getattr(request, "parent_task_id", None)
            if parent_task_id:
                ingest_kwargs["parent_task_id"] = parent_task_id
            task = self._create_task(
                self._incremental_ingest(
                    request.agent_id,
                    request.session_id,
                    ingested_user_content,
                    done_content,
                    turn.run_id,
                    **ingest_kwargs,
                )
            )
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
        except Exception:
            self._logger.exception(
                "Failed to schedule turn ingestion for %s",
                turn.run_id,
            )

        try:
            await self._maybe_auto_compact(
                request.session_id,
                request.agent_id,
                overhead_tokens=(
                    request.prompt_tokens + request.summary_tokens
                ),
            )
        except Exception:
            self._logger.exception(
                "Failed to auto-compact after turn %s",
                turn.run_id,
            )
