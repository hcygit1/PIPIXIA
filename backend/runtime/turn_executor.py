"""Execute one Agent turn against one concrete model."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable

from infra.event_bus import Events
from llm.model_selection import ModelCandidateError
from llm.models_config import ModelRef
from runtime.source_sink_guard import is_untrusted_source_tool
from runtime.text_tool_call_fallback import TextToolCallFallbackProcessor
from runtime.tool_call_parser import (
    parse_text_tool_calls,
    strip_tool_call_patterns,
)
from runtime.tool_execution import invoke_tool_async
from runtime.turn_completion import TurnCompletionService
from runtime.turn_event_stream import (
    TerminalTurnError as _TerminalTurnError,
    TurnEventStreamProcessor,
)
from runtime.turn_models import TurnExecutionRequest
from sandbox.loop_detection import LoopDetector
from tools.error_utils import format_tool_error


logger = logging.getLogger(__name__)


def should_persist_input_message(persist_input_role: str) -> bool:
    return bool((persist_input_role or "").strip())


def _new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex[:12]}"


def _infer_tool_result_status(
    output: str,
) -> tuple[str, str | None]:
    text = output or ""
    lowered = text.lower()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            status = str(parsed.get("status") or "").lower()
            if status == "error":
                return (
                    "error",
                    str(parsed.get("error") or "")[:500] or None,
                )
    except Exception:
        pass
    if (
        "命令被拒绝" in text
        or "command rejected" in lowered
        or "已拒绝" in text
    ):
        return "denied", text[:500]
    if "timed out" in lowered or "超时" in text:
        return "timeout", text[:500]
    if (
        "execution error" in lowered
        or "执行错误" in text
        or "执行出错" in text
    ):
        return "error", text[:500]
    return "success", None


def _loop_warning_is_breaker(warning: str) -> bool:
    return "全局熔断" in warning or "circuit" in warning.lower()


class TurnExecutor:
    def __init__(
        self,
        *,
        create_llm: Callable[[ModelRef], Any],
        build_messages: Callable[
            [list[dict[str, Any]], str],
            list[Any],
        ],
        get_lifecycle_hooks: Callable[[], Any],
        get_run_tracker: Callable[[], Any],
        get_audit_logger: Callable[[], Any],
        save_message: Callable[..., Any],
        write_skills_snapshot: Callable[[str], None],
        emit_event: Callable[[str, dict[str, Any]], None],
        count_tokens: Callable[[str], int],
        incremental_ingest: Callable[..., Awaitable[None]],
        get_pending_tasks: Callable[[], set[asyncio.Task]],
        maybe_auto_compact: Callable[..., Awaitable[None]],
    ) -> None:
        self._create_llm = create_llm
        self._build_messages = build_messages
        self._get_lifecycle_hooks = get_lifecycle_hooks
        self._get_run_tracker = get_run_tracker
        self._get_audit_logger = get_audit_logger
        self._save_message = save_message
        self._write_skills_snapshot = write_skills_snapshot
        self._emit_event = emit_event
        self._count_tokens = count_tokens
        self._incremental_ingest = incremental_ingest
        self._get_pending_tasks = get_pending_tasks
        self._maybe_auto_compact = maybe_auto_compact

    async def execute(
        self,
        request: TurnExecutionRequest,
    ) -> AsyncGenerator[dict[str, Any], None]:
        ref = ModelRef(
            provider=request.provider,
            model=request.model,
        )
        try:
            llm = self._create_llm(ref)
        except Exception as error:
            raise ModelCandidateError(
                str(error)
            ) from error

        try:
            from langgraph.prebuilt import create_react_agent

            agent = create_react_agent(
                model=llm,
                tools=request.tools,
                prompt=request.system_prompt,
            )
        except ImportError:
            yield {
                "type": "error",
                "error": "langgraph 未安装",
            }
            return
        except Exception as error:
            yield Events.turn_error(error=str(error))
            yield {
                "type": "error",
                "error": str(error),
            }
            return

        turn = None
        run_tracker = None
        audit_logger = None
        try:
            messages = self._build_messages(
                request.history,
                request.message,
            )
            run_tracker = self._get_run_tracker()
            audit_logger = self._get_audit_logger()
            turn = run_tracker.start_turn(
                request.agent_id,
                request.session_id,
            )
            audit_logger.log_turn_start(
                request.agent_id,
                turn.run_id,
                request.session_id,
            )
        except Exception as error:
            error_text = str(error)
            if turn is not None and run_tracker is not None:
                run_tracker.error_turn(
                    turn.run_id,
                    error_text,
                )
                if audit_logger is not None:
                    try:
                        audit_logger.log_turn_error(
                            request.agent_id,
                            turn.run_id,
                            error_text,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to audit setup error "
                            "for turn %s",
                            turn.run_id,
                        )
            yield Events.turn_error(error=error_text)
            yield {
                "type": "error",
                "error": error_text,
            }
            return
        turn_start_event = Events.turn_start(
            run_id=turn.run_id,
            model=str(ref),
        )
        turn_start_emitted = False

        loop_detector = LoopDetector()
        event_stream = TurnEventStreamProcessor(
            agent=agent,
            request=request,
            messages=messages,
            turn=turn,
            turn_start_event=turn_start_event,
            run_tracker=run_tracker,
            audit_logger=audit_logger,
            get_lifecycle_hooks=self._get_lifecycle_hooks,
            emit_event=self._emit_event,
            loop_detector=loop_detector,
            new_tool_call_id=_new_tool_call_id,
            infer_tool_result_status=_infer_tool_result_status,
            loop_warning_is_breaker=_loop_warning_is_breaker,
            parse_text_tool_calls=parse_text_tool_calls,
            strip_tool_call_patterns=strip_tool_call_patterns,
        )
        full_response = ""
        tool_calls_log: list[dict[str, Any]] = []
        step_count = 0
        content_refresh_sent = False
        recent_untrusted_content = (
            event_stream.state.recent_untrusted_content
        )

        try:
            async for output_event in event_stream.stream():
                turn_start_emitted = (
                    event_stream.state.turn_start_emitted
                )
                step_count = event_stream.state.step_count
                yield output_event
            stream_state = event_stream.state
            turn_start_emitted = stream_state.turn_start_emitted
            full_response = stream_state.full_response
            tool_calls_log = stream_state.tool_calls_log
            step_count = stream_state.step_count
            content_refresh_sent = stream_state.content_refresh_sent
            recent_untrusted_content = (
                stream_state.recent_untrusted_content
            )
        except Exception as error:
            root_error = (
                error.__cause__
                if isinstance(error, ModelCandidateError)
                and error.__cause__ is not None
                else error
            )
            error_text = str(root_error)
            is_recursion = (
                "recursion" in error_text.lower()
                or "GraphRecursionError"
                in type(root_error).__name__
            )
            run_tracker.error_turn(turn.run_id, error_text)
            audit_logger.log_turn_error(
                request.agent_id,
                turn.run_id,
                error_text,
            )
            if is_recursion:
                if not turn_start_emitted:
                    yield turn_start_event
                    turn_start_emitted = True
                yield Events.recursion_limit_reached(
                    step=step_count,
                    max_steps=request.recursion_limit,
                )
                yield {
                    "type": "error",
                    "error": (
                        "Agent 达到最大迭代次数 "
                        f"({request.recursion_limit})，已自动停止。"
                        f"已执行 {step_count} 步工具调用。"
                    ),
                }
            elif (
                isinstance(error, _TerminalTurnError)
                or not isinstance(error, ModelCandidateError)
            ):
                if not turn_start_emitted:
                    yield turn_start_event
                    turn_start_emitted = True
                yield Events.turn_error(error=error_text)
                yield {
                    "type": "error",
                    "error": error_text,
                }
            else:
                raise
            return

        try:
            fallback = TextToolCallFallbackProcessor(
                request=request,
                turn=turn,
                state=stream_state,
                run_tracker=run_tracker,
                audit_logger=audit_logger,
                emit_event=self._emit_event,
                loop_detector=loop_detector,
                parse_text_tool_calls=parse_text_tool_calls,
                strip_tool_call_patterns=strip_tool_call_patterns,
                invoke_tool_async=invoke_tool_async,
                format_tool_error=format_tool_error,
                is_untrusted_source_tool=is_untrusted_source_tool,
                new_tool_call_id=_new_tool_call_id,
                infer_tool_result_status=_infer_tool_result_status,
                loop_warning_is_breaker=_loop_warning_is_breaker,
                log=logger,
            )
            async for output_event in fallback.stream():
                yield output_event
            fallback_state = fallback.state
            full_response = fallback_state.full_response
            tool_calls_log = fallback_state.tool_calls_log
            step_count = fallback_state.step_count
            content_refresh_sent = (
                fallback_state.content_refresh_sent
            )
            recent_untrusted_content = (
                fallback_state.recent_untrusted_content
            )

            completion_service = TurnCompletionService(
                save_message=self._save_message,
                write_skills_snapshot=self._write_skills_snapshot,
                count_tokens=self._count_tokens,
                parse_text_tool_calls=parse_text_tool_calls,
                strip_tool_call_patterns=strip_tool_call_patterns,
                should_persist_input_message=(
                    should_persist_input_message
                ),
                create_task=asyncio.create_task,
                incremental_ingest=self._incremental_ingest,
                get_pending_tasks=self._get_pending_tasks,
                maybe_auto_compact=self._maybe_auto_compact,
                log=logger,
            )
            completion = completion_service.finalize(
                request=request,
                turn=turn,
                model_ref=str(ref),
                full_response=full_response,
                tool_calls_log=tool_calls_log,
                run_tracker=run_tracker,
                audit_logger=audit_logger,
            )
        except Exception as error:
            error_text = str(error)
            run_tracker.error_turn(
                turn.run_id,
                error_text,
            )
            audit_logger.log_turn_error(
                request.agent_id,
                turn.run_id,
                error_text,
            )
            yield Events.turn_error(error=error_text)
            yield {
                "type": "error",
                "error": error_text,
            }
            return

        yield Events.turn_end(
            run_id=turn.run_id,
            usage=completion.usage_info,
        )
        yield {
            "type": "done",
            "content": completion.done_content,
            "session_id": request.session_id,
            "usage": completion.usage_info,
            "context_utilization": completion.context_utilization,
        }

        await completion_service.run_follow_up(
            request=request,
            turn=turn,
            done_content=completion.done_content,
        )
