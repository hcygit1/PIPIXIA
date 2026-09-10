from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.turn_completion import TurnCompletionService
from runtime.turn_models import TurnExecutionRequest


class TurnCompletionServiceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _request(state: object | None = None) -> TurnExecutionRequest:
        return TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=state or SimpleNamespace(record_turn=Mock()),
            provider="fake",
            model="model",
            message="question",
            persist_input_role="user",
            system_prompt="system",
            tools=[],
            history=[],
            recursion_limit=10,
            prompt_tokens=10,
            summary_tokens=4,
            history_tokens=20,
            active_tokens=100,
        )

    def test_finalize_persists_messages_and_records_usage(self) -> None:
        state = SimpleNamespace(record_turn=Mock())
        save_message = Mock()
        parse_text_tool_calls = Mock(return_value=[])
        run_tracker = SimpleNamespace(
            complete_turn=Mock(
                return_value=SimpleNamespace(
                    input_tokens=3,
                    output_tokens=5,
                    total_tokens=8,
                    duration_ms=12,
                )
            )
        )
        audit = SimpleNamespace(log_turn_end=Mock())
        service = TurnCompletionService(
            save_message=save_message,
            write_skills_snapshot=Mock(),
            count_tokens=lambda text: {
                "question": 7,
                "answer": 5,
                "tool output": 8,
            }[text],
            parse_text_tool_calls=parse_text_tool_calls,
            strip_tool_call_patterns=lambda content: content,
            should_persist_input_message=lambda role: bool(role.strip()),
            create_task=asyncio.create_task,
            incremental_ingest=AsyncMock(),
            get_pending_tasks=lambda: set(),
            maybe_auto_compact=AsyncMock(),
        )
        tool_calls = [
            {
                "tool": "read",
                "output": "tool output",
            }
        ]

        result = service.finalize(
            request=self._request(state),
            turn=SimpleNamespace(run_id="turn-1"),
            model_ref="fake/model",
            full_response="answer",
            tool_calls_log=tool_calls,
            run_tracker=run_tracker,
            audit_logger=audit,
        )

        self.assertEqual(result.done_content, "answer")
        self.assertEqual(result.context_utilization, 0.5)
        self.assertEqual(
            result.usage_info,
            {
                "input_tokens": 3,
                "output_tokens": 5,
                "total_tokens": 8,
                "duration_ms": 12,
                "model": "fake/model",
            },
        )
        self.assertEqual(
            save_message.call_args_list,
            [
                call("s1", "main", "user", "question"),
                call(
                    "s1",
                    "main",
                    "assistant",
                    "answer",
                    tool_calls=tool_calls,
                ),
            ],
        )
        self.assertEqual(parse_text_tool_calls.call_count, 2)
        run_tracker.complete_turn.assert_called_once_with("turn-1")
        state.record_turn.assert_called_once_with(3, 5)
        audit.log_turn_end.assert_called_once_with(
            "main",
            "turn-1",
            "s1",
            tokens={"input": 3, "output": 5},
            tool_calls=1,
            duration_ms=12,
        )

    async def test_follow_up_schedules_ingest_then_compacts(self) -> None:
        incremental_ingest = AsyncMock()
        maybe_auto_compact = AsyncMock()
        pending_tasks: set[asyncio.Task] = set()
        service = TurnCompletionService(
            save_message=Mock(),
            write_skills_snapshot=Mock(),
            count_tokens=lambda _text: 0,
            parse_text_tool_calls=lambda _content: [],
            strip_tool_call_patterns=lambda content: content,
            should_persist_input_message=lambda _role: True,
            create_task=asyncio.create_task,
            incremental_ingest=incremental_ingest,
            get_pending_tasks=lambda: pending_tasks,
            maybe_auto_compact=maybe_auto_compact,
        )
        request = self._request()

        await service.run_follow_up(
            request=request,
            turn=SimpleNamespace(run_id="turn-1"),
            done_content="answer",
        )
        await asyncio.gather(*tuple(pending_tasks))
        await asyncio.sleep(0)

        incremental_ingest.assert_awaited_once_with(
            "main",
            "s1",
            "question",
            "answer",
            "turn-1",
        )
        maybe_auto_compact.assert_awaited_once_with(
            "s1",
            "main",
            overhead_tokens=14,
        )
        self.assertEqual(pending_tasks, set())

    async def test_evaluation_mode_does_not_persist_or_ingest(self) -> None:
        save_message = Mock()
        incremental_ingest = AsyncMock()
        maybe_auto_compact = AsyncMock()
        service = TurnCompletionService(
            save_message=save_message,
            write_skills_snapshot=Mock(),
            count_tokens=lambda _text: 1,
            parse_text_tool_calls=lambda _content: [],
            strip_tool_call_patterns=lambda content: content,
            should_persist_input_message=lambda _role: True,
            create_task=asyncio.create_task,
            incremental_ingest=incremental_ingest,
            get_pending_tasks=lambda: set(),
            maybe_auto_compact=maybe_auto_compact,
        )
        request = TurnExecutionRequest(**{**self._request().__dict__, "evaluation_mode": True})
        service.finalize(
            request=request, turn=SimpleNamespace(run_id="eval-turn"), model_ref="fake/model",
            full_response="evaluation answer", tool_calls_log=[],
            run_tracker=SimpleNamespace(complete_turn=Mock(return_value=None)),
            audit_logger=SimpleNamespace(log_turn_end=Mock()),
        )
        await service.run_follow_up(request=request, turn=SimpleNamespace(run_id="eval-turn"), done_content="evaluation answer")
        save_message.assert_not_called()
        incremental_ingest.assert_not_awaited()
        maybe_auto_compact.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
