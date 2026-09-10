from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.agent import AgentManager
from runtime.turn_executor import TurnExecutor
from runtime.turn_models import TurnExecutionRequest


class TurnExecutorTests(unittest.IsolatedAsyncioTestCase):
    def _build_executor(
        self,
        *,
        create_llm=None,
        tracker=None,
        audit=None,
        save_message=None,
        count_tokens=None,
        maybe_auto_compact=None,
    ) -> TurnExecutor:
        tracker = tracker or SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = audit or SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        return TurnExecutor(
            create_llm=create_llm or (lambda _ref: object()),
            build_messages=lambda history, message: [
                *history,
                {"role": "user", "content": message},
            ],
            get_lifecycle_hooks=lambda: None,
            get_run_tracker=lambda: tracker,
            get_audit_logger=lambda: audit,
            save_message=save_message or Mock(),
            write_skills_snapshot=lambda _agent_id: None,
            emit_event=lambda _agent_id, _event: None,
            count_tokens=(
                count_tokens
                or (lambda _text: 0)
            ),
            incremental_ingest=AsyncMock(),
            get_pending_tasks=lambda: set(),
            maybe_auto_compact=(
                maybe_auto_compact
                or AsyncMock()
            ),
        )

    @staticmethod
    def _build_request(
        *,
        state=None,
        tools=None,
    ) -> TurnExecutionRequest:
        return TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=state or SimpleNamespace(record_turn=Mock()),
            provider="fake",
            model="model",
            message="question",
            persist_input_role="user",
            system_prompt="system",
            tools=tools or [],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )

    async def test_execute_raises_llm_initialization_error_for_fallback(
        self,
    ) -> None:
        executor = self._build_executor(
            create_llm=Mock(
                side_effect=RuntimeError(
                    "503 upstream unavailable"
                )
            )
        )

        with self.assertRaises(RuntimeError) as raised:
            async for _event in executor.execute(
                self._build_request()
            ):
                pass

        self.assertEqual(
            str(raised.exception),
            "503 upstream unavailable",
        )

    async def test_execute_raises_model_stream_error_for_fallback(
        self,
    ) -> None:
        class _FailingReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                if False:
                    yield {}
                raise RuntimeError("503 upstream unavailable")

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FailingReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
        )

        events = []
        with (
            patch.dict(
                sys.modules,
                {
                    "langgraph": fake_langgraph,
                    "langgraph.prebuilt": fake_prebuilt,
                },
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "503 upstream unavailable",
            ),
        ):
            async for event in executor.execute(
                self._build_request()
            ):
                events.append(event)

        self.assertEqual(events, [])
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "503 upstream unavailable",
        )
        audit.log_turn_error.assert_called_once_with(
            "main",
            "turn-1",
            "503 upstream unavailable",
        )

    async def test_execute_keeps_recursion_limit_as_terminal_event(
        self,
    ) -> None:
        class _RecursiveReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                if False:
                    yield {}
                raise RuntimeError("recursion limit reached")

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _RecursiveReactAgent()
        )
        executor = self._build_executor()

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "lifecycle", "error"],
        )
        self.assertIn("最大迭代次数", events[-1]["error"])

    async def test_execute_keeps_runtime_hook_error_terminal(
        self,
    ) -> None:
        class _ToolReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_tool_start",
                    "run_id": "tool-run",
                    "name": "read",
                    "data": {"input": {"path": "note.txt"}},
                }

        class _FailingHooks:
            async def on_before_tool_call(
                self,
                *_args,
                **_kwargs,
            ):
                raise RuntimeError("hook failed")

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _ToolReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
        )
        executor._get_lifecycle_hooks = (
            lambda: _FailingHooks()
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "lifecycle", "error"],
        )
        self.assertEqual(events[-1]["error"], "hook failed")
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "hook failed",
        )
        audit.log_turn_error.assert_called_once_with(
            "main",
            "turn-1",
            "hook failed",
        )

    async def test_text_tool_breaker_records_error_not_completion(
        self,
    ) -> None:
        text_call = 'functions.read:0{"path":"note.txt"}'

        class _TextToolReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=text_call,
                            usage_metadata=None,
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        class _ReadTool:
            name = "read"

            async def ainvoke(self, _tool_input):
                return "content"

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _TextToolReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
        )
        warning = "[安全警告] 触发全局熔断。"

        with (
            patch.dict(
                sys.modules,
                {
                    "langgraph": fake_langgraph,
                    "langgraph.prebuilt": fake_prebuilt,
                },
            ),
            patch(
                "runtime.turn_executor.LoopDetector.record",
                return_value=warning,
            ),
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request(
                        tools=[_ReadTool()]
                    )
                )
            ]

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["error"], warning)
        tracker.complete_turn.assert_not_called()
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            warning,
        )
        audit.log_turn_end.assert_not_called()
        audit.log_turn_error.assert_called_once_with(
            "main",
            "turn-1",
            warning,
        )

    async def test_text_tool_runtime_error_closes_turn(
        self,
    ) -> None:
        text_call = 'functions.read:0{"path":"note.txt"}'

        class _TextToolReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content=text_call,
                            usage_metadata=None,
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        class _ReadTool:
            name = "read"

            async def ainvoke(self, _tool_input):
                return "content"

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _TextToolReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "langgraph": fake_langgraph,
                    "langgraph.prebuilt": fake_prebuilt,
                },
            ),
            patch(
                "runtime.turn_executor.LoopDetector.record",
                side_effect=RuntimeError(
                    "loop detector failed"
                ),
            ),
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request(
                        tools=[_ReadTool()]
                    )
                )
            ]

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["error"],
            "loop detector failed",
        )
        tracker.complete_turn.assert_not_called()
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "loop detector failed",
        )
        audit.log_turn_end.assert_not_called()
        audit.log_turn_error.assert_called_once_with(
            "main",
            "turn-1",
            "loop detector failed",
        )

    async def test_persistence_error_closes_turn_before_completion(
        self,
    ) -> None:
        class _FakeReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=None,
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FakeReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
            save_message=Mock(
                side_effect=RuntimeError("save failed")
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["error"], "save failed")
        tracker.complete_turn.assert_not_called()
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "save failed",
        )
        audit.log_turn_end.assert_not_called()
        audit.log_turn_error.assert_called_once_with(
            "main",
            "turn-1",
            "save failed",
        )

    async def test_audit_start_error_closes_started_turn(
        self,
    ) -> None:
        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: object()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(
                side_effect=RuntimeError(
                    "audit start failed"
                )
            ),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "error"],
        )
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "audit start failed",
        )
        tracker.complete_turn.assert_not_called()

    async def test_token_count_error_closes_turn_before_completion(
        self,
    ) -> None:
        class _FakeReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=None,
                        )
                    },
                }

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FakeReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
            count_tokens=Mock(
                side_effect=RuntimeError(
                    "token count failed"
                )
            ),
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(
            events[-1]["error"],
            "token count failed",
        )
        tracker.complete_turn.assert_not_called()
        tracker.error_turn.assert_called_once_with(
            "turn-1",
            "token count failed",
        )

    async def test_auto_compaction_error_does_not_reverse_done(
        self,
    ) -> None:
        class _FakeReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=None,
                        )
                    },
                }

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FakeReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        maybe_auto_compact = AsyncMock(
            side_effect=RuntimeError(
                "auto compact failed"
            )
        )
        executor = self._build_executor(
            tracker=tracker,
            audit=audit,
            maybe_auto_compact=maybe_auto_compact,
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(
                    self._build_request()
                )
            ]

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "token", "lifecycle", "done"],
        )
        tracker.error_turn.assert_not_called()
        maybe_auto_compact.assert_awaited_once()

    async def test_agent_manager_resolves_turn_state_once(
        self,
    ) -> None:
        class _FakeReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=None,
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        async def _run_with_fallback(
            _candidates,
            run_model,
            _agent_id,
        ):
            async for event in run_model("fake", "model"):
                yield event

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FakeReactAgent()
        )
        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(
                return_value=SimpleNamespace(
                    input_tokens=3,
                    output_tokens=5,
                    total_tokens=8,
                    duration_ms=10,
                )
            ),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
            log=Mock(),
        )
        state = SimpleNamespace(record_turn=Mock())
        manager = AgentManager()
        get_state = Mock(return_value=state)
        manager.get_state = get_state

        with (
            patch.object(
                manager,
                "_get_or_build_tool_names",
                return_value=(),
            ),
            patch.object(
                manager,
                "_get_or_build_prompt",
                return_value=(
                    "system prompt",
                    SimpleNamespace(summary=lambda: ""),
                    0,
                ),
            ),
            patch.object(
                manager,
                "_get_or_build_session_context",
                return_value=SimpleNamespace(
                    pruned_history=[],
                    summary_tokens=0,
                    history_tokens=0,
                ),
            ),
            patch.object(manager, "_build_tools", return_value=[]),
            patch.object(
                manager,
                "_incremental_ingest",
                new=AsyncMock(),
            ),
            patch.object(
                manager,
                "_maybe_auto_compact",
                new=AsyncMock(),
            ),
            patch(
                "tools.skills_scanner.write_skills_snapshot"
            ),
            patch(
                "runtime.workspace.has_bootstrap",
                return_value=False,
            ),
            patch(
                "runtime.agent.resolve_fallback_candidates",
                return_value=[
                    SimpleNamespace(
                        provider="fake",
                        model="model",
                    )
                ],
            ),
            patch(
                "runtime.agent.run_with_fallback_stream",
                side_effect=_run_with_fallback,
            ),
            patch(
                "runtime.agent.resolve_agent_config",
                return_value={"recursion_limit": 10},
            ),
            patch(
                "runtime.agent.create_llm",
                return_value=object(),
            ),
            patch("runtime.agent.run_tracker", tracker),
            patch("runtime.agent.audit_logger", audit),
            patch(
                "runtime.agent.session_manager.save_message"
            ),
            patch(
                "runtime.context_budget.resolve_budget",
                return_value=SimpleNamespace(
                    active_tokens=1000
                ),
            ),
            patch(
                "runtime.agent.count_tokens",
                return_value=0,
            ),
            patch.dict(
                sys.modules,
                {
                    "langgraph": fake_langgraph,
                    "langgraph.prebuilt": fake_prebuilt,
                },
            ),
        ):
            events = [
                event
                async for event in manager.astream(
                    "question",
                    "s1",
                    agent_id="main",
                )
            ]
        await asyncio.sleep(0)

        self.assertEqual(events[-1]["type"], "done")
        get_state.assert_called_once_with("main")
        state.record_turn.assert_called_once_with(3, 5)

    async def test_execute_streams_and_persists_a_complete_turn(
        self,
    ) -> None:
        class _FakeReactAgent:
            async def astream_events(
                self,
                _payload,
                version="v2",
                config=None,
            ):
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "model-run",
                    "data": {
                        "chunk": SimpleNamespace(
                            content="answer",
                            usage_metadata=None,
                        )
                    },
                }
                yield {
                    "event": "on_chat_model_end",
                    "run_id": "model-run",
                    "data": {},
                }

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = (
            lambda **_kwargs: _FakeReactAgent()
        )

        tracker = SimpleNamespace(
            start_turn=Mock(
                return_value=SimpleNamespace(run_id="turn-1")
            ),
            complete_turn=Mock(return_value=None),
            error_turn=Mock(),
            record_tokens=Mock(),
            record_tool_start=Mock(),
            record_tool_end=Mock(),
        )
        audit = SimpleNamespace(
            log_turn_start=Mock(),
            log_turn_end=Mock(),
            log_turn_error=Mock(),
            log_tool_call=Mock(),
            log_tool_loop_warning=Mock(),
        )
        state = SimpleNamespace(record_turn=Mock())
        save_message = Mock()
        incremental_ingest = AsyncMock()
        maybe_auto_compact = AsyncMock()
        pending_tasks: set[asyncio.Task] = set()
        executor = TurnExecutor(
            create_llm=lambda _ref: object(),
            build_messages=lambda history, message: [
                *history,
                {"role": "user", "content": message},
            ],
            get_lifecycle_hooks=lambda: None,
            get_run_tracker=lambda: tracker,
            get_audit_logger=lambda: audit,
            save_message=save_message,
            write_skills_snapshot=lambda _agent_id: None,
            emit_event=lambda _agent_id, _event: None,
            count_tokens=lambda _text: 0,
            incremental_ingest=incremental_ingest,
            get_pending_tasks=lambda: pending_tasks,
            maybe_auto_compact=maybe_auto_compact,
        )
        request = TurnExecutionRequest(
            agent_id="main",
            session_id="s1",
            state=state,
            provider="fake",
            model="model",
            message="question",
            persist_input_role="user",
            system_prompt="system",
            tools=[],
            history=[],
            recursion_limit=10,
            prompt_tokens=0,
            summary_tokens=0,
            history_tokens=0,
            active_tokens=1000,
        )

        with patch.dict(
            sys.modules,
            {
                "langgraph": fake_langgraph,
                "langgraph.prebuilt": fake_prebuilt,
            },
        ):
            events = [
                event
                async for event in executor.execute(request)
            ]
        await asyncio.sleep(0)

        self.assertEqual(
            [event["type"] for event in events],
            ["lifecycle", "token", "lifecycle", "done"],
        )
        self.assertEqual(events[1]["content"], "answer")
        self.assertEqual(events[-1]["content"], "answer")
        self.assertEqual(
            save_message.call_args_list[0].args,
            ("s1", "main", "user", "question"),
        )
        self.assertEqual(
            save_message.call_args_list[1].args[:4],
            ("s1", "main", "assistant", "answer"),
        )
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
            overhead_tokens=0,
        )


if __name__ == "__main__":
    unittest.main()
