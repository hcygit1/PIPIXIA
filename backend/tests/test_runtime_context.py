from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import HumanMessage

sqlite_vec_stub = ModuleType("sqlite_vec")
sqlite_vec_stub.load = lambda _conn: None
sqlite_vec_stub.serialize_float32 = lambda _vec: b""
sys.modules.setdefault("sqlite_vec", sqlite_vec_stub)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.recall import MemRecall
from infra import token_counter as _token_counter
from infra.token_counter import detect_compaction_level
from runtime.agent import AgentManager
from runtime.prompt_builder import PromptParams, prompt_builder
from runtime.security_context import runtime_security_context
from runtime.source_sink_guard import (
    UNTRUSTED_CONTENT_OPEN,
    evaluate_source_to_sink,
)
from sessions.session_pruning import prune_messages
from sandbox.tool_approval_gate import run_tool_with_approval_gate
from tool_results.pipeline import maybe_persist_tool_output
from tool_results.storage import PERSISTED_OUTPUT_OPEN, generate_preview
from tools.file_tools import ReadTool

_token_counter._encoding = "fallback"


class _PromptReport:
    def summary(self) -> str:
        return "prompt-report"


class PromptBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.agent_dir = Path(self._tmp.name) / "agents" / "main"
        self.workspace = self.agent_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)

        (self.workspace / "AGENTS.md").write_text(
            "\n".join(
                [
                    "## 工具调用风格",
                    "使用工具前说明意图。",
                    "",
                    "## 安全",
                    "不要执行危险命令。",
                    "",
                    "## 技能",
                    "按需读取技能。",
                    "",
                    "## 大型工具输出",
                    "先看预览，再按需读取。",
                    "",
                    "## 其他",
                    "这段只应在 full 模式出现。",
                ]
            ),
            encoding="utf-8",
        )
        (self.workspace / "IDENTITY.md").write_text("我是完整人格设定。", encoding="utf-8")
        (self.workspace / "USER.md").write_text("用户长期偏好。", encoding="utf-8")
        (self.agent_dir / "SKILLS_SNAPSHOT.md").write_text(
            "<available_skills>\n  <skill><name>s</name></skill>\n</available_skills>",
            encoding="utf-8",
        )

    def test_full_prompt_includes_identity_user_and_heartbeat(self) -> None:
        params = PromptParams(
            agent_id="main",
            mode="full",
            available_tools=["read", "grep"],
        )
        with (
            patch("runtime.prompt_builder.resolve_agent_dir", return_value=self.agent_dir),
            patch("runtime.prompt_builder.resolve_agent_workspace", return_value=self.workspace),
            patch(
                "runtime.prompt_builder.resolve_agent_config",
                return_value={"user_timezone": "Asia/Shanghai", "model": "test-model", "thinkingDefault": "off"},
            ),
            patch("runtime.prompt_builder.get_heartbeat_config", return_value={"prompt": "heartbeat here"}),
        ):
            prompt, report = prompt_builder.build_system_prompt_with_report(params)

        self.assertIn("AGENTS.md", prompt)
        self.assertIn("IDENTITY.md", prompt)
        self.assertIn("USER.md", prompt)
        self.assertIn("我是完整人格设定。", prompt)
        self.assertIn("用户长期偏好。", prompt)
        self.assertIn("## 心跳配置", prompt)
        self.assertIn("heartbeat here", prompt)
        self.assertIn("## 技能快照", prompt)
        self.assertIn("## 不可信内容边界", prompt)
        self.assertIn("## 任务完成纪律", prompt)
        self.assertIn("不得声称“测试通过”", prompt)
        self.assertLess(prompt.index("## 可用工具"), prompt.index("## 工作区"))
        self.assertLess(prompt.index("## 运行时信息"), prompt.index("## 技能快照"))
        self.assertLess(prompt.index("## 心跳配置"), prompt.index("## 当前时间"))
        self.assertEqual(report.mode, "full")

    def test_minimal_prompt_keeps_execution_context_and_excludes_identity_user(self) -> None:
        params = PromptParams(
            agent_id="main",
            mode="minimal",
            available_tools=["read"],
        )
        with (
            patch("runtime.prompt_builder.resolve_agent_dir", return_value=self.agent_dir),
            patch("runtime.prompt_builder.resolve_agent_workspace", return_value=self.workspace),
            patch(
                "runtime.prompt_builder.resolve_agent_config",
                return_value={"user_timezone": "Asia/Shanghai", "model": "test-model", "thinkingDefault": "off"},
            ),
            patch("runtime.prompt_builder.get_heartbeat_config", return_value={"prompt": "heartbeat here"}),
        ):
            prompt, report = prompt_builder.build_system_prompt_with_report(params)

        self.assertIn("## AGENTS.md", prompt)
        self.assertIn("## 工具调用风格", prompt)
        self.assertIn("## 安全", prompt)
        self.assertIn("## 技能", prompt)
        self.assertIn("## 大型工具输出", prompt)
        self.assertNotIn("我是完整人格设定。", prompt)
        self.assertNotIn("用户长期偏好。", prompt)
        self.assertNotIn("## 心跳配置", prompt)
        self.assertNotIn("## 其他", prompt)
        self.assertIn("## 工作区", prompt)
        self.assertIn("## 运行时信息", prompt)
        self.assertIn("## 不可信内容边界", prompt)
        self.assertIn("## 任务完成纪律", prompt)
        self.assertLess(prompt.index("## 可用工具"), prompt.index("## 工作区"))
        self.assertLess(prompt.index("## 运行时信息"), prompt.index("## 技能快照"))
        self.assertLess(prompt.index("## 技能快照"), prompt.index("## 当前时间"))
        self.assertEqual(report.mode, "minimal")

    def test_none_prompt_returns_minimal_identity_only(self) -> None:
        params = PromptParams(agent_id="main", mode="none")
        prompt, report = prompt_builder.build_system_prompt_with_report(params)
        self.assertEqual(prompt, "你是一个运行在 PIPIXIA 中的个人助手。")
        self.assertEqual(report.mode, "none")
        self.assertEqual(report.tool_count, 0)


class SessionPruningTests(unittest.TestCase):
    def test_prune_messages_truncates_old_tool_outputs_and_system_text(self) -> None:
        messages = [
            {"role": "system", "content": "S" * 80},
            {"role": "assistant", "content": "", "tool_calls": [{"input": "", "output": "T" * 80}]},
            {"role": "user", "content": "recent-1"},
            {"role": "assistant", "content": "recent-2"},
            {"role": "user", "content": "recent-3"},
            {"role": "assistant", "content": "", "tool_calls": [{"input": "", "output": "U" * 80}]},
        ]
        fake_budget = SimpleNamespace(session_summary_chars=20, jit_tool_output_chars=20)
        with patch("sessions.session_pruning.resolve_budget", return_value=fake_budget):
            pruned = prune_messages(messages, recent_preserve=4, agent_id="main")

        self.assertTrue(pruned[0]["content"].endswith("\n...[已修剪]"))
        self.assertIn("[工具输出已修剪", pruned[1]["tool_calls"][0]["output"])
        self.assertEqual(pruned[5]["tool_calls"][0]["output"], "U" * 80)


class CompactionLevelTests(unittest.TestCase):
    def test_detect_compaction_level_includes_overhead_tokens(self) -> None:
        with (
            patch("infra.token_counter.resolve_compaction_thresholds", return_value=(100, 120)),
            patch("infra.token_counter.count_messages_tokens", return_value=90),
        ):
            self.assertEqual(detect_compaction_level([{"role": "user", "content": "x"}], overhead_tokens=15), "sliding")
            self.assertEqual(detect_compaction_level([{"role": "user", "content": "x"}], overhead_tokens=35), "forced")
            self.assertEqual(detect_compaction_level([{"role": "user", "content": "x"}], overhead_tokens=5), "none")


class AgentOverflowRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 这些用例只关心溢出重试逻辑，不关心模型解析。真实环境里
        # config.json 被 gitignore，CI 上不存在，resolve_default_model_ref
        # 会直接抛 "未配置可用模型"，导致用例在准备 prompt 阶段就失败。
        patcher = patch(
            "runtime.agent.AgentManager.get_current_model_ref",
            return_value="fake/model",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_astream_does_not_retry_after_partial_output(
        self,
    ) -> None:
        manager = AgentManager()
        attempts = {"count": 0}

        async def _fake_run_with_fallback_stream(
            _candidates,
            _run_fn,
            _agent_id,
        ):
            attempts["count"] += 1
            yield {"type": "token", "content": "partial"}
            error = RuntimeError("503 upstream unavailable")
            error.committed = True
            raise error

        with (
            patch.object(manager, "_build_tools", return_value=[]),
            patch(
                "tools.skills_scanner.write_skills_snapshot",
                return_value=None,
            ),
            patch(
                "runtime.workspace.has_bootstrap",
                return_value=False,
            ),
            patch(
                "runtime.agent.resolve_fallback_candidates",
                return_value=[],
            ),
            patch(
                "runtime.agent.run_with_fallback_stream",
                side_effect=_fake_run_with_fallback_stream,
            ),
            patch(
                "runtime.agent.resolve_agent_config",
                return_value={"recursion_limit": 10},
            ),
            patch(
                "runtime.agent.prompt_builder.build_system_prompt_with_report",
                return_value=(
                    "system prompt",
                    _PromptReport(),
                ),
            ),
            patch(
                "runtime.agent.session_manager.load_session_for_agent",
                return_value=[],
            ),
            patch(
                "runtime.agent.prune_messages",
                side_effect=(
                    lambda history, agent_id=None: history
                ),
            ),
            patch(
                "runtime.agent.count_tokens",
                return_value=0,
            ),
            patch(
                "runtime.agent.asyncio.sleep",
                new=AsyncMock(),
            ),
        ):
            events = [
                event
                async for event in manager.astream(
                    "hello",
                    "s1",
                    agent_id="main",
                )
            ]

        self.assertEqual(attempts["count"], 1)
        self.assertEqual(
            [event["type"] for event in events],
            ["token", "lifecycle", "error"],
        )

    async def test_astream_retries_once_after_forced_compaction_on_context_overflow(self) -> None:
        manager = AgentManager()
        attempts = {"count": 0}

        async def _fake_run_with_fallback_stream(_candidates, _run_fn, _agent_id):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("context overflow: too many tokens")
            yield {"type": "done", "content": "ok", "session_id": "s1"}

        with (
            patch.object(manager, "_build_tools", return_value=[]),
            patch.object(manager, "compress_session", new=AsyncMock(return_value={"compress": {}, "post_compaction": {}})) as compress_mock,
            patch("tools.skills_scanner.write_skills_snapshot", return_value=None),
            patch("runtime.workspace.has_bootstrap", return_value=False),
            patch("runtime.agent.resolve_fallback_candidates", return_value=[]),
            patch("runtime.agent.run_with_fallback_stream", side_effect=_fake_run_with_fallback_stream),
            patch("runtime.agent.resolve_agent_config", return_value={"recursion_limit": 10}),
            patch("runtime.agent.prompt_builder.build_system_prompt_with_report", return_value=("system prompt", _PromptReport())),
            patch("runtime.agent.session_manager.load_session_for_agent", return_value=[]),
            patch("runtime.agent.prune_messages", side_effect=lambda history, agent_id=None: history),
            patch("runtime.agent.count_tokens", return_value=0),
        ):
            events = []
            async for event in manager.astream("hello", "s1", agent_id="main"):
                events.append(event)

        self.assertEqual(events, [{"type": "done", "content": "ok", "session_id": "s1"}])
        compress_mock.assert_awaited_once_with("s1", "main", level="forced")
        self.assertEqual(attempts["count"], 2)


class AgentSessionSummaryInjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # 同 AgentOverflowRetryTests：不依赖真实模型配置，否则无 config.json 的
        # 环境（CI）会在构建 prompt 时因 "未配置可用模型" 报错。
        patcher = patch(
            "runtime.agent.AgentManager.get_current_model_ref",
            return_value="fake/model",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_astream_injects_session_summary_before_model_call(self) -> None:
        manager = AgentManager()
        captured = {"messages": None}

        class _FakeReactAgent:
            async def astream_events(self, payload, version="v2", config=None):
                captured["messages"] = payload["messages"]
                yield {
                    "event": "on_chat_model_stream",
                    "run_id": "run-1",
                    "data": {"chunk": SimpleNamespace(content="ok")},
                }
                yield {"event": "on_chat_model_end", "run_id": "run-1", "data": {}}

        async def _fake_run_with_fallback_stream(_candidates, run_fn, _agent_id):
            async for item in run_fn("fake", "model"):
                yield item

        fake_langgraph = ModuleType("langgraph")
        fake_prebuilt = ModuleType("langgraph.prebuilt")
        fake_prebuilt.create_react_agent = lambda **kwargs: _FakeReactAgent()

        fake_store = SimpleNamespace(
            get_session_summary=lambda session_id, agent_id: {
                "goal": "排查 sqlite 初始化失败",
                "progress": "已经确认错误来自扩展加载",
                "decisions": ["优先检查扩展路径"],
                "open_items": ["确认版本兼容性"],
                "entities": ["sqlite-vec", "app.db"],
                "user_preferences": ["命令行优先"],
                "raw_summary": "",
            }
        )
        manager.mem_stores["main"] = fake_store

        with (
            patch.object(manager, "_build_tools", return_value=[]),
            patch.object(manager, "_incremental_ingest", new=AsyncMock()),
            patch.object(manager, "_maybe_auto_compact", new=AsyncMock()),
            patch("tools.skills_scanner.write_skills_snapshot", return_value=None),
            patch("runtime.workspace.has_bootstrap", return_value=False),
            patch("runtime.agent.resolve_fallback_candidates", return_value=[SimpleNamespace(provider="fake", model="model")]),
            patch("runtime.agent.run_with_fallback_stream", side_effect=_fake_run_with_fallback_stream),
            patch("runtime.agent.resolve_agent_config", return_value={"recursion_limit": 10}),
            patch("runtime.agent.prompt_builder.build_system_prompt_with_report", return_value=("system prompt", _PromptReport())),
            patch("runtime.agent.session_manager.load_session_for_agent", return_value=[{"role": "user", "content": "历史问题"}]),
            patch("runtime.agent.prune_messages", side_effect=lambda history, agent_id=None: history),
            patch("runtime.agent.create_llm", return_value=object()),
            patch("runtime.agent.count_tokens", return_value=0),
            patch("runtime.agent.count_messages_tokens", return_value=0),
            patch.dict(sys.modules, {"langgraph": fake_langgraph, "langgraph.prebuilt": fake_prebuilt}),
        ):
            events = []
            async for event in manager.astream("最新问题", "s1", agent_id="main"):
                events.append(event)

        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["content"], "ok")
        self.assertEqual(events[-1]["session_id"], "s1")
        messages = captured["messages"]
        self.assertIsNotNone(messages)
        self.assertTrue(messages[0].content.startswith("[会话摘要"))
        self.assertIn("排查 sqlite 初始化失败", messages[0].content)
        self.assertEqual(messages[-1], HumanMessage(content="最新问题"))


class _FakeRecallEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeRecallStore:
    def __init__(self) -> None:
        self.tasks = {
            "task-1": SimpleNamespace(
                id="task-1",
                title="排查 sqlite-vec 初始化失败",
                summary=(
                    "检查扩展路径与版本兼容性，最终确认需要重新安装匹配版本，"
                    "并补充了动态库加载顺序、版本校验、回滚步骤、环境变量检查、"
                    "符号链接确认、二进制兼容性验证以及最终验证流程。"
                ),
                status="completed",
                started_at=1710000000000,
            )
        }
        self.chunks = {
            "chunk-1": SimpleNamespace(
                id="chunk-1",
                summary="错误来自扩展版本不匹配",
                content=(
                    "最终确认 sqlite-vec 与 sqlite 版本不匹配，需要重新安装扩展包。"
                    "同时验证了扩展路径与动态库符号链接，确认问题不在数据文件本身。"
                ),
                role="assistant",
                session_key="s-old",
                task_id="task-1",
                created_at=1710000001000,
            ),
            "orphan-1": SimpleNamespace(
                id="orphan-1",
                summary="用户提到 app.db 路径变化",
                content="app.db 路径从 data/app.db 改到了 storage/app.db，需要同步调整配置。",
                role="user",
                session_key="s-old",
                task_id=None,
                created_at=1710000002000,
            ),
            "orphan-2": SimpleNamespace(
                id="orphan-2",
                summary="用户提到日志目录迁移",
                content="日志目录从 logs/ 迁移到了 runtime/logs/，需要同步修改诊断脚本。",
                role="user",
                session_key="s-old",
                task_id=None,
                created_at=1710000003000,
            ),
        }

    def fts_search_tasks(self, query: str, limit: int = 10, owner: str | None = None):
        return [SimpleNamespace(task_id="task-1", score=1.0)]

    def ann_search_tasks(self, query_vec: list[float], top_k: int = 10, owner: str | None = None):
        return []

    def get_task(self, task_id: str):
        return self.tasks.get(task_id)

    def fts_search_orphan_chunks(self, query: str, limit: int = 10, exclude_session: str | None = None, owner: str | None = None):
        return [
            SimpleNamespace(chunk_id="orphan-1", score=1.0),
            SimpleNamespace(chunk_id="orphan-2", score=0.9),
        ]

    def ann_search_orphan_chunks(self, query_vec: list[float], top_k: int = 10, exclude_session: str | None = None, owner: str | None = None):
        return []

    def get_chunk(self, chunk_id: str):
        return self.chunks.get(chunk_id)

    def fts_search_chunks_in_tasks(self, query: str, task_ids: list[str], limit: int = 10, owner: str | None = None):
        return [SimpleNamespace(chunk_id="chunk-1", score=1.0, task_id="task-1")]

    def exact_search_chunks_in_tasks(self, query_vec: list[float], task_ids: list[str], top_k: int = 10, owner: str | None = None):
        return []


class RecallBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_recall_applies_budget_after_global_chunk_ranking(self) -> None:
        recall = MemRecall(
            store=_FakeRecallStore(),
            embedder=_FakeRecallEmbedder(),
            config={
                "recall": {
                    "budget_chars": 80,
                    "max_task_results": 5,
                    "min_task_hits": 3,
                    "chunks_per_task": 1,
                    "max_orphan_chunks": 3,
                    "min_task_score": 0.0,
                    "min_inject_score": 0.0,
                }
            },
            agent_id="main",
        )

        result = await recall.search("sqlite vec 初始化失败", session_id="s-now", owner="agent:main")

        self.assertEqual(len(result.task_groups), 1)
        self.assertEqual(result.task_groups[0].task_id, "task-1")
        self.assertEqual(len(result.task_groups[0].chunks), 1)
        self.assertEqual([hit.chunk_id for hit in result.ranked_hits], ["chunk-1"])
        self.assertTrue(result.orphan_search_triggered)
        self.assertGreater(result.candidate_chunk_count, len(result.ranked_hits))


class ToolPersistenceTests(unittest.TestCase):
    def test_maybe_persist_tool_output_keeps_small_output_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "short output"
            result = maybe_persist_tool_output(
                text,
                tool_name="read",
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )
        self.assertEqual(result, text)

    def test_maybe_persist_tool_output_persists_large_output_and_returns_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "x" * 40050
            result = maybe_persist_tool_output(
                text,
                tool_name="read",
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
                file_stem="read_test",
            )
            persisted_path = Path(tmp) / "main" / "sessions" / "s1" / "tool-results" / "read_test.txt"

            self.assertIn(UNTRUSTED_CONTENT_OPEN, result)
            self.assertIn(PERSISTED_OUTPUT_OPEN, result)
            self.assertTrue(persisted_path.exists())
            self.assertEqual(persisted_path.read_text(encoding="utf-8"), text)

    def test_never_persist_tool_returns_inline_even_if_large(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "x" * 100000
            result = maybe_persist_tool_output(
                text,
                tool_name="memory_get",
                data_dir=tmp,
                agent_id="main",
                session_id="s1",
            )
        self.assertEqual(result, text)

    def test_generate_preview_prefers_newline_boundary(self) -> None:
        text = ("a" * 120) + "\n" + ("b" * 500)
        preview, has_more = generate_preview(text, max_chars=200)
        self.assertTrue(has_more)
        self.assertEqual(len(preview), 120)
        self.assertLessEqual(len(preview), 200)

    def test_read_tool_wraps_file_content_as_untrusted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello\nworld", encoding="utf-8")
            tool = ReadTool(root_dir=str(root))
            result = tool._run("note.txt")
        self.assertIn(UNTRUSTED_CONTENT_OPEN, result)
        self.assertIn("Source: file_read", result)
        self.assertIn("hello", result)


class CompressSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_compress_session_uses_keep_recent_turns_for_sliding_and_forced(self) -> None:
        manager = AgentManager()
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ]
        store = SimpleNamespace(upsert_session_summary=Mock())
        manager.mem_stores["main"] = store
        manager._pending_tasks = set()

        with (
            patch("runtime.agent.resolve_agent_config", return_value={"compaction": {"keepRecentTurns": 2, "forcedKeepRecentTurns": 1}}),
            patch("runtime.agent.session_manager.load_session", return_value={"messages": messages}),
            patch("runtime.agent.session_manager.compress_history", return_value={"archived_count": 4, "remaining_count": 4}) as compress_mock,
            patch.object(manager, "_generate_structured_summary", new=AsyncMock(return_value={"raw_summary": "sum"})),
            patch.object(manager, "_batch_ingest_messages", new=AsyncMock()),
        ):
            sliding_result = await manager.compress_session("s1", "main", level="sliding")
            forced_result = await manager.compress_session("s1", "main", level="forced")

        self.assertEqual(sliding_result["compress"]["level"], "sliding")
        self.assertEqual(forced_result["compress"]["level"], "forced")
        self.assertEqual(compress_mock.call_args_list[0].args, ("s1", "main", 4))
        self.assertEqual(compress_mock.call_args_list[1].args, ("s1", "main", 6))

    async def test_calc_compress_count_by_turns_respects_turn_pairs(self) -> None:
        manager = AgentManager()
        messages = [
            {"role": "system", "content": "summary"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        self.assertEqual(manager._calc_compress_count_by_turns(messages, keep_turns=2), 3)
        self.assertEqual(manager._calc_compress_count_by_turns(messages, keep_turns=3), 0)


class AgentManagerCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_releases_background_tasks_and_runtime_resources(self) -> None:
        manager = AgentManager()
        manager._initialized = True

        store_main = SimpleNamespace(close=Mock())
        store_writer = SimpleNamespace(close=Mock())
        manager.mem_stores = {
            "main": store_main,
            "writer": store_writer,
        }
        manager.mem_embedders["main"] = object()
        manager.mem_workers["main"] = object()
        manager.mem_recalls["main"] = object()
        manager._states["main"] = SimpleNamespace()
        manager._prompt_cache[("main",)] = SimpleNamespace()
        manager._session_context_cache[("main", "s1")] = SimpleNamespace()
        manager._tool_name_cache[("main",)] = SimpleNamespace()

        state_task = asyncio.create_task(asyncio.sleep(3600))
        pending_finished = asyncio.Event()

        async def _finish_pending() -> None:
            await asyncio.sleep(0)
            pending_finished.set()

        pending_task = asyncio.create_task(_finish_pending())
        manager._state_save_tasks["main"] = state_task
        manager._pending_tasks.add(pending_task)

        with patch.object(manager, "_save_all_states", new=AsyncMock()):
            await manager.close(timeout=1)

        self.assertTrue(pending_finished.is_set())
        self.assertTrue(state_task.cancelled())
        store_main.close.assert_called_once_with()
        store_writer.close.assert_called_once_with()
        self.assertFalse(manager._initialized)
        self.assertEqual(manager.mem_stores, {})
        self.assertEqual(manager.mem_embedders, {})
        self.assertEqual(manager.mem_workers, {})
        self.assertEqual(manager.mem_recalls, {})
        self.assertEqual(manager._states, {})
        self.assertEqual(manager._state_save_tasks, {})
        self.assertEqual(manager._pending_tasks, set())
        self.assertEqual(manager._prompt_cache, {})
        self.assertEqual(manager._session_context_cache, {})
        self.assertEqual(manager._tool_name_cache, {})

    async def test_close_cancels_state_tasks_when_final_save_fails(self) -> None:
        manager = AgentManager()
        manager._initialized = True
        state_task = asyncio.create_task(asyncio.sleep(3600))
        manager._state_save_tasks["main"] = state_task
        store = SimpleNamespace(close=Mock())
        manager.mem_stores["main"] = store

        with patch.object(
            manager,
            "_save_all_states",
            new=AsyncMock(side_effect=RuntimeError("save failed")),
        ):
            await manager.close(timeout=1)

        self.assertTrue(state_task.cancelled())
        store.close.assert_called_once_with()
        self.assertEqual(manager._state_save_tasks, {})

    async def test_manager_can_initialize_again_after_close(self) -> None:
        manager = AgentManager()

        with (
            patch("runtime.agent.list_agents", return_value=[]),
            patch.object(manager, "_save_all_states", new=AsyncMock()),
        ):
            await manager.initialize("/tmp/pipixia-first")
            self.assertTrue(manager._initialized)

            await manager.close(timeout=1)
            self.assertFalse(manager._initialized)

            await manager.initialize("/tmp/pipixia-second")
            self.assertTrue(manager._initialized)
            self.assertEqual(manager.data_dir, "/tmp/pipixia-second")

            await manager.close(timeout=1)
            self.assertFalse(manager._initialized)


class MessageBuildTests(unittest.TestCase):
    def test_build_messages_keeps_first_system_and_downgrades_later_system(self) -> None:
        manager = AgentManager()
        history = [
            {"role": "system", "content": "system-1"},
            {"role": "user", "content": "user-1"},
            {"role": "system", "content": "[System Message] internal"},
            {"role": "assistant", "content": "assistant-1"},
        ]

        messages = manager._build_messages(history, "new-user")

        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[0].content, "system-1")
        self.assertEqual(messages[2].type, "human")
        self.assertEqual(messages[2].content, "[System Message] internal")
        self.assertEqual(messages[-1], HumanMessage(content="new-user"))


class SessionSummaryFormattingTests(unittest.TestCase):
    def test_format_session_summary_renders_structured_fields(self) -> None:
        summary = {
            "goal": "完成 sqlite-vec 初始化",
            "progress": "已经确认错误来自扩展版本不匹配",
            "decisions": ["优先检查扩展路径"],
            "open_items": ["确认最终安装命令"],
            "entities": ["sqlite-vec", "app.db"],
            "user_preferences": ["命令行优先"],
            "raw_summary": "",
        }

        rendered = prompt_builder.format_session_summary(summary)

        self.assertIn("[会话摘要", rendered)
        self.assertIn("🎯 会话目标：完成 sqlite-vec 初始化", rendered)
        self.assertIn("💡 关键决策：", rendered)
        self.assertIn("📌 待办事项：", rendered)
        self.assertIn("🏷️ 关键实体：sqlite-vec, app.db", rendered)
        self.assertIn("👤 用户偏好：命令行优先", rendered)


class RuntimeCacheTests(unittest.TestCase):
    def test_prompt_cache_reuses_built_prompt_when_signature_is_unchanged(self) -> None:
        manager = AgentManager()
        report = _PromptReport()
        build_mock = Mock(return_value=("prompt-text", report))

        with (
            patch.object(manager, "_project_context_signature", return_value=(("AGENTS.md", 1.0),)),
            patch.object(
                manager,
                "_prompt_runtime_signature",
                return_value=("runtime", "2026-07-16 10:00"),
            ),
            patch("runtime.agent.prompt_builder.build_system_prompt_with_report", build_mock),
            patch("runtime.agent.count_tokens", return_value=123),
        ):
            first = manager._get_or_build_prompt(
                agent_id="main",
                prompt_mode="full",
                available_tool_names=["read", "grep"],
                extra_system_prompt=None,
                locale="zh-CN",
            )
            second = manager._get_or_build_prompt(
                agent_id="main",
                prompt_mode="full",
                available_tool_names=["grep", "read"],
                extra_system_prompt=None,
                locale="zh-CN",
            )

        self.assertEqual(first, second)
        self.assertEqual(build_mock.call_count, 1)

    def test_session_context_cache_reuses_and_invalidates_on_summary_change(self) -> None:
        manager = AgentManager()
        manager.mem_stores["main"] = SimpleNamespace(
            get_session_summary=lambda session_id, agent_id: {
                "goal": "g1",
                "raw_summary": "",
            }
        )

        load_mock = Mock(return_value=[{"role": "user", "content": "hello"}])
        prune_mock = Mock(side_effect=lambda history, agent_id=None: history)

        with (
            patch.object(manager, "_safe_mtime", return_value=1.0),
            patch("runtime.agent.session_manager.load_session_for_agent", load_mock),
            patch("runtime.agent.prune_messages", prune_mock),
            patch("runtime.agent.count_tokens", return_value=10),
            patch("runtime.agent.count_messages_tokens", return_value=20),
        ):
            first = manager._get_or_build_session_context(agent_id="main", session_id="s1")
            second = manager._get_or_build_session_context(agent_id="main", session_id="s1")

        self.assertIs(first, second)
        self.assertEqual(load_mock.call_count, 1)
        self.assertEqual(prune_mock.call_count, 1)

        manager.mem_stores["main"] = SimpleNamespace(
            get_session_summary=lambda session_id, agent_id: {
                "goal": "g2",
                "raw_summary": "",
            }
        )

        with (
            patch.object(manager, "_safe_mtime", return_value=1.0),
            patch("runtime.agent.session_manager.load_session_for_agent", load_mock),
            patch("runtime.agent.prune_messages", prune_mock),
            patch("runtime.agent.count_tokens", return_value=10),
            patch("runtime.agent.count_messages_tokens", return_value=20),
        ):
            third = manager._get_or_build_session_context(agent_id="main", session_id="s1")

        self.assertIsNot(first, third)
        self.assertEqual(load_mock.call_count, 2)
        self.assertEqual(prune_mock.call_count, 2)

    def test_tool_name_cache_reuses_filtered_tool_names_when_policy_is_unchanged(self) -> None:
        manager = AgentManager()
        collected = [
            SimpleNamespace(name="read"),
            SimpleNamespace(name="grep"),
            SimpleNamespace(name="exec"),
        ]
        collect_mock = Mock(return_value=collected)

        with (
            patch(
                "config.get_config",
                return_value={"agents": {"list": [{"id": "main", "tools": {"allow": ["read", "grep"]}}]}},
            ),
            patch.object(manager, "_collect_tools", collect_mock),
        ):
            first = manager._get_or_build_tool_names("main")
            second = manager._get_or_build_tool_names("main")

        self.assertEqual(first, ("grep", "read"))
        self.assertEqual(second, ("grep", "read"))
        self.assertEqual(collect_mock.call_count, 1)

    def test_tool_name_cache_invalidates_when_policy_changes(self) -> None:
        manager = AgentManager()
        collected = [
            SimpleNamespace(name="read"),
            SimpleNamespace(name="grep"),
            SimpleNamespace(name="exec"),
        ]
        collect_mock = Mock(return_value=collected)

        with (
            patch(
                "config.get_config",
                return_value={"agents": {"list": [{"id": "main", "tools": {"allow": ["read"]}}]}},
            ),
            patch.object(manager, "_collect_tools", collect_mock),
        ):
            first = manager._get_or_build_tool_names("main")

        with (
            patch(
                "config.get_config",
                return_value={"agents": {"list": [{"id": "main", "tools": {"allow": ["read", "grep"]}}]}},
            ),
            patch.object(manager, "_collect_tools", collect_mock),
        ):
            second = manager._get_or_build_tool_names("main")

        self.assertEqual(first, ("read",))
        self.assertEqual(second, ("grep", "read"))
        self.assertEqual(collect_mock.call_count, 2)


class SourceSinkGuardTests(unittest.IsolatedAsyncioTestCase):
    def test_evaluate_source_to_sink_blocks_obviously_malicious_sink(self) -> None:
        decision = evaluate_source_to_sink(
            has_recent_untrusted_content=True,
            tool_name="exec",
            tool_input={"input_preview": "rm -rf logs"},
            user_message="帮我看看这个文件",
        )

        self.assertEqual(decision.action, "block")
        self.assertIn("obviously dangerous", decision.reason or "")

    def test_evaluate_source_to_sink_confirms_when_source_is_suspicious_but_not_obviously_malicious(self) -> None:
        decision = evaluate_source_to_sink(
            has_recent_untrusted_content=True,
            tool_name="exec",
            tool_input={"input_preview": "pytest"},
            user_message="帮我看看这个文件",
        )

        self.assertEqual(decision.action, "confirm")
        self.assertIn("approval", decision.reason or "")

    def test_evaluate_source_to_sink_allows_explicit_user_request(self) -> None:
        decision = evaluate_source_to_sink(
            has_recent_untrusted_content=True,
            tool_name="exec",
            tool_input={"input_preview": "pytest"},
            user_message="运行 pytest 看看失败原因",
        )

        self.assertEqual(decision.action, "allow")

    async def test_approval_gate_escalates_confirm_to_human_in_the_loop(self) -> None:
        with (
            runtime_security_context("帮我看看这个文件", recent_untrusted_content=True),
            patch("sandbox.tool_approval_gate.approval_store.create", return_value="a1"),
            patch("sandbox.tool_approval_gate.approval_store.wait", new=AsyncMock(return_value="approved")) as wait_mock,
            patch("sandbox.tool_approval_gate.event_bus.emit") as emit_mock,
        ):
            result = await run_tool_with_approval_gate(
                agent_id="main",
                tool_name="exec",
                input_preview="pytest",
                locale="zh-CN",
                base_needs_approval=False,
                deny_reason=None,
                execute_fn=lambda: "ok",
            )

        self.assertEqual(result, "ok")
        emit_mock.assert_called_once()
        wait_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
