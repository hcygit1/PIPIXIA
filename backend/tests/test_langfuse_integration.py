from __future__ import annotations

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class LangfuseIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = SimpleNamespace(
            agent_id="main",
            session_id="session-1",
            provider="openai",
            model="gpt-test",
            observability_metadata={
                "skill_names": ["检索增强"],
                "skill_versions": {"检索增强": "2"},
                "skill_count": 1,
                "task_family": "知识问答",
            },
        )

    def test_missing_credentials_keeps_agent_config_without_callbacks(self) -> None:
        from runtime.langfuse_integration import build_langfuse_config

        with patch.dict(os.environ, {}, clear=True):
            config = build_langfuse_config(request=self.request, run_id="run-1")

        self.assertNotIn("callbacks", config)
        self.assertEqual(config["metadata"]["pipixia_run_id"], "run-1")
        self.assertEqual(config["metadata"]["langfuse_session_id"], "session-1")
        self.assertEqual(config["metadata"]["skill_names"], ["检索增强"])
        self.assertEqual(config["metadata"]["skill_versions"], {"检索增强": "2"})
        self.assertEqual(config["metadata"]["task_family"], "知识问答")
        self.assertEqual(config["run_name"], "pipixia-agent-turn")

    def test_empty_business_metadata_is_omitted(self) -> None:
        from runtime.langfuse_integration import build_langfuse_config

        request = SimpleNamespace(
            agent_id="main",
            session_id="session-1",
            provider="openai",
            model="gpt-test",
            observability_metadata={"task_family": "", "failure_type": None},
        )
        with patch.dict(os.environ, {}, clear=True):
            config = build_langfuse_config(request=request, run_id="run-empty")

        self.assertNotIn("task_family", config["metadata"])
        self.assertNotIn("failure_type", config["metadata"])

    def test_langfuse_handler_is_added_when_credentials_exist(self) -> None:
        from runtime.langfuse_integration import build_langfuse_config

        fake_module = ModuleType("langfuse")
        fake_module.__path__ = []
        fake_langchain = ModuleType("langfuse.langchain")
        fake_langchain.CallbackHandler = lambda: "langfuse-handler"
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }, clear=True), patch.dict(sys.modules, {
            "langfuse": fake_module,
            "langfuse.langchain": fake_langchain,
        }):
            config = build_langfuse_config(request=self.request, run_id="run-2")

        self.assertEqual(config["callbacks"], ["langfuse-handler"])
        self.assertIn("pipixia-agent", config["tags"])

    def test_langfuse_initialization_failure_does_not_break_turn(self) -> None:
        from runtime.langfuse_integration import build_langfuse_config

        fake_module = ModuleType("langfuse")
        fake_module.__path__ = []
        fake_langchain = ModuleType("langfuse.langchain")
        fake_langchain.CallbackHandler = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
        with patch.dict(os.environ, {
            "LANGFUSE_PUBLIC_KEY": "pk-test",
            "LANGFUSE_SECRET_KEY": "sk-test",
        }, clear=True), patch.dict(sys.modules, {
            "langfuse": fake_module,
            "langfuse.langchain": fake_langchain,
        }):
            config = build_langfuse_config(request=self.request, run_id="run-3")

        self.assertNotIn("callbacks", config)

    def test_flush_langfuse_config_flushes_callback_client(self) -> None:
        from runtime.langfuse_integration import flush_langfuse_config

        class Client:
            def __init__(self) -> None:
                self.flushed = False

            def flush(self) -> None:
                self.flushed = True

        class Handler:
            def __init__(self) -> None:
                self.client = Client()

        handler = Handler()
        flush_langfuse_config({"callbacks": [handler]})
        self.assertTrue(handler.client.flushed)


if __name__ == "__main__":
    unittest.main()
