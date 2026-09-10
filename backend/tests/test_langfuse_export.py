from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class LangfuseExportCliTests(unittest.TestCase):
    def test_cli_passes_filters_and_writes_output(self) -> None:
        calls = []

        class Client:
            def __init__(self, **kwargs):
                calls.append(("client", kwargs))

        fake_module = ModuleType("langfuse")
        fake_module.Langfuse = Client
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"langfuse": fake_module}
        ), patch.dict(
            os.environ,
            {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"},
            clear=True,
        ), patch(
            "evaluation.langfuse_export.export_langfuse_traces",
            return_value=2,
        ) as export, patch(
            "evaluation.langfuse_export._build_memory_store"
        ) as build_memory_store:
            from evaluation.langfuse_export import main

            output = Path(directory) / "samples.jsonl"
            self.assertEqual(
                main([
                    "--output", str(output),
                    "--session-id", "session-1",
                    "--from-timestamp", "2026-01-01T00:00:00Z",
                    "--limit", "25",
                    "--max-pages", "2",
                ]),
                0,
            )

        self.assertEqual(calls[0][1]["public_key"], "pk")
        kwargs = export.call_args.kwargs
        self.assertEqual(kwargs["limit"], 25)
        self.assertEqual(kwargs["max_pages"], 2)
        self.assertEqual(kwargs["session_id"], "session-1")
        self.assertIs(kwargs["memory_store"], build_memory_store.return_value)
        build_memory_store.return_value.close.assert_called_once_with()

    def test_build_client_loads_backend_env(self) -> None:
        from evaluation import langfuse_export

        fake_module = ModuleType("langfuse")
        fake_module.Langfuse = lambda **kwargs: kwargs
        with patch("evaluation.langfuse_export.Path.exists", return_value=True), patch(
            "evaluation.langfuse_export.os.getenv",
            side_effect=lambda key, default="": {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
                "LANGFUSE_HOST": "https://cloud.langfuse.com",
            }.get(key, default),
        ), patch.dict(sys.modules, {"langfuse": fake_module}):
            client = langfuse_export._build_client()

        self.assertEqual(client["public_key"], "pk-test")


if __name__ == "__main__":
    unittest.main()
