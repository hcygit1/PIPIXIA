from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


MODEL_NAMES = (
    "Chunk",
    "Task",
    "Skill",
    "SearchHit",
    "TaskSearchHit",
    "SkillSearchHit",
    "SessionSummary",
)


class MemoryModelBoundaryTests(unittest.TestCase):
    def test_importing_models_does_not_load_storage_or_pipeline(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.models; "
                    "blocked = sorted(name for name in sys.modules "
                    "if name in {'mem.store', 'mem.worker', 'mem.recall', "
                    "'mem.task_processor', 'mem.skill_evolver'}); "
                    "assert not blocked, blocked"
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_memory_models_have_neutral_ownership_and_compatibility_exports(
        self,
    ) -> None:
        models_path = BACKEND_DIR / "mem" / "models.py"
        self.assertTrue(models_path.is_file(), "mem.models should own memory models")

        models = importlib.import_module("mem.models")
        store = importlib.import_module("mem.store")
        package = importlib.import_module("mem")

        for name in MODEL_NAMES:
            self.assertIs(getattr(store, name), getattr(models, name))
        for name in MODEL_NAMES[:-1]:
            self.assertIs(getattr(package, name), getattr(models, name))

        self.assertEqual(
            [field.name for field in fields(models.Chunk)],
            [
                "id",
                "session_key",
                "turn_id",
                "seq",
                "role",
                "content",
                "kind",
                "summary",
                "task_id",
                "parent_task_id",
                "source_type",
                "skill_id",
                "owner",
                "content_hash",
                "dedup_status",
                "dedup_target",
                "dedup_reason",
                "summary_source",
                "embedding_status",
                "embedding_error",
                "created_at",
                "updated_at",
            ],
        )

        model_names = set(MODEL_NAMES)
        for relative_path in (
            "mem/worker.py",
            "mem/task_processor.py",
            "mem/skill_evolver.py",
        ):
            source = (BACKEND_DIR / relative_path).read_text(encoding="utf-8")
            self.assertIn("from mem.models import", source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module == "mem.store":
                    imported_names = {alias.name for alias in node.names}
                    self.assertNotIn("*", imported_names)
                    self.assertTrue(model_names.isdisjoint(imported_names))


if __name__ == "__main__":
    unittest.main()
