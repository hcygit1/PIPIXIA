from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.schema import MemorySchema


class MemoryStorageBoundaryTests(unittest.TestCase):
    def _run_isolated(self, source: str) -> dict:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_existing_columns_do_not_report_an_fts_rebuild(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE chunks (
                id TEXT PRIMARY KEY,
                summary_source TEXT NOT NULL DEFAULT 'llm',
                embedding_status TEXT NOT NULL DEFAULT 'ok',
                embedding_error TEXT
            )"""
        )
        schema = MemorySchema(connection, dimensions=3)

        with self.assertNoLogs("mem.schema", level="INFO"):
            schema.ensure_chunk_columns()

        connection.close()

    def test_schema_and_fts_have_separate_owners(self) -> None:
        schema_path = BACKEND_DIR / "mem" / "schema.py"
        fts_path = BACKEND_DIR / "mem" / "fts_index.py"

        self.assertTrue(schema_path.is_file(), "mem.schema should own schema setup")
        self.assertTrue(fts_path.is_file(), "mem.fts_index should own FTS maintenance")

        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("MemorySchema(", store_source)
        self.assertIn("MemoryFtsIndex(", store_source)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS chunks", store_source)
        self.assertNotIn("INSERT INTO chunks_fts", store_source)

    def test_real_store_preserves_schema_and_fts_contract(self) -> None:
        result = self._run_isolated(
            """
            import json
            import tempfile
            from pathlib import Path
            from mem.models import Chunk, Skill, Task
            from mem.store import MemStore

            with tempfile.TemporaryDirectory() as root:
                store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                store.insert_chunk(Chunk(
                    id="chunk-1", session_key="s1", turn_id="t1", seq=0,
                    role="user", content="postgres database",
                    summary="postgres connection",
                ))
                store.insert_task(Task(
                    id="task-1", session_key="s1", title="postgres repair",
                    summary="database connection restored", status="completed",
                    started_at=1,
                ))
                store.insert_skill(Skill(
                    id="skill-1", name="postgres-repair",
                    description="repair postgres connection",
                ))

                before = {
                    "chunks": [hit.chunk_id for hit in store.fts_search_chunks("postgres")],
                    "tasks": [hit.task_id for hit in store.fts_search_tasks("postgres")],
                    "skills": [hit.skill_id for hit in store.fts_search_skills("postgres")],
                }
                store._conn.execute("DELETE FROM chunks_fts")
                store._conn.execute("DELETE FROM tasks_fts")
                store._conn.execute("DELETE FROM skills_fts")
                store.rebuild_fts_indexes()
                after = {
                    "chunks": [hit.chunk_id for hit in store.fts_search_chunks("postgres")],
                    "tasks": [hit.task_id for hit in store.fts_search_tasks("postgres")],
                    "skills": [hit.skill_id for hit in store.fts_search_skills("postgres")],
                }

                columns = {}
                for table in ("chunks", "tasks", "skills", "session_summaries"):
                    columns[table] = [
                        row["name"]
                        for row in store._conn.execute(f"PRAGMA table_info({table})")
                    ]
                indexes = sorted(
                    row[0]
                    for row in store._conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' "
                        "AND name NOT LIKE 'sqlite_%'"
                    )
                )
                vec_sql = store._conn.execute(
                    "SELECT sql FROM sqlite_master WHERE name='vec_chunks'"
                ).fetchone()[0]
                result = {
                    "before": before,
                    "after": after,
                    "columns": columns,
                    "indexes": indexes,
                    "vec_sql": vec_sql,
                }
                store.close()
                print(json.dumps(result, sort_keys=True))
            """
        )

        self.assertEqual(
            result["before"],
            {"chunks": ["chunk-1"], "tasks": ["task-1"], "skills": ["skill-1"]},
        )
        self.assertEqual(result["after"], result["before"])
        self.assertEqual(
            result["columns"]["chunks"],
            [
                "id", "session_key", "turn_id", "seq", "role", "content",
                "kind", "summary", "task_id", "parent_task_id", "source_type",
                "skill_id", "owner",
                "content_hash", "dedup_status", "dedup_target", "dedup_reason",
                "summary_source", "embedding_status", "embedding_error",
                "created_at", "updated_at",
            ],
        )
        self.assertEqual(
            result["columns"]["tasks"],
            [
                "id", "session_key", "owner", "title", "summary",
                "boundary_summary", "boundary_compacted_count", "status",
                "started_at", "ended_at", "updated_at",
            ],
        )
        self.assertEqual(
            result["columns"]["skills"],
            [
                "id", "name", "description", "dir_path", "version", "status",
                "installed", "owner", "visibility", "quality_score",
                "created_at", "updated_at",
            ],
        )
        self.assertEqual(
            result["columns"]["session_summaries"],
            [
                "id", "session_id", "agent_id", "version", "goal", "decisions",
                "progress", "open_items", "entities", "user_preferences",
                "raw_summary", "token_count", "created_at", "updated_at",
            ],
        )
        self.assertIn("embedding float[3]", result["vec_sql"])
        self.assertTrue(
            {
                "idx_chunks_session", "idx_chunks_turn", "idx_chunks_created",
                "idx_chunks_dedup", "idx_chunks_dedup_status", "idx_chunks_owner",
                "idx_chunks_task", "idx_tasks_session", "idx_tasks_status",
                "idx_tasks_owner", "idx_skills_status", "idx_skills_name",
                "idx_skills_owner",
            }.issubset(result["indexes"])
        )

    def test_legacy_store_columns_are_migrated_in_place(self) -> None:
        result = self._run_isolated(
            """
            import json
            import sqlite3
            import tempfile
            from pathlib import Path
            from mem.store import MemStore

            with tempfile.TemporaryDirectory() as root:
                path = Path(root) / "legacy.db"
                conn = sqlite3.connect(path)
                conn.executescript('''
                    CREATE TABLE chunks (
                        id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                        turn_id TEXT NOT NULL, seq INTEGER NOT NULL,
                        role TEXT NOT NULL, content TEXT NOT NULL,
                        kind TEXT NOT NULL DEFAULT 'paragraph',
                        summary TEXT NOT NULL DEFAULT '', task_id TEXT, skill_id TEXT,
                        owner TEXT NOT NULL DEFAULT 'agent:main',
                        content_hash TEXT NOT NULL DEFAULT '',
                        dedup_status TEXT NOT NULL DEFAULT 'active',
                        dedup_target TEXT, dedup_reason TEXT,
                        created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
                    );
                    CREATE TABLE tasks (
                        id TEXT PRIMARY KEY, session_key TEXT NOT NULL,
                        owner TEXT NOT NULL DEFAULT 'agent:main',
                        title TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'active', started_at INTEGER NOT NULL,
                        ended_at INTEGER, updated_at INTEGER NOT NULL
                    );
                ''')
                conn.commit()
                conn.close()

                store = MemStore(str(path), dimensions=3)
                result = {
                    "chunks": [
                        row["name"]
                        for row in store._conn.execute("PRAGMA table_info(chunks)")
                    ],
                    "tasks": [
                        row["name"]
                        for row in store._conn.execute("PRAGMA table_info(tasks)")
                    ],
                }
                store.close()
                print(json.dumps(result, sort_keys=True))
            """
        )

        self.assertTrue(
            {"summary_source", "embedding_status", "embedding_error"}.issubset(
                result["chunks"]
            )
        )
        self.assertTrue(
            {"boundary_summary", "boundary_compacted_count"}.issubset(
                result["tasks"]
            )
        )


if __name__ == "__main__":
    unittest.main()
