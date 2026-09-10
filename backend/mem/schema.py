"""SQLite schema initialization and compatibility migrations for memory."""

from __future__ import annotations

import sqlite3


class MemorySchema:
    def __init__(self, connection: sqlite3.Connection, dimensions: int) -> None:
        self._connection = connection
        self._dimensions = dimensions

    def initialize(self) -> None:
        connection = self._connection
        dimensions = self._dimensions

        connection.executescript(f"""
            -- Chunks
            CREATE TABLE IF NOT EXISTS chunks (
                id           TEXT PRIMARY KEY,
                session_key  TEXT NOT NULL,
                turn_id      TEXT NOT NULL,
                seq          INTEGER NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT NOT NULL,
                kind         TEXT NOT NULL DEFAULT 'paragraph',
                summary      TEXT NOT NULL DEFAULT '',
                task_id      TEXT,
                parent_task_id TEXT,
                source_type  TEXT NOT NULL DEFAULT 'main_agent',
                skill_id     TEXT,
                owner        TEXT NOT NULL DEFAULT 'agent:main',
                content_hash TEXT NOT NULL DEFAULT '',
                dedup_status TEXT NOT NULL DEFAULT 'active',
                dedup_target TEXT,
                dedup_reason TEXT,
                summary_source TEXT NOT NULL DEFAULT 'llm',
                embedding_status TEXT NOT NULL DEFAULT 'ok',
                embedding_error TEXT,
                created_at   INTEGER NOT NULL,
                updated_at   INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_session
                ON chunks(session_key);
            CREATE INDEX IF NOT EXISTS idx_chunks_turn
                ON chunks(session_key, turn_id, seq);
            CREATE INDEX IF NOT EXISTS idx_chunks_created
                ON chunks(created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_dedup
                ON chunks(session_key, role, content_hash);
            CREATE INDEX IF NOT EXISTS idx_chunks_dedup_status
                ON chunks(dedup_status, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chunks_owner
                ON chunks(owner);
            CREATE INDEX IF NOT EXISTS idx_chunks_task
                ON chunks(task_id);

            -- Tasks
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                owner       TEXT NOT NULL DEFAULT 'agent:main',
                title       TEXT NOT NULL DEFAULT '',
                summary     TEXT NOT NULL DEFAULT '',
                boundary_summary TEXT NOT NULL DEFAULT '',
                boundary_compacted_count INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'active',
                started_at  INTEGER NOT NULL,
                ended_at    INTEGER,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_key);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner);

            -- Semantic task-boundary decisions requiring human confirmation.
            CREATE TABLE IF NOT EXISTS boundary_reviews (
                id              TEXT PRIMARY KEY,
                session_key     TEXT NOT NULL,
                owner           TEXT NOT NULL DEFAULT 'agent:main',
                current_task_id TEXT NOT NULL,
                turn_id         TEXT NOT NULL,
                confidence      REAL NOT NULL DEFAULT 0,
                reason          TEXT NOT NULL DEFAULT '',
                retry_count     INTEGER NOT NULL DEFAULT 0,
                status          TEXT NOT NULL DEFAULT 'pending',
                resolution      TEXT NOT NULL DEFAULT '',
                target_task_id  TEXT,
                note            TEXT NOT NULL DEFAULT '',
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL,
                resolved_at     INTEGER,
                UNIQUE(owner, session_key, turn_id)
            );
            CREATE INDEX IF NOT EXISTS idx_boundary_reviews_status
                ON boundary_reviews(owner, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_boundary_reviews_session
                ON boundary_reviews(owner, session_key, status);

            -- Skills
            CREATE TABLE IF NOT EXISTS skills (
                id            TEXT PRIMARY KEY,
                name          TEXT NOT NULL UNIQUE,
                description   TEXT NOT NULL DEFAULT '',
                dir_path      TEXT NOT NULL DEFAULT '',
                version       INTEGER NOT NULL DEFAULT 1,
                status        TEXT NOT NULL DEFAULT 'active',
                installed     INTEGER NOT NULL DEFAULT 0,
                owner         TEXT NOT NULL DEFAULT 'agent:main',
                visibility    TEXT NOT NULL DEFAULT 'private',
                quality_score REAL,
                created_at    INTEGER NOT NULL,
                updated_at    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);
            CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
            CREATE INDEX IF NOT EXISTS idx_skills_owner ON skills(owner);

            -- Session Summaries (结构化压缩摘要, 每个活跃 session 至多 1 行)
            CREATE TABLE IF NOT EXISTS session_summaries (
                id               TEXT PRIMARY KEY,
                session_id       TEXT NOT NULL,
                agent_id         TEXT NOT NULL,
                version          INTEGER DEFAULT 1,
                goal             TEXT,
                decisions        TEXT,
                progress         TEXT,
                open_items       TEXT,
                entities         TEXT,
                user_preferences TEXT,
                raw_summary      TEXT,
                token_count      INTEGER,
                created_at       REAL,
                updated_at       REAL,
                UNIQUE(session_id, agent_id)
            );
        """)

        self.ensure_chunk_columns()
        self.ensure_task_columns()

        for statement in [
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                summary, content,
                tokenize='unicode61'
            )""",
            """CREATE VIRTUAL TABLE IF NOT EXISTS tasks_fts USING fts5(
                title, summary,
                tokenize='unicode61'
            )""",
            """CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                name, description,
                tokenize='unicode61'
            )""",
        ]:
            try:
                connection.execute(statement)
            except sqlite3.OperationalError:
                pass

        for statement in [
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{dimensions}])",
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_tasks USING vec0(task_id TEXT PRIMARY KEY, embedding float[{dimensions}])",
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_skills USING vec0(skill_id TEXT PRIMARY KEY, embedding float[{dimensions}])",
        ]:
            connection.execute(statement)

        connection.commit()

    def ensure_chunk_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(chunks)"
            ).fetchall()
        }
        additions = {
            "summary_source": "ALTER TABLE chunks ADD COLUMN summary_source TEXT NOT NULL DEFAULT 'llm'",
            "embedding_status": "ALTER TABLE chunks ADD COLUMN embedding_status TEXT NOT NULL DEFAULT 'ok'",
            "embedding_error": "ALTER TABLE chunks ADD COLUMN embedding_error TEXT",
            "parent_task_id": "ALTER TABLE chunks ADD COLUMN parent_task_id TEXT",
            "source_type": "ALTER TABLE chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'main_agent'",
        }
        for name, statement in additions.items():
            if name not in columns:
                self._connection.execute(statement)
        self._connection.commit()

    def ensure_task_columns(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute(
                "PRAGMA table_info(tasks)"
            ).fetchall()
        }
        additions = {
            "boundary_summary": "ALTER TABLE tasks ADD COLUMN boundary_summary TEXT NOT NULL DEFAULT ''",
            "boundary_compacted_count": "ALTER TABLE tasks ADD COLUMN boundary_compacted_count INTEGER NOT NULL DEFAULT 0",
        }
        for name, statement in additions.items():
            if name not in columns:
                self._connection.execute(statement)
        self._connection.commit()
