"""Persistence boundary for memory chunks."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

from mem.models import Chunk, DedupStatus, EmbeddingStatus, SummarySource


class ChunkRepository:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        sync_fts: Callable[[str], None],
        now_ms: Callable[[], int],
        content_hash: Callable[[str], str],
    ) -> None:
        self._connection = connection
        self._sync_fts = sync_fts
        self._now_ms = now_ms
        self._content_hash = content_hash

    def insert(self, chunk: Chunk) -> None:
        if not chunk.content_hash:
            chunk.content_hash = self._content_hash(chunk.content)
        now = self._now_ms()
        if not chunk.created_at:
            chunk.created_at = now
        if not chunk.updated_at:
            chunk.updated_at = now

        cursor = self._connection.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        cursor.close()
        has_provenance = {
            "parent_task_id",
            "source_type",
        }.issubset(columns)
        insert_columns = [
            "id", "session_key", "turn_id", "seq", "role", "content",
            "kind", "summary", "task_id",
        ]
        insert_values: list[Any] = [
            chunk.id, chunk.session_key, chunk.turn_id, chunk.seq,
            chunk.role, chunk.content, chunk.kind, chunk.summary,
            chunk.task_id,
        ]
        if has_provenance:
            insert_columns.extend(["parent_task_id", "source_type"])
            insert_values.extend([chunk.parent_task_id, chunk.source_type])
        insert_columns.extend([
            "skill_id", "owner", "content_hash", "dedup_status",
            "dedup_target", "dedup_reason", "summary_source",
            "embedding_status", "embedding_error", "created_at", "updated_at",
        ])
        insert_values.extend([
            chunk.skill_id, chunk.owner, chunk.content_hash, chunk.dedup_status,
            chunk.dedup_target, chunk.dedup_reason, chunk.summary_source,
            chunk.embedding_status, chunk.embedding_error, chunk.created_at,
            chunk.updated_at,
        ])
        update_columns = [column for column in insert_columns if column != "id"]
        sql = (
            "INSERT INTO chunks (" + ", ".join(insert_columns) + ") VALUES ("
            + ", ".join("?" for _ in insert_columns)
            + ") ON CONFLICT(id) DO UPDATE SET "
            + ", ".join(
                f"{column}=excluded.{column}" for column in update_columns
            )
        )
        self._connection.execute(sql, insert_values)
        self._sync_fts(chunk.id)
        self._connection.commit()

    def update_summary(
        self,
        chunk_id: str,
        summary: str,
        *,
        summary_source: SummarySource | None = None,
    ) -> None:
        if summary_source is None:
            self._connection.execute(
                "UPDATE chunks SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, self._now_ms(), chunk_id),
            )
        else:
            self._connection.execute(
                """UPDATE chunks
                SET summary = ?, summary_source = ?, updated_at = ? WHERE id = ?""",
                (summary, summary_source, self._now_ms(), chunk_id),
            )
        self._sync_fts(chunk_id)
        self._connection.commit()

    def update_embedding_status(
        self,
        chunk_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> None:
        self._connection.execute(
            """UPDATE chunks
            SET embedding_status = ?, embedding_error = ?, updated_at = ?
            WHERE id = ?""",
            (status, error, self._now_ms(), chunk_id),
        )
        self._connection.commit()

    def mark_dedup_status(
        self,
        chunk_id: str,
        status: DedupStatus,
        target: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._connection.execute(
            """UPDATE chunks
            SET dedup_status=?, dedup_target=?, dedup_reason=?, updated_at=?
            WHERE id=?""",
            (status, target, reason, self._now_ms(), chunk_id),
        )
        self._connection.commit()

    def orphan(self, chunk_id: str, reason: str | None = None) -> None:
        self._connection.execute(
            """UPDATE chunks
            SET dedup_status='orphaned', task_id=NULL, dedup_reason=?, updated_at=?
            WHERE id=?""",
            (reason, self._now_ms(), chunk_id),
        )
        self._connection.commit()

    def find_active_by_hash(self, content: str, owner: str) -> str | None:
        content_hash = self._content_hash(content)
        row = self._connection.execute(
            """SELECT id FROM chunks
            WHERE content_hash=? AND dedup_status='active' AND owner=? LIMIT 1""",
            (content_hash, owner),
        ).fetchone()
        return row["id"] if row else None

    def get(self, chunk_id: str) -> Chunk | None:
        row = self._connection.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return row_to_chunk(row) if row else None

    def get_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE task_id=? AND dedup_status='active' "
            "ORDER BY created_at"
        )
        params: list[Any] = [task_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._connection.execute(sql, params).fetchall()
        return [row_to_chunk(row) for row in rows]

    def get_by_turn(self, session_key: str, turn_id: str) -> list[Chunk]:
        rows = self._connection.execute(
            """SELECT * FROM chunks
            WHERE session_key=? AND turn_id=? AND dedup_status='active'
            ORDER BY seq, created_at""",
            (session_key, turn_id),
        ).fetchall()
        return [row_to_chunk(row) for row in rows]

    def get_for_embedding_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE dedup_status='active' "
            "AND embedding_status='failed' ORDER BY updated_at LIMIT ?"
        )
        params: list[Any] = [limit]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE dedup_status='active' "
                "AND embedding_status='failed' AND owner=? "
                "ORDER BY updated_at LIMIT ?"
            )
            params = [owner, limit]
        rows = self._connection.execute(sql, params).fetchall()
        return [row_to_chunk(row) for row in rows]

    def get_for_summary_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        sql = (
            "SELECT * FROM chunks WHERE dedup_status='active' "
            "AND summary_source='fallback' ORDER BY updated_at LIMIT ?"
        )
        params: list[Any] = [limit]
        if owner:
            sql = (
                "SELECT * FROM chunks WHERE dedup_status='active' "
                "AND summary_source='fallback' AND owner=? "
                "ORDER BY updated_at LIMIT ?"
            )
            params = [owner, limit]
        rows = self._connection.execute(sql, params).fetchall()
        return [row_to_chunk(row) for row in rows]

    def get_in_range(
        self,
        session_key: str,
        turn_id: str,
        seq: int,
        window: int = 2,
        owner: str | None = None,
    ) -> list[Chunk]:
        params: list[Any] = [session_key]
        sql = "SELECT * FROM chunks WHERE session_key = ?"
        if owner:
            sql += " AND owner = ?"
            params.append(owner)
        sql += " ORDER BY created_at, seq"
        rows = self._connection.execute(sql, params).fetchall()

        target_index = -1
        for index, row in enumerate(rows):
            if row["turn_id"] == turn_id and row["seq"] == seq:
                target_index = index
                break
        if target_index == -1:
            return []

        radius = window * 3
        start = max(0, target_index - radius)
        end = min(len(rows), target_index + radius + 1)
        return [row_to_chunk(rows[index]) for index in range(start, end)]


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        session_key=row["session_key"],
        turn_id=row["turn_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        kind=row["kind"] or "paragraph",
        summary=row["summary"] or "",
        task_id=row["task_id"],
        parent_task_id=row["parent_task_id"] if "parent_task_id" in row.keys() else None,
        source_type=row["source_type"] if "source_type" in row.keys() else "main_agent",
        skill_id=row["skill_id"],
        owner=row["owner"] or "agent:main",
        content_hash=row["content_hash"] or "",
        dedup_status=row["dedup_status"] or "active",
        dedup_target=row["dedup_target"],
        dedup_reason=row["dedup_reason"],
        summary_source=row["summary_source"] or "llm",
        embedding_status=row["embedding_status"] or "ok",
        embedding_error=row["embedding_error"],
        created_at=row["created_at"] or 0,
        updated_at=row["updated_at"] or 0,
    )
