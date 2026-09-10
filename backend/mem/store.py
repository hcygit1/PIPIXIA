"""记忆存储层 — 单 SQLite 文件 + FTS5 + sqlite-vec ANN

Schema 与 docs/memory-system-refactor.md §3.1 一致:
  chunks / chunks_fts / vec_chunks
  tasks  / tasks_fts  / vec_tasks
  skills / skills_fts / vec_skills
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import sqlite_vec
from sqlite_vec import serialize_float32

from mem.boundary_review_repository import BoundaryReviewRepository
from mem.chunk_repository import ChunkRepository, row_to_chunk as _row_to_chunk
from mem.dashboard_queries import MemoryDashboardQueries
from mem.fts_index import MemoryFtsIndex
from mem.models import (
    BoundaryReview,
    Chunk,
    DedupStatus,
    EmbeddingStatus,
    SearchHit,
    SessionSummary,
    Skill,
    SkillSearchHit,
    SkillStatus,
    SkillVisibility,
    SummarySource,
    Task,
    TaskSearchHit,
    TaskStatus,
)
from mem.persistence_values import content_hash as _content_hash
from mem.persistence_values import now_ms as _now_ms
from mem.schema import MemorySchema
from mem.search_queries import MemorySearchQueries
from mem.session_summary_repository import SessionSummaryRepository
from mem.skill_repository import SkillRepository, row_to_skill as _row_to_skill
from mem.task_repository import TaskRepository, row_to_task as _row_to_task
from mem.vector_index import MemoryVectorIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FTS5 tokenizer helper (jieba if available, else regex)
# ---------------------------------------------------------------------------

_HAS_JIEBA = False
try:
    import jieba  # type: ignore

    _HAS_JIEBA = True
except ImportError:
    pass


def _tokenize_for_fts(text: str) -> str:
    text_lower = text.lower()
    if _HAS_JIEBA:
        words = jieba.cut_for_search(text_lower)
        return " ".join(w.strip() for w in words if w.strip() and len(w.strip()) > 1)
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text_lower)
    return " ".join(t for t in tokens if len(t) > 1)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# MemStore
# ---------------------------------------------------------------------------


class MemStore:
    """SQLite schema + sqlite-vec + FTS5，所有 CRUD 方法。"""

    def __init__(self, db_path: str, dimensions: int = 1536):
        self.db_path = db_path
        self.dimensions = dimensions
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA foreign_keys = ON")

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._schema = MemorySchema(self._conn, dimensions)
        self._fts_index = MemoryFtsIndex(
            self._conn,
            lambda text: _tokenize_for_fts(text),
        )
        self._vector_index = MemoryVectorIndex(
            self._conn,
            serialize_vector=lambda vector: serialize_float32(vector),
        )
        self._chunks = ChunkRepository(
            self._conn,
            sync_fts=lambda chunk_id: self._sync_chunk_fts(chunk_id),
            now_ms=lambda: _now_ms(),
            content_hash=lambda content: _content_hash(content),
        )
        self._tasks = TaskRepository(
            self._conn,
            sync_fts=lambda task_id: self._sync_task_fts(task_id),
            now_ms=lambda: _now_ms(),
        )
        self._boundary_reviews = BoundaryReviewRepository(
            self._conn,
            now_ms=lambda: _now_ms(),
        )
        self._skills = SkillRepository(
            self._conn,
            sync_fts=lambda skill_id: self._sync_skill_fts(skill_id),
            now_ms=lambda: _now_ms(),
        )
        self._dashboard_queries = MemoryDashboardQueries(self._conn)
        self._search_queries = MemorySearchQueries(
            self._conn,
            lambda query: _sanitize_fts(query),
        )
        self._session_summaries = SessionSummaryRepository(
            self._conn,
            now=lambda: time.time(),
            new_id=lambda: str(uuid.uuid4()),
        )
        self._init_schema()
        logger.info("MemStore initialized: %s (dim=%d)", db_path, dimensions)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._schema.initialize()

    # ------------------------------------------------------------------
    # FTS sync helpers — 预分词写入 FTS 索引
    # ------------------------------------------------------------------

    def _sync_chunk_fts(self, chunk_id: str) -> None:
        self._fts_index.sync_chunk(chunk_id)

    def _sync_task_fts(self, task_id: str) -> None:
        self._fts_index.sync_task(task_id)

    def _sync_skill_fts(self, skill_id: str) -> None:
        self._fts_index.sync_skill(skill_id)

    def rebuild_fts_indexes(self) -> None:
        """重建所有 FTS 索引（迁移或修复时使用）。"""
        self._fts_index.rebuild()

    def _ensure_chunk_columns(self) -> None:
        self._schema.ensure_chunk_columns()

    def _ensure_task_columns(self) -> None:
        self._schema.ensure_task_columns()

    # ------------------------------------------------------------------
    # Chunks — Write
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: Chunk) -> None:
        self._chunks.insert(chunk)

    def update_chunk_summary(
        self,
        chunk_id: str,
        summary: str,
        *,
        summary_source: SummarySource | None = None,
    ) -> None:
        self._chunks.update_summary(
            chunk_id,
            summary,
            summary_source=summary_source,
        )

    def update_chunk_embedding_status(
        self,
        chunk_id: str,
        status: EmbeddingStatus,
        error: str | None = None,
    ) -> None:
        self._chunks.update_embedding_status(chunk_id, status, error)

    def mark_dedup_status(
        self,
        chunk_id: str,
        status: DedupStatus,
        target: str | None = None,
        reason: str | None = None,
    ) -> None:
        self._chunks.mark_dedup_status(
            chunk_id,
            status,
            target,
            reason,
        )

    def orphan_chunk(self, chunk_id: str, reason: str | None = None) -> None:
        self._chunks.orphan(chunk_id, reason)

    def find_active_chunk_by_hash(self, content: str, owner: str) -> str | None:
        return self._chunks.find_active_by_hash(content, owner)

    # ------------------------------------------------------------------
    # Chunks — Read
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def get_chunks_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        return self._chunks.get_by_task(task_id, limit)

    def get_chunks_by_turn(self, session_key: str, turn_id: str) -> list[Chunk]:
        return self._chunks.get_by_turn(session_key, turn_id)

    def get_chunks_for_embedding_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        return self._chunks.get_for_embedding_retry(owner, limit)

    def get_chunks_for_summary_retry(
        self,
        owner: str | None = None,
        limit: int = 100,
    ) -> list[Chunk]:
        return self._chunks.get_for_summary_retry(owner, limit)

    def get_chunks_in_range(
        self,
        session_key: str,
        turn_id: str,
        seq: int,
        window: int = 2,
        owner: str | None = None,
    ) -> list[Chunk]:
        """Get neighboring chunks for timeline display."""
        return self._chunks.get_in_range(
            session_key,
            turn_id,
            seq,
            window,
            owner,
        )

    # ------------------------------------------------------------------
    # Embeddings (sqlite-vec)
    # ------------------------------------------------------------------

    def upsert_chunk_embedding(self, chunk_id: str, vec: list[float]) -> None:
        self._vector_index.upsert_chunk(chunk_id, vec)

    def delete_chunk_embedding(self, chunk_id: str) -> None:
        self._vector_index.delete_chunk(chunk_id)

    def upsert_task_embedding(self, task_id: str, vec: list[float]) -> None:
        self._vector_index.upsert_task(task_id, vec)

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        self._vector_index.upsert_skill(skill_id, vec)

    # ------------------------------------------------------------------
    # ANN Search (sqlite-vec)
    # ------------------------------------------------------------------

    def ann_search_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        """ANN search on vec_chunks, join back to chunks for metadata."""
        return self._search_queries.ann_search_chunks(
            query_vec,
            top_k=top_k,
            exclude_session=exclude_session,
        )

    def ann_search_tasks(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        return self._search_queries.ann_search_tasks(
            query_vec,
            top_k=top_k,
            owner=owner,
        )

    def ann_search_skills(
        self, query_vec: list[float], top_k: int = 5,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        return self._search_queries.ann_search_skills(
            query_vec,
            top_k=top_k,
            owner=owner,
        )

    def ann_dedup_candidates(
        self,
        query_vec: list[float],
        threshold: float,
        top_k: int = 5,
        owner: str | None = None,
    ) -> list[SearchHit]:
        """Find top-K similar active chunks above threshold for dedup."""
        return self._search_queries.ann_dedup_candidates(
            query_vec,
            threshold,
            top_k=top_k,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # FTS Search
    # ------------------------------------------------------------------

    def fts_search_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_chunks(
            query,
            limit=limit,
            exclude_session=exclude_session,
        )

    def fts_search_tasks(
        self, query: str, limit: int = 10,
        owner: str | None = None,
    ) -> list[TaskSearchHit]:
        return self._search_queries.fts_search_tasks(
            query,
            limit=limit,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # Tasks — CRUD
    # ------------------------------------------------------------------

    def insert_task(self, task: Task) -> None:
        self._tasks.insert(task)

    def get_active_task(self, owner: str = "agent:main") -> Task | None:
        return self._tasks.get_active(owner)

    def finalize_task(
        self, task_id: str, title: str, summary: str, status: TaskStatus = "completed"
    ) -> None:
        self._tasks.finalize(task_id, title, summary, status)

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def assign_chunks_to_task(self, chunk_ids: list[str], task_id: str) -> None:
        self._tasks.assign_chunks(chunk_ids, task_id)

    def get_all_active_tasks(self, owner: str = "agent:main") -> list[Task]:
        return self._tasks.get_all_active(owner)

    def get_active_task_by_session(
        self, session_key: str, owner: str = "agent:main"
    ) -> Task | None:
        return self._tasks.get_active_by_session(session_key, owner)

    def get_unassigned_chunks(
        self, session_key: str, owner: str | None = None,
    ) -> list[Chunk]:
        return self._tasks.get_unassigned_chunks(session_key, owner)

    def update_task(self, task_id: str, **fields: Any) -> None:
        self._tasks.update(task_id, **fields)

    # ------------------------------------------------------------------
    # Boundary reviews
    # ------------------------------------------------------------------

    def create_boundary_review(self, review: BoundaryReview) -> BoundaryReview:
        return self._boundary_reviews.create(review)

    def get_boundary_review(self, review_id: str) -> BoundaryReview | None:
        return self._boundary_reviews.get(review_id)

    def get_pending_boundary_review(
        self, session_key: str, owner: str,
    ) -> BoundaryReview | None:
        return self._boundary_reviews.get_pending_for_session(session_key, owner)

    def list_pending_boundary_reviews(
        self, owner: str, limit: int = 100,
    ) -> list[BoundaryReview]:
        return self._boundary_reviews.list_pending(owner, limit)

    def resolve_boundary_review(
        self,
        review_id: str,
        *,
        resolution: str,
        target_task_id: str | None = None,
        note: str = "",
    ) -> BoundaryReview | None:
        return self._boundary_reviews.resolve(
            review_id,
            resolution=resolution,
            target_task_id=target_task_id,
            note=note,
        )

    # ------------------------------------------------------------------
    # Skills — CRUD
    # ------------------------------------------------------------------

    def insert_skill(self, skill: Skill) -> None:
        self._skills.insert(skill)

    def get_skill(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def update_skill(self, skill_id: str, **fields: Any) -> None:
        self._skills.update(skill_id, **fields)

    def fts_search_skills(
        self, query: str, limit: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        return self._search_queries.fts_search_skills(
            query,
            limit=limit,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # Task-scoped Chunk Search (for recall)
    # ------------------------------------------------------------------

    def ann_search_chunks_in_tasks(
        self,
        query_vec: list[float],
        task_ids: list[str],
        top_k: int = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.ann_search_chunks_in_tasks(
            query_vec,
            task_ids,
            top_k=top_k,
            owner=owner,
        )

    def exact_search_chunks_in_tasks(
        self,
        query_vec: list[float],
        task_ids: list[str],
        top_k: int | None = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.exact_search_chunks_in_tasks(
            query_vec,
            task_ids,
            top_k=top_k,
            owner=owner,
        )

    def fts_search_chunks_in_tasks(
        self,
        query: str,
        task_ids: list[str],
        limit: int | None = 10,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_chunks_in_tasks(
            query,
            task_ids,
            limit=limit,
            owner=owner,
        )

    def ann_search_orphan_chunks(
        self,
        query_vec: list[float],
        top_k: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.ann_search_orphan_chunks(
            query_vec,
            top_k=top_k,
            exclude_session=exclude_session,
            owner=owner,
        )

    def fts_search_orphan_chunks(
        self,
        query: str,
        limit: int = 10,
        exclude_session: str | None = None,
        owner: str | None = None,
    ) -> list[SearchHit]:
        return self._search_queries.fts_search_orphan_chunks(
            query,
            limit=limit,
            exclude_session=exclude_session,
            owner=owner,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_dashboard_stats(self) -> dict[str, Any]:
        return self._dashboard_queries.get_stats()

    def list_dashboard_tasks(
        self,
        *,
        status: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        return self._dashboard_queries.list_tasks(
            status=status,
            limit=limit,
            offset=offset,
        )

    def list_dashboard_skills(self, *, status: str = "") -> list[dict[str, Any]]:
        return self._dashboard_queries.list_skills(status=status)

    def list_dashboard_memories(
        self,
        *,
        limit: int = 40,
        offset: int = 0,
        session: str = "",
        role: str = "",
    ) -> tuple[list[dict[str, Any]], int]:
        return self._dashboard_queries.list_memories(
            limit=limit,
            offset=offset,
            session=session,
            role=role,
        )

    def close(self) -> None:
        self._conn.close()


    # ------------------------------------------------------------------
    # Session Summaries (结构化压缩摘要)
    # ------------------------------------------------------------------

    def upsert_session_summary(
        self,
        session_id: str,
        agent_id: str,
        summary: dict[str, Any],
    ) -> SessionSummary:
        return self._session_summaries.upsert(
            session_id,
            agent_id,
            summary,
        )

    def delete_session_summary(self, session_id: str, agent_id: str) -> bool:
        return self._session_summaries.delete(session_id, agent_id)

    def get_session_summary(
        self, session_id: str, agent_id: str,
    ) -> SessionSummary | None:
        return self._session_summaries.get(session_id, agent_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_fts(query: str) -> str:
    """Sanitize query string for FTS5 MATCH."""
    tokenized = _tokenize_for_fts(query)
    if not tokenized.strip():
        return ""
    words = tokenized.split()
    safe = [w for w in words if re.match(r"^[\w\u4e00-\u9fff]+$", w)]
    if not safe:
        return ""
    return " OR ".join(safe)
