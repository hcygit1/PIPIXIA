"""Domain and query-result models shared by the memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DedupStatus = Literal["active", "duplicate", "merged", "orphaned"]
SummarySource = Literal["llm", "fallback"]
EmbeddingStatus = Literal["ok", "failed", "skipped"]
TaskStatus = Literal["active", "completed", "skipped"]
TaskSourceType = Literal["main_agent", "subagent"]
BoundaryReviewStatus = Literal["pending", "resolved"]
SkillStatus = Literal["active", "archived", "draft"]
SkillVisibility = Literal["private", "public"]


@dataclass
class Chunk:
    id: str
    session_key: str
    turn_id: str
    seq: int
    role: str
    content: str
    kind: str = "paragraph"
    summary: str = ""
    task_id: str | None = None
    parent_task_id: str | None = None
    source_type: TaskSourceType = "main_agent"
    skill_id: str | None = None
    owner: str = "agent:main"
    content_hash: str = ""
    dedup_status: DedupStatus = "active"
    dedup_target: str | None = None
    dedup_reason: str | None = None
    summary_source: SummarySource = "llm"
    embedding_status: EmbeddingStatus = "ok"
    embedding_error: str | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class Task:
    id: str
    session_key: str
    owner: str = "agent:main"
    title: str = ""
    summary: str = ""
    boundary_summary: str = ""
    boundary_compacted_count: int = 0
    status: TaskStatus = "active"
    started_at: int = 0
    ended_at: int | None = None
    updated_at: int = 0


@dataclass
class BoundaryReview:
    id: str
    session_key: str
    owner: str
    current_task_id: str
    turn_id: str
    confidence: float = 0.0
    reason: str = ""
    retry_count: int = 0
    status: BoundaryReviewStatus = "pending"
    resolution: str = ""
    target_task_id: str | None = None
    note: str = ""
    created_at: int = 0
    updated_at: int = 0
    resolved_at: int | None = None


@dataclass
class Skill:
    id: str
    name: str
    description: str = ""
    dir_path: str = ""
    version: int = 1
    status: SkillStatus = "active"
    installed: int = 0
    owner: str = "agent:main"
    visibility: SkillVisibility = "private"
    quality_score: float | None = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    summary: str = ""
    content_excerpt: str = ""
    role: str = ""
    session_key: str = ""
    task_id: str | None = None
    created_at: int = 0


@dataclass
class TaskSearchHit:
    task_id: str
    score: float
    title: str = ""
    summary: str = ""
    status: str = ""
    started_at: int = 0
    ended_at: int | None = None


@dataclass
class SkillSearchHit:
    skill_id: str
    score: float
    name: str = ""
    description: str = ""


@dataclass
class SessionSummary:
    id: str
    session_id: str
    agent_id: str
    version: int = 1
    goal: str = ""
    decisions: str = ""
    progress: str = ""
    open_items: str = ""
    entities: str = ""
    user_preferences: str = ""
    raw_summary: str = ""
    token_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
