"""Storage port consumed by the memory task processor."""

from __future__ import annotations

from typing import Any, Protocol

from mem.models import BoundaryReview, Chunk, Task, TaskStatus


class MemTaskProcessorStore(Protocol):
    def get_active_task_by_session(
        self,
        session_key: str,
        owner: str = "agent:main",
    ) -> Task | None:
        ...

    def get_all_active_tasks(self, owner: str = "agent:main") -> list[Task]:
        ...

    def get_unassigned_chunks(
        self,
        session_key: str,
        owner: str | None = None,
    ) -> list[Chunk]:
        ...

    def get_chunks_by_task(
        self,
        task_id: str,
        limit: int | None = None,
    ) -> list[Chunk]:
        ...

    def insert_task(self, task: Task) -> None:
        ...

    def assign_chunks_to_task(self, chunk_ids: list[str], task_id: str) -> None:
        ...

    def finalize_task(
        self,
        task_id: str,
        title: str,
        summary: str,
        status: TaskStatus = "completed",
    ) -> None:
        ...

    def orphan_chunk(self, chunk_id: str, reason: str | None = None) -> None:
        ...

    def upsert_task_embedding(self, task_id: str, vec: list[float]) -> None:
        ...

    def get_task(self, task_id: str) -> Task | None:
        ...

    def update_task(self, task_id: str, **fields: Any) -> None:
        ...

    def create_boundary_review(self, review: BoundaryReview) -> BoundaryReview:
        ...

    def get_boundary_review(self, review_id: str) -> BoundaryReview | None:
        ...

    def get_pending_boundary_review(
        self, session_key: str, owner: str,
    ) -> BoundaryReview | None:
        ...

    def resolve_boundary_review(
        self,
        review_id: str,
        *,
        resolution: str,
        target_task_id: str | None = None,
        note: str = "",
    ) -> BoundaryReview | None:
        ...

    def get_chunks_by_turn(self, session_key: str, turn_id: str) -> list[Chunk]:
        ...
