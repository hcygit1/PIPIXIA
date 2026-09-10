"""Persistence for unresolved semantic task-boundary decisions."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from mem.models import BoundaryReview


class BoundaryReviewRepository:
    def __init__(self, connection: sqlite3.Connection, *, now_ms: Callable[[], int]) -> None:
        self._connection = connection
        self._now_ms = now_ms

    def create(self, review: BoundaryReview) -> BoundaryReview:
        now = self._now_ms()
        review.created_at = review.created_at or now
        review.updated_at = review.updated_at or now
        self._connection.execute(
            """INSERT INTO boundary_reviews
               (id, session_key, owner, current_task_id, turn_id, confidence,
                reason, retry_count, status, resolution, target_task_id, note,
                created_at, updated_at, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(owner, session_key, turn_id) DO NOTHING""",
            (
                review.id, review.session_key, review.owner, review.current_task_id,
                review.turn_id, review.confidence, review.reason, review.retry_count,
                review.status, review.resolution, review.target_task_id, review.note,
                review.created_at, review.updated_at, review.resolved_at,
            ),
        )
        self._connection.commit()
        return self.get_by_turn(review.owner, review.session_key, review.turn_id) or review

    def get(self, review_id: str) -> BoundaryReview | None:
        row = self._connection.execute(
            "SELECT * FROM boundary_reviews WHERE id=?", (review_id,)
        ).fetchone()
        return row_to_boundary_review(row) if row else None

    def get_by_turn(
        self, owner: str, session_key: str, turn_id: str,
    ) -> BoundaryReview | None:
        row = self._connection.execute(
            """SELECT * FROM boundary_reviews
               WHERE owner=? AND session_key=? AND turn_id=?""",
            (owner, session_key, turn_id),
        ).fetchone()
        return row_to_boundary_review(row) if row else None

    def get_pending_for_session(
        self, session_key: str, owner: str,
    ) -> BoundaryReview | None:
        row = self._connection.execute(
            """SELECT * FROM boundary_reviews
               WHERE owner=? AND session_key=? AND status='pending'
               ORDER BY created_at LIMIT 1""",
            (owner, session_key),
        ).fetchone()
        return row_to_boundary_review(row) if row else None

    def list_pending(self, owner: str, limit: int = 100) -> list[BoundaryReview]:
        rows = self._connection.execute(
            """SELECT * FROM boundary_reviews
               WHERE owner=? AND status='pending'
               ORDER BY created_at LIMIT ?""",
            (owner, limit),
        ).fetchall()
        return [row_to_boundary_review(row) for row in rows]

    def resolve(
        self,
        review_id: str,
        *,
        resolution: str,
        target_task_id: str | None = None,
        note: str = "",
    ) -> BoundaryReview | None:
        now = self._now_ms()
        self._connection.execute(
            """UPDATE boundary_reviews
               SET status='resolved', resolution=?, target_task_id=?, note=?,
                   resolved_at=?, updated_at=?
               WHERE id=? AND status='pending'""",
            (resolution, target_task_id, note, now, now, review_id),
        )
        self._connection.commit()
        return self.get(review_id)


def row_to_boundary_review(row: sqlite3.Row) -> BoundaryReview:
    return BoundaryReview(
        id=row["id"], session_key=row["session_key"], owner=row["owner"],
        current_task_id=row["current_task_id"], turn_id=row["turn_id"],
        confidence=float(row["confidence"] or 0), reason=row["reason"] or "",
        retry_count=int(row["retry_count"] or 0), status=row["status"] or "pending",
        resolution=row["resolution"] or "", target_task_id=row["target_task_id"],
        note=row["note"] or "", created_at=int(row["created_at"] or 0),
        updated_at=int(row["updated_at"] or 0), resolved_at=row["resolved_at"],
    )
