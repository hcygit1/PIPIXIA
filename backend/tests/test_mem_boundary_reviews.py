from __future__ import annotations

import sqlite3
import unittest

from mem.boundary_review_repository import BoundaryReviewRepository
from mem.models import BoundaryReview


class BoundaryReviewStoreTests(unittest.TestCase):
    def test_pending_review_is_persisted_once_and_resolved_idempotently(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("""CREATE TABLE boundary_reviews (
            id TEXT PRIMARY KEY, session_key TEXT NOT NULL, owner TEXT NOT NULL,
            current_task_id TEXT NOT NULL, turn_id TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0, reason TEXT NOT NULL DEFAULT '',
            retry_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
            resolution TEXT NOT NULL DEFAULT '', target_task_id TEXT, note TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, resolved_at INTEGER,
            UNIQUE(owner, session_key, turn_id)
        )""")
        repository = BoundaryReviewRepository(connection, now_ms=lambda: 100)
        try:
            review = repository.create(BoundaryReview(
                id="review-1",
                session_key="session-1",
                owner="agent:main",
                current_task_id="task-1",
                turn_id="turn-2",
                confidence=0.35,
                reason="low confidence",
                retry_count=2,
            ))
            duplicate = repository.create(BoundaryReview(
                id="review-duplicate",
                session_key="session-1",
                owner="agent:main",
                current_task_id="task-1",
                turn_id="turn-2",
            ))

            self.assertEqual(review.id, "review-1")
            self.assertEqual(duplicate.id, "review-1")
            self.assertEqual(
                [item.id for item in repository.list_pending("agent:main")],
                ["review-1"],
            )

            resolved = repository.resolve(
                "review-1",
                resolution="assign_current",
                target_task_id="task-1",
                note="人工确认",
            )
            repeated = repository.resolve(
                "review-1",
                resolution="orphan",
            )

            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.status, "resolved")
            self.assertEqual(resolved.resolution, "assign_current")
            self.assertEqual(repeated.resolution, "assign_current")
            self.assertEqual(repository.list_pending("agent:main"), [])
        finally:
            connection.close()
