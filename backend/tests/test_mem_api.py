from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.dependencies import get_agent_manager
from api.mem_api import (
    BoundaryReviewResolution,
    mem_boundary_reviews,
    mem_memories,
    mem_skills,
    mem_stats,
    mem_tasks,
    resolve_mem_boundary_review,
    router,
)


class MemoryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_stats_selects_requested_agent_and_uses_public_query(self) -> None:
        store = Mock(spec=["get_dashboard_stats"])
        store.get_dashboard_stats.return_value = {
            "totalChunks": 2,
            "totalTasks": 1,
        }
        manager = Mock()
        manager.mem_stores = {"worker": store}

        result = await mem_stats(
            agent_id="worker",
            agent_manager=manager,
        )

        store.get_dashboard_stats.assert_called_once_with()
        self.assertTrue(result["ok"])
        self.assertEqual(result["totalChunks"], 2)

    async def test_collection_endpoints_delegate_without_connection_access(self) -> None:
        store = Mock(
            spec=[
                "list_dashboard_tasks",
                "list_dashboard_skills",
                "list_dashboard_memories",
            ]
        )
        store.list_dashboard_tasks.return_value = ([{"id": "task-1"}], 1)
        store.list_dashboard_skills.return_value = [{"id": "skill-1"}]
        store.list_dashboard_memories.return_value = ([{"id": "chunk-1"}], 1)
        manager = Mock()
        manager.mem_stores = {"worker": store}

        tasks = await mem_tasks(
            agent_id="worker",
            status="",
            limit=10,
            offset=0,
            agent_manager=manager,
        )
        skills = await mem_skills(
            agent_id="worker",
            status="",
            agent_manager=manager,
        )
        memories = await mem_memories(
            agent_id="worker",
            limit=10,
            page=1,
            session="",
            role="",
            agent_manager=manager,
        )

        self.assertEqual(tasks["total"], 1)
        self.assertEqual(skills["skills"][0]["id"], "skill-1")
        self.assertEqual(memories["total"], 1)

    def test_http_dependency_override_does_not_change_openapi(self) -> None:
        store = Mock()
        store.get_dashboard_stats.return_value = {"source": "override"}
        manager = Mock(mem_stores={"worker": store})
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_agent_manager] = lambda: manager
        client = TestClient(app)

        response = client.get("/api/mem/stats?agent_id=worker")
        operation = app.openapi()["paths"]["/api/mem/stats"]["get"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"], "override")
        self.assertEqual(
            {item["name"] for item in operation.get("parameters", [])},
            {"agent_id"},
        )

    async def test_boundary_review_list_includes_context_and_pending_turn(self) -> None:
        from types import SimpleNamespace

        review = SimpleNamespace(
            id="review-1", session_key="session-1", current_task_id="task-1",
            turn_id="turn-2", confidence=0.3, reason="uncertain",
            retry_count=2, created_at=100,
        )
        store = Mock()
        store.list_pending_boundary_reviews.return_value = [review]
        store.get_task.return_value = SimpleNamespace(
            title="数据库修复", boundary_summary="正在检查端口", summary="",
        )
        store.get_chunks_by_task.return_value = [
            SimpleNamespace(id="old", role="user", content="检查数据库"),
        ]
        store.get_chunks_by_turn.return_value = [
            SimpleNamespace(id="new", role="user", content="鱼香肉丝怎么做"),
        ]
        manager = Mock(mem_stores={"main": store})

        result = await mem_boundary_reviews(
            agent_id="main", limit=100, agent_manager=manager,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["reviews"][0]["pendingChunks"][0]["id"], "new")
        self.assertEqual(result["reviews"][0]["currentTaskSummary"], "正在检查端口")

    async def test_boundary_review_resolution_delegates_to_processor(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        processor = SimpleNamespace(resolve_boundary_review=AsyncMock(return_value=SimpleNamespace(
            id="review-1", status="resolved", resolution="create_new",
            target_task_id="task-2",
        )))
        manager = Mock(mem_task_processors={"main": processor})

        result = await resolve_mem_boundary_review(
            "review-1",
            BoundaryReviewResolution(action="create_new", note="主题不同"),
            agent_id="main",
            agent_manager=manager,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["targetTaskId"], "task-2")
        processor.resolve_boundary_review.assert_awaited_once_with(
            "review-1", action="create_new", target_task_id=None, note="主题不同",
        )


if __name__ == "__main__":
    unittest.main()
