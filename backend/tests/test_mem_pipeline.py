from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path

sqlite_vec_stub = types.ModuleType("sqlite_vec")
sqlite_vec_stub.load = lambda _conn: None
sqlite_vec_stub.serialize_float32 = lambda _vec: b""
sys.modules.setdefault("sqlite_vec", sqlite_vec_stub)

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.skill_evolver import MemSkillEvolver
from mem.models import BoundaryReview
from mem.store import Chunk, SearchHit, Skill, SkillSearchHit, Task
from mem.task_processor import MemTaskProcessor
from mem.worker import IngestMessage, MemWorker


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeStoreForTaskProcessor:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.chunks: dict[str, Chunk] = {}
        self.unassigned: list[Chunk] = []
        self.boundary_reviews: dict[str, BoundaryReview] = {}

    def insert_task(self, task: Task) -> None:
        self.tasks[task.id] = task

    def insert_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.id] = chunk
        if chunk.task_id is None:
            self.unassigned.append(chunk)

    def get_chunks_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        chunks = [c for c in self.chunks.values() if c.task_id == task_id and c.dedup_status == "active"]
        chunks.sort(key=lambda c: c.created_at)
        return chunks if limit is None else chunks[:limit]

    def finalize_task(self, task_id: str, title: str, summary: str, status: str = "completed") -> None:
        task = self.tasks[task_id]
        task.title = title
        task.summary = summary
        task.status = status

    def orphan_chunk(self, chunk_id: str, reason: str | None = None) -> None:
        chunk = self.chunks[chunk_id]
        chunk.dedup_status = "orphaned"
        chunk.task_id = None
        chunk.dedup_reason = reason

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def update_task(self, task_id: str, **fields) -> None:
        task = self.tasks[task_id]
        for key, value in fields.items():
            setattr(task, key, value)

    def get_unassigned_chunks(self, session_key: str, owner: str | None = None) -> list[Chunk]:
        rows = [
            c for c in self.unassigned
            if c.session_key == session_key
            and c.task_id is None
            and c.dedup_status == "active"
            and (owner is None or c.owner == owner)
        ]
        rows.sort(key=lambda c: c.created_at)
        return rows

    def assign_chunks_to_task(self, chunk_ids: list[str], task_id: str) -> None:
        for chunk_id in chunk_ids:
            self.chunks[chunk_id].task_id = task_id
        self.unassigned = [c for c in self.unassigned if c.id not in set(chunk_ids)]

    def get_pending_boundary_review(
        self, session_key: str, owner: str,
    ) -> BoundaryReview | None:
        return next((
            review for review in self.boundary_reviews.values()
            if review.session_key == session_key
            and review.owner == owner
            and review.status == "pending"
        ), None)

    def create_boundary_review(self, review: BoundaryReview) -> BoundaryReview:
        existing = next((
            item for item in self.boundary_reviews.values()
            if item.owner == review.owner
            and item.session_key == review.session_key
            and item.turn_id == review.turn_id
        ), None)
        if existing:
            return existing
        self.boundary_reviews[review.id] = review
        return review

    def get_boundary_review(self, review_id: str) -> BoundaryReview | None:
        return self.boundary_reviews.get(review_id)

    def resolve_boundary_review(
        self, review_id: str, *, resolution: str,
        target_task_id: str | None = None, note: str = "",
    ) -> BoundaryReview | None:
        review = self.boundary_reviews.get(review_id)
        if review and review.status == "pending":
            review.status = "resolved"
            review.resolution = resolution
            review.target_task_id = target_task_id
            review.note = note
        return review

    def get_chunks_by_turn(self, session_key: str, turn_id: str) -> list[Chunk]:
        return [
            chunk for chunk in self.chunks.values()
            if chunk.session_key == session_key and chunk.turn_id == turn_id
            and chunk.dedup_status == "active"
        ]

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def get_active_task_by_session(
        self, session_key: str, owner: str = "agent:main",
    ) -> Task | None:
        return next((
            task for task in reversed(list(self.tasks.values()))
            if task.session_key == session_key
            and task.owner == owner
            and task.status == "active"
        ), None)

    def upsert_task_embedding(self, task_id: str, vec: list[float]) -> None:
        pass


class _FakeStoreForWorker:
    def __init__(self) -> None:
        self.active_hashes: dict[tuple[str, str], str] = {}
        self.inserted: dict[str, Chunk] = {}

    def find_active_chunk_by_hash(self, content: str, owner: str) -> str | None:
        from mem.persistence_values import content_hash

        return self.active_hashes.get((owner, content_hash(content)))

    def ann_dedup_candidates(self, embedding, threshold: float, top_k: int = 5, owner: str | None = None):
        return []

    def insert_chunk(self, chunk: Chunk) -> None:
        self.inserted[chunk.id] = chunk

    def upsert_chunk_embedding(self, chunk_id: str, vec: list[float]) -> None:
        chunk = self.inserted[chunk_id]
        chunk.embedding_status = "ok"
        chunk.embedding_error = None

    def get_chunks_for_summary_retry(self, owner: str | None = None, limit: int = 100) -> list[Chunk]:
        rows = [
            c for c in self.inserted.values()
            if c.dedup_status == "active" and c.summary_source == "fallback"
            and (owner is None or c.owner == owner)
        ]
        return rows[:limit]

    def get_chunks_for_embedding_retry(self, owner: str | None = None, limit: int = 100) -> list[Chunk]:
        rows = [
            c for c in self.inserted.values()
            if c.dedup_status == "active" and c.embedding_status == "failed"
            and (owner is None or c.owner == owner)
        ]
        return rows[:limit]

    def update_chunk_summary(self, chunk_id: str, summary: str, *, summary_source: str | None = None) -> None:
        chunk = self.inserted[chunk_id]
        chunk.summary = summary
        if summary_source is not None:
            chunk.summary_source = summary_source

    def update_chunk_embedding_status(self, chunk_id: str, status: str, error: str | None = None) -> None:
        chunk = self.inserted[chunk_id]
        chunk.embedding_status = status
        chunk.embedding_error = error


class _SpyStoreForSkillEvolver:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = {s.id: s for s in skills}
        self.last_fts_owner: str | None = None
        self.last_ann_owner: str | None = None

    def fts_search_skills(self, query: str, limit: int = 10, owner: str | None = None) -> list[SkillSearchHit]:
        self.last_fts_owner = owner
        return [
            SkillSearchHit(skill_id=s.id, score=1.0, name=s.name, description=s.description)
            for s in self.skills.values()
        ]

    def ann_search_skills(self, query_vec: list[float], top_k: int = 10, owner: str | None = None) -> list[SkillSearchHit]:
        self.last_ann_owner = owner
        return [
            SkillSearchHit(skill_id=s.id, score=1.0, name=s.name, description=s.description)
            for s in self.skills.values()
        ]

    def get_skill(self, skill_id: str) -> Skill | None:
        return self.skills.get(skill_id)


class _SpySkillEvolver(MemSkillEvolver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidates: list[Skill] = []

    async def _judge_related(self, task: Task, candidates: list[Skill]) -> Skill | None:
        self.candidates = candidates
        return candidates[0] if candidates else None


class _QueueConcurrencySkillEvolver(MemSkillEvolver):
    def __init__(self) -> None:
        super().__init__(store=None, embedder=_FakeEmbedder())  # type: ignore[arg-type]
        self.started = 0
        self.finished_tasks: list[str] = []
        self.enter_event = asyncio.Event()
        self.release_event = asyncio.Event()

    async def _process_one(self, task: Task) -> None:
        self.started += 1
        if self.started == 1:
            self.enter_event.set()
            await self.release_event.wait()
        self.finished_tasks.append(task.id)


class _SpyMemWorker(MemWorker):
    def __init__(self, on_chunks_ingested) -> None:
        super().__init__(store=_FakeStoreForWorker(), embedder=_FakeEmbedder(), on_chunks_ingested=on_chunks_ingested)


class _WorkerWithStubLLM(MemWorker):
    def __init__(self, store, embedder) -> None:
        super().__init__(
            store=store,
            embedder=embedder,
            llm_api_key="test-key",
            on_chunks_ingested=None,
        )
        self.summary_calls = 0

    async def _llm_chat(self, system: str, user: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
        self.summary_calls += 1
        return f"summary:{user[:20]}"


class MemPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_mem_worker_triggers_callback_for_each_touched_session(self) -> None:
        touched: list[tuple[str, bool]] = []

        async def _on_chunks_ingested(session_key: str, session_end: bool) -> None:
            touched.append((session_key, session_end))

        worker = _SpyMemWorker(_on_chunks_ingested)
        await worker.enqueue(
            [
                IngestMessage(role="user", content="a1", session_key="s1", turn_id="t1"),
                IngestMessage(role="assistant", content="a2", session_key="s1", turn_id="t1"),
                IngestMessage(role="user", content="b1", session_key="s2", turn_id="t2"),
                IngestMessage(role="assistant", content="b2", session_key="s2", turn_id="t2"),
                IngestMessage(role="user", content="a3", session_key="s1", turn_id="t3"),
            ],
            session_end=True,
        )

        self.assertEqual(touched, [("s1", True), ("s2", True)])

    async def test_low_value_message_is_skipped_before_ingest_chain(self) -> None:
        store = _FakeStoreForWorker()
        worker = _WorkerWithStubLLM(store, _FakeEmbedder())

        stats = await worker.enqueue([
            IngestMessage(role="assistant", content="好的", session_key="s1", turn_id="t1")
        ])

        self.assertEqual(stats["low_value_skipped"], 1)
        self.assertEqual(stats["stored"], 0)
        self.assertEqual(len(store.inserted), 0)

    async def test_retry_failed_chunks_recovers_summary_and_embedding(self) -> None:
        store = _FakeStoreForWorker()
        store.insert_chunk(
            Chunk(
                id="chunk-1",
                session_key="s1",
                turn_id="t1",
                seq=0,
                role="user",
                content="正式环境端口改为 6432",
                summary="正式环境端口改为 6432",
                owner="agent:main",
                summary_source="fallback",
                embedding_status="failed",
                embedding_error="network",
            )
        )
        worker = _WorkerWithStubLLM(store, _FakeEmbedder())

        stats = await worker.retry_failed_chunks(owner="agent:main")

        self.assertEqual(stats["summary_retry_count"], 1)
        self.assertEqual(stats["summary_retry_recovered"], 1)
        self.assertEqual(stats["embedding_retry_count"], 1)
        self.assertEqual(stats["embedding_retry_recovered"], 1)
        chunk = store.inserted["chunk-1"]
        self.assertEqual(chunk.summary_source, "llm")
        self.assertEqual(chunk.embedding_status, "ok")
        self.assertIsNone(chunk.embedding_error)

    async def test_skipped_task_chunks_become_orphan(self) -> None:
        store = _FakeStoreForTaskProcessor()
        task = Task(id="task-1", session_key="s1", owner="agent:main", status="active")
        store.insert_task(task)
        store.insert_chunk(
            Chunk(
                id="chunk-1",
                session_key="s1",
                turn_id="t1",
                seq=0,
                role="user",
                content="数据库端口是 6432",
                summary="数据库端口是 6432",
                task_id=task.id,
                owner="agent:main",
                dedup_status="active",
            )
        )

        processor = MemTaskProcessor(store, _FakeEmbedder())
        await processor._finalize_task(task)

        chunk = store.get_chunk("chunk-1")
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.dedup_status, "orphaned")
        self.assertIsNone(chunk.task_id)
        self.assertEqual(chunk.dedup_reason, "task_skipped")

    async def test_subagent_chunks_do_not_create_or_finalize_independent_task(self) -> None:
        store = _FakeStoreForTaskProcessor()
        parent = Task(
            id="task-main",
            session_key="main-session",
            owner="main",
            status="active",
        )
        store.insert_task(parent)
        store.insert_chunk(
            Chunk(
                id="child-user",
                session_key="child-session",
                turn_id="child-turn",
                seq=0,
                role="user",
                content="检查子任务",
                task_id="task-main",
                parent_task_id="task-main",
                source_type="subagent",
                owner="worker",
            )
        )
        store.insert_chunk(
            Chunk(
                id="child-asst",
                session_key="child-session",
                turn_id="child-turn",
                seq=1,
                role="assistant",
                content="子任务完成",
                task_id="task-main",
                parent_task_id="task-main",
                source_type="subagent",
                owner="worker",
            )
        )

        processor = MemTaskProcessor(store, _FakeEmbedder())
        await processor.on_chunks_ingested(
            "child-session",
            session_end=False,
            owner="worker",
        )

        self.assertEqual(list(store.tasks), ["task-main"])
        self.assertEqual(store.tasks["task-main"].status, "active")
        self.assertEqual(
            {store.chunks["child-user"].task_id, store.chunks["child-asst"].task_id},
            {"task-main"},
        )

    async def test_long_task_builds_boundary_summary(self) -> None:
        class _SpyTaskProcessor(MemTaskProcessor):
            async def _llm_call(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
                return "task-state-summary"

        store = _FakeStoreForTaskProcessor()
        task = Task(id="task-1", session_key="s1", owner="agent:main", status="active")
        store.insert_task(task)
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            store.insert_chunk(
                Chunk(
                    id=f"chunk-{i}",
                    session_key="s1",
                    turn_id=f"t{i//2}",
                    seq=i,
                    role=role,
                    content=f"{role}-{i}",
                    summary=f"{role}-summary-{i}",
                    owner="agent:main",
                    task_id="task-1",
                    created_at=i,
                    updated_at=i,
                )
            )

        processor = _SpyTaskProcessor(store=store, embedder=_FakeEmbedder(), llm_api_key="x")
        context = await processor._build_boundary_context(task, store.get_chunks_by_task("task-1"))

        self.assertEqual(task.boundary_summary, "task-state-summary")
        self.assertEqual(task.boundary_compacted_count, 10)
        self.assertIn("task-state-summary", context)
        self.assertIn("user-10", context)

    async def test_time_gap_no_longer_directly_splits_task(self) -> None:
        class _SameTopicProcessor(MemTaskProcessor):
            async def _judge_new_topic(self, context: str, new_message: str, *, gap_ms: int | None = None) -> bool | None:
                self.last_gap_ms = gap_ms
                return False

        store = _FakeStoreForTaskProcessor()
        task = Task(id="task-1", session_key="s1", owner="agent:main", status="active")
        store.insert_task(task)
        store.insert_chunk(
            Chunk(
                id="base-user",
                session_key="s1",
                turn_id="t0",
                seq=0,
                role="user",
                content="先看数据库端口",
                summary="数据库端口",
                owner="agent:main",
                task_id="task-1",
                created_at=0,
                updated_at=0,
            )
        )
        store.insert_chunk(
            Chunk(
                id="base-asst",
                session_key="s1",
                turn_id="t0",
                seq=1,
                role="assistant",
                content="好的，先检查配置",
                summary="检查配置",
                owner="agent:main",
                task_id="task-1",
                created_at=1,
                updated_at=1,
            )
        )
        store.insert_chunk(
            Chunk(
                id="new-user",
                session_key="s1",
                turn_id="t1",
                seq=0,
                role="user",
                content="继续看正式环境端口",
                summary="继续看正式环境端口",
                owner="agent:main",
                task_id=None,
                created_at=10 * 3600 * 1000,
                updated_at=10 * 3600 * 1000,
            )
        )

        processor = _SameTopicProcessor(store=store, embedder=_FakeEmbedder(), llm_api_key="x")
        await processor._process_chunks_incrementally(task, "s1", "agent:main")

        self.assertEqual(store.chunks["new-user"].task_id, "task-1")
        self.assertEqual(store.tasks["task-1"].status, "active")
        self.assertGreater(processor.last_gap_ms or 0, 0)

    async def test_uncertain_boundary_stays_unassigned_for_human_review(self) -> None:
        class _UncertainProcessor(MemTaskProcessor):
            async def _judge_new_topic(self, context: str, new_message: str, *, gap_ms: int | None = None):
                return None

        store = _FakeStoreForTaskProcessor()
        task = Task(id="task-1", session_key="s1", owner="agent:main", status="active")
        store.insert_task(task)
        for chunk in (
            Chunk(id="old-user", session_key="s1", turn_id="t1", seq=0, role="user", content="修复数据库", task_id="task-1", owner="agent:main", created_at=1),
            Chunk(id="old-asst", session_key="s1", turn_id="t1", seq=1, role="assistant", content="正在修复", task_id="task-1", owner="agent:main", created_at=2),
            Chunk(id="new-user", session_key="s1", turn_id="t2", seq=0, role="user", content="鱼香肉丝怎么做", owner="agent:main", created_at=3),
            Chunk(id="new-asst", session_key="s1", turn_id="t2", seq=1, role="assistant", content="准备食材", owner="agent:main", created_at=4),
        ):
            store.insert_chunk(chunk)

        processor = _UncertainProcessor(store=store, embedder=_FakeEmbedder())
        await processor._process_chunks_incrementally(task, "s1", "agent:main")

        self.assertIsNone(store.chunks["new-user"].task_id)
        self.assertIsNone(store.chunks["new-asst"].task_id)
        reviews = list(store.boundary_reviews.values())
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].turn_id, "t2")
        self.assertEqual(reviews[0].status, "pending")

    def test_boundary_judgment_parser_requires_valid_structured_output(self) -> None:
        parsed = MemTaskProcessor._parse_boundary_judgment(
            '{"decision":"NEW","confidence":0.96,"reason":"目标发生变化"}'
        )
        empty = MemTaskProcessor._parse_boundary_judgment("")

        self.assertEqual(parsed.decision, "NEW")
        self.assertEqual(parsed.confidence, 0.96)
        self.assertEqual(empty.decision, "UNCERTAIN")
        self.assertEqual(empty.reason, "empty_model_response")

    async def test_human_create_new_resolution_finalizes_old_task_and_assigns_turn(self) -> None:
        class _SummaryProcessor(MemTaskProcessor):
            async def _llm_call(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.1) -> str:
                return "📌 Title\n数据库修复\n\n✅ Outcome\n修复完成"

        store = _FakeStoreForTaskProcessor()
        old_task = Task(id="task-1", session_key="s1", owner="agent:main", status="active")
        store.insert_task(old_task)
        for chunk in (
            Chunk(id="old-user", session_key="s1", turn_id="t1", seq=0, role="user", content="请修复生产环境数据库连接失败的问题，并检查端口、用户名和连接超时配置", summary="修复数据库", task_id="task-1", owner="agent:main", created_at=1),
            Chunk(id="old-asst", session_key="s1", turn_id="t1", seq=1, role="assistant", content="已经检查数据库配置并修复错误端口，连接测试目前可以正常建立", summary="修复完成", task_id="task-1", owner="agent:main", created_at=2),
            Chunk(id="old-user-2", session_key="s1", turn_id="t1b", seq=0, role="user", content="继续验证生产环境数据库连接是否完全恢复，并运行集成测试确认没有回归", summary="验证连接", task_id="task-1", owner="agent:main", created_at=3),
            Chunk(id="old-asst-2", session_key="s1", turn_id="t1b", seq=1, role="assistant", content="生产环境连接验证成功，相关集成测试全部通过，没有发现新的连接错误", summary="验证成功", task_id="task-1", owner="agent:main", created_at=4),
            Chunk(id="new-user", session_key="s1", turn_id="t2", seq=0, role="user", content="鱼香肉丝怎么做", owner="agent:main", created_at=5),
            Chunk(id="new-asst", session_key="s1", turn_id="t2", seq=1, role="assistant", content="准备食材", owner="agent:main", created_at=6),
        ):
            store.insert_chunk(chunk)
        review = store.create_boundary_review(BoundaryReview(
            id="review-1", session_key="s1", owner="agent:main",
            current_task_id="task-1", turn_id="t2",
        ))
        processor = _SummaryProcessor(store=store, embedder=_FakeEmbedder())

        resolved = await processor.resolve_boundary_review(
            review.id, action="create_new", note="目标完全不同",
        )

        self.assertEqual(store.tasks["task-1"].status, "completed")
        self.assertTrue(store.tasks["task-1"].summary)
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(resolved.resolution, "create_new")
        new_task = next(task for task in store.tasks.values() if task.id != "task-1")
        self.assertEqual(store.chunks["new-user"].task_id, new_task.id)
        self.assertEqual(store.chunks["new-asst"].task_id, new_task.id)

        repeated = await processor.resolve_boundary_review(review.id, action="orphan")
        self.assertEqual(repeated.resolution, "create_new")
        self.assertEqual(store.chunks["new-user"].task_id, new_task.id)

    async def test_get_chunks_by_task_default_behavior_is_unbounded(self) -> None:
        store = _FakeStoreForTaskProcessor()
        task = Task(id="task-2", session_key="s2", owner="agent:main", status="active")
        store.insert_task(task)

        for i in range(60):
            store.insert_chunk(
                Chunk(
                    id=f"chunk-{i}",
                    session_key="s2",
                    turn_id=f"t-{i}",
                    seq=0,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"message {i}",
                    summary=f"summary {i}",
                    task_id=task.id,
                    owner="agent:main",
                    dedup_status="active",
                    created_at=i,
                )
            )

        chunks = store.get_chunks_by_task(task.id)
        self.assertEqual(len(chunks), 60)

    async def test_skill_related_candidates_only_include_same_owner_active_skills(self) -> None:
        task = Task(
            id="task-3",
            session_key="s3",
            owner="agent:main",
            title="排查数据库连接问题",
            summary="修复数据库连接，端口改为 6432",
            status="completed",
        )

        skills = [
            Skill(
                id="skill-1",
                name="db-troubleshooting",
                description="排查数据库连接问题",
                owner="agent:main",
                status="active",
            ),
            Skill(
                id="skill-2",
                name="db-draft",
                description="数据库草稿技能",
                owner="agent:main",
                status="draft",
            ),
            Skill(
                id="skill-3",
                name="db-other-owner",
                description="其他 owner 的数据库技能",
                owner="agent:other",
                status="active",
            ),
        ]
        store = _SpyStoreForSkillEvolver(skills)
        evolver = _SpySkillEvolver(store, _FakeEmbedder())

        related = await evolver._find_related_skill(task)

        self.assertIsNotNone(related)
        self.assertEqual(related.id, "skill-1")
        self.assertEqual(store.last_fts_owner, "agent:main")
        self.assertEqual(store.last_ann_owner, "agent:main")
        self.assertEqual([s.id for s in evolver.candidates], ["skill-1"])

    async def test_skill_evolver_concurrent_entry_starts_single_drain(self) -> None:
        evolver = _QueueConcurrencySkillEvolver()
        task_a = Task(id="task-a", session_key="s1", owner="agent:main", status="completed")
        task_b = Task(id="task-b", session_key="s1", owner="agent:main", status="completed")

        task_a_runner = asyncio.create_task(evolver.on_task_completed(task_a))
        await evolver.enter_event.wait()
        task_b_runner = asyncio.create_task(evolver.on_task_completed(task_b))
        await asyncio.sleep(0)

        self.assertTrue(evolver._processing)
        self.assertEqual(len(evolver._queue), 1)
        self.assertEqual(evolver._queue[0].id, "task-b")

        evolver.release_event.set()
        await asyncio.gather(task_a_runner, task_b_runner)

        self.assertEqual(evolver.started, 2)
        self.assertEqual(evolver.finished_tasks, ["task-a", "task-b"])
        self.assertFalse(evolver._processing)

    async def test_skill_evidence_prefers_goal_and_high_value_signals(self) -> None:
        evolver = _SpySkillEvolver(_SpyStoreForSkillEvolver([]), _FakeEmbedder())
        chunks = [
            Chunk(id="c1", session_key="s1", turn_id="t1", seq=0, role="user", content="帮我修复 postgres 连接问题", owner="agent:main"),
            Chunk(id="c2", session_key="s1", turn_id="t1", seq=1, role="assistant", content="好的", owner="agent:main"),
            Chunk(id="c3", session_key="s1", turn_id="t2", seq=0, role="user", content="报错 ECONNREFUSED 127.0.0.1:5432", owner="agent:main"),
            Chunk(id="c4", session_key="s1", turn_id="t2", seq=1, role="assistant", content="把 DB_PORT 改为 6432，并执行 `docker compose restart api`", summary="DB_PORT 改为 6432，重启 api", owner="agent:main"),
            Chunk(id="c5", session_key="s1", turn_id="t3", seq=0, role="assistant", content="最终验证成功，连接恢复正常，配置文件是 /app/.env", summary="验证成功，配置文件 /app/.env", owner="agent:main"),
        ]

        original_goal = evolver._extract_original_goal(chunks)
        evidence = evolver._build_skill_evidence(chunks)

        self.assertEqual(original_goal, "帮我修复 postgres 连接问题")
        self.assertIn("报错 ECONNREFUSED 127.0.0.1:5432", evidence)
        self.assertIn("DB_PORT 改为 6432", evidence)
        self.assertIn("验证成功", evidence)


if __name__ == "__main__":
    unittest.main()
