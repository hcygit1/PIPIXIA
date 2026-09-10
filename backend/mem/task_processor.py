"""任务层处理器 — 边界检测 + finalize + Task embedding

按 docs/memory-system-refactor.md §4.2 ⑥:
  - session 变化 → finalize 旧 Task，创建新 Task
  - 时间间隔作为弱信号参与 SAME/NEW 判断
  - LLM 判断话题切换 → finalize 旧 Task，创建新 Task
  - finalize: LLM 结构化摘要 → embedding → tasks + tasks_fts + vec_tasks
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Literal

import httpx

from mem.embedder import MemEmbedder
from mem.models import BoundaryReview, Chunk, Task
from mem.task_processor_store import MemTaskProcessorStore

logger = logging.getLogger(__name__)

BOUNDARY_WINDOW_SIZE = 10
BOUNDARY_ASSISTANT_MAX_LEN = 300

TRIVIAL_RE = re.compile(
    r"^(test|testing|hello|hi|hey|ok|okay|yes|no|yeah|nope|sure|thanks|thx|ping|pong|"
    r"哈哈|好的|嗯|是的|不是|谢谢|你好|测试)\s*[.!?。！？]*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TASK_SUMMARY_PROMPT = (
    "You create a structured high-value task memory from a multi-turn conversation. "
    "This summary will be used for later retrieval and skill generation, so it must "
    "preserve the final effective state and the most important operational details.\n\n"
    "CRITICAL LANGUAGE RULE: Write in the SAME language as the user's messages. "
    "Chinese input → Chinese output. English input → English output.\n\n"
    "Output EXACTLY this structure:\n\n"
    "📌 Title\n<short descriptive title, max 60 chars>\n\n"
    "🎯 Goal\n<what the user was trying to achieve>\n\n"
    "📋 Key Steps\n<numbered list of only the important steps, commands, edits, checks>\n\n"
    "✅ Outcome\n<final result — success/failure/partial, and the final effective state/value>\n\n"
    "💡 Insights\n<key lessons, gotchas, important decisions, fixes>\n\n"
    "Rules:\n"
    "- Preserve exact commands, file paths, config values, version numbers, error codes/messages when important\n"
    "- Preserve the FINAL effective value/state if something changed during the conversation\n"
    "- Keep failed attempts only if they explain the final fix or key lesson\n"
    "- Remove greetings, filler, repeated back-and-forth, and duplicated details\n"
    "- Replace secrets with [REDACTED]\n"
    "- Make it dense and retrieval-friendly, not conversational\n"
    "- Output summary only"
)

TOPIC_JUDGE_PROMPT = (
    "You are a conservative conversation task-boundary detector. "
    "Given the CURRENT task context and a single NEW user message, "
    "decide whether the new message belongs to the SAME task or clearly starts a NEW task.\n\n"
    "Return one JSON object with decision, confidence, and reason.\n"
    "decision must be SAME, NEW, or UNCERTAIN. confidence must be 0..1.\n\n"
    "Choose SAME when the new message:\n"
    "- continues, follows up on, retries, corrects, refines, or asks for the next step\n"
    "- references the same file, service, error, config, command, object, or workflow\n"
    "- is a natural continuation of the same troubleshooting or implementation thread\n"
    "- is ambiguous but can reasonably be interpreted as a continuation\n\n"
    "Choose NEW only when the new message clearly:\n"
    "- switches to a different goal, object, system, or workflow\n"
    "- starts a separate request that does not depend on the current task context\n"
    "- cannot reasonably be treated as a follow-up\n\n"
    "Important:\n"
    "- A long time gap is only a weak signal, not a hard boundary\n"
    "- Even after hours, if the user is clearly continuing the same troubleshooting or workflow, choose SAME\n"
    "- Same topic area does not automatically mean NEW\n"
    "- Follow-up questions on the same issue should be SAME\n"
    "- When evidence is insufficient, choose UNCERTAIN instead of guessing\n\n"
    'Output: {"decision":"SAME|NEW|UNCERTAIN","confidence":0.0,"reason":"brief reason"}'
)

BOUNDARY_SUMMARY_PROMPT = (
    "You maintain a compact task-state summary for future SAME/NEW boundary detection. "
    "Summarize the current task state, not the conversational style.\n\n"
    "Include only:\n"
    "- the current goal\n"
    "- the current object/system/file/service/error/config being worked on\n"
    "- the latest effective state or conclusion\n"
    "- the key steps or checks already performed if they still matter\n"
    "- key commands, paths, config values, versions, or error codes when important\n\n"
    "Do not include greetings, filler, repeated back-and-forth, or superseded details.\n"
    "Keep it dense and short. Output plain text only."
)

OnTaskCompleted = Callable[[Task], Coroutine[Any, Any, None]]


@dataclass(frozen=True)
class BoundaryJudgment:
    decision: Literal["SAME", "NEW", "UNCERTAIN"]
    confidence: float
    reason: str


# ---------------------------------------------------------------------------
# MemTaskProcessor
# ---------------------------------------------------------------------------

class MemTaskProcessor:
    """Detects task boundaries and finalizes tasks with LLM summaries."""

    def __init__(
        self,
        store: MemTaskProcessorStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        idle_timeout_ms: int = 2 * 3600 * 1000,
        on_task_completed: OnTaskCompleted | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.idle_timeout_ms = idle_timeout_ms
        self._on_task_completed = on_task_completed
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public entry — called after chunks are ingested
    # ------------------------------------------------------------------

    async def on_chunks_ingested(
        self, session_key: str, session_end: bool, owner: str = "agent:main",
    ) -> None:
        async with self._lock:
            await self._detect_and_process(session_key, owner)
            if session_end:
                active = self.store.get_active_task_by_session(session_key, owner)
                if active:
                    await self._finalize_task(active)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    async def _detect_and_process(self, session_key: str, owner: str) -> None:
        active = self.store.get_active_task_by_session(session_key, owner)
        unassigned = self.store.get_unassigned_chunks(session_key, owner)

        # A sub-agent writes chunks pre-bound to its parent's business task.
        # It owns no task in its own session, so it must not create or close one.
        if not active and not unassigned:
            return
        if not active:
            active = self._create_task(session_key, owner)

        await self._process_chunks_incrementally(active, session_key, owner)

    async def _process_chunks_incrementally(
        self, active_task: Task, session_key: str, owner: str,
    ) -> None:
        pending_review = self.store.get_pending_boundary_review(session_key, owner)
        if pending_review:
            logger.info(
                "Boundary processing paused: review=%s session=%s",
                pending_review.id,
                session_key,
            )
            return

        unassigned = self.store.get_unassigned_chunks(session_key, owner)
        if not unassigned:
            return

        task_chunks = self.store.get_chunks_by_task(active_task.id)

        turns = self._group_into_turns(unassigned)
        if not turns:
            self._assign_chunks(unassigned, active_task.id)
            return

        current_task = active_task
        current_chunks = list(task_chunks)

        for turn in turns:
            user_chunk = next((c for c in turn if c.role == "user"), None)

            if not user_chunk:
                self._assign_chunks(turn, current_task.id)
                current_chunks.extend(turn)
                continue

            gap_ms: int | None = None
            if current_chunks:
                last_ts = max(c.created_at for c in current_chunks)
                gap_ms = user_chunk.created_at - last_ts

            existing_user_count = sum(1 for c in current_chunks if c.role == "user")
            if existing_user_count < 1:
                self._assign_chunks(turn, current_task.id)
                current_chunks.extend(turn)
                continue

            context = await self._build_boundary_context(current_task, current_chunks)
            judgment = await self._judge_boundary_with_retry(
                context,
                user_chunk.content,
                gap_ms=gap_ms,
            )

            if judgment.decision == "UNCERTAIN":
                review = self.store.create_boundary_review(BoundaryReview(
                    id=str(uuid.uuid4()),
                    session_key=session_key,
                    owner=owner,
                    current_task_id=current_task.id,
                    turn_id=user_chunk.turn_id,
                    confidence=judgment.confidence,
                    reason=judgment.reason,
                    retry_count=2,
                ))
                logger.warning(
                    "Task boundary pending review=%s task=%s turn=%s confidence=%.2f reason=%s",
                    review.id,
                    current_task.id,
                    user_chunk.turn_id,
                    judgment.confidence,
                    judgment.reason,
                )
                return

            if judgment.decision == "NEW":
                logger.info("Task boundary: LLM judged new topic")
                await self._finalize_task(current_task)
                current_task = self._create_task(session_key, owner)
                current_chunks = []
            else:
                logger.info(
                    "Task boundary: SAME confidence=%.2f reason=%s",
                    judgment.confidence,
                    judgment.reason,
                )

            self._assign_chunks(turn, current_task.id)
            current_chunks.extend(turn)

    # ------------------------------------------------------------------
    # Task CRUD helpers
    # ------------------------------------------------------------------

    def _create_task(self, session_key: str, owner: str) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            session_key=session_key,
            owner=owner,
            title="",
            summary="",
            status="active",
        )
        self.store.insert_task(task)
        logger.info("Created new task=%s session=%s", task.id, session_key)
        return task

    def _assign_chunks(self, chunks: list[Chunk], task_id: str) -> None:
        if not chunks:
            return
        self.store.assign_chunks_to_task([c.id for c in chunks], task_id)

    async def _finalize_task(self, task: Task, *, notify_completed: bool = True) -> None:
        chunks = self.store.get_chunks_by_task(task.id)
        fallback_title = self._extract_title(chunks)

        if not chunks:
            self.store.finalize_task(task.id, fallback_title, "无对话内容", "skipped")
            return

        skip_reason = self._should_skip_summary(chunks)
        if skip_reason:
            logger.info("Task %s skipped: %s", task.id, skip_reason)
            self.store.finalize_task(task.id, fallback_title, skip_reason, "skipped")
            for c in chunks:
                self.store.orphan_chunk(c.id, reason="task_skipped")
            return

        conversation_text = self._build_conversation_text(chunks)
        try:
            summary = await self._llm_call(TASK_SUMMARY_PROMPT, conversation_text, max_tokens=4096, temperature=0.1)
        except Exception as e:
            logger.warning("Task summary failed for %s: %s", task.id, e)
            summary = self._fallback_summary(chunks)

        title, body = self._parse_title_from_summary(summary)
        title = title or fallback_title
        self.store.finalize_task(task.id, title, body, "completed")

        try:
            vec = await self.embedder.embed_query(f"{title} {body[:300]}")
            self.store.upsert_task_embedding(task.id, vec)
        except Exception as e:
            logger.warning("Task embedding failed for %s: %s", task.id, e)

        logger.info("Finalized task=%s title='%s' chunks=%d", task.id, title[:60], len(chunks))

        if notify_completed and self._on_task_completed:
            finalized = self.store.get_task(task.id)
            if finalized:
                try:
                    await self._on_task_completed(finalized)
                except Exception as e:
                    logger.warning("on_task_completed callback error: %s", e)

    # ------------------------------------------------------------------
    # Boundary detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_into_turns(chunks: list[Chunk]) -> list[list[Chunk]]:
        turns: list[list[Chunk]] = []
        current: list[Chunk] = []
        for c in chunks:
            if c.role == "user" and current:
                turns.append(current)
                current = []
            current.append(c)
        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _boundary_chunks(chunks: list[Chunk]) -> list[Chunk]:
        return [c for c in chunks if c.role in ("user", "assistant")]

    @staticmethod
    def _format_boundary_chunk(chunk: Chunk) -> str:
        label = "User" if chunk.role == "user" else "Assistant"
        if chunk.role == "user":
            text = chunk.content
        else:
            text = chunk.summary or chunk.content[:BOUNDARY_ASSISTANT_MAX_LEN]
        return f"[{label}]: {text}"

    @staticmethod
    def _fallback_boundary_summary(existing_summary: str, chunks: list[Chunk]) -> str:
        parts: list[str] = []
        if existing_summary.strip():
            parts.append(existing_summary.strip())
        parts.extend(MemTaskProcessor._format_boundary_chunk(c) for c in chunks)
        return "\n".join(parts)[:4000]

    async def _refresh_boundary_summary(self, task: Task, chunks: list[Chunk]) -> None:
        conv = self._boundary_chunks(chunks)
        if len(conv) <= BOUNDARY_WINDOW_SIZE:
            if task.boundary_summary or task.boundary_compacted_count:
                task.boundary_summary = ""
                task.boundary_compacted_count = 0
                self.store.update_task(task.id, boundary_summary="", boundary_compacted_count=0)
            return

        changed = False
        while len(conv) - task.boundary_compacted_count >= BOUNDARY_WINDOW_SIZE:
            next_batch = conv[
                task.boundary_compacted_count: task.boundary_compacted_count + BOUNDARY_WINDOW_SIZE
            ]
            if task.boundary_summary.strip():
                user_content = (
                    f"EXISTING TASK STATE SUMMARY:\n{task.boundary_summary.strip()}\n\n"
                    f"NEW CONVERSATION MESSAGES:\n"
                    + "\n".join(self._format_boundary_chunk(c) for c in next_batch)
                )
            else:
                user_content = (
                    "TASK CONVERSATION MESSAGES:\n"
                    + "\n".join(self._format_boundary_chunk(c) for c in next_batch)
                )
            try:
                new_summary = await self._llm_call(
                    BOUNDARY_SUMMARY_PROMPT,
                    user_content,
                    max_tokens=512,
                    temperature=0.1,
                )
            except Exception as e:
                logger.warning("Boundary summary refresh failed for task %s: %s", task.id, e)
                new_summary = self._fallback_boundary_summary(task.boundary_summary, next_batch)

            task.boundary_summary = new_summary.strip() or self._fallback_boundary_summary(task.boundary_summary, next_batch)
            task.boundary_compacted_count += len(next_batch)
            changed = True

        if changed:
            self.store.update_task(
                task.id,
                boundary_summary=task.boundary_summary,
                boundary_compacted_count=task.boundary_compacted_count,
            )

    async def _build_boundary_context(self, task: Task, chunks: list[Chunk]) -> str:
        conv = self._boundary_chunks(chunks)
        if not conv:
            return ""

        await self._refresh_boundary_summary(task, chunks)
        pending = conv[task.boundary_compacted_count:]
        if not task.boundary_summary:
            return "\n".join(self._format_boundary_chunk(c) for c in pending)

        sections = [f"--- Task state summary ---\n{task.boundary_summary.strip()}"]
        if pending:
            sections.append(
                "--- Recent conversation ---\n"
                + "\n".join(self._format_boundary_chunk(c) for c in pending)
            )
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_skip_summary(chunks: list[Chunk]) -> str | None:
        user_chunks = [c for c in chunks if c.role == "user"]
        asst_chunks = [c for c in chunks if c.role == "assistant"]

        if len(chunks) < 4:
            return f"消息过少（{len(chunks)} < 4）"

        turns = min(len(user_chunks), len(asst_chunks))
        if turns < 2:
            return f"对话轮次不足（{turns} < 2）"

        if not user_chunks:
            return "无用户消息"

        total_len = sum(len(c.content) for c in chunks)
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", user_chunks[0].content))
        min_len = 80 if has_cjk else 200
        if total_len < min_len:
            return f"内容过短（{total_len} < {min_len}）"

        user_content = " ".join(c.content for c in user_chunks)
        lines = [l.strip() for l in user_content.split("\n") if l.strip()]
        if lines:
            trivial = sum(1 for l in lines if len(l) < 5 or TRIVIAL_RE.match(l))
            if trivial / len(lines) > 0.7:
                return "用户内容为简单问候或测试"

        if len(user_chunks) >= 3:
            unique = set(c.content.strip().lower() for c in user_chunks)
            if len(unique) / len(user_chunks) < 0.4:
                return "用户消息高度重复"

        return None

    @staticmethod
    def _build_conversation_text(chunks: list[Chunk]) -> str:
        lines = []
        for c in chunks:
            label = "User" if c.role == "user" else ("Assistant" if c.role == "assistant" else c.role)
            lines.append(f"[{label}]: {c.content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_title_from_summary(summary: str) -> tuple[str, str]:
        m = re.search(r"📌\s*(?:Title|标题)\s*\n?(.+)", summary)
        if m:
            title = m.group(1).strip()[:80]
            body = (summary[:m.start()].strip() + "\n" + summary[m.end():].strip()).strip()
            return title, body
        return "", summary

    @staticmethod
    def _extract_title(chunks: list[Chunk]) -> str:
        first_user = next((c for c in chunks if c.role == "user"), None)
        if not first_user:
            return "Untitled Task"
        text = first_user.content.strip()
        return text[:60] if len(text) <= 60 else text[:57] + "..."

    @staticmethod
    def _fallback_summary(chunks: list[Chunk]) -> str:
        title_chunk = next((c for c in chunks if c.role == "user"), None)
        title = title_chunk.content[:60].strip() if title_chunk else "Untitled"
        summaries = [f"- {c.summary}" for c in chunks if c.summary]
        return "\n".join([f"🎯 Goal\n{title}", "", "📋 Key Steps", *summaries[:20]])

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _judge_new_topic(
        self,
        context: str,
        new_message: str,
        *,
        gap_ms: int | None = None,
    ) -> BoundaryJudgment:
        gap_text = ""
        if gap_ms is not None and gap_ms > 0:
            gap_text = f"\nTIME GAP FROM CURRENT TASK: {gap_ms / 3600000:.1f} hours"
        user_content = (
            f"CURRENT TASK CONTEXT:\n{context}"
            f"{gap_text}\n\n"
            f"NEW USER MESSAGE:\n{new_message}"
        )
        try:
            result = await self._llm_call(
                TOPIC_JUDGE_PROMPT, user_content, max_tokens=200, temperature=0,
            )
            return self._parse_boundary_judgment(result)
        except Exception as e:
            logger.warning("Topic judge failed: %s", e)
            return BoundaryJudgment("UNCERTAIN", 0.0, f"model_error: {str(e)[:160]}")

    async def _judge_boundary_with_retry(
        self,
        context: str,
        new_message: str,
        *,
        gap_ms: int | None,
    ) -> BoundaryJudgment:
        last = BoundaryJudgment("UNCERTAIN", 0.0, "no_decision")
        for attempt in range(2):
            result = await self._judge_new_topic(
                context if attempt == 0 else context + "\n\nRe-evaluate conservatively using only task semantics.",
                new_message,
                gap_ms=gap_ms,
            )
            last = self._normalize_boundary_judgment(result)
            if last.decision != "UNCERTAIN" and last.confidence >= 0.8:
                return last
        return BoundaryJudgment(
            "UNCERTAIN",
            last.confidence,
            last.reason or "low_confidence_after_retry",
        )

    @staticmethod
    def _normalize_boundary_judgment(result: Any) -> BoundaryJudgment:
        # Preserve compatibility with lightweight test doubles and older plugins.
        if result is True:
            return BoundaryJudgment("NEW", 1.0, "legacy_boolean")
        if result is False:
            return BoundaryJudgment("SAME", 1.0, "legacy_boolean")
        if isinstance(result, BoundaryJudgment):
            return result
        return BoundaryJudgment("UNCERTAIN", 0.0, "invalid_judgment")

    @staticmethod
    def _parse_boundary_judgment(raw: str) -> BoundaryJudgment:
        text = (raw or "").strip()
        if not text:
            return BoundaryJudgment("UNCERTAIN", 0.0, "empty_model_response")
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                payload = json.loads(match.group(0))
                decision = str(payload.get("decision", "UNCERTAIN")).upper()
                if decision not in {"SAME", "NEW", "UNCERTAIN"}:
                    decision = "UNCERTAIN"
                confidence = max(0.0, min(1.0, float(payload.get("confidence", 0))))
                reason = str(payload.get("reason", ""))[:500]
                return BoundaryJudgment(decision, confidence, reason)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        upper = text.upper()
        if upper == "NEW":
            return BoundaryJudgment("NEW", 1.0, "legacy_text")
        if upper == "SAME":
            return BoundaryJudgment("SAME", 1.0, "legacy_text")
        return BoundaryJudgment("UNCERTAIN", 0.0, "invalid_model_response")

    async def resolve_boundary_review(
        self,
        review_id: str,
        *,
        action: str,
        target_task_id: str | None = None,
        note: str = "",
    ) -> BoundaryReview:
        allowed = {"assign_current", "create_new", "assign_other", "orphan"}
        if action not in allowed:
            raise ValueError(f"unsupported boundary action: {action}")

        async with self._lock:
            review = self.store.get_boundary_review(review_id)
            if not review:
                raise ValueError("boundary review not found")
            if review.status == "resolved":
                return review

            chunks = self.store.get_chunks_by_turn(review.session_key, review.turn_id)
            pending_chunks = [chunk for chunk in chunks if chunk.task_id is None]
            chunk_ids = [chunk.id for chunk in pending_chunks]
            resolved_target = target_task_id

            if action == "assign_current":
                task = self.store.get_task(review.current_task_id)
                if not task:
                    raise ValueError("current task not found")
                resolved_target = task.id
                self._assign_chunks(pending_chunks, task.id)
            elif action == "create_new":
                current = self.store.get_task(review.current_task_id)
                if current and current.status == "active":
                    await self._finalize_task(current)
                new_task = self._create_task(review.session_key, review.owner)
                resolved_target = new_task.id
                self._assign_chunks(pending_chunks, new_task.id)
            elif action == "assign_other":
                if not target_task_id:
                    raise ValueError("target_task_id is required")
                target = self.store.get_task(target_task_id)
                if not target or target.owner != review.owner:
                    raise ValueError("target task not found")
                self._assign_chunks(pending_chunks, target.id)
                if target.status == "completed":
                    await self._finalize_task(target, notify_completed=False)
            else:
                for chunk_id in chunk_ids:
                    self.store.orphan_chunk(chunk_id, reason="boundary_review_orphan")

            resolved = self.store.resolve_boundary_review(
                review.id,
                resolution=action,
                target_task_id=resolved_target,
                note=note,
            )
            if not resolved:
                raise RuntimeError("failed to resolve boundary review")

            active = self.store.get_active_task_by_session(review.session_key, review.owner)
            if active:
                await self._process_chunks_incrementally(
                    active,
                    review.session_key,
                    review.owner,
                )
            return resolved

    async def _llm_call(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        endpoint = f"{self.llm_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_api_key}",
        }
        body = {
            "model": self.llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        store: MemTaskProcessorStore,
        embedder: MemEmbedder,
        on_task_completed: OnTaskCompleted | None = None,
    ) -> MemTaskProcessor:
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        task_cfg = config.get("task", {})
        idle_hours = task_cfg.get("idle_timeout_hours", 2)
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            idle_timeout_ms=int(idle_hours * 3600 * 1000),
            on_task_completed=on_task_completed,
        )
