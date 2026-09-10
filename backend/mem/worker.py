"""记忆入库队列 — hash 去重 + LLM summary + sqlite-vec ANN 语义去重

流程严格按 docs/memory-system-refactor.md §4.2:
  ① 精确 hash 去重
  ② LLM 生成 summary（保留数字/配置值/结论）
  ③ Embedder 对 summary 嵌入
  ④ sqlite-vec ANN 候选 → LLM judgeDedup（summary + content 前 300 字）
  ⑤ INSERT chunks + vec_chunks
  ⑥ 通知 TaskProcessor（阶段四接入）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Literal

import httpx

from mem.embedder import MemEmbedder
from mem.models import Chunk, SearchHit
from mem.persistence_values import content_hash, now_ms
from mem.worker_store import MemWorkerStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "Convert the input into ONE single-sentence memory record for retrieval (max 120 characters). "
    "IMPORTANT: Use the SAME language as the input text — if the input is Chinese, write Chinese; "
    "if English, write English. "
    "Keep only durable facts and actionable information. "
    "Preserve ALL numbers, config values, version numbers, file paths, commands, identifiers, "
    "error codes, state changes, and conclusions. "
    "Do NOT remove or generalize away concrete details. "
    "Remove greetings, filler, hedging, conversational wording, repetition, and background chatter. "
    "No bullet points, no preamble — output only the sentence."
)

DEDUP_JUDGE_PROMPT = """\
You are a memory deduplication system.

LANGUAGE RULE (MUST FOLLOW): You MUST reply in the SAME language as the input memories. \
如果输入是中文，reason 必须用中文。If input is English, reply in English.

Given a NEW memory and several EXISTING memories (each with summary and content excerpt), \
determine the relationship.

For each EXISTING memory, the NEW memory is either:
- "DUPLICATE": Content is identical or conveys the same information with ZERO new factual information. \
Only choose DUPLICATE when the NEW memory adds absolutely no new facts.
- "NEW": NEW either adds meaningful new information or covers a genuinely different topic/event

Pick the BEST match among all candidates. If none match well, choose "NEW".

CRITICAL RULES:
- Same topic does NOT mean DUPLICATE
- Similar wording does NOT mean DUPLICATE
- A follow-up on the same issue is still NEW if it adds any concrete detail
- Any of the following counts as NEW factual information:
  - new numbers or counts
  - new config values or parameter values
  - new version numbers
  - new error codes or error messages
  - new file paths, commands, identifiers, function names, table names, endpoint names
  - new state changes (failed -> success, pending -> done, old value -> new value)
  - new conclusions or decisions
  - new actionable steps or fixes
- When in doubt, choose NEW
- Be conservative: false duplicate is worse than false new

Output a single JSON object:
- If DUPLICATE: {"action":"DUPLICATE","targetIndex":2,"reason":"与已有记忆内容相同"}
- If NEW: {"action":"NEW","reason":"有新增信息或属于不同主题"}

Output ONLY the JSON object, no other text."""

LOW_VALUE_RE = re.compile(
    r"^(ok|okay|好的|收到|明白|知道了|谢谢|感谢|thx|thanks|yes|no|是的|不是|嗯|哦|好|test|测试|ping|pong)[.!?。！？]*$",
    re.IGNORECASE,
)
STRUCTURED_SIGNAL_RE = re.compile(
    r"(\d|/|\\|`|::|->|=>|[A-Z]{2,}\d+|v\d+(?:\.\d+)*|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_]+|--?[a-zA-Z0-9_-]+)"
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class IngestMessage:
    role: str
    content: str
    session_key: str
    turn_id: str
    owner: str = "agent:main"
    task_id: str | None = None
    parent_task_id: str | None = None
    source_type: Literal["main_agent", "subagent"] = "main_agent"
    timestamp: int = 0


@dataclass
class DedupResult:
    action: Literal["DUPLICATE", "NEW"]
    target_index: int | None = None
    reason: str = ""


@dataclass
class PreparedChunk:
    msg: IngestMessage
    chunk_id: str
    kind: str
    content: str
    summary: str
    summary_source: Literal["llm", "fallback"]
    embedding: list[float] | None = None
    embedding_status: Literal["ok", "failed", "skipped"] = "ok"
    embedding_error: str | None = None


@dataclass
class IngestOutcome:
    action: IngestAction
    metrics: dict[str, int]
    prepared: PreparedChunk | None = None


IngestAction = Literal["stored", "duplicate", "skipped", "error"]

OnChunksIngested = Callable[[str, bool], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# MemWorker
# ---------------------------------------------------------------------------


class MemWorker:
    """异步入库队列，按 §4.2 流程处理每条消息。"""

    def __init__(
        self,
        store: MemWorkerStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        dedup_threshold: float = 0.60,
        on_chunks_ingested: OnChunksIngested | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.dedup_threshold = dedup_threshold
        self._on_chunks_ingested = on_chunks_ingested

        self._queue: deque[tuple[IngestMessage, bool]] = deque()
        self._processing = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        messages: list[IngestMessage],
        session_end: bool = False,
    ) -> dict[str, int]:
        """Enqueue messages for ingestion. Returns stats after processing."""
        for msg in messages:
            self._queue.append((msg, session_end))

        return await self._process_queue()

    # ------------------------------------------------------------------
    # Queue processor
    # ------------------------------------------------------------------

    async def _process_queue(self) -> dict[str, int]:
        async with self._lock:
            if self._processing:
                return {"queued": len(self._queue)}
            self._processing = True

        stats = {
            "stored": 0,
            "duplicate": 0,
            "skipped": 0,
            "error": 0,
            "exact_hash_skipped": 0,
            "low_value_skipped": 0,
            "summary_fallback_count": 0,
            "embedding_failed_count": 0,
            "embedding_batched_items": 0,
            "embedding_retry_count": 0,
            "summary_retry_count": 0,
        }
        sessions_touched: dict[str, bool] = {}

        try:
            prepared_items: list[PreparedChunk] = []
            while self._queue:
                msg, is_session_end = self._queue.popleft()
                sessions_touched[msg.session_key] = (
                    sessions_touched.get(msg.session_key, False) or is_session_end
                )
                try:
                    outcome = await self._prepare_one(msg)
                    for key, value in outcome.metrics.items():
                        stats[key] = stats.get(key, 0) + value
                    if outcome.prepared is not None:
                        prepared_items.append(outcome.prepared)
                    else:
                        stats[outcome.action] += 1
                except Exception:
                    logger.exception("Failed to ingest message turn=%s", msg.turn_id)
                    stats["error"] += 1

            if prepared_items:
                await self._embed_prepared_items(prepared_items, stats)
                for prepared in prepared_items:
                    try:
                        outcome = await self._store_prepared(prepared)
                        for key, value in outcome.metrics.items():
                            stats[key] = stats.get(key, 0) + value
                        stats[outcome.action] += 1
                    except Exception:
                        logger.exception("Failed to store prepared chunk turn=%s", prepared.msg.turn_id)
                        stats["error"] += 1

            if sessions_touched and self._on_chunks_ingested:
                try:
                    for session_key, session_end in sessions_touched.items():
                        await self._on_chunks_ingested(session_key, session_end)
                except Exception:
                    logger.exception("on_chunks_ingested callback failed")
        finally:
            self._processing = False

        logger.info("Ingest batch: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Per-message pipeline
    # ------------------------------------------------------------------

    async def _prepare_one(self, msg: IngestMessage) -> IngestOutcome:
        content = msg.content.strip()
        if not content:
            return IngestOutcome("skipped", {})

        if self._is_low_value_message(content):
            return IngestOutcome("skipped", {"low_value_skipped": 1})

        # ① 精确 hash 去重
        existing = self.store.find_active_chunk_by_hash(content, msg.owner)
        if existing:
            logger.debug("Exact hash dup → %s", existing)
            return IngestOutcome("skipped", {"exact_hash_skipped": 1})

        chunk_id = uuid.uuid4().hex[:16]
        kind = "tool_result" if msg.role == "tool" else "paragraph"

        # ② LLM summary
        summary, summary_source = await self._generate_summary(content)
        metrics: dict[str, int] = {}
        if summary_source == "fallback":
            metrics["summary_fallback_count"] = 1

        prepared = PreparedChunk(
            msg=msg,
            chunk_id=chunk_id,
            kind=kind,
            content=content,
            summary=summary,
            summary_source=summary_source,
        )
        return IngestOutcome("stored", metrics, prepared=prepared)

    async def _store_prepared(self, prepared: PreparedChunk) -> IngestOutcome:
        # ④ 语义去重 (ANN + LLM judge)
        dedup_status: Literal["active", "duplicate"] = "active"
        dedup_target: str | None = None
        dedup_reason: str | None = None

        if prepared.embedding:
            candidates = self.store.ann_dedup_candidates(
                prepared.embedding, self.dedup_threshold, top_k=5, owner=prepared.msg.owner
            )
            if candidates:
                result = await self._judge_dedup(prepared.summary, prepared.content, candidates)
                if result and result.action == "DUPLICATE" and result.target_index is not None:
                    idx = result.target_index - 1
                    if 0 <= idx < len(candidates):
                        dedup_status = "duplicate"
                        dedup_target = candidates[idx].chunk_id
                        dedup_reason = result.reason
                        logger.debug("Dedup DUPLICATE → %s", dedup_target)

        # ⑤ INSERT
        ts = prepared.msg.timestamp or now_ms()
        chunk = Chunk(
            id=prepared.chunk_id,
            session_key=prepared.msg.session_key,
            turn_id=prepared.msg.turn_id,
            seq=0,
            role=prepared.msg.role,
            content=prepared.content,
            kind=prepared.kind,
            summary=prepared.summary,
            task_id=prepared.msg.task_id,
            parent_task_id=prepared.msg.parent_task_id,
            source_type=prepared.msg.source_type,
            owner=prepared.msg.owner,
            content_hash=content_hash(prepared.content),
            dedup_status=dedup_status,
            dedup_target=dedup_target,
            dedup_reason=dedup_reason,
            summary_source=prepared.summary_source,
            embedding_status="skipped" if dedup_status == "duplicate" else prepared.embedding_status,
            embedding_error=prepared.embedding_error,
            created_at=ts,
            updated_at=ts,
        )
        self.store.insert_chunk(chunk)

        if prepared.embedding and dedup_status == "active":
            self.store.upsert_chunk_embedding(prepared.chunk_id, prepared.embedding)

        logger.debug(
            "Stored chunk=%s role=%s dedup=%s len=%d summary_source=%s embedding_status=%s",
            prepared.chunk_id,
            prepared.msg.role,
            dedup_status,
            len(prepared.content),
            prepared.summary_source,
            chunk.embedding_status,
        )

        if dedup_status == "duplicate":
            return IngestOutcome("duplicate", {})
        return IngestOutcome("stored", {})

    async def _embed_prepared_items(
        self,
        prepared_items: list[PreparedChunk],
        stats: dict[str, int],
    ) -> None:
        to_embed = [item for item in prepared_items if item.summary.strip()]
        if not to_embed:
            return
        try:
            vectors = await self.embedder.embed([item.summary for item in to_embed])
            stats["embedding_batched_items"] += len(to_embed)
            for item, vec in zip(to_embed, vectors):
                item.embedding = vec
                item.embedding_status = "ok"
        except Exception as e:
            logger.warning("Batch embedding failed, storing without vectors: %s", e)
            for item in to_embed:
                item.embedding = None
                item.embedding_status = "failed"
                item.embedding_error = str(e)[:200]
                stats["embedding_failed_count"] += 1

    async def retry_failed_chunks(
        self,
        *,
        owner: str | None = None,
        summary_limit: int = 50,
        embedding_limit: int = 100,
    ) -> dict[str, int]:
        stats = {
            "summary_retry_count": 0,
            "embedding_retry_count": 0,
            "summary_retry_recovered": 0,
            "embedding_retry_recovered": 0,
        }
        if self.llm_api_key:
            for chunk in self.store.get_chunks_for_summary_retry(owner=owner, limit=summary_limit):
                stats["summary_retry_count"] += 1
                summary, source = await self._generate_summary(chunk.content)
                if source == "llm":
                    self.store.update_chunk_summary(chunk.id, summary, summary_source="llm")
                    stats["summary_retry_recovered"] += 1

        retry_chunks = self.store.get_chunks_for_embedding_retry(owner=owner, limit=embedding_limit)
        if retry_chunks:
            stats["embedding_retry_count"] = len(retry_chunks)
            try:
                vectors = await self.embedder.embed([c.summary for c in retry_chunks])
                for chunk, vec in zip(retry_chunks, vectors):
                    self.store.upsert_chunk_embedding(chunk.id, vec)
                    self.store.update_chunk_embedding_status(chunk.id, "ok", None)
                    stats["embedding_retry_recovered"] += 1
            except Exception as e:
                logger.warning("Embedding retry batch failed: %s", e)
                for chunk in retry_chunks:
                    self.store.update_chunk_embedding_status(chunk.id, "failed", str(e)[:200])
        logger.info("Ingest retry batch: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # LLM: summary generation
    # ------------------------------------------------------------------

    async def _generate_summary(self, content: str) -> tuple[str, Literal["llm", "fallback"]]:
        """Generate a concise summary via LLM. Falls back to truncation."""
        if not self.llm_api_key:
            return self._fallback_summary(content), "fallback"

        try:
            text = content[:2000] if len(content) > 2000 else content
            result = await self._llm_chat(
                system=SUMMARY_SYSTEM_PROMPT,
                user=text,
                max_tokens=200,
                temperature=0.0,
            )
            if result:
                return result.strip(), "llm"
            return self._fallback_summary(content), "fallback"
        except Exception:
            logger.warning("Summary LLM failed, using fallback")
            return self._fallback_summary(content), "fallback"

    @staticmethod
    def _fallback_summary(content: str) -> str:
        lines = content.strip().split("\n")
        first = lines[0].strip() if lines else content[:120]
        if len(first) > 120:
            first = first[:117] + "..."
        return first

    # ------------------------------------------------------------------
    # LLM: dedup judge
    # ------------------------------------------------------------------

    async def _judge_dedup(
        self,
        new_summary: str,
        new_content: str,
        candidates: list[SearchHit],
    ) -> DedupResult | None:
        """Ask LLM to judge dedup (summary + content 前 300 字)."""
        if not self.llm_api_key:
            return None

        new_excerpt = new_content[:300]
        candidate_lines: list[str] = []
        for i, c in enumerate(candidates, 1):
            candidate_lines.append(
                f"{i}. Summary: {c.summary}\n   Excerpt: {c.content_excerpt[:300]}"
            )
        candidate_text = "\n".join(candidate_lines)

        user_msg = (
            f"NEW MEMORY:\nSummary: {new_summary}\nExcerpt: {new_excerpt}\n\n"
            f"EXISTING MEMORIES:\n{candidate_text}"
        )

        try:
            raw = await self._llm_chat(
                system=DEDUP_JUDGE_PROMPT,
                user=user_msg,
                max_tokens=300,
                temperature=0.0,
            )
            return self._parse_dedup_result(raw)
        except Exception:
            logger.warning("Dedup judge LLM failed, treating as NEW")
            return DedupResult(action="NEW", reason="llm_failed")

    @staticmethod
    def _parse_dedup_result(raw: str) -> DedupResult:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                action = obj.get("action", "NEW").upper()
                if action not in ("DUPLICATE", "NEW"):
                    action = "NEW"
                return DedupResult(
                    action=action,
                    target_index=obj.get("targetIndex"),
                    reason=obj.get("reason", ""),
                )
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse dedup result: %s", raw[:200])
        return DedupResult(action="NEW", reason="parse_failed")

    @staticmethod
    def _is_low_value_message(content: str) -> bool:
        text = content.strip()
        if len(text) > 24:
            return False
        if STRUCTURED_SIGNAL_RE.search(text):
            return False
        if "\n" in text:
            return False
        return bool(LOW_VALUE_RE.match(text))

    # ------------------------------------------------------------------
    # LLM: generic chat completion
    # ------------------------------------------------------------------

    async def _llm_chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 300,
        temperature: float = 0.0,
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

        async with httpx.AsyncClient(timeout=30.0) as client:
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
        store: MemStore,
        embedder: MemEmbedder,
        on_chunks_ingested: OnChunksIngested | None = None,
    ) -> MemWorker:
        """Build from the full `mem` config dict."""
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        dedup_cfg = config.get("dedup", {})
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            dedup_threshold=dedup_cfg.get("similarity_threshold", 0.60),
            on_chunks_ingested=on_chunks_ingested,
        )
