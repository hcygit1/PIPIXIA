"""Skill 进化器 — 评估 + 生成 + 升级 + 安装

按 docs/memory-system-refactor.md §6:
  Task 完成 → 规则过滤 → LLM 评估 → 四步流水线生成 Skill
  相似 Task → 搜索 Skill → LLM 判断升级
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import httpx

from mem.embedder import MemEmbedder
from mem.models import Chunk, Skill, Task
from mem.skill_artifact import (
    build_new_skill,
    extract_skill_description as _extract_description,
    write_skill_file,
)
from mem.skill_evaluation import (
    CREATE_EVAL_PROMPT,
    UPGRADE_EVAL_PROMPT,
    CreateEvalResult,
    UpgradeEvalResult,
    evaluate_skill_creation,
    evaluate_skill_upgrade,
)
from mem.skill_evidence import (
    build_skill_evidence,
    chunk_signal_score,
    extract_original_goal,
)
from mem.skill_evolver_store import MemSkillEvolverStore
from mem.skill_generation import (
    SKILL_GENERATE_PROMPT,
    build_skill_generation_prompt,
    generate_skill_content,
)
from mem.skill_persistence import persist_new_skill
from mem.skill_quality import score_skill_quality
from mem.skill_relation import find_related_skill, judge_related_skill
from mem.skill_version_store import SkillVersionStore
from mem.skill_upgrade import (
    SKILL_UPGRADE_PROMPT,
    execute_skill_upgrade,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MemSkillEvolver
# ---------------------------------------------------------------------------

class MemSkillEvolver:
    """Evaluates completed Tasks and generates/upgrades Skills."""

    def __init__(
        self,
        store: MemSkillEvolverStore,
        embedder: MemEmbedder,
        *,
        llm_base_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "gpt-4o-mini",
        skill_store_dir: str = "",
        min_chunks_for_eval: int = 6,
        min_confidence: float = 0.7,
        auto_install: bool = False,
        enabled: bool = True,
    ):
        self.store = store
        self.embedder = embedder
        self.llm_base_url = (llm_base_url or "https://api.openai.com/v1").rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.skill_store_dir = skill_store_dir
        self.min_chunks_for_eval = min_chunks_for_eval
        self.min_confidence = min_confidence
        self.auto_install = auto_install
        self.enabled = enabled
        self._lock = asyncio.Lock()
        self._queue: list[Task] = []
        self._processing = False
        self.last_llm_response = ""

    # ------------------------------------------------------------------
    # Public entry — called by MemTaskProcessor
    # ------------------------------------------------------------------

    async def on_task_completed(self, task: Task) -> None:
        if not self.enabled:
            return
        should_start_drain = False
        async with self._lock:
            if self._processing:
                self._queue.append(task)
                return
            self._processing = True
            should_start_drain = True
        if should_start_drain:
            await self._drain(task)

    async def _drain(self, task: Task) -> None:
        try:
            await self._process_one(task)
            while self._queue:
                next_task = self._queue.pop(0)
                await self._process_one(next_task)
        finally:
            self._processing = False

    async def _process_one(self, task: Task) -> None:
        try:
            await self._process(task)
        except Exception as e:
            logger.error("SkillEvolver error for task %s: %s", task.id, e)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _process(self, task: Task) -> None:
        chunks = self.store.get_chunks_by_task(task.id)

        skip = self._rule_filter(chunks, task)
        if skip:
            logger.debug("SkillEvolver: task %s skipped by rule filter: %s", task.id, skip)
            return

        related = await self._find_related_skill(task)

        if related:
            await self._handle_existing_skill(task, chunks, related)
        else:
            await self._handle_new_skill(task, chunks)

    # ------------------------------------------------------------------
    # Rule filter (§6.2)
    # ------------------------------------------------------------------

    def _rule_filter(self, chunks: list[Chunk], task: Task) -> str | None:
        if task.status == "skipped":
            return "task状态为skipped"
        if len(chunks) < self.min_chunks_for_eval:
            return f"chunk数量不足 ({len(chunks)} < {self.min_chunks_for_eval})"
        if len(task.summary or "") < 100:
            return f"summary过短 ({len(task.summary or '')} < 100)"
        if not any(c.role == "user" for c in chunks):
            return "无用户消息"
        if not any(c.role == "assistant" for c in chunks):
            return "无助手回复"
        return None

    # ------------------------------------------------------------------
    # Find related existing Skill (FTS + ANN → LLM judge)
    # ------------------------------------------------------------------

    async def _find_related_skill(self, task: Task) -> Skill | None:
        return await find_related_skill(
            task,
            store=self.store,
            embedder=self.embedder,
            judge_related=self._judge_related,
        )

    async def _judge_related(self, task: Task, candidates: list[Skill]) -> Skill | None:
        try:
            return await judge_related_skill(
                task,
                candidates,
                llm_call=self._llm_call,
                parse_json=_parse_json,
            )
        except Exception as e:
            logger.warning("Skill relation judge failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Handle new skill
    # ------------------------------------------------------------------

    async def _handle_new_skill(self, task: Task, chunks: list[Chunk]) -> None:
        eval_result = await self._evaluate_create(task)

        if not eval_result.should_generate or eval_result.confidence < self.min_confidence:
            logger.debug(
                "SkillEvolver: not generating for '%s' (confidence=%.2f)",
                task.title, eval_result.confidence,
            )
            return

        logger.info("SkillEvolver: generating skill '%s'", eval_result.suggested_name)
        skill = await self._generate_skill(task, chunks, eval_result)
        if skill:
            logger.info("SkillEvolver: skill '%s' created (id=%s)", skill.name, skill.id)

    # ------------------------------------------------------------------
    # Handle existing skill (upgrade)
    # ------------------------------------------------------------------

    async def _handle_existing_skill(
        self, task: Task, chunks: list[Chunk], skill: Skill,
    ) -> None:
        eval_result = await self._evaluate_upgrade(task, skill)

        if not eval_result.should_upgrade or eval_result.confidence < self.min_confidence:
            return

        logger.info("SkillEvolver: upgrading skill '%s' — %s", skill.name, eval_result.reason)
        await self._upgrade_skill(task, skill, eval_result)

    # ------------------------------------------------------------------
    # LLM Evaluator
    # ------------------------------------------------------------------

    async def _evaluate_create(self, task: Task) -> CreateEvalResult:
        try:
            return await evaluate_skill_creation(
                task,
                llm_call=self._llm_call,
                parse_json=_parse_json,
            )
        except Exception as e:
            logger.warning("Skill create eval failed: %s", e)
            return CreateEvalResult(reason=f"error: {e}")

    async def _evaluate_upgrade(self, task: Task, skill: Skill) -> UpgradeEvalResult:
        try:
            return await evaluate_skill_upgrade(
                task,
                skill,
                llm_call=self._llm_call,
                parse_json=_parse_json,
            )
        except Exception as e:
            logger.warning("Skill upgrade eval failed: %s", e)
            return UpgradeEvalResult(reason=f"error: {e}")

    # ------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------

    async def _generate_skill(
        self, task: Task, chunks: list[Chunk], eval_result: CreateEvalResult,
    ) -> Skill | None:
        original_goal = self._extract_original_goal(chunks)
        evidence = self._build_skill_evidence(chunks)
        prompt = build_skill_generation_prompt(
            task,
            eval_result,
            original_goal=original_goal,
            evidence=evidence,
            prompt_template=SKILL_GENERATE_PROMPT,
        )

        try:
            skill_content = await generate_skill_content(
                prompt,
                llm_call=self._llm_call,
            )
        except Exception as e:
            logger.error("Skill generation LLM call failed: %s", e)
            return None

        if not skill_content or len(skill_content.strip()) < 50:
            logger.warning("Skill generation returned empty/short content")
            return None

        skill_id = str(uuid.uuid4())
        name = eval_result.suggested_name or f"skill-{skill_id[:8]}"
        description = _extract_description(skill_content) or (task.summary or "")[:200]

        skill_dir = write_skill_file(
            self.skill_store_dir,
            name,
            skill_content,
            path_provider=lambda value: Path(value),
        )

        quality_score = await self._score_quality(skill_content, task)

        skill = build_new_skill(
            skill_id=skill_id,
            name=name,
            description=description,
            skill_dir=skill_dir,
            task=task,
            quality_score=quality_score,
            skill_factory=Skill,
        )
        persistence_error = await persist_new_skill(
            skill,
            store=self.store,
            embedder=self.embedder,
        )
        if persistence_error is not None:
            logger.warning("Skill embedding failed: %s", persistence_error)

        return skill

    async def _generate_candidate(
        self, task: Task, chunks: list[Chunk], eval_result: CreateEvalResult,
        *, source: str = "offline-distill",
    ) -> Skill | None:
        """Generate an offline candidate without publishing or indexing it."""
        original_goal = self._extract_original_goal(chunks)
        evidence = self._build_skill_evidence(chunks)
        prompt = build_skill_generation_prompt(
            task, eval_result, original_goal=original_goal, evidence=evidence,
            prompt_template=SKILL_GENERATE_PROMPT,
        )
        try:
            content = await generate_skill_content(prompt, llm_call=self._llm_call)
        except Exception as error:
            logger.error("Candidate Skill generation failed: %s", error)
            return None
        if not content or not content.strip():
            return None
        # Offline candidates get two format-only repair attempts before the
        # evolution attempt is abandoned. Security failures are never repaired.
        from evaluation.skill_candidate import repair_candidate_content

        content, static_check, repair_attempts = await repair_candidate_content(
            content, llm_call=self._llm_call, max_attempts=2,
        )
        name = eval_result.suggested_name or f"skill-{uuid.uuid4().hex[:8]}"
        description = _extract_description(content) or (task.summary or "")[:200]
        if static_check.blocking_reasons:
            logger.warning(
                "Unsafe Candidate Skill '%s' discarded: %s",
                name, "; ".join(static_check.blocking_reasons),
            )
            return None
        store = SkillVersionStore(
            evolution_root=Path(self.skill_store_dir).parent / "skill_evolution",
            active_root=Path(self.skill_store_dir),
        )
        candidate = store.create_candidate(
            skill_id=name, content=content, source=source,
            reason=(
                f"{eval_result.reason or 'offline candidate'}; "
                f"static_repair_attempts={repair_attempts}"
            ),
        )
        if not static_check.passed:
            store.set_candidate_status(candidate, "abandoned")
            logger.warning(
                "Candidate Skill '%s' abandoned after static checks: %s",
                name, "; ".join(static_check.reasons),
            )
            return None
        quality_score = await self._score_quality(content, task)
        return build_new_skill(
            skill_id=str(uuid.uuid4()), name=name, description=description,
            skill_dir=str(store.candidate_dir(name, candidate.version)), task=task,
            quality_score=quality_score, skill_factory=Skill,
        )

    @staticmethod
    def _extract_original_goal(chunks: list[Chunk]) -> str:
        return extract_original_goal(chunks)

    @staticmethod
    def _chunk_signal_score(chunk: Chunk, index: int, total: int) -> int:
        return chunk_signal_score(chunk, index, total)

    def _build_skill_evidence(self, chunks: list[Chunk]) -> str:
        return build_skill_evidence(
            chunks,
            signal_score=self._chunk_signal_score,
        )

    async def _score_quality(self, content: str, task: Task) -> float:
        return await score_skill_quality(
            content,
            task,
            llm_call=self._llm_call,
            parse_json=_parse_json,
        )

    # ------------------------------------------------------------------
    # Upgrader
    # ------------------------------------------------------------------

    async def _upgrade_skill(
        self, task: Task, skill: Skill, eval_result: UpgradeEvalResult,
    ) -> None:
        outcome = await execute_skill_upgrade(
            task,
            skill,
            eval_result,
            store=self.store,
            embedder=self.embedder,
            llm_call=self._llm_call,
            extract_description=_extract_description,
            path_provider=lambda value: Path(value),
            prompt_template=SKILL_UPGRADE_PROMPT,
        )
        if outcome.llm_error is not None:
            logger.error("Skill upgrade LLM call failed: %s", outcome.llm_error)
            return
        if outcome.upgraded:
            logger.info("Skill '%s' upgraded to v%d", skill.name, outcome.version)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    async def _llm_call(
        self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.1,
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
            "messages": [{"role": "user", "content": prompt}],
        }
        if "dashscope.aliyuncs.com" in self.llm_base_url or self.llm_model.lower().startswith("glm-"):
            body["enable_thinking"] = False
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        self.last_llm_response = str(content or "").strip()
        return self.last_llm_response

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        store: MemSkillEvolverStore,
        embedder: MemEmbedder,
        skill_store_dir: str = "",
    ) -> MemSkillEvolver:
        llm_cfg = config.get("llm", {})
        embedding_cfg = config.get("embedding", {})
        skill_cfg = config.get("skill_evolution", {})
        return cls(
            store=store,
            embedder=embedder,
            llm_base_url=llm_cfg.get("base_url") or embedding_cfg.get("base_url", ""),
            llm_api_key=llm_cfg.get("api_key") or embedding_cfg.get("api_key", ""),
            llm_model=llm_cfg.get("model") or "qwen-plus",
            skill_store_dir=skill_store_dir,
            min_chunks_for_eval=skill_cfg.get("min_chunks_for_eval", 6),
            min_confidence=skill_cfg.get("min_confidence", 0.7),
            auto_install=skill_cfg.get("auto_install", False),
            enabled=skill_cfg.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return fallback

    candidates = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    first_brace = text.find("{")
    if first_brace >= 0:
        candidates.append(text[first_brace:])
    candidates.append(text)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            try:
                from json_repair import loads as repair_json

                parsed = repair_json(candidate)
            except (ImportError, ValueError, TypeError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return fallback
