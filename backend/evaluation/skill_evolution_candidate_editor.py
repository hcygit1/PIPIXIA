"""Candidate generation and review artifacts for the evolution workspace."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import load_config
from evaluation.skill_candidate import build_candidate_record, repair_candidate_content, save_candidate_manifest
from mem.skill_generation import SKILL_GENERATE_PROMPT


BENCHMARK_SKILL_GENERATE_PROMPT = """\
You are designing an initial reusable Agent Skill from one seed task instance.

The seed is a task specification, not a verified successful demonstration. Infer a robust workflow
that can solve similar instances, then express it as an executable SKILL.md.

Rules:
- Use only the task instruction and observable environment inventory below.
- Never infer or mention hidden tests, verifier implementation, solution files, or reference skills.
- Do not copy instance-specific filenames as fixed answers; generalize discovery and classification.
- Give an action-first workflow: inspect minimally, act in bounded batches, verify outputs, then finish.
- Each step must state its goal, tool/action strategy, and completion condition.
- Include safeguards against excessive file reading, repeated exploration, and stopping before artifacts exist.
- Include a final acceptance checklist derived only from the user-visible task requirements.
- Capture one coherent workflow. Keep it concise and operational.

Output exactly one complete SKILL.md with this structure:
---
name: "{NAME}"
description: "What this workflow does and when it should trigger"
---

# {TITLE}
## Purpose
## Preconditions
## Workflow
## Tool strategy
## Completion checks
## Failure handling
## Boundaries

Seed task material:
{EVIDENCE}

Output ONLY the complete SKILL.md content."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_samples(root: Path) -> list[dict[str, Any]]:
    path = root / "datasets" / "v1" / "samples.jsonl"
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            row = json.loads(raw)
            if row.get("split") == "seed":
                rows.append(row)
    return rows


def _skill_name(task_family: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", task_family.lower()).strip("-")
    return value or "generated-workflow"


def _next_version(root: Path) -> int:
    versions = [
        int(path.name[1:])
        for path in (root / "candidates").glob("v*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    return max(versions, default=0) + 1


async def generate(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    seed = _seed_samples(root)
    if not seed:
        raise ValueError("冻结数据集没有 Seed 样本")
    version = _next_version(root)
    candidate_dir = root / "candidates" / f"v{version}"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    skill_path = candidate_dir / "SKILL.md"
    evidence = json.dumps([{"sample_id": row.get("sample_id"), "task_snapshot": row.get("task_snapshot"), "trajectory": row.get("trajectory"), "verifier_result": row.get("verifier_result")} for row in seed], ensure_ascii=False, indent=2)[:14000]
    if task.get("data_source") == "skilllearnbench":
        prompt = BENCHMARK_SKILL_GENERATE_PROMPT.replace("{NAME}", _skill_name(task["task_family"])).replace("{TITLE}", task["task_family"]).replace("{EVIDENCE}", evidence)
    else:
        prompt = SKILL_GENERATE_PROMPT.replace("{NAME}", _skill_name(task["task_family"])).replace("{TITLE}", task["task_family"]).replace("{SUMMARY}", f"从 {len(seed)} 条成功 Seed 轨迹中提炼可复用流程").replace("{ORIGINAL_GOAL}", task["task_family"]).replace("{EVIDENCE}", evidence)
    llm = load_config().get("mem", {}).get("llm", {})
    base_url = str(llm.get("base_url") or "").rstrip("/")
    api_key = str(llm.get("api_key") or "")
    model = str(llm.get("model") or "qwen-plus")
    if not base_url or not api_key:
        raise ValueError("记忆模型未配置，无法生成 Candidate")
    async def llm_call(request_prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        body: dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": request_prompt}]}
        if "dashscope.aliyuncs.com" in base_url:
            body["enable_thinking"] = False
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=body)
            response.raise_for_status()
        choices = response.json().get("choices", [])
        return str(choices[0].get("message", {}).get("content", "") if choices else "").strip()

    content = await llm_call(prompt)
    fence = re.fullmatch(r"```(?:markdown|md)?\s*\n([\s\S]*?)\n```", content, re.IGNORECASE)
    content = fence.group(1).strip() if fence else content
    if not content:
        raise ValueError("模型未返回 Candidate 内容")
    content, _check, repair_attempts = await repair_candidate_content(content, llm_call=llm_call, max_attempts=2)
    skill_path.write_text(content, encoding="utf-8")
    record = build_candidate_record(family=task["task_family"], source_samples=seed, skill_id=f"{task['task_family']}-skill", version=version, skill_path=skill_path)
    manifest_path = candidate_dir / "candidate.json"
    save_candidate_manifest(manifest_path, record)
    result = {"candidate_id": record.candidate_id, "skill_path": str(skill_path), "manifest_path": str(manifest_path), "version": version, "source_sample_ids": list(record.source_sample_ids), "checks": {"passed": record.checks.passed, "reasons": list(record.checks.reasons), "warnings": list(record.checks.warnings), "repairable": record.checks.repairable, "repair_attempts": repair_attempts}}
    (candidate_dir / "result.json").write_text(json.dumps({**result, "generated_at": _now()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_result(root: Path) -> dict[str, Any] | None:
    versions = sorted(
        (path for path in (root / "candidates").glob("v*") if path.name[1:].isdigit()),
        key=lambda path: int(path.name[1:]),
        reverse=True,
    )
    path = versions[0] / "result.json" if versions else root / "candidates" / "v1" / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
