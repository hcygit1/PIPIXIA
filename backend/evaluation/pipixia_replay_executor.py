"""Isolated PIPIXIA runtime replay used by offline Skill evaluations."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx

from config import load_config


def _instruction(sample: dict[str, Any]) -> str:
    snapshot = sample.get("task_snapshot")
    if isinstance(snapshot, dict):
        messages = snapshot.get("messages")
        if isinstance(messages, list):
            content = [str(row.get("content", "")) for row in messages if isinstance(row, dict) and row.get("role") == "user" and row.get("content")]
            if content:
                return "\n".join(content)
        for key in ("goal", "instruction", "title", "summary"):
            if snapshot.get(key):
                return str(snapshot[key])
    if isinstance(snapshot, str) and snapshot.strip():
        return snapshot.strip()
    raise ValueError("样本缺少可回放的用户任务")


async def _verify(sample: dict[str, Any], variant: str, instruction: str, output: str) -> tuple[bool | None, str | None, str | None]:
    verifier = sample.get("verifier")
    if isinstance(verifier, dict):
        expected = verifier.get("expected_contains")
        if isinstance(expected, str):
            return expected in output, None, None
        if isinstance(expected, list):
            return all(str(value) in output for value in expected), None, None
    try:
        llm = load_config().get("mem", {}).get("llm", {})
        base_url, api_key, model = str(llm.get("base_url") or "").rstrip("/"), str(llm.get("api_key") or ""), str(llm.get("model") or "qwen-plus")
        if base_url and api_key:
            prompt = f"判断 Agent 输出是否完成用户任务。只返回 JSON：{{\"passed\":true或false,\"reason\":\"简短原因\"}}。\n\n用户任务：\n{instruction[:5000]}\n\nAgent 输出：\n{output[:8000]}"
            body: dict[str, Any] = {"model": model, "temperature": 0, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]}
            if "dashscope.aliyuncs.com" in base_url:
                body["enable_thinking"] = False
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=body)
                response.raise_for_status()
            raw = str(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                import json
                judged = json.loads(raw[start:end + 1])
                if isinstance(judged.get("passed"), bool):
                    return judged["passed"], "semantic_judge", str(judged.get("reason") or "") or None
    except Exception:
        pass
    fallback = sample.get("verifier_result")
    if isinstance(fallback, dict):
        value = fallback.get(variant, fallback.get("passed"))
        if isinstance(value, bool):
            return value, "stored_verifier_fallback", None
    if isinstance(fallback, bool):
        return fallback, "stored_verifier_fallback", None
    return None, "missing_verifier", "Agent 已执行，但样本没有可用于验证本次输出的规则"


async def replay(sample: dict[str, Any], variant: str, skill_path: Path | None, *, agent_id: str = "main") -> dict[str, Any]:
    from runtime.agent import agent_manager

    instruction = _instruction(sample)
    skill_content = skill_path.read_text(encoding="utf-8") if skill_path is not None else ""
    skill_prompt = "本次评估禁止使用任何 Skill。" if not skill_content else f"本次评估只允许使用下面这一份 Skill：\n\n<skill>\n{skill_content}\n</skill>"
    session_id = f"eval-{uuid.uuid4().hex}"
    output = ""
    usage: dict[str, Any] = {}
    async for event in agent_manager.astream(
        instruction,
        session_id,
        agent_id=agent_id,
        prompt_mode="evaluation",
        persist_input_role="",
        extra_system_prompt=f"你正在执行离线 Skill 评估。完成用户任务并给出最终结果。{skill_prompt}",
        evaluation_mode=True,
    ):
        if event.get("type") == "error":
            return {"passed": False, "external_failure": True, "failure_type": "agent_runtime_error", "error": str(event.get("error", "Agent 执行失败"))}
        if event.get("type") == "done":
            output = str(event.get("content", ""))
            usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
    passed, verifier_type, verifier_error = await _verify(sample, variant, instruction, output)
    return {
        "passed": bool(passed) if passed is not None else False,
        "external_failure": passed is None,
        "failure_type": verifier_type if passed is None else None,
        "error": verifier_error,
        "tokens": float(usage.get("total_tokens", 0) or 0),
        "duration_ms": float(usage.get("duration_ms", 0) or 0),
        "output": output[:8000],
        "verifier": verifier_type or "explicit",
    }
