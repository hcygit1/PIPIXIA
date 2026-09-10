"""Data models shared by Agent turn runtime services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnExecutionRequest:
    agent_id: str
    session_id: str
    state: Any
    provider: str
    model: str
    message: str
    persist_input_role: str
    system_prompt: str
    tools: list[Any]
    history: list[dict[str, Any]]
    recursion_limit: int
    prompt_tokens: int
    summary_tokens: int
    history_tokens: int
    active_tokens: int
    observability_metadata: dict[str, Any] = field(default_factory=dict)
    parent_task_id: str | None = None
    evaluation_mode: bool = False
