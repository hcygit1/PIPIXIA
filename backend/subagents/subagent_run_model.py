"""Domain model for subagent runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


class SubagentCapacityError(Exception):
    pass


@dataclass
class SubagentRunRecord:
    run_id: str
    child_session_key: str
    requester_session_key: str
    requester_agent_id: str
    target_agent_id: str
    task: str
    label: str | None = None
    model: str | None = None
    parent_task_id: str | None = None
    cleanup: Literal["delete", "keep"] = "keep"
    spawn_depth: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    outcome: str | None = None
    result_summary: str | None = None
    asyncio_task: Any = field(default=None, repr=False)
    archive_at_ms: float | None = None
    announce_retry_count: int = 0
    last_announce_retry_at: float | None = None
    state: str = "running"
    terminal_reason: str | None = None
    result_delivery_state: str = "pending"
    delivery_work_id: str | None = None
