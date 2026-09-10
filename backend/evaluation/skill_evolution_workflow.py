"""Persistent, human-controlled stages for offline Skill evolution."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = (
    "created",
    "data_exported",
    "dataset_confirmed",
    "candidate_generated",
    "candidate_confirmed",
    "dev_evaluated",
    "revision_pending",
    "regression_pending",
    "regression_verified",
    "holdout_pending",
    "holdout_verified",
    "approved",
    "published",
    "abandoned",
)

_NEXT: dict[str, set[str]] = {
    "created": {"data_exported", "abandoned"},
    "data_exported": {"dataset_confirmed", "abandoned"},
    "dataset_confirmed": {"candidate_generated", "abandoned"},
    "candidate_generated": {"candidate_confirmed", "abandoned"},
    "candidate_confirmed": {"dev_evaluated", "abandoned"},
    "dev_evaluated": {"revision_pending", "regression_pending", "abandoned"},
    "revision_pending": {"candidate_generated", "abandoned"},
    "regression_pending": {"regression_verified", "revision_pending", "abandoned"},
    "regression_verified": {"holdout_pending", "revision_pending", "abandoned"},
    "holdout_pending": {"holdout_verified", "revision_pending", "abandoned"},
    "holdout_verified": {"approved", "abandoned"},
    "approved": {"published", "abandoned"},
    "published": set(),
    "abandoned": set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SkillEvolutionWorkflow:
    """Small JSON-backed state machine; execution is deliberately separate."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "workflow-tasks.json"
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)

    def list(self, agent_id: str = "main") -> list[dict[str, Any]]:
        with self._lock:
            return [item for item in self._read() if item.get("agent_id") == agent_id]

    def get(self, task_id: str, agent_id: str = "main") -> dict[str, Any] | None:
        return next((item for item in self.list(agent_id) if item["id"] == task_id), None)

    def create(
        self, *, agent_id: str, task_family: str, title: str = "",
        data_source: str = "langfuse",
    ) -> dict[str, Any]:
        if data_source not in {"langfuse", "skilllearnbench"}:
            raise ValueError(f"unsupported evolution data source: {data_source}")
        with self._lock:
            now = _now()
            item = {
                "id": f"evo-{uuid.uuid4().hex[:12]}",
                "agent_id": agent_id,
                "task_family": task_family,
                "data_source": data_source,
                "title": title or f"{task_family} Skill 进化",
                "stage": "created",
                "artifacts": {},
                "execution": {
                    "operation": None, "status": "idle", "started_at": None,
                    "finished_at": None, "result": None, "error": None,
                },
                "history": [{"stage": "created", "at": now, "note": ""}],
                "created_at": now,
                "updated_at": now,
            }
            items = self._read()
            items.append(item)
            self._write(items)
            return item

    def record_execution(
        self, task_id: str, *, agent_id: str, operation: str, status: str,
        result: dict[str, Any] | None = None, error: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"unsupported execution status: {status}")
        with self._lock:
            items = self._read()
            item = next((row for row in items if row["id"] == task_id and row.get("agent_id") == agent_id), None)
            if item is None:
                raise KeyError(task_id)
            now = _now()
            previous = item.get("execution") if isinstance(item.get("execution"), dict) else {}
            item["execution"] = {
                "operation": operation,
                "status": status,
                "started_at": now if status == "running" else previous.get("started_at"),
                "finished_at": None if status == "running" else now,
                "result": result if result is not None else previous.get("result"),
                "error": error if status in {"failed", "cancelled"} else None,
            }
            item["updated_at"] = now
            self._write(items)
            return item

    def transition(
        self, task_id: str, *, agent_id: str, stage: str, note: str = "",
        artifacts: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unsupported workflow stage: {stage}")
        with self._lock:
            items = self._read()
            item = next((row for row in items if row["id"] == task_id and row.get("agent_id") == agent_id), None)
            if item is None:
                raise KeyError(task_id)
            current = str(item["stage"])
            if stage != current and stage not in _NEXT.get(current, set()):
                raise ValueError(f"invalid transition: {current} -> {stage}")
            now = _now()
            item["stage"] = stage
            item["updated_at"] = now
            if artifacts:
                item.setdefault("artifacts", {}).update({str(k): str(v) for k, v in artifacts.items()})
            item.setdefault("history", []).append({"stage": stage, "at": now, "note": note})
            self._write(items)
            return item


def build_workflow(root: Path) -> SkillEvolutionWorkflow:
    return SkillEvolutionWorkflow(root)
