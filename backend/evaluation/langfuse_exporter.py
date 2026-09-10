"""Convert Langfuse traces into the versioned Skill-evolution sample format."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+"),
)


def _value(trace: Any, key: str, default: Any = None) -> Any:
    if isinstance(trace, dict):
        return trace.get(key, default)
    return getattr(trace, key, default)


def _redact(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _redact(model_dump(mode="json"))
    if isinstance(value, str):
        result = value
        for pattern in _SECRET_PATTERNS:
            result = pattern.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]" if match.lastindex and match.lastindex >= 2 else "[REDACTED]", result)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    return value


def _metadata(trace: Any) -> dict[str, Any]:
    metadata = _value(trace, "metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _category(metadata: dict[str, Any]) -> str:
    result = str(metadata.get("task_result", "")).lower()
    if result in {"success", "succeeded", "completed", "pass", "passed"}:
        return "success"
    if result in {"boundary", "edge", "ambiguous"}:
        return "boundary"
    return "bad_case" if result else "unclassified"


def _chunk_message(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": str(getattr(chunk, "id", "")),
        "turn_id": str(getattr(chunk, "turn_id", "")),
        "role": str(getattr(chunk, "role", "")),
        "content": getattr(chunk, "content", ""),
    }


def _memory_scope(trace: Any, memory_store: Any) -> dict[str, Any] | None:
    metadata = _metadata(trace)
    session_id = str(metadata.get("langfuse_session_id") or "")
    turn_id = str(
        metadata.get("memory_turn_id") or metadata.get("pipixia_run_id") or ""
    )
    if not session_id or not turn_id:
        return None
    turn_chunks = memory_store.get_chunks_by_turn(session_id, turn_id)
    if not turn_chunks:
        return None
    task_id = next((chunk.task_id for chunk in turn_chunks if chunk.task_id), None)
    chunks = memory_store.get_chunks_by_task(task_id) if task_id else turn_chunks
    task = memory_store.get_task(task_id) if task_id else None
    messages = [_chunk_message(chunk) for chunk in chunks]
    return {
        "task_id": task_id,
        "skill_id": next((chunk.skill_id for chunk in chunks if chunk.skill_id), None),
        "task_snapshot": {
            "task_id": task_id,
            "title": getattr(task, "title", "") if task else "",
            "summary": getattr(task, "summary", "") if task else "",
            "status": getattr(task, "status", "unassigned") if task else "unassigned",
            "messages": messages,
        },
        "trajectory": messages,
    }


def trace_to_sample(trace: Any, memory_store: Any = None) -> dict[str, Any]:
    """Map one Langfuse trace to the existing evolution sample contract."""
    metadata = _metadata(trace)
    trace_id = str(_value(trace, "id", ""))
    task_snapshot = _value(trace, "input", _value(trace, "observations", ""))
    result = _value(trace, "output", _value(trace, "result", None))
    source = trace_id or json.dumps([task_snapshot, result], ensure_ascii=False, sort_keys=True)
    sample_id = "lf-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    skill_names = metadata.get("skill_names") or []
    skill_name = metadata.get("skill_name")
    if not skill_name and isinstance(skill_names, list) and len(skill_names) == 1:
        skill_name = skill_names[0]
    versions = metadata.get("skill_versions") or {}
    skill_version = metadata.get("skill_version")
    if not skill_version and skill_name and isinstance(versions, dict):
        skill_version = versions.get(skill_name)
    sample = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "task_family": metadata.get("task_family") or metadata.get("family"),
        "split": metadata.get("split"),
        "task_id": metadata.get("task_id"),
        "skill_id": metadata.get("skill_id"),
        "skill_version": skill_version,
        "category": _category(metadata),
        "failure_reason": metadata.get("failure_type") or metadata.get("failure_reason"),
        "task_snapshot": task_snapshot,
        "trace_path": metadata.get("trace_path") or trace_id,
        "result_path": metadata.get("result_path"),
        "verifier_result": metadata.get("verifier_result"),
        "user_feedback": metadata.get("user_feedback"),
        "created_at": _value(trace, "timestamp", _value(trace, "created_at")),
        "trajectory": _value(trace, "trajectory", _value(trace, "observations", [])),
    }
    if memory_store is not None:
        memory_scope = _memory_scope(trace, memory_store)
        if memory_scope:
            for key, value in memory_scope.items():
                if value is not None:
                    sample[key] = value
    return _redact(sample)


def export_traces(
    traces: Iterable[Any], output_path: Path, *, memory_store: Any = None
) -> int:
    """Write converted traces as UTF-8 JSONL, skipping duplicate sample IDs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            sample = trace_to_sample(trace, memory_store=memory_store)
            if sample["sample_id"] in seen:
                continue
            seen.add(sample["sample_id"])
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    return count


def fetch_langfuse_traces(
    client: Any,
    *,
    limit: int = 100,
    max_pages: int = 1,
    **filters: Any,
) -> list[Any]:
    """Fetch traces through either Langfuse SDK v2/v3 client shape."""
    api = getattr(client, "api", client)
    trace_api = getattr(api, "trace", api)
    list_traces = getattr(trace_api, "list", None)
    if not callable(list_traces):
        raise TypeError("Langfuse client does not expose trace.list()")

    traces: list[Any] = []
    page = 1
    while page <= max_pages:
        params = {"limit": limit, "page": page, **filters}
        response = list_traces(**params)
        rows = getattr(response, "data", response)
        if isinstance(rows, dict):
            rows = rows.get("data", [])
        if not isinstance(rows, (list, tuple)):
            rows = list(rows or [])
        traces.extend(rows)
        if len(rows) < limit:
            break
        page += 1
    return traces


def export_langfuse_traces(
    client: Any,
    output_path: Path,
    *,
    limit: int = 100,
    max_pages: int = 1,
    memory_store: Any = None,
    **filters: Any,
) -> int:
    """Fetch and export Langfuse traces in one deterministic operation."""
    traces = fetch_langfuse_traces(
        client, limit=limit, max_pages=max_pages, **filters
    )
    return export_traces(traces, output_path, memory_store=memory_store)


def fetch_langfuse_traces_by_ids(client: Any, trace_ids: Iterable[str]) -> list[Any]:
    """Fetch an explicit user selection; never broadens it to nearby traces."""
    api = getattr(client, "api", client)
    trace_api = getattr(api, "trace", api)
    get_trace = getattr(trace_api, "get", None)
    if not callable(get_trace):
        raise TypeError("Langfuse client does not expose trace.get()")
    result = []
    seen: set[str] = set()
    for raw_id in trace_ids:
        trace_id = str(raw_id).strip()
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        result.append(get_trace(trace_id))
    return result


def trace_preview(trace: Any) -> dict[str, Any]:
    """Return a compact, redacted row suitable for manual selection."""
    sample = trace_to_sample(trace)
    metadata = _metadata(trace)
    task_snapshot = sample.get("task_snapshot")
    if isinstance(task_snapshot, dict):
        title = str(task_snapshot.get("title") or metadata.get("task_title") or "")
        preview_source = task_snapshot.get("summary") or task_snapshot.get("messages") or task_snapshot
    else:
        title = str(metadata.get("task_title") or metadata.get("title") or "")
        preview_source = task_snapshot
    output = _redact(_value(trace, "output", _value(trace, "result", None)))
    return {
        "trace_id": str(_value(trace, "id", "")),
        "sample_id": sample["sample_id"],
        "title": title,
        "task_family": sample.get("task_family"),
        "category": sample.get("category"),
        "session_id": str(metadata.get("langfuse_session_id") or _value(trace, "session_id", "") or ""),
        "created_at": sample.get("created_at"),
        "input_preview": json.dumps(preview_source, ensure_ascii=False)[:500],
        "output_preview": json.dumps(output, ensure_ascii=False)[:500],
        "skill_names": metadata.get("skill_names") or [],
    }
