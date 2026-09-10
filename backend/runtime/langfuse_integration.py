"""Optional Langfuse tracing for LangChain/LangGraph agent runs."""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def _business_metadata(request: Any) -> dict[str, Any]:
    """Return JSON-friendly business fields without leaking empty values."""
    raw = getattr(request, "observability_metadata", {})
    metadata = dict(raw) if isinstance(raw, dict) else {}
    for key in (
        "skill_id",
        "skill_name",
        "skill_version",
        "task_type",
        "task_family",
        "task_result",
        "sample_type",
        "failure_type",
        "candidate_version",
        "parent_version",
    ):
        value = getattr(request, key, None)
        if value is not None and value != "":
            metadata[key] = value
    return {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ""
    }


def build_langfuse_config(*, request: Any, run_id: str) -> dict[str, Any]:
    """Build a LangGraph config with optional Langfuse callbacks.

    Observability is deliberately best-effort: missing credentials or an
    unavailable SDK must never prevent the agent turn from running.
    """
    metadata = {
            "pipixia_run_id": run_id,
            "memory_turn_id": run_id,
            "agent_id": str(request.agent_id),
            # Langfuse uses the prefixed key to group traces into Sessions.
            "langfuse_session_id": str(request.session_id),
            "provider": str(request.provider),
            "model": str(request.model),
        }
    metadata.update(_business_metadata(request))
    config: dict[str, Any] = {
        "metadata": metadata,
        "run_name": "pipixia-agent-turn",
        "tags": ["pipixia-agent", f"agent:{request.agent_id}"],
    }

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return config

    try:
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
    except Exception as error:
        logger.warning("Langfuse tracing disabled for this turn: %s", error)
    return config


def flush_langfuse_config(config: dict[str, Any]) -> None:
    """Flush callbacks created for one turn without affecting the turn."""
    for callback in config.get("callbacks", []):
        client = getattr(callback, "client", None)
        flush = getattr(client, "flush", None)
        if not callable(flush):
            continue
        try:
            flush()
        except Exception as error:
            logger.warning("Langfuse flush failed: %s", error)
