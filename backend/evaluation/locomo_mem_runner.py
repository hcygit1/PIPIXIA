"""Run LoCoMo against PIPIXIA's real MemStore and MemRecall."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from evaluation.locomo_adapter import AdaptedConversation, load_pilot, load_source
from evaluation.locomo_runner import evaluate_case
from infra.token_counter import count_tokens
from mem.embedder import MemEmbedder
from mem.recall import MemRecall, expand_query, rrf_fuse
from mem.store import MemStore


TokenCounter = Callable[[str], int]

STRUCTURED_SUMMARY_PROMPT = """You create a reusable structured memory summary from one multi-turn conversation.
The summary will be used for semantic and keyword retrieval. Preserve concrete facts exactly.
Do not infer or invent facts. Keep names, dates, places, organizations, numbers, decisions,
events, outcomes, and next steps when present. Use the same language as the conversation.

Return JSON only, with exactly these fields:
{
  "title": "short descriptive title",
  "context": "one sentence describing the conversation context",
  "goal": "the user's goal, or an empty string",
  "participants": ["people or organizations"],
  "facts": ["concrete facts and events"],
  "steps": ["important actions or steps"],
  "decisions": ["decisions or conclusions"],
  "outcome": "result or current state, or an empty string",
  "next_steps": ["planned follow-up actions"],
  "insights": ["useful lessons or constraints"],
  "keywords": ["important names, dates, places, topics, and exact terms"]
}

Rules:
- Use empty arrays/strings when a field does not apply.
- Prefer several short factual items over one vague paragraph.
- Preserve original spellings of names and proper nouns.
- Output no Markdown and no explanatory text."""


def _summary_index_text(summary: dict[str, Any]) -> str:
    labels = {
        "title": "标题", "context": "背景", "goal": "目标", "participants": "参与者",
        "facts": "事实与事件", "steps": "步骤", "decisions": "决定", "outcome": "结果",
        "next_steps": "后续计划", "insights": "经验", "keywords": "关键词",
    }
    parts: list[str] = []
    for key, label in labels.items():
        value = summary.get(key, "")
        if isinstance(value, list):
            value = "；".join(str(item).strip() for item in value if str(item).strip())
        value = str(value).strip()
        if value:
            parts.append(f"{label}：{value}")
    return "\n".join(parts)


def _parse_summary_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("structured summary response does not contain JSON")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("structured summary must be a JSON object")
    return parsed


async def _generate_structured_summaries(
    source: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    cache_path: Path,
) -> tuple[dict[str, str], dict[str, int]]:
    sample_id = str(source.get("sample_id") or source.get("id") or "locomo-sample")
    source_hash = hashlib.sha256(json.dumps(source.get("conversation") or {}, sort_keys=True).encode()).hexdigest()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("sample_id") == sample_id and cached.get("source_hash") == source_hash:
            return cached.get("summaries", {}), cached.get("usage", {})

    if not base_url or not api_key:
        raise ValueError("structured summary requires --summary-base-url and --summary-api-key")
    base_url = base_url.rstrip("/")
    sessions = sorted(
        ((key, value) for key, value in (source.get("conversation") or {}).items()
         if key.startswith("session_") and isinstance(value, list)),
        key=lambda item: int(item[0].split("_")[-1]),
    )
    summaries: dict[str, str] = {}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "sessions": 0}
    async with httpx.AsyncClient(timeout=120.0) as client:
        for session_key, turns in sessions:
            transcript = "\n".join(
                f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}" for turn in turns
            )
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "temperature": 0, "max_tokens": 1200,
                      "messages": [{"role": "system", "content": STRUCTURED_SUMMARY_PROMPT},
                                   {"role": "user", "content": transcript}]},
            )
            response.raise_for_status()
            payload = response.json()
            parsed = _parse_summary_json(payload["choices"][0]["message"]["content"])
            summaries[f"{session_key}_summary"] = _summary_index_text(parsed)
            usage = payload.get("usage") or {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_totals[key] += int(usage.get(key) or 0)
            usage_totals["sessions"] += 1
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"sample_id": sample_id, "source_hash": source_hash,
                                      "model": model, "summaries": summaries, "usage": usage_totals},
                                     ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summaries, usage_totals


def build_token_usage(
    *,
    index_texts: list[str],
    queries: list[str],
    contexts: list[str],
    count_tokens: TokenCounter,
) -> dict[str, int]:
    index_tokens = sum(count_tokens(text) for text in index_texts)
    query_tokens = sum(count_tokens(text) for text in queries)
    context_tokens = sum(count_tokens(text) for text in contexts)
    return {
        "index_tokens": index_tokens,
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "estimated_llm_input_tokens": query_tokens + context_tokens,
        "llm_output_tokens": 0,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_runs(
    *,
    system: str,
    runs: list[dict[str, Any]],
    index_tokens: int,
) -> dict[str, Any]:
    count = max(1, len(runs))
    avg_query = sum(item["query_tokens"] for item in runs) / count
    avg_context = sum(item["context_tokens"] for item in runs) / count
    return {
        "system": system,
        "total_cases": len(runs),
        "hit_rate_at_k": sum(item["hit_rate_at_k"] for item in runs) / count,
        "evidence_recall_at_k": sum(item["evidence_recall_at_k"] for item in runs) / count,
        "mrr": sum(item["mrr"] for item in runs) / count,
        "task_hit_rate_at_k": sum(item["task_hit_rate_at_k"] for item in runs) / count,
        "task_evidence_recall_at_k": sum(item["task_evidence_recall_at_k"] for item in runs) / count,
        "context_evidence_coverage": sum(item["context_evidence_coverage"] for item in runs) / count,
        "avg_latency_ms": sum(item["latency_ms"] for item in runs) / count,
        "p95_latency_ms": _percentile(
            [item["latency_ms"] for item in runs], 0.95
        ),
        "index_tokens": index_tokens,
        "avg_query_tokens": avg_query,
        "avg_context_tokens": avg_context,
        "avg_estimated_llm_input_tokens": avg_query + avg_context,
        "total_query_tokens": sum(item["query_tokens"] for item in runs),
        "total_context_tokens": sum(item["context_tokens"] for item in runs),
        "total_estimated_llm_input_tokens": sum(
            item["query_tokens"] + item["context_tokens"] for item in runs
        ),
        "llm_output_tokens": 0,
    }


async def _build_index(
    adapted: AdaptedConversation,
    *,
    db_path: Path,
    model: str,
    device: str,
) -> tuple[MemStore, MemEmbedder, float]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    embedder = MemEmbedder(
        provider="local",
        model=model,
    )
    embedder._local_model = embedder._local_model.to(device)
    store = MemStore(str(db_path), dimensions=embedder.dimensions)

    started = time.perf_counter()
    for task in adapted.tasks:
        store.insert_task(task)
    for chunk in adapted.chunks:
        store.insert_chunk(chunk)

    task_texts = [f"{task.title}\n{task.summary}" for task in adapted.tasks]
    chunk_texts = [f"{chunk.summary}\n{chunk.content}" for chunk in adapted.chunks]
    task_vectors = await embedder.embed(task_texts)
    chunk_vectors = await embedder.embed(chunk_texts)
    for task, vector in zip(adapted.tasks, task_vectors):
        store.upsert_task_embedding(task.id, vector)
    for chunk, vector in zip(adapted.chunks, chunk_vectors):
        store.upsert_chunk_embedding(chunk.id, vector)
    return store, embedder, (time.perf_counter() - started) * 1000


async def _direct_retrieve(
    store: MemStore,
    query: str,
    query_vector: list[float],
    top_k: int,
    reranker: Any | None = None,
) -> tuple[list[str], str]:
    pool_size = max(30, top_k * 6)
    fts_hits = store.fts_search_chunks(query, limit=pool_size)
    ann_hits = store.ann_search_chunks(query_vector, top_k=pool_size)
    fused = rrf_fuse([
        [(hit.chunk_id, hit.score) for hit in fts_hits],
        [(hit.chunk_id, hit.score) for hit in ann_hits],
    ])
    candidate_ids = [
        chunk_id for chunk_id, _ in sorted(fused.items(), key=lambda item: -item[1])
    ][:top_k * 5]
    ranked_ids = candidate_ids
    if reranker and candidate_ids:
        rerank_inputs: list[tuple[str, str]] = []
        for chunk_id in candidate_ids:
            chunk = store.get_chunk(chunk_id)
            if chunk:
                rerank_inputs.append((chunk_id, _merge_context_text(
                    chunk.summary, chunk.content[:300]
                )))
        scores = await asyncio.to_thread(reranker.score, query, rerank_inputs)
        ranked_ids = [
            chunk_id
            for (chunk_id, _), _score in sorted(
                zip(rerank_inputs, scores), key=lambda item: -item[1]
            )
        ]
    ranked_ids = ranked_ids[:top_k]
    context_parts: list[str] = []
    for chunk_id in ranked_ids:
        chunk = store.get_chunk(chunk_id)
        if chunk:
            context_parts.append(_merge_context_text(
                chunk.summary,
                chunk.content[:300],
            ))
    return ranked_ids, "\n".join(context_parts)


def _merge_context_text(summary: str, excerpt: str) -> str:
    """Keep summary and excerpt once when one is a prefix of the other."""
    summary = (summary or "").strip()
    excerpt = (excerpt or "").strip()
    if not summary:
        return excerpt
    if not excerpt:
        return summary
    if summary.startswith(excerpt):
        return summary
    if excerpt.startswith(summary):
        return excerpt
    return f"{summary}\n{excerpt}"


def _waterfall_context(result: Any) -> tuple[list[str], list[str], str]:
    retrieved_ids = [hit.chunk_id for hit in result.ranked_hits]
    retrieved_task_ids = list(result.retrieved_task_ids)
    context_parts: list[str] = []
    for hit in result.ranked_hits:
        context_parts.append(_merge_context_text(hit.summary, hit.content_excerpt))
    return retrieved_ids, retrieved_task_ids, "\n".join(part for part in context_parts if part)


def _id_metrics(
    retrieved_ids: list[str],
    golden_ids: list[str],
    *,
    k: int,
) -> tuple[float, float]:
    if not golden_ids:
        return 0.0, 0.0
    matched = set(retrieved_ids[:k]).intersection(golden_ids)
    return (1.0 if matched else 0.0, len(matched) / len(set(golden_ids)))


def _id_mrr(retrieved_ids: list[str], golden_ids: list[str], *, k: int) -> float:
    golden = set(golden_ids)
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in golden:
            return 1.0 / rank
    return 0.0


async def _task_route_rankings(
    store: MemStore,
    embedder: MemEmbedder,
    query: str,
    query_vector: list[float],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> dict[str, list[str]]:
    """Rank Tasks independently through FTS5, Dense, and their RRF fusion."""
    pool_size = max(top_k * 5, top_k)
    fts_hits = store.fts_search_tasks(query, limit=pool_size, owner="eval:locomo")
    fts_ranked = [(hit.task_id, hit.score) for hit in fts_hits]

    vector_scores: dict[str, float] = {}
    for sub_query in expand_query(query):
        vector = query_vector if sub_query == query else await embedder.embed_query(sub_query)
        for hit in store.ann_search_tasks(vector, top_k=pool_size, owner="eval:locomo"):
            vector_scores[hit.task_id] = max(vector_scores.get(hit.task_id, 0.0), hit.score)
    dense_ranked = sorted(vector_scores.items(), key=lambda item: -item[1])
    fused = rrf_fuse([fts_ranked, dense_ranked], k=rrf_k)
    rrf_ranked = sorted(fused.items(), key=lambda item: -item[1])
    return {
        "fts5": [task_id for task_id, _ in fts_ranked[:top_k]],
        "dense": [task_id for task_id, _ in dense_ranked[:top_k]],
        "rrf": [task_id for task_id, _ in rrf_ranked[:top_k]],
    }


def _summarize_task_routes(
    route_runs: dict[str, list[dict[str, float]]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for route in ("fts5", "dense", "rrf"):
        runs = route_runs[route]
        count = max(1, len(runs))
        summaries.append({
            "route": route,
            "top_k": top_k,
            "total_cases": len(runs),
            "task_hit_rate_at_k": sum(run["hit"] for run in runs) / count,
            "task_evidence_recall_at_k": sum(run["recall"] for run in runs) / count,
            "task_mrr": sum(run["mrr"] for run in runs) / count,
        })
    return summaries


def _chunk_route_ids(
    store: MemStore,
    task_ids: list[str],
    query: str,
    sub_queries: list[str],
    query_vectors: dict[str, list[float]],
    *,
    chunks_per_task: int,
    final_top_k: int,
    rrf_k: int = 60,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Rank every Chunk in the retrieved Task scope and expose scope audits."""
    if not task_ids:
        empty_routes = {
            route: {"full_ids": [], "candidate_ids": [], "final_ids": []}
            for route in ("fts5", "dense", "rrf")
        }
        return empty_routes, {
            "scoped_chunk_count": 0,
            "dense_ranked_chunk_count": 0,
            "dense_missing_chunk_count": 0,
            "dense_out_of_scope_count": 0,
            "scope_complete": True,
        }
    candidate_limit = len(task_ids) * chunks_per_task
    scoped_ids = {
        chunk.id
        for task_id in task_ids
        for chunk in store.get_chunks_by_task(task_id)
        if chunk.owner == "eval:locomo" and chunk.dedup_status == "active"
    }
    fts_hits = store.fts_search_chunks_in_tasks(
        query, task_ids, limit=None, owner="eval:locomo"
    )
    fts_ranked = [(hit.chunk_id, hit.score) for hit in fts_hits]

    vector_scores: dict[str, float] = {}
    for sub_query in sub_queries:
        vector = query_vectors.get(sub_query)
        if not vector:
            continue
        for hit in store.exact_search_chunks_in_tasks(
            vector, task_ids, top_k=None, owner="eval:locomo"
        ):
            vector_scores[hit.chunk_id] = max(
                vector_scores.get(hit.chunk_id, 0.0), hit.score
            )
    dense_ranked = sorted(vector_scores.items(), key=lambda item: -item[1])
    rrf_ranked = sorted(
        rrf_fuse([fts_ranked, dense_ranked], k=rrf_k).items(),
        key=lambda item: -item[1],
    )

    route_ranked = {
        "fts5": [chunk_id for chunk_id, _ in fts_ranked],
        "dense": [chunk_id for chunk_id, _ in dense_ranked],
        "rrf": [chunk_id for chunk_id, _ in rrf_ranked],
    }
    dense_ids = set(route_ranked["dense"])
    dense_missing_ids = scoped_ids - dense_ids
    dense_out_of_scope_ids = dense_ids - scoped_ids
    routes = {
        route: {
            "full_ids": ids,
            "candidate_ids": ids[:candidate_limit],
            "final_ids": ids[:final_top_k],
        }
        for route, ids in route_ranked.items()
    }
    audit = {
        "scoped_chunk_count": len(scoped_ids),
        "dense_ranked_chunk_count": len(dense_ids),
        "dense_missing_chunk_count": len(dense_missing_ids),
        "dense_out_of_scope_count": len(dense_out_of_scope_ids),
        "scope_complete": not dense_missing_ids and not dense_out_of_scope_ids,
    }
    return routes, audit


def _summarize_chunk_routes(
    route_runs: dict[str, list[dict[str, float]]],
    *,
    top_k: int,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for route in ("fts5", "dense", "rrf"):
        runs = route_runs[route]
        count = max(1, len(runs))
        summaries.append({
            "route": route,
            "candidate_top_k": candidate_limit,
            "final_top_k": top_k,
            "total_cases": len(runs),
            "avg_full_ranked_chunks": sum(
                run["full_ranked_count"] for run in runs
            ) / count,
            "full_range_evidence_recall": sum(
                run["full_recall"] for run in runs
            ) / count,
            "candidate_evidence_recall_at_k": sum(
                run["candidate_recall"] for run in runs
            ) / count,
            "final_evidence_recall_at_k": sum(
                run["final_recall"] for run in runs
            ) / count,
            "final_hit_rate_at_k": sum(
                run["final_hit"] for run in runs
            ) / count,
            "reachable_evidence_rate": sum(
                run["reachable_rate"] for run in runs
            ) / count,
        })
    return summaries


def _summarize_chunk_scope_audits(
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    count = max(1, len(audits))
    scoped_counts = [audit["scoped_chunk_count"] for audit in audits]
    return {
        "total_cases": len(audits),
        "all_cases_scope_complete": all(
            audit["scope_complete"] for audit in audits
        ),
        "avg_scoped_chunk_count": sum(scoped_counts) / count,
        "min_scoped_chunk_count": min(scoped_counts, default=0),
        "max_scoped_chunk_count": max(scoped_counts, default=0),
        "total_dense_missing_chunks": sum(
            audit["dense_missing_chunk_count"] for audit in audits
        ),
        "total_dense_out_of_scope_chunks": sum(
            audit["dense_out_of_scope_count"] for audit in audits
        ),
    }


def _context_evidence_coverage(
    injected_chunk_ids: list[str],
    golden_chunk_ids: list[str],
) -> float:
    if not golden_chunk_ids:
        return 0.0
    matched = set(injected_chunk_ids).intersection(golden_chunk_ids)
    return len(matched) / len(set(golden_chunk_ids))


async def run_real_pilot(
    data_path: str | Path,
    *,
    db_path: str | Path,
    sample_index: int = 0,
    max_questions: int = 20,
    top_k: int = 5,
    model: str = "BAAI/bge-m3",
    device: str = "cpu",
    seed: int = 42,
    structured_summary: bool = False,
    summary_base_url: str = "",
    summary_api_key: str = "",
    summary_model: str = "gpt-4o-mini",
    summary_cache: str | Path | None = None,
) -> dict[str, Any]:
    summary_usage: dict[str, int] = {}
    session_summaries: dict[str, str] | None = None
    source = load_source(data_path, sample_index=sample_index)
    if structured_summary:
        cache_path = Path(summary_cache or Path(data_path).with_name(
            f"{Path(data_path).stem}-structured-summaries-{sample_index}.json"
        ))
        session_summaries, summary_usage = await _generate_structured_summaries(
            source,
            base_url=summary_base_url,
            api_key=summary_api_key,
            model=summary_model,
            cache_path=cache_path,
        )
    adapted = load_pilot(
        data_path,
        sample_index=sample_index,
        max_questions=max_questions,
        include_abstention=False,
        seed=seed,
        session_summaries=session_summaries,
    )
    store, embedder, index_latency_ms = await _build_index(
        adapted,
        db_path=Path(db_path),
        model=model,
        device=device,
    )
    recall = MemRecall(
        store=store,
        embedder=embedder,
        config={
            "recall": {
                "max_task_results": 5,
                "min_task_hits": 3,
                "chunks_per_task": top_k,
                "max_orphan_chunks": top_k,
                "final_chunk_top_k": top_k,
                "rerank_enabled": True,
                "rerank_model": "BAAI/bge-reranker-v2-m3",
                "rerank_device": "cpu",
                "budget_chars": 20000,
                "min_task_score": 0.0,
                "rrf_k": 60,
                "recency_half_life_days": 14.0,
                "min_inject_score": 0.0,
            }
        },
    )

    direct_runs: list[dict[str, Any]] = []
    waterfall_runs: list[dict[str, Any]] = []
    task_route_runs: dict[str, list[dict[str, float]]] = {
        "fts5": [],
        "dense": [],
        "rrf": [],
    }
    chunk_route_runs: dict[str, list[dict[str, float]]] = {
        "fts5": [],
        "dense": [],
        "rrf": [],
    }
    chunk_scope_audits: list[dict[str, Any]] = []
    try:
        for case in adapted.cases:
            query_token_count = count_tokens(case.query)

            started = time.perf_counter()
            query_vector = await embedder.embed_query(case.query)
            direct_ids, direct_context = await _direct_retrieve(
                store, case.query, query_vector, top_k, recall.reranker
            )
            direct_latency = (time.perf_counter() - started) * 1000
            task_routes = await _task_route_rankings(
                store,
                embedder,
                case.query,
                query_vector,
                top_k=top_k,
            )
            for route, retrieved_task_ids in task_routes.items():
                task_hit, task_recall = _id_metrics(
                    retrieved_task_ids,
                    case.golden_task_ids,
                    k=top_k,
                )
                task_route_runs[route].append({
                    "hit": task_hit,
                    "recall": task_recall,
                    "mrr": _id_mrr(
                        retrieved_task_ids,
                        case.golden_task_ids,
                        k=top_k,
                    ),
                })
            direct_metrics = evaluate_case(
                retrieved_ids=direct_ids,
                golden_ids=case.golden_chunk_ids,
                answer=case.answer,
                context=direct_context,
                k=top_k,
            )
            direct_task_ids = list(dict.fromkeys(
                chunk.task_id
                for chunk_id in direct_ids
                if (chunk := store.get_chunk(chunk_id)) and chunk.task_id
            ))
            direct_task_hit, direct_task_recall = _id_metrics(
                direct_task_ids, case.golden_task_ids, k=top_k
            )
            direct_runs.append({
                "case_id": case.case_id,
                "category": case.category,
                "gold_chunk_ids": case.golden_chunk_ids,
                "retrieved_chunk_ids": direct_ids,
                "gold_task_ids": case.golden_task_ids,
                "retrieved_task_ids": direct_task_ids,
                "hit_rate_at_k": direct_metrics.hit_rate_at_k,
                "evidence_recall_at_k": direct_metrics.evidence_recall_at_k,
                "mrr": direct_metrics.mrr,
                "task_hit_rate_at_k": direct_task_hit,
                "task_evidence_recall_at_k": direct_task_recall,
                "context_evidence_coverage": _context_evidence_coverage(
                    direct_ids, case.golden_chunk_ids
                ),
                "latency_ms": direct_latency,
                "query_tokens": query_token_count,
                "context_tokens": count_tokens(direct_context),
            })

            started = time.perf_counter()
            result = await recall.search(
                case.query,
                owner="eval:locomo",
                max_results=top_k,
            )
            waterfall_latency = (time.perf_counter() - started) * 1000
            waterfall_ids, waterfall_task_ids, waterfall_context = _waterfall_context(result)
            chunk_routes, chunk_scope_audit = _chunk_route_ids(
                store,
                waterfall_task_ids,
                case.query,
                expand_query(case.query),
                {
                    sub_query: (
                        query_vector
                        if sub_query == case.query
                        else await embedder.embed_query(sub_query)
                    )
                    for sub_query in expand_query(case.query)
                },
                chunks_per_task=top_k,
                final_top_k=top_k,
            )
            chunk_scope_audits.append(chunk_scope_audit)
            reachable_ids = {
                chunk.id for chunk in adapted.chunks
                if chunk.task_id in set(waterfall_task_ids)
                and chunk.id in set(case.golden_chunk_ids)
            }
            reachable_rate = (
                len(reachable_ids) / len(set(case.golden_chunk_ids))
                if case.golden_chunk_ids else 0.0
            )
            for route, route_ids in chunk_routes.items():
                full_ids = route_ids["full_ids"]
                candidate_ids = route_ids["candidate_ids"]
                final_ids = route_ids["final_ids"]
                full_matched = set(full_ids).intersection(reachable_ids)
                candidate_matched = set(candidate_ids).intersection(reachable_ids)
                final_matched = set(final_ids).intersection(case.golden_chunk_ids)
                chunk_route_runs[route].append({
                    "full_ranked_count": float(len(full_ids)),
                    "full_recall": (
                        len(full_matched) / len(reachable_ids)
                        if reachable_ids else 0.0
                    ),
                    "candidate_recall": (
                        len(candidate_matched) / len(reachable_ids)
                        if reachable_ids else 0.0
                    ),
                    "final_recall": (
                        len(final_matched) / len(set(case.golden_chunk_ids))
                        if case.golden_chunk_ids else 0.0
                    ),
                    "final_hit": 1.0 if final_matched else 0.0,
                    "reachable_rate": reachable_rate,
                })
            waterfall_metrics = evaluate_case(
                retrieved_ids=waterfall_ids,
                golden_ids=case.golden_chunk_ids,
                answer=case.answer,
                context=waterfall_context,
                k=top_k,
            )
            waterfall_task_hit, waterfall_task_recall = _id_metrics(
                waterfall_task_ids, case.golden_task_ids, k=top_k
            )
            waterfall_runs.append({
                "case_id": case.case_id,
                "category": case.category,
                "gold_chunk_ids": case.golden_chunk_ids,
                "retrieved_chunk_ids": waterfall_ids,
                "gold_task_ids": case.golden_task_ids,
                "retrieved_task_ids": waterfall_task_ids,
                "hit_rate_at_k": waterfall_metrics.hit_rate_at_k,
                "evidence_recall_at_k": waterfall_metrics.evidence_recall_at_k,
                "mrr": waterfall_metrics.mrr,
                "task_hit_rate_at_k": waterfall_task_hit,
                "task_evidence_recall_at_k": waterfall_task_recall,
                "context_evidence_coverage": _context_evidence_coverage(
                    waterfall_ids, case.golden_chunk_ids
                ),
                "expanded_task_count": result.expanded_task_count,
                "candidate_chunk_count": result.candidate_chunk_count,
                "orphan_search_triggered": result.orphan_search_triggered,
                "latency_ms": waterfall_latency,
                "query_tokens": query_token_count,
                "context_tokens": count_tokens(waterfall_context),
            })
    finally:
        store.close()

    chunk_index_texts = [f"{chunk.summary}\n{chunk.content}" for chunk in adapted.chunks]
    task_index_texts = [f"{task.title}\n{task.summary}" for task in adapted.tasks]
    direct_index_tokens = sum(count_tokens(text) for text in chunk_index_texts)
    waterfall_index_tokens = direct_index_tokens + sum(
        count_tokens(text) for text in task_index_texts
    )
    return {
        "source": str(data_path),
        "sample_id": adapted.sample_id,
        "model": model,
        "embedding_dimensions": embedder.dimensions,
        "device": device,
        "sampling_seed": seed,
        "task_count": len(adapted.tasks),
        "chunk_count": len(adapted.chunks),
        "case_count": len(adapted.cases),
        "top_k": top_k,
        "rerank": {
            "enabled_for": ["direct", "waterfall"],
            "model": "BAAI/bge-reranker-v2-m3",
            "candidate_top_k": top_k * 5,
            "final_top_k": top_k,
        },
        "summary_mode": "structured_llm" if structured_summary else "dataset_summary",
        "summary_model": summary_model if structured_summary else None,
        "summary_usage": summary_usage,
        "task_retrieval_routes": _summarize_task_routes(
            task_route_runs,
            top_k=top_k,
        ),
        "chunk_retrieval_routes": _summarize_chunk_routes(
            chunk_route_runs,
            top_k=top_k,
            candidate_limit=top_k * 5,
        ),
        "chunk_scope_audit": _summarize_chunk_scope_audits(
            chunk_scope_audits
        ),
        "index_latency_ms": index_latency_ms,
        "token_scope": {
            "index_tokens": "one-time embedding/index text",
            "query_tokens": "retrieval query text",
            "context_tokens": "retrieved text injected into the Agent",
            "llm_output_tokens": "0 because this run does not call an answer model",
        },
        "systems": [
            summarize_runs(system="direct", runs=direct_runs, index_tokens=direct_index_tokens),
            summarize_runs(system="waterfall", runs=waterfall_runs, index_tokens=waterfall_index_tokens),
        ],
        "cases": {"direct": direct_runs, "waterfall": waterfall_runs},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoCoMo with PIPIXIA MemRecall")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--structured-summary", action="store_true", help="Generate structured Task summaries with an LLM")
    parser.add_argument("--summary-base-url", default=os.getenv("PIPIXIA_LLM_BASE_URL", ""))
    parser.add_argument("--summary-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--summary-model", default=os.getenv("PIPIXIA_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--summary-cache", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = asyncio.run(run_real_pilot(
        args.data,
        db_path=args.db,
        sample_index=args.sample_index,
        max_questions=0 if args.all_questions else args.max_questions,
        top_k=args.top_k,
        model=args.model,
        device=args.device,
        seed=args.seed,
        structured_summary=args.structured_summary,
        summary_base_url=args.summary_base_url,
        summary_api_key=args.summary_api_key,
        summary_model=args.summary_model,
        summary_cache=args.summary_cache,
    ))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
