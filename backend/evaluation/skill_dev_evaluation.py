"""Development-set comparison for offline Skill evolution.

This module owns the *evaluation contract*, not a particular agent runtime.
The caller supplies one executor that can replay a sample with a selected
Skill configuration.  Keeping that seam here lets the same Dev gate work for
PIPIXIA replays and benchmark adapters, while producing one stable report
format for the release gate.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from evaluation.skill_candidate import CandidateRecord, check_candidate, validate_candidate_for_evaluation
from evaluation.skill_evolution_dataset import validate_manifest


VariantName = str
CaseExecutor = Callable[
    [dict[str, Any], VariantName, Path | None],
    Mapping[str, Any] | bool | Awaitable[Mapping[str, Any] | bool],
]

WITHOUT_SKILL = "without_skill"
ACTIVE_SKILL = "active_skill"
CANDIDATE_SKILL = "candidate_skill"


def _family_splits(manifest: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    for entry in manifest.get("families", []):
        if isinstance(entry, Mapping) and entry.get("task_family") == family:
            splits = entry.get("splits")
            if isinstance(splits, Mapping):
                return splits
    raise ValueError(f"task family not found in dataset manifest: {family}")


async def _execute_case(
    executor: CaseExecutor,
    sample: dict[str, Any],
    variant: VariantName,
    skill_path: Path | None,
) -> dict[str, Any]:
    """Execute and normalize one replay without treating task failure as infra failure."""
    started = time.perf_counter()
    try:
        raw = executor(sample, variant, skill_path)
        if inspect.isawaitable(raw):
            raw = await raw
    except Exception as exc:  # The report must preserve failures for the release gate.
        return {
            "sample_id": str(sample["sample_id"]),
            "variant": variant,
            "passed": None,
            "external_failure": True,
            "failure_type": "executor_error",
            "error": f"{type(exc).__name__}: {exc}",
            "tokens": 0.0,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    if isinstance(raw, bool):
        raw = {"passed": raw}
    if not isinstance(raw, Mapping) or not isinstance(raw.get("passed"), bool):
        return {
            "sample_id": str(sample["sample_id"]),
            "variant": variant,
            "passed": None,
            "external_failure": True,
            "failure_type": "invalid_executor_result",
            "error": "executor must return a bool or a mapping with boolean 'passed'",
            "tokens": 0.0,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    tokens = raw.get("tokens", raw.get("total_tokens", 0))
    duration_ms = raw.get("duration_ms", (time.perf_counter() - started) * 1000)
    try:
        tokens = float(tokens or 0)
    except (TypeError, ValueError):
        tokens = 0.0
    try:
        duration_ms = float(duration_ms or 0)
    except (TypeError, ValueError):
        duration_ms = 0.0
    external_failure = bool(raw.get("external_failure", False))
    result = {
        "sample_id": str(sample["sample_id"]),
        "variant": variant,
        "passed": raw["passed"] if not external_failure else None,
        "external_failure": external_failure,
        "failure_type": str(raw.get("failure_type", "")) or None,
        "error": str(raw.get("error", "")) or None,
        "tokens": tokens,
        "duration_ms": round(duration_ms, 3),
    }
    if raw.get("output") is not None:
        result["output"] = str(raw["output"])[:8000]
    if raw.get("verifier") is not None:
        result["verifier"] = str(raw["verifier"])
    return result


def _summarize(variant: VariantName, rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if isinstance(row["passed"], bool)]
    passed = sum(row["passed"] is True for row in evaluated)
    tokens = sum(float(row["tokens"]) for row in rows)
    durations = sum(float(row["duration_ms"]) for row in rows)
    return {
        "system": variant,
        "total_cases": len(rows),
        "evaluated_cases": len(evaluated),
        "passed": passed,
        "failed": len(evaluated) - passed,
        "external_failures": len(rows) - len(evaluated),
        "pass_rate": passed / len(evaluated) if evaluated else 0.0,
        "total_tokens": tokens,
        "avg_tokens": tokens / len(rows) if rows else 0.0,
        "avg_duration_ms": durations / len(rows) if rows else 0.0,
    }


async def evaluate_dataset_split(
    *,
    candidate_manifest: Path,
    dataset_manifest: Mapping[str, Any],
    samples: Iterable[dict[str, Any]],
    executor: CaseExecutor,
    split: str,
    active_skill_path: Path | None = None,
) -> dict[str, Any]:
    """Run one named dataset split under comparable Skill configurations.

    ``executor`` receives ``(sample, variant, skill_path)`` and returns either
    a bool or a mapping with ``passed`` plus optional ``tokens``,
    ``duration_ms``, ``external_failure``, ``failure_type`` and ``error``.
    An executor exception is retained as an external failure instead of being
    silently counted as a normal task failure.
    """
    candidate = validate_candidate_for_evaluation(candidate_manifest)
    fresh_check = check_candidate(Path(candidate.skill_path))
    if not fresh_check.passed:
        raise ValueError("Candidate no longer passes static checks: " + "; ".join(fresh_check.reasons))

    sample_rows = list(samples)
    sample_by_id = {str(sample["sample_id"]): sample for sample in sample_rows}
    errors = validate_manifest(dict(dataset_manifest), set(sample_by_id))
    if errors:
        raise ValueError("Invalid dataset manifest: " + "; ".join(errors))
    splits = _family_splits(dataset_manifest, candidate.task_family)
    if split not in {"dev", "regression", "holdout"}:
        raise ValueError(f"unsupported evaluation split: {split}")
    sample_ids = [str(value) for value in splits.get(split, [])]
    if not sample_ids:
        raise ValueError(f"task family has no {split} samples: {candidate.task_family}")
    missing = [sample_id for sample_id in sample_ids if sample_id not in sample_by_id]
    if missing:
        raise ValueError(f"{split} samples not found: " + ", ".join(missing))

    seed_ids = {str(value) for value in splits.get("seed", [])}
    unexpected_sources = sorted(set(candidate.source_sample_ids) - seed_ids)
    if unexpected_sources:
        raise ValueError("Candidate source samples must belong to Seed: " + ", ".join(unexpected_sources))

    candidate_path = Path(candidate.skill_path)
    variants: list[tuple[VariantName, Path | None]] = [(WITHOUT_SKILL, None)]
    if active_skill_path is not None:
        active_path = Path(active_skill_path)
        if not active_path.is_file():
            raise ValueError(f"Active Skill file not found: {active_path}")
        variants.append((ACTIVE_SKILL, active_path))
    variants.append((CANDIDATE_SKILL, candidate_path))

    cases: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant, _ in variants}
    for sample_id in sample_ids:
        sample = sample_by_id[sample_id]
        for variant, skill_path in variants:
            row = await _execute_case(executor, sample, variant, skill_path)
            cases.append(row)
            by_variant[variant].append(row)

    systems = [_summarize(variant, by_variant[variant]) for variant, _ in variants]
    return {
        "schema_version": "1.0",
        "stage": split,
        "task_family": candidate.task_family,
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "skill_id": candidate.skill_id,
            "version": candidate.version,
            "skill_path": candidate.skill_path,
        },
        f"{split}_sample_ids": sample_ids,
        "active_skill_available": active_skill_path is not None,
        "systems": systems,
        "cases": cases,
    }


async def evaluate_dev(
    *,
    candidate_manifest: Path,
    dataset_manifest: Mapping[str, Any],
    samples: Iterable[dict[str, Any]],
    executor: CaseExecutor,
    active_skill_path: Path | None = None,
) -> dict[str, Any]:
    """Run Dev samples under no Skill, optional Active Skill, and Candidate Skill.

    This compatibility wrapper keeps the original public entry point while
    allowing regression and holdout to use the exact same report contract.
    """
    return await evaluate_dataset_split(
        candidate_manifest=candidate_manifest,
        dataset_manifest=dataset_manifest,
        samples=samples,
        executor=executor,
        split="dev",
        active_skill_path=active_skill_path,
    )


def write_dev_report(path: Path, report: Mapping[str, Any]) -> None:
    """Persist the immutable Dev report consumed by the next release gate."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
