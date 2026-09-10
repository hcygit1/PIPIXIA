"""Bounded local Candidate revisions for Skill-evolution failures."""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from evaluation.skill_candidate import (
    CandidateRecord,
    UnsafeCandidateContentError,
    build_candidate_record,
    repair_candidate_content,
    save_candidate_manifest,
    validate_candidate_for_evaluation,
)
from evaluation.skill_failure_attribution import FailureAttribution, attribute_failure
from evaluation.skill_release_gate import ReleaseDecision


LOCAL_REVISION_PROMPT = """You are making a LOCAL revision to an offline Candidate Skill.

Rules:
- Return the complete updated SKILL.md, not a patch or explanation.
- Preserve verified behavior and change only the sections related to the supplied failures.
- Generalize the correction; do not hard-code one task's incidental values.
- Keep frontmatter, name, description and Markdown sections valid.
- Never add credentials, secrets, or unverified procedures.

Current Candidate:
{CURRENT_CONTENT}

Failed Dev evidence:
{FAILURE_EVIDENCE}

Parent Candidate: {PARENT_ID}
Revision number: {ITERATION}
"""


LocalRevisionLlmCall = Callable[[str], str | Awaitable[str]]


@dataclass(frozen=True)
class IterationPolicy:
    max_iterations: int = 3
    no_improvement_limit: int = 2
    minimum_improvement: float = 0.001


@dataclass(frozen=True)
class IterationResult:
    status: str
    candidate_manifest: Path
    report: Mapping[str, Any]
    iterations: int
    no_improvement_count: int
    attribution: FailureAttribution


def promote_failures_to_regression(
    manifest: Mapping[str, Any], *, family: str, sample_ids: Iterable[str],
) -> dict[str, Any]:
    """Move failed Dev IDs into Regression without duplicating split membership."""
    updated = copy.deepcopy(dict(manifest))
    ids = {str(value) for value in sample_ids}
    if not ids:
        return updated
    for entry in updated.get("families", []):
        if not isinstance(entry, dict) or entry.get("task_family") != family:
            continue
        splits = entry.setdefault("splits", {})
        dev = [str(value) for value in splits.get("dev", [])]
        regression = [str(value) for value in splits.get("regression", [])]
        moved = [value for value in dev if value in ids]
        splits["dev"] = [value for value in dev if value not in ids]
        for value in moved:
            if value not in regression:
                regression.append(value)
        splits["regression"] = regression
        return updated
    raise ValueError(f"task family not found in manifest: {family}")


def _failure_evidence(
    report: Mapping[str, Any], samples: Mapping[str, dict[str, Any]], sample_ids: Iterable[str],
) -> str:
    rows = []
    wanted = {str(value) for value in sample_ids}
    for row in report.get("cases", []) if isinstance(report.get("cases"), list) else []:
        if not isinstance(row, Mapping) or str(row.get("sample_id")) not in wanted:
            continue
        sample = samples.get(str(row["sample_id"]), {})
        rows.append({
            "sample_id": row.get("sample_id"),
            "result": {key: row.get(key) for key in ("passed", "error", "failure_type")},
            "task_snapshot": sample.get("task_snapshot"),
            "trajectory": sample.get("trajectory"),
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)[:12000]


async def create_local_revision(
    *,
    parent_manifest: Path,
    report: Mapping[str, Any],
    samples: Iterable[dict[str, Any]],
    evolution_root: Path,
    output_manifest: Path,
    llm_call: LocalRevisionLlmCall,
    no_improvement_count: int | None = None,
) -> CandidateRecord:
    """Generate one isolated child Candidate from Candidate-only failures."""
    parent = validate_candidate_for_evaluation(parent_manifest)
    attribution = attribute_failure(
        report,
        ReleaseDecision("rejected", ("candidate requires local revision",)),
    )
    if attribution.classification != "skill_defect":
        raise ValueError("local revision requires a skill_defect attribution")
    sample_rows = {str(row["sample_id"]): row for row in samples}
    evidence = _failure_evidence(report, sample_rows, attribution.evidence_sample_ids)
    current_content = Path(parent.skill_path).read_text(encoding="utf-8", errors="replace")
    prompt = LOCAL_REVISION_PROMPT.format(
        CURRENT_CONTENT=current_content[:12000],
        FAILURE_EVIDENCE=evidence,
        PARENT_ID=parent.candidate_id,
        ITERATION=parent.iteration + 1,
    )
    result = llm_call(prompt)
    if inspect.isawaitable(result):
        result = await result
    content, check, repair_attempts = await repair_candidate_content(
        str(result or "").strip(), llm_call=llm_call, max_attempts=2,
    )
    if check.blocking_reasons:
        raise UnsafeCandidateContentError(check)
    versions = [
        int(path.name[1:])
        for path in (evolution_root / "candidates").glob("v*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    version = max([parent.version, *versions], default=0) + 1
    destination = evolution_root / "candidates" / f"v{version}"
    destination.mkdir(parents=True, exist_ok=False)
    skill_path = destination / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    record = CandidateRecord(
        candidate_id="",
        skill_id=parent.skill_id,
        version=version,
        task_family=parent.task_family,
        source_sample_ids=parent.source_sample_ids,
        skill_path=str(skill_path),
        status="ready_for_dev" if check.passed else "static_check_failed",
        checks=check,
        parent_candidate_id=parent.candidate_id,
        iteration=parent.iteration + 1,
        failure_sample_ids=attribution.evidence_sample_ids,
        no_improvement_count=(
            parent.no_improvement_count if no_improvement_count is None else no_improvement_count
        ),
        static_repair_attempts=repair_attempts,
    )
    # Reuse the canonical content hash/id generator while preserving lineage.
    generated = build_candidate_record(
        family=record.task_family,
        source_samples=[{"sample_id": value} for value in record.source_sample_ids],
        skill_id=record.skill_id,
        version=record.version,
        skill_path=skill_path,
    )
    record = replace(
        generated,
        parent_candidate_id=record.parent_candidate_id,
        iteration=record.iteration,
        failure_sample_ids=record.failure_sample_ids,
        no_improvement_count=record.no_improvement_count,
        static_repair_attempts=record.static_repair_attempts,
    )
    # The candidate directory is the publishable artifact; the optional
    # external manifest is a convenient pointer for the evaluation command.
    save_candidate_manifest(destination / "candidate.json", record)
    # Keep the historical skill-scoped manifest as a compatibility pointer for
    # older callers; the task-level candidates/vN directory is canonical.
    legacy_destination = evolution_root / record.skill_id / "candidates" / f"v{version}"
    legacy_destination.mkdir(parents=True, exist_ok=True)
    save_candidate_manifest(legacy_destination / "candidate.json", record)
    if version != 1:
        legacy_v1 = evolution_root / record.skill_id / "candidates" / "v1"
        legacy_v1.mkdir(parents=True, exist_ok=True)
        save_candidate_manifest(legacy_v1 / "candidate.json", record)
    if output_manifest.resolve() != (destination / "candidate.json").resolve():
        save_candidate_manifest(output_manifest, record)
    return record


def _write_iteration_state(evolution_root: Path, result: IterationResult) -> None:
    candidate = json.loads(result.candidate_manifest.read_text(encoding="utf-8"))
    path = evolution_root / str(candidate["skill_id"]) / "iteration-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_manifest": str(result.candidate_manifest),
        "status": result.status,
        "iterations": result.iterations,
        "no_improvement_count": result.no_improvement_count,
        "classification": result.attribution.classification,
        "next_action": result.attribution.next_action,
        "evidence_sample_ids": list(result.attribution.evidence_sample_ids),
        "failure_types": list(result.attribution.failure_types),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def iterate_skill_candidate(
    *,
    candidate_manifest: Path,
    dataset_manifest: Mapping[str, Any],
    samples: Iterable[dict[str, Any]],
    initial_report: Mapping[str, Any],
    initial_decision: ReleaseDecision,
    evolution_root: Path,
    manifest_writer: Callable[[Mapping[str, Any]], None],
    evaluate_candidate: Callable[[Path, Mapping[str, Any]], Awaitable[tuple[Mapping[str, Any], ReleaseDecision]] | tuple[Mapping[str, Any], ReleaseDecision]],
    llm_call: LocalRevisionLlmCall,
    policy: IterationPolicy = IterationPolicy(),
) -> IterationResult:
    """Run bounded local revisions; abandon an unproductive evolution attempt."""
    current_manifest = candidate_manifest
    current_report: Mapping[str, Any] = initial_report
    current_decision = initial_decision
    no_improvement = 0
    iterations = 0
    previous_rate = None
    sample_rows = list(samples)
    while True:
        attribution = attribute_failure(current_report, current_decision)
        if attribution.classification != "skill_defect":
            result = IterationResult(
                "stopped", current_manifest, current_report, iterations, no_improvement, attribution,
            )
            _write_iteration_state(evolution_root, result)
            return result
        candidate_row = next(
            (row for row in current_report.get("systems", [])
             if isinstance(row, Mapping) and row.get("system") == "candidate_skill"),
            {},
        )
        rate = float(candidate_row.get("pass_rate", 0.0) or 0.0)
        if previous_rate is not None and rate - previous_rate < policy.minimum_improvement:
            no_improvement += 1
        else:
            no_improvement = 0
        if iterations >= policy.max_iterations or no_improvement >= policy.no_improvement_limit:
            result = IterationResult(
                "abandoned", current_manifest, current_report, iterations, no_improvement,
                FailureAttribution(
                    "iteration_limit", "retain_artifacts_and_wait_for_more_samples",
                    attribution.evidence_sample_ids, attribution.failure_types,
                ),
            )
            _write_iteration_state(evolution_root, result)
            return result
        regression_manifest = promote_failures_to_regression(
            dataset_manifest, family=str(current_report.get("task_family", "")),
            sample_ids=attribution.evidence_sample_ids,
        )
        manifest_writer(regression_manifest)
        next_manifest = current_manifest.with_name(f"{current_manifest.stem}.v{iterations + 1}.json")
        try:
            revised = await create_local_revision(
                parent_manifest=current_manifest,
                report=current_report,
                samples=sample_rows,
                evolution_root=evolution_root,
                output_manifest=next_manifest,
                llm_call=llm_call,
                no_improvement_count=no_improvement,
            )
        except UnsafeCandidateContentError as error:
            result = IterationResult(
                "abandoned", current_manifest, current_report, iterations, no_improvement,
                FailureAttribution(
                    "security_check_failed",
                    "discard_unsafe_candidate_content",
                    attribution.evidence_sample_ids,
                    tuple(error.check.blocking_reasons),
                ),
            )
            _write_iteration_state(evolution_root, result)
            return result
        if not revised.checks.passed:
            current_manifest = next_manifest
            result = IterationResult(
                "abandoned", current_manifest, current_report, iterations + 1, no_improvement,
                FailureAttribution(
                    "static_repair_exhausted",
                    "retain_artifacts_and_wait_for_more_samples",
                    attribution.evidence_sample_ids,
                    tuple(revised.checks.reasons),
                ),
            )
            _write_iteration_state(evolution_root, result)
            return result
        previous_rate = rate
        iterations += 1
        current_manifest = next_manifest
        evaluated = evaluate_candidate(current_manifest, regression_manifest)
        if inspect.isawaitable(evaluated):
            current_report, current_decision = await evaluated
        else:
            current_report, current_decision = evaluated
