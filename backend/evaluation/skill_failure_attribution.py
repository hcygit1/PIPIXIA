"""Classify a failed offline Skill evaluation before changing any Skill."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from evaluation.skill_release_gate import ReleaseDecision


_EVALUATION_FAILURE_TYPES = {"verifier_error", "dataset_error", "evaluation_error", "invalid_executor_result"}


@dataclass(frozen=True)
class FailureAttribution:
    classification: str
    next_action: str
    evidence_sample_ids: tuple[str, ...] = ()
    failure_types: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _case_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("cases", [])
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def attribute_failure(
    report: Mapping[str, Any], decision: ReleaseDecision,
) -> FailureAttribution:
    """Choose the safe next action from a completed Dev/release report.

    This function deliberately never recommends changing a Skill after an
    infrastructure or verifier failure.  A Skill patch is allowed only when
    the evaluation itself is healthy and the candidate missed task cases.
    """
    if decision.status == "accepted":
        return FailureAttribution("ready_for_review", "run_holdout_and_request_review")
    if decision.status == "pending_regression":
        return FailureAttribution("pending_regression", "run_regression_evaluation")

    systems = report.get("systems", [])
    if not isinstance(systems, list):
        return FailureAttribution("evaluation_defect", "repair_evaluation_and_rerun")
    required = {"without_skill", "candidate_skill"}
    names = {
        str(row.get("system"))
        for row in systems
        if isinstance(row, Mapping)
    }
    if not required.issubset(names):
        return FailureAttribution("evaluation_defect", "repair_evaluation_and_rerun")

    external_rows = [
        row for row in _case_rows(report)
        if row.get("external_failure") is True or row.get("passed") is None
    ]
    has_metric_external_failure = any(
        isinstance(row, Mapping) and int(row.get("external_failures", 0) or 0) > 0
        for row in systems
    )
    if external_rows or has_metric_external_failure or decision.status == "external_failure":
        failure_types = tuple(sorted({
            str(row.get("failure_type") or "external_failure")
            for row in external_rows
        })) or ("external_failure",)
        sample_ids = tuple(sorted({
            str(row.get("sample_id")) for row in external_rows if row.get("sample_id")
        }))
        if any(kind in _EVALUATION_FAILURE_TYPES for kind in failure_types):
            return FailureAttribution(
                "evaluation_defect", "repair_evaluation_and_rerun", sample_ids, failure_types,
            )
        return FailureAttribution(
            "external_failure", "repair_environment_and_rerun", sample_ids, failure_types,
        )

    candidate_failures = [
        row for row in _case_rows(report)
        if row.get("variant") == "candidate_skill" and row.get("passed") is False
    ]
    # A local Skill patch is justified only for a failure that the Candidate
    # introduced.  If every comparable variant also fails, changing the Skill
    # would overfit an environment, task, or verifier problem.
    passed_by_sample = {
        str(row.get("sample_id"))
        for row in _case_rows(report)
        if row.get("variant") != "candidate_skill" and row.get("passed") is True
    }
    candidate_only_failures = [
        row for row in candidate_failures
        if str(row.get("sample_id")) in passed_by_sample
    ]
    sample_ids = tuple(sorted({
        str(row.get("sample_id"))
        for row in candidate_only_failures
        if row.get("sample_id")
    }))
    if not sample_ids:
        failed_ids = tuple(sorted({
            str(row.get("sample_id")) for row in candidate_failures if row.get("sample_id")
        }))
        return FailureAttribution(
            "inconclusive", "manual_failure_analysis", failed_ids,
        )
    return FailureAttribution(
        "skill_defect", "create_local_candidate_revision", sample_ids,
    )
