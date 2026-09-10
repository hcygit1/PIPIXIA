"""Deterministic first-phase release gate for offline Skill evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from pathlib import Path


@dataclass(frozen=True)
class VariantMetrics:
    pass_rate: float
    avg_tokens: float = 0.0
    external_failures: int = 0
    avg_duration_ms: float = 0.0
    avg_tool_calls: float = 0.0
    stability_sample_count: int = 0


@dataclass(frozen=True)
class ReleaseDecision:
    status: str
    reasons: tuple[str, ...]


def static_skill_content_check(content: str) -> tuple[bool, tuple[str, ...]]:
    """Validate deterministic Skill structure without requiring a file."""
    reasons: list[str] = []
    if len(content.strip()) < 50:
        reasons.append("Skill content is too short")
    if not content.lstrip().startswith("---"):
        reasons.append("Skill frontmatter is missing")
    if "name:" not in content:
        reasons.append("Skill name is missing")
    if "description:" not in content:
        reasons.append("Skill description is missing")
    return not reasons, tuple(reasons)


def static_skill_check(skill_path: Path) -> tuple[bool, tuple[str, ...]]:
    """Validate the deterministic parts of a candidate before running agents."""
    if not skill_path.is_file():
        return False, (f"missing Skill file: {skill_path}",)
    return static_skill_content_check(
        skill_path.read_text(encoding="utf-8", errors="replace")
    )


def evaluate_release_gate(
    *,
    static_check_passed: bool,
    validation: Mapping[str, VariantMetrics],
    regression_candidate_passed: bool | None,
    max_token_increase_ratio: float = 0.5,
) -> ReleaseDecision:
    """Accept only candidates that improve the baseline without regressing active."""
    candidate = validation.get("candidate_skill")
    baseline = validation.get("without_skill")
    active = validation.get("active_skill")
    if candidate is None or baseline is None:
        return ReleaseDecision("needs_eval_fix", ("missing required variant metrics",))
    if candidate.external_failures or baseline.external_failures or (active and active.external_failures):
        return ReleaseDecision("external_failure", ("one or more runs had external failures",))
    if not static_check_passed:
        return ReleaseDecision("rejected", ("static quality check failed",))
    if candidate.pass_rate <= baseline.pass_rate:
        return ReleaseDecision("rejected", ("candidate did not outperform without_skill",))
    if active is not None and candidate.pass_rate < active.pass_rate:
        return ReleaseDecision("rejected", ("candidate regressed against active_skill",))
    if regression_candidate_passed is None:
        return ReleaseDecision("pending_regression", ("regression evaluation has not run",))
    if not regression_candidate_passed:
        return ReleaseDecision("rejected", ("candidate failed regression cases",))
    reference_tokens = active.avg_tokens if active is not None else baseline.avg_tokens
    if reference_tokens and candidate.avg_tokens > reference_tokens * (1 + max_token_increase_ratio):
        return ReleaseDecision("rejected", ("candidate token cost exceeds budget",))
    return ReleaseDecision("accepted", ())
