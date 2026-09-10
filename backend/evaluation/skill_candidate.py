"""Seed-to-Candidate generation and pre-evaluation static checks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from evaluation.skill_evolution_dataset import load_samples, validate_manifest
from evaluation.skill_release_gate import static_skill_content_check
from mem.skill_version_store import SkillVersionStore


@dataclass(frozen=True)
class CandidateCheck:
    passed: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    repairable: bool = False
    blocking_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    skill_id: str
    version: int
    task_family: str
    source_sample_ids: tuple[str, ...]
    skill_path: str
    status: str
    checks: CandidateCheck
    parent_candidate_id: str | None = None
    iteration: int = 0
    failure_sample_ids: tuple[str, ...] = ()
    no_improvement_count: int = 0
    static_repair_attempts: int = 0


StaticRepairLlmCall = Callable[[str], str | Awaitable[str]]


class UnsafeCandidateContentError(ValueError):
    """Raised when generated content must not be persisted or sent for repair."""

    def __init__(self, check: CandidateCheck):
        self.check = check
        super().__init__("; ".join(check.blocking_reasons or check.reasons))

STATIC_REPAIR_PROMPT = """Repair only the static format problems in this Candidate Skill.

Return the complete corrected SKILL.md without a code fence or explanation.
Do not add new behavior or task-specific facts. Preserve all valid content.

Static check failures:
{REASONS}

Current SKILL.md:
{CONTENT}
"""


def check_candidate(skill_path: Path) -> CandidateCheck:
    """Run deterministic checks before a candidate can enter Dev evaluation."""
    if not skill_path.is_file():
        reason = f"missing Skill file: {skill_path}"
        return CandidateCheck(False, (reason,), repairable=False, blocking_reasons=(reason,))
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    return check_candidate_content(content)


def check_candidate_content(content: str) -> CandidateCheck:
    """Classify static failures as format-repairable or security-blocking."""
    passed, reasons = static_skill_content_check(content)
    extra: list[str] = []
    warnings: list[str] = []
    blocking: list[str] = []
    if re.search(r"(?i)(api[_-]?key|secret|password|bearer\s+)[^\n]{0,100}", content):
        blocking.append("possible secret or credential in Skill content")
    if len(content.splitlines()) > 400:
        warnings.append("Skill content exceeds 400 lines; review token cost and duplication")
    if not re.search(r"(?m)^##\s+", content):
        extra.append("Skill must contain at least one Markdown section")
    all_reasons = (*reasons, *extra, *blocking)
    return CandidateCheck(
        not all_reasons,
        tuple(all_reasons),
        tuple(warnings),
        repairable=bool(all_reasons) and not blocking,
        blocking_reasons=tuple(blocking),
    )


def _strip_skill_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*", content, re.IGNORECASE)
    return match.group(1).strip() if match else content.strip()


async def repair_candidate_content(
    content: str,
    *,
    llm_call: StaticRepairLlmCall,
    max_attempts: int = 2,
) -> tuple[str, CandidateCheck, int]:
    """Apply bounded format-only repairs; security failures are never sent back to the LLM."""
    current = content
    check = check_candidate_content(current)
    attempts = 0
    while not check.passed and check.repairable and attempts < max_attempts:
        prompt = STATIC_REPAIR_PROMPT.format(
            REASONS="\n".join(f"- {reason}" for reason in check.reasons),
            CONTENT=current[:16000],
        )
        repaired = llm_call(prompt)
        if inspect.isawaitable(repaired):
            repaired = await repaired
        attempts += 1
        current = _strip_skill_fence(str(repaired or ""))
        check = check_candidate_content(current)
    return current, check, attempts


def _seed_ids(manifest: dict[str, Any], family: str) -> list[str]:
    for entry in manifest.get("families", []):
        if entry.get("task_family") == family:
            return [str(value) for value in entry.get("splits", {}).get("seed", [])]
    raise ValueError(f"task family not found: {family}")


def _candidate_id(family: str, source_ids: list[str], content: str) -> str:
    raw = json.dumps([family, source_ids, content], ensure_ascii=False, sort_keys=True)
    return "candidate-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def save_candidate_manifest(path: Path, record: CandidateRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_candidate_record(
    *,
    family: str,
    source_samples: list[dict[str, Any]],
    skill_id: str,
    version: int,
    skill_path: Path,
) -> CandidateRecord:
    check = check_candidate(skill_path)
    status = "ready_for_dev" if check.passed else "static_check_failed"
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    return CandidateRecord(
        candidate_id=_candidate_id(family, [str(s["sample_id"]) for s in source_samples], content),
        skill_id=skill_id,
        version=version,
        task_family=family,
        source_sample_ids=tuple(str(s["sample_id"]) for s in source_samples),
        skill_path=str(skill_path),
        status=status,
        checks=check,
    )


def validate_candidate_for_evaluation(candidate_manifest: Path) -> CandidateRecord:
    data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    checks = data.get("checks") or {}
    record = CandidateRecord(
        candidate_id=str(data["candidate_id"]),
        skill_id=str(data["skill_id"]),
        version=int(data["version"]),
        task_family=str(data["task_family"]),
        source_sample_ids=tuple(str(value) for value in data.get("source_sample_ids", [])),
        skill_path=str(data["skill_path"]),
        status=str(data["status"]),
        checks=CandidateCheck(
            bool(checks.get("passed")),
            tuple(checks.get("reasons", [])),
            tuple(checks.get("warnings", [])),
            bool(checks.get("repairable", False)),
            tuple(checks.get("blocking_reasons", [])),
        ),
        parent_candidate_id=(str(data["parent_candidate_id"]) if data.get("parent_candidate_id") else None),
        iteration=int(data.get("iteration", 0) or 0),
        failure_sample_ids=tuple(str(value) for value in data.get("failure_sample_ids", [])),
        no_improvement_count=int(data.get("no_improvement_count", 0) or 0),
        static_repair_attempts=int(data.get("static_repair_attempts", 0) or 0),
    )
    if record.status != "ready_for_dev" or not record.checks.passed:
        raise ValueError("Candidate has not passed static checks")
    if not Path(record.skill_path).is_file():
        raise ValueError(f"Candidate Skill file not found: {record.skill_path}")
    return record


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成并静态检查离线 Skill Candidate")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--skill", type=Path, required=True, help="已生成的 SKILL.md")
    parser.add_argument("--evolution-root", type=Path, required=True)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    samples = load_samples(args.samples)
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest, set(sample_by_id))
    if errors:
        raise SystemExit("\n".join(errors))
    source_ids = _seed_ids(manifest, args.family)
    source_samples = [sample_by_id[sample_id] for sample_id in source_ids]
    store = SkillVersionStore(evolution_root=args.evolution_root, active_root=args.active_root)
    version = store.next_candidate_version(args.skill_id)
    record = build_candidate_record(
        family=args.family,
        source_samples=source_samples,
        skill_id=args.skill_id,
        version=version,
        skill_path=args.skill,
    )
    save_candidate_manifest(args.output, record)
    print(json.dumps(asdict(record), ensure_ascii=False, indent=2))
    return 0 if record.checks.passed else 2


if __name__ == "__main__":
    raise SystemExit(_main())
