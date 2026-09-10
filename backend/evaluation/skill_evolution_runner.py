"""Offline first-phase Skill candidate gate and release command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.skill_candidate import validate_candidate_for_evaluation
from evaluation.skill_failure_attribution import attribute_failure
from evaluation.skill_release_gate import VariantMetrics, evaluate_release_gate, static_skill_check
from mem.skill_version_store import SkillVersion, SkillVersionStore


def _metrics(report: dict[str, object]) -> dict[str, VariantMetrics]:
    rows = report.get("systems") or []
    result: dict[str, VariantMetrics] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("system", ""))
        result[name] = VariantMetrics(
            pass_rate=float(row.get("pass_rate", 0)),
            avg_tokens=float(row.get("avg_tokens", 0)),
            external_failures=int(row.get("external_failures", 0)),
            avg_duration_ms=float(row.get("avg_duration_ms", 0) or 0),
            avg_tool_calls=float(row.get("avg_tool_calls", 0) or 0),
            stability_sample_count=int(row.get("stability_sample_count", row.get("total_cases", 0)) or 0),
        )
    return result


def gate_report(
    report_path: Path,
    *,
    static_check_passed: bool | None = None,
    regression_candidate_passed: bool | None = None,
    candidate_path: Path | None = None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    static_reasons: tuple[str, ...] = ()
    if candidate_path is None:
        candidate_data = report.get("candidate")
        if isinstance(candidate_data, dict) and candidate_data.get("skill_path"):
            candidate_path = Path(str(candidate_data["skill_path"]))
    if candidate_path is not None:
        static_check_passed, static_reasons = static_skill_check(candidate_path)
    decision = evaluate_release_gate(
        static_check_passed=static_check_passed,
        validation=_metrics(report),
        regression_candidate_passed=regression_candidate_passed,
    )
    reasons = [*static_reasons, *decision.reasons]
    attribution = attribute_failure(report, decision)
    return {
        "status": decision.status,
        "reasons": reasons,
        "classification": attribution.classification,
        "next_action": attribution.next_action,
        "evidence_sample_ids": list(attribution.evidence_sample_ids),
        "failure_types": list(attribution.failure_types),
        "report": str(report_path),
    }


def publish_candidate(
    *, evolution_root: Path, active_root: Path, skill_id: str, version: int,
) -> None:
    store = SkillVersionStore(evolution_root=evolution_root, active_root=active_root)
    path = store.candidate_dir(skill_id, version) / "candidate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    candidate = SkillVersion(
        skill_id=str(data["skill_id"]), version=int(data["version"]),
        status=str(data["status"]), parent_version=data.get("parent_version"),
        source=str(data.get("source", "")), reason=str(data.get("reason", "")),
    )
    store.publish(candidate)


def run_dev_evaluation(
    *,
    benchmark_manifest_path: Path,
    candidate_manifest_path: Path,
    family: str,
    report_path: Path,
    trials_dir: Path,
    active_skill_root: Path | None = None,
    skill_root: Path | None = None,
    agent: str = "pipixia-bailian",
    model: str = "glm-5.2",
    repeats: int = 1,
    max_steps: int = 100,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run Dev's three-version comparison and persist one gate report."""
    candidate = validate_candidate_for_evaluation(candidate_manifest_path)
    if candidate.task_family != family:
        raise ValueError(
            f"candidate task family {candidate.task_family!r} does not match {family!r}"
        )
    manifest = json.loads(benchmark_manifest_path.read_text(encoding="utf-8"))
    if not any(str(entry.get("family")) == family for entry in manifest.get("families", [])):
        raise ValueError(f"task family not found in SkillLearnBench manifest: {family}")

    candidate_path = Path(candidate.skill_path).resolve()
    resolved_skill_root = (skill_root or candidate_path.parent).resolve()
    resolved_trials_dir = trials_dir.resolve()
    # Keep this module importable without Docker/benchmark dependencies for
    # gate, publish and rollback commands.
    from evaluation.skilllearnbench_runner import evaluate, summarize_trials

    exit_code = evaluate(
        manifest,
        family=family,
        skill_root=resolved_skill_root,
        active_skill_root=active_skill_root,
        agent=agent,
        model=model,
        repeats=repeats,
        max_steps=max_steps,
        trials_dir=resolved_trials_dir,
        dry_run=dry_run,
    )
    if exit_code:
        raise RuntimeError(f"Dev evaluation failed with exit code {exit_code}")

    summary = summarize_trials(resolved_trials_dir)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "stage": "dev",
        "task_family": family,
        "candidate_manifest": str(candidate_manifest_path.resolve()),
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "skill_id": candidate.skill_id,
            "version": candidate.version,
            "skill_path": str(candidate_path),
        },
        "benchmark_manifest": str(benchmark_manifest_path.resolve()),
        "trials_dir": str(resolved_trials_dir),
        "systems": summary.get("systems", []),
        "cases": summary.get("cases", []),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PIPIXIA offline Skill evolution")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--report", type=Path, required=True)
    gate.add_argument("--candidate", type=Path)
    gate.add_argument("--evolution-root", type=Path)
    gate.add_argument("--skill")
    gate.add_argument("--version", type=int)
    gate.add_argument("--static-check-passed", action="store_true")
    gate.add_argument("--regression-passed", action="store_true")
    publish = sub.add_parser("publish")
    publish.add_argument("--evolution-root", type=Path, required=True)
    publish.add_argument("--active-root", type=Path, required=True)
    publish.add_argument("--skill", required=True)
    publish.add_argument("--version", type=int, required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--active-root", type=Path, required=True)
    rollback.add_argument("--skill", required=True)
    rollback.add_argument("--version", type=int)
    dev = sub.add_parser("dev", help="校验 Candidate 并运行 Dev 三版本对照评估")
    dev.add_argument("--benchmark-manifest", type=Path, required=True)
    dev.add_argument("--candidate-manifest", type=Path, required=True)
    dev.add_argument("--family", required=True)
    dev.add_argument("--report", type=Path, required=True)
    dev.add_argument("--trials-dir", type=Path, required=True)
    dev.add_argument("--skill-root", type=Path, help="可选；默认使用 Candidate 所在目录")
    dev.add_argument("--active-skill-root", type=Path)
    dev.add_argument("--agent", default="pipixia-bailian")
    dev.add_argument("--model", default="glm-5.2")
    dev.add_argument("--repeats", type=int, default=1)
    dev.add_argument("--max-steps", type=int, default=100)
    dev.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "gate":
        result = gate_report(
            args.report,
            static_check_passed=args.static_check_passed,
            regression_candidate_passed=args.regression_passed,
            candidate_path=getattr(args, "candidate", None),
        )
        if args.evolution_root and args.skill and args.version:
            store = SkillVersionStore(
                evolution_root=args.evolution_root,
                active_root=args.evolution_root.parent / "skills",
            )
            metadata = store.candidate_dir(args.skill, args.version) / "candidate.json"
            data = json.loads(metadata.read_text(encoding="utf-8"))
            candidate = SkillVersion(
                skill_id=str(data["skill_id"]), version=int(data["version"]),
                status=str(data["status"]), parent_version=data.get("parent_version"),
                source=str(data.get("source", "")), reason=str(data.get("reason", "")),
            )
            updated = store.set_candidate_status(candidate, str(result["status"]))
            result["candidate_status"] = updated.status
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "rollback":
        store = SkillVersionStore(evolution_root=args.active_root.parent / "skill_evolution", active_root=args.active_root)
        print(json.dumps({"active_version": store.rollback(args.skill, args.version)}))
        return
    if args.command == "dev":
        report_data = run_dev_evaluation(
            benchmark_manifest_path=args.benchmark_manifest,
            candidate_manifest_path=args.candidate_manifest,
            family=args.family,
            report_path=args.report,
            trials_dir=args.trials_dir,
            skill_root=args.skill_root,
            active_skill_root=args.active_skill_root,
            agent=args.agent,
            model=args.model,
            repeats=args.repeats,
            max_steps=args.max_steps,
            dry_run=args.dry_run,
        )
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
        return
    publish_candidate(
        evolution_root=args.evolution_root, active_root=args.active_root,
        skill_id=args.skill, version=args.version,
    )


if __name__ == "__main__":
    main()
