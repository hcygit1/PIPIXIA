"""Workspace adapter for running the Dev three-variant comparison."""

from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import resolve_agent_skills_dir
from evaluation.skill_dev_evaluation import evaluate_dataset_split, write_dev_report
from evaluation.pipixia_replay_executor import replay


def _split_ids(manifest: dict[str, Any], family: str, split: str) -> list[str]:
    entry = next(
        (row for row in manifest.get("families", []) if row.get("task_family") == family),
        None,
    )
    if entry is None:
        raise ValueError(f"数据集不包含任务族: {family}")
    return [str(value) for value in entry.get("splits", {}).get(split, [])]


@contextmanager
def _benchmark_skill_roots(
    family: str, candidate_skill: Path, active_skill: Path | None,
):
    temp = Path(tempfile.mkdtemp(prefix="pipixia_skill_configs_"))
    try:
        candidate_root = temp / "candidate_skill"
        candidate_family = candidate_root / family
        candidate_family.mkdir(parents=True)
        shutil.copy2(candidate_skill, candidate_family / "SKILL.md")
        active_root = None
        if active_skill is not None and active_skill.is_file():
            active_root = temp / "active_skill"
            active_family = active_root / family
            active_family.mkdir(parents=True)
            shutil.copy2(active_skill, active_family / "SKILL.md")
        yield candidate_root, active_root
    finally:
        shutil.rmtree(temp, ignore_errors=True)


async def _run_skilllearnbench(
    root: Path, task: dict[str, Any], manifest: dict[str, Any],
    candidate_manifest: Path, active_path: Path, *, split: str = "dev",
) -> dict[str, Any]:
    import asyncio
    from evaluation.skilllearnbench_runner import evaluate, summarize_trials

    candidate_data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    candidate_skill = Path(str(candidate_data["skill_path"]))
    instance_ids = _split_ids(manifest, str(task["task_family"]), split)
    if not instance_ids:
        raise ValueError(f"{split} 集没有 SkillLearnBench 评估实例")
    benchmark_manifest = Path(str(task.get("artifacts", {}).get("benchmark_manifest", "")))
    if not benchmark_manifest.is_file():
        raise ValueError("缺少 SkillLearnBench benchmark manifest")
    benchmark = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
    trials_dir = root / "benchmark" / f"{split}-trials"
    if trials_dir.exists():
        shutil.rmtree(trials_dir)
    with _benchmark_skill_roots(
        str(task["task_family"]), candidate_skill,
        active_path if active_path.exists() else None,
    ) as (candidate_root, active_root):
        exit_code = await asyncio.to_thread(
            evaluate,
            benchmark,
            family=str(task["task_family"]),
            skill_root=candidate_root,
            active_skill_root=active_root,
            human_authored_skill_root=(Path(str(benchmark["benchmark_root"])) / "skills" / "human_authored") if split == "holdout" and (Path(str(benchmark["benchmark_root"])) / "skills" / "human_authored").is_dir() else None,
            agent="pipixia-bailian",
            model=str(__import__("os").getenv("PIPIXIA_LLM_MODEL", "glm-5.2")),
            repeats=1,
            max_steps=100,
            trials_dir=trials_dir,
            dry_run=False,
            instance_ids=instance_ids,
        )
    # A non-zero exit means one or more task variants failed their verifier;
    # those are evaluation results, not an infrastructure error.
    summary = summarize_trials(trials_dir)
    if not summary["cases"]:
        raise RuntimeError(f"SkillLearnBench 未生成评估结果，返回码: {exit_code}")
    return {
        "schema_version": "1.0",
        "stage": split,
        "task_family": task["task_family"],
        "candidate": {
            "candidate_id": candidate_data.get("candidate_id"),
            "skill_id": candidate_data.get("skill_id"),
            "version": candidate_data.get("version"),
            "skill_path": str(candidate_skill),
        },
        f"{split}_sample_ids": instance_ids,
        "active_skill_available": active_root is not None,
        "human_authored_available": any(row.get("system") == "human_authored" for row in summary["systems"]),
        "evaluator": "skilllearnbench_official_verifier",
        "systems": summary["systems"],
        "cases": summary["cases"],
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run_dev(root: Path, task: dict[str, Any], *, candidate_manifest: Path | None = None, report_name: str = "dev-v1.json", dataset_manifest_path: Path | None = None, split: str = "dev") -> dict[str, Any]:
    manifest_path = dataset_manifest_path or (root / "datasets" / "v1" / "manifest.json")
    samples_path = root / "datasets" / "v1" / "samples.jsonl"
    candidate_manifest = candidate_manifest or Path(str(task.get("artifacts", {}).get("candidate_manifest") or root / "candidates" / "v1" / "candidate.json"))
    if not manifest_path.exists() or not samples_path.exists():
        raise ValueError("缺少冻结数据集 v1")
    if not candidate_manifest.exists():
        raise ValueError("缺少 Candidate 清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = _load_jsonl(samples_path)

    async def executor(sample: dict[str, Any], variant: str, skill_path: Path | None) -> dict[str, Any]:
        return await replay(sample, variant, skill_path, agent_id=str(task.get("agent_id") or "main"))

    candidate_data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    # Active Skill is resolved independently from the Candidate lineage. The
    # candidate may have a generated skill_id that is not the live skill name.
    active_root = resolve_agent_skills_dir(str(task.get("agent_id") or "main"))
    active_path = active_root / str(task.get("task_family", "")) / "SKILL.md"
    if not active_path.exists():
        active_path = active_root / str(candidate_data.get("skill_id", "")) / "SKILL.md"

    if task.get("data_source") == "skilllearnbench":
        report = await _run_skilllearnbench(
            root, task, manifest, candidate_manifest, active_path, split=split,
        )
    else:
        report = await evaluate_dataset_split(
            candidate_manifest=candidate_manifest,
            dataset_manifest=manifest,
            samples=samples,
            executor=executor,
            split=split,
            active_skill_path=active_path if active_path.exists() else None,
        )
    report_path = root / "reports" / report_name
    write_dev_report(report_path, report)
    report["report_path"] = str(report_path)
    return report
