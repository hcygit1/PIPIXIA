"""Failure attribution and one bounded local Candidate revision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from config import load_config
from evaluation.skill_candidate_iteration import create_local_revision
from evaluation.skill_candidate_iteration import IterationPolicy, promote_failures_to_regression
from evaluation.skill_evolution_dev_editor import run_dev
from evaluation.skill_failure_attribution import attribute_failure
from evaluation.skill_release_gate import ReleaseDecision


def load_dev_report(root: Path) -> dict[str, Any]:
    reports = sorted((root / "reports").glob("dev-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    path = reports[0] if reports else root / "reports" / "dev-v1.json"
    if not path.exists():
        raise ValueError("开发集报告不存在")
    return json.loads(path.read_text(encoding="utf-8"))


def attribution(root: Path) -> dict[str, Any]:
    report = load_dev_report(root)
    decision = ReleaseDecision("rejected", ("开发集未达到发布门禁",))
    result = attribute_failure(report, decision).to_dict()
    (root / "reports" / "failure-attribution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


async def create_revision(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    report = load_dev_report(root)
    result = attribute(root)
    if result.get("classification") != "skill_defect":
        raise ValueError(f"当前不属于可自动修改的 Skill 问题: {result.get('next_action')}")
    base_manifest_path = root / "datasets" / "v1" / "manifest.json"
    if base_manifest_path.exists():
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        regression_manifest = promote_failures_to_regression(
            base_manifest,
            family=str(task.get("task_family") or report.get("task_family") or ""),
            sample_ids=result.get("evidence_sample_ids", []),
        )
        regression_path = root / "datasets" / "v1" / "manifest-with-regression.json"
        regression_path.write_text(json.dumps(regression_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path = root / "datasets" / "v1" / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parent = Path(task.get("artifacts", {}).get("candidate_manifest") or root / "candidates" / "v1" / "candidate.json")
    output = root / "candidates" / "latest-revision.json"
    llm_cfg = load_config().get("mem", {}).get("llm", {})
    base_url, api_key, model = str(llm_cfg.get("base_url") or "").rstrip("/"), str(llm_cfg.get("api_key") or ""), str(llm_cfg.get("model") or "qwen-plus")
    if not base_url or not api_key:
        raise ValueError("记忆模型未配置，无法执行局部修改")

    async def llm_call(prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        body: dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if "dashscope.aliyuncs.com" in base_url:
            body["enable_thinking"] = False
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=body)
            response.raise_for_status()
        choices = response.json().get("choices", [])
        return str(choices[0].get("message", {}).get("content", "") if choices else "").strip()

    record = await create_local_revision(parent_manifest=parent, report=report, samples=samples, evolution_root=root, output_manifest=output, llm_call=llm_call)
    result = {"candidate_id": record.candidate_id, "candidate_manifest": str(output), "skill_path": record.skill_path, "version": record.version, "iteration": record.iteration, "checks": {"passed": record.checks.passed, "reasons": list(record.checks.reasons)}, "failure_sample_ids": list(record.failure_sample_ids), "regression_manifest": str(root / "datasets" / "v1" / "manifest-with-regression.json")}
    (root / "reports" / "revision-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


async def iterate(root: Path, task: dict[str, Any], policy: IterationPolicy = IterationPolicy()) -> dict[str, Any]:
    """Run bounded local revisions and re-evaluate each child Candidate."""
    report = load_dev_report(root)
    current_manifest = Path(task.get("artifacts", {}).get("candidate_manifest") or root / "candidates" / "v1" / "candidate.json")
    samples_path = root / "datasets" / "v1" / "samples.jsonl"
    samples = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    base_manifest = json.loads((root / "datasets" / "v1" / "manifest.json").read_text(encoding="utf-8"))
    history: list[dict[str, Any]] = []
    previous_rate: float | None = None
    no_improvement = 0
    for iteration in range(1, policy.max_iterations + 1):
        attribution_result = attribute_failure(report, ReleaseDecision("rejected", ("开发集未达到发布门禁",))).to_dict()
        if attribution_result.get("classification") != "skill_defect":
            result = {"status": "stopped", "reason": attribution_result, "iterations": history, "candidate_manifest": str(current_manifest)}
            return _save_iteration_state(root, result)
        rate = next((float(row.get("pass_rate", 0.0) or 0.0) for row in report.get("systems", []) if row.get("system") == "candidate_skill"), 0.0)
        if previous_rate is not None and rate - previous_rate < policy.minimum_improvement:
            no_improvement += 1
        else:
            no_improvement = 0
        if no_improvement >= policy.no_improvement_limit:
            result = {"status": "abandoned", "reason": {"classification": "iteration_limit", "next_action": "retain_artifacts_and_wait_for_more_samples"}, "iterations": history, "no_improvement_count": no_improvement, "candidate_manifest": str(current_manifest)}
            return _save_iteration_state(root, result)
        ids = attribution_result.get("evidence_sample_ids", [])
        regression_manifest = json.loads(json.dumps(base_manifest))
        family_entry = next((entry for entry in regression_manifest.get("families", []) if entry.get("task_family") == str(report.get("task_family", task.get("task_family", "")))), None)
        if family_entry is None:
            raise ValueError("task family not found in frozen dataset")
        regression_ids = list(family_entry.setdefault("splits", {}).get("regression", []))
        for sample_id in ids:
            if sample_id not in regression_ids:
                regression_ids.append(sample_id)
        family_entry["splits"]["regression"] = regression_ids
        regression_path = root / "datasets" / "v1" / "manifest-with-regression.json"
        regression_path.write_text(json.dumps(regression_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = root / "reports" / f"dev-v{iteration + 1}.json"
        output = root / "candidates" / "latest-revision.json"
        task_view = {**task, "artifacts": {**task.get("artifacts", {}), "candidate_manifest": str(current_manifest)}}
        # create_local_revision uses the current report and preserves parent lineage.
        revised = await _create_revision_from(root, task_view, report, samples, current_manifest, output)
        if not revised["checks"]["passed"]:
            result = {"status": "abandoned", "reason": {"classification": "static_repair_exhausted", "next_action": "retain_artifacts_and_wait_for_more_samples"}, "iterations": history + [revised], "candidate_manifest": revised["candidate_manifest"]}
            return _save_iteration_state(root, result)
        new_report = await run_dev(root, task, candidate_manifest=Path(revised["candidate_manifest"]), report_name=report_path.name, dataset_manifest_path=regression_path, split="regression")
        new_rate = next((float(row.get("pass_rate", 0.0) or 0.0) for row in new_report.get("systems", []) if row.get("system") == "candidate_skill"), 0.0)
        history.append({"iteration": iteration, "candidate_manifest": revised["candidate_manifest"], "report_path": str(report_path), "before_pass_rate": rate, "after_pass_rate": new_rate, "improvement": new_rate - rate, "failure_sample_ids": ids})
        previous_rate, current_manifest, report = new_rate, Path(revised["candidate_manifest"]), new_report
    return _save_iteration_state(root, {"status": "max_iterations_reached", "reason": {"classification": "iteration_limit", "next_action": "manual_failure_analysis"}, "iterations": history, "candidate_manifest": str(current_manifest), "no_improvement_count": no_improvement})


def _save_iteration_state(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    path = root / "reports" / "iteration-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["iteration_state_path"] = str(path)
    return result


async def _create_revision_from(root: Path, task: dict[str, Any], report: dict[str, Any], samples: list[dict[str, Any]], parent: Path, output: Path) -> dict[str, Any]:
    from evaluation.skill_candidate_iteration import create_local_revision
    llm_cfg = load_config().get("mem", {}).get("llm", {})
    base_url, api_key, model = str(llm_cfg.get("base_url") or "").rstrip("/"), str(llm_cfg.get("api_key") or ""), str(llm_cfg.get("model") or "qwen-plus")
    if not base_url or not api_key:
        raise ValueError("记忆模型未配置，无法执行局部修改")
    async def llm_call(prompt: str, *, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        body: dict[str, Any] = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if "dashscope.aliyuncs.com" in base_url: body["enable_thinking"] = False
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{base_url}/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=body); response.raise_for_status()
        choices = response.json().get("choices", [])
        return str(choices[0].get("message", {}).get("content", "") if choices else "").strip()
    record = await create_local_revision(parent_manifest=parent, report=report, samples=samples, evolution_root=root, output_manifest=output, llm_call=llm_call)
    result = {"candidate_id": record.candidate_id, "candidate_manifest": str(output), "skill_path": record.skill_path, "version": record.version, "iteration": record.iteration, "checks": {"passed": record.checks.passed, "reasons": list(record.checks.reasons)}, "failure_sample_ids": list(record.failure_sample_ids)}
    (root / "reports" / f"revision-v{record.iteration}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
