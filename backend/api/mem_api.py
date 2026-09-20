"""记忆系统 API — 为前端 MemoryModal 提供数据"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.dependencies import get_agent_manager
from config import DATA_DIR
from evaluation.skill_evolution_workflow import build_workflow

router = APIRouter(tags=["mem"])
logger = logging.getLogger(__name__)
_evolution_workflow = build_workflow(DATA_DIR / "evaluation" / "skill_evolution")
_evaluation_jobs: dict[str, subprocess.Popen[str]] = {}
_evaluation_jobs_lock = threading.RLock()
_evaluation_timeout_seconds = int(os.getenv("PIPIXIA_SKILL_EVAL_TIMEOUT_SECONDS", "3600"))


def _get_store(agent_id: str, agent_manager: Any):
    return agent_manager.mem_stores.get(agent_id)


class BoundaryReviewResolution(BaseModel):
    action: Literal["assign_current", "create_new", "assign_other", "orphan"]
    target_task_id: str | None = None
    note: str = ""


class EvolutionTaskCreate(BaseModel):
    task_family: str
    title: str = ""
    data_source: Literal["langfuse", "skilllearnbench"] = "langfuse"


class EvolutionExportRequest(BaseModel):
    trace_ids: list[str]


class EvolutionStageUpdate(BaseModel):
    stage: str
    note: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


class EvolutionDatasetSave(BaseModel):
    assignments: dict[str, list[str]]


@router.get("/mem/evolution/tasks")
async def mem_evolution_tasks(agent_id: str = Query("main")):
    return {"ok": True, "tasks": _evolution_workflow.list(agent_id)}


@router.post("/mem/evolution/tasks")
async def create_mem_evolution_task(
    request: EvolutionTaskCreate, agent_id: str = Query("main"),
):
    if request.data_source == "skilllearnbench":
        from evaluation.skilllearnbench_dataset_importer import list_families
        available = {row["family"] for row in list_families()}
        if request.task_family not in available:
            raise HTTPException(status_code=422, detail="SkillLearnBench 任务族不存在")
    return {"ok": True, "task": _evolution_workflow.create(
        agent_id=agent_id, task_family=request.task_family, title=request.title,
        data_source=request.data_source,
    )}


@router.get("/mem/evolution/skilllearnbench/families")
async def mem_evolution_skilllearnbench_families():
    try:
        from evaluation.skilllearnbench_dataset_importer import list_families
        return {"ok": True, "families": list_families()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/mem/evolution/tasks/{task_id}")
async def mem_evolution_task(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    return {"ok": True, "task": task}


@router.post("/mem/evolution/tasks/{task_id}/stage")
async def update_mem_evolution_stage(
    task_id: str, request: EvolutionStageUpdate, agent_id: str = Query("main"),
):
    if request.stage in {"approved", "published"}:
        raise HTTPException(status_code=403, detail="发布状态必须通过发布门禁和人工审核接口变更")
    try:
        task = _evolution_workflow.transition(
            task_id, agent_id=agent_id, stage=request.stage,
            note=request.note, artifacts=request.artifacts,
        )
        return {"ok": True, "task": task}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="evolution task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/mem/evolution/tasks/{task_id}/export")
async def export_mem_evolution_task(
    task_id: str,
    request: EvolutionExportRequest,
    agent_id: str = Query("main"),
):
    """Export Langfuse traces for one workflow task, then advance its stage."""
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "created":
        raise HTTPException(status_code=409, detail="data export is only available from created stage")
    if task.get("data_source", "langfuse") != "langfuse":
        raise HTTPException(status_code=409, detail="当前任务使用 SkillLearnBench 数据源")
    trace_ids = list(dict.fromkeys(value.strip() for value in request.trace_ids if value.strip()))
    if not trace_ids:
        raise HTTPException(status_code=422, detail="请至少选择一条轨迹")
    if len(trace_ids) > 200:
        raise HTTPException(status_code=422, detail="单次最多选择 200 条轨迹")
    _evolution_workflow.record_execution(
        task_id, agent_id=agent_id, operation="export_traces", status="running",
    )
    try:
        from evaluation.langfuse_export import _build_client, _build_memory_store
        from evaluation.langfuse_exporter import export_traces, fetch_langfuse_traces_by_ids
        import asyncio

        output = DATA_DIR / "evaluation" / "skill_evolution" / task_id / "traces.jsonl"
        client = _build_client()
        traces = await asyncio.to_thread(fetch_langfuse_traces_by_ids, client, trace_ids)
        memory_store = _build_memory_store()
        try:
            count = await asyncio.to_thread(export_traces, traces, output, memory_store=memory_store)
        finally:
            memory_store.close()
        updated = _evolution_workflow.transition(
            task_id, agent_id=agent_id, stage="data_exported",
            note=f"导出 {count} 条 Langfuse 样本",
            artifacts={"traces": str(output), "sample_count": str(count)},
        )
        updated = _evolution_workflow.record_execution(
            task_id, agent_id=agent_id, operation="export_traces", status="succeeded",
            result={"sample_count": count, "output_path": str(output), "trace_ids": trace_ids},
        )
        return {"ok": True, "task": updated, "sample_count": count}
    except Exception as exc:
        _evolution_workflow.record_execution(
            task_id, agent_id=agent_id, operation="export_traces", status="failed",
            error=str(exc),
        )
        logger.warning("mem_evolution_export error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/mem/evolution/tasks/{task_id}/traces")
async def list_mem_evolution_traces(
    task_id: str,
    agent_id: str = Query("main"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    session_id: str = Query(""),
    name: str = Query(""),
):
    """List redacted Langfuse summaries without exporting or changing state."""
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "created":
        raise HTTPException(status_code=409, detail="trace selection is only available before export")
    if task.get("data_source", "langfuse") != "langfuse":
        raise HTTPException(status_code=409, detail="当前任务使用 SkillLearnBench 数据源")
    try:
        from evaluation.langfuse_export import _build_client
        from evaluation.langfuse_exporter import fetch_langfuse_traces, trace_preview
        import asyncio

        filters = {key: value for key, value in {"session_id": session_id, "name": name}.items() if value}
        rows = await asyncio.to_thread(
            fetch_langfuse_traces, _build_client(), limit=limit, max_pages=page, **filters,
        )
        page_rows = rows[(page - 1) * limit:page * limit]
        return {"ok": True, "traces": [trace_preview(row) for row in page_rows], "page": page, "has_more": len(rows) >= page * limit}
    except Exception as exc:
        logger.warning("mem_evolution_traces error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/mem/evolution/tasks/{task_id}/skilllearnbench/prepare")
async def prepare_mem_evolution_skilllearnbench(
    task_id: str, agent_id: str = Query("main"),
):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "created":
        raise HTTPException(status_code=409, detail="标准任务集只能在数据准备阶段执行")
    if task.get("data_source") != "skilllearnbench":
        raise HTTPException(status_code=409, detail="当前任务不是 SkillLearnBench 数据源")
    _evolution_workflow.record_execution(
        task_id, agent_id=agent_id, operation="prepare_skilllearnbench", status="running",
    )
    try:
        import asyncio
        from evaluation.skilllearnbench_dataset_importer import prepare_family

        result = await asyncio.to_thread(
            prepare_family, _dataset_root(task_id), str(task["task_family"]),
        )
        updated = _evolution_workflow.transition(
            task_id, agent_id=agent_id, stage="data_exported",
            note=f"已准备 {result['sample_count']} 个标准实例，Seed 用于生成 Candidate",
            artifacts={
                "traces": result["source_path"],
                "sample_count": str(result["sample_count"]),
                "seed_instance_id": result["seed_instance_id"],
                "benchmark_manifest": result["benchmark_manifest"],
                "benchmark_revision": result["benchmark_revision"],
            },
        )
        updated = _evolution_workflow.record_execution(
            task_id, agent_id=agent_id, operation="prepare_skilllearnbench",
            status="succeeded", result=result,
        )
        return {"ok": True, "task": updated, "result": result}
    except Exception as exc:
        _evolution_workflow.record_execution(
            task_id, agent_id=agent_id, operation="prepare_skilllearnbench",
            status="failed", error=str(exc),
        )
        logger.warning("mem_evolution_skilllearnbench_prepare error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _dataset_root(task_id: str):
    return DATA_DIR / "evaluation" / "skill_evolution" / task_id


def _dataset_draft(task: dict[str, Any]):
    from evaluation.skill_evolution_dataset_editor import load_or_create
    source = task.get("artifacts", {}).get("traces")
    if not source:
        raise HTTPException(status_code=409, detail="请先完成轨迹导出")
    path = Path(source)
    if not path.exists():
        raise HTTPException(status_code=404, detail="导出的轨迹文件不存在")
    return load_or_create(
        _dataset_root(task["id"]), task["id"], task["task_family"], path,
        source=str(task.get("data_source") or "langfuse"),
    )


@router.get("/mem/evolution/tasks/{task_id}/dataset")
async def get_mem_evolution_dataset(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] not in {"data_exported", "dataset_confirmed"}:
        raise HTTPException(status_code=409, detail="当前阶段不能编辑数据集")
    from evaluation.skill_evolution_dataset_editor import validate_draft
    draft = _dataset_draft(task)
    return {"ok": True, "draft": draft, "validation": validate_draft(draft)}


@router.post("/mem/evolution/tasks/{task_id}/dataset/save")
async def save_mem_evolution_dataset(task_id: str, request: EvolutionDatasetSave, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "data_exported":
        raise HTTPException(status_code=409, detail="数据集草稿只允许在导出后修改")
    from evaluation.skill_evolution_dataset_editor import save, validate_draft
    draft = _dataset_draft(task)
    draft["assignments"] = request.assignments
    save(_dataset_root(task_id), draft)
    return {"ok": True, "draft": draft, "validation": validate_draft(draft)}


@router.post("/mem/evolution/tasks/{task_id}/dataset/confirm")
async def confirm_mem_evolution_dataset(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "data_exported":
        raise HTTPException(status_code=409, detail="数据集只能从轨迹已导出阶段确认")
    from evaluation.skill_evolution_dataset_editor import confirm
    draft = _dataset_draft(task)
    confirmed, errors = confirm(_dataset_root(task_id), draft)
    if errors:
        raise HTTPException(status_code=422, detail={"message": "数据集校验未通过", "errors": errors})
    updated = _evolution_workflow.transition(
        task_id, agent_id=agent_id, stage="dataset_confirmed", note="人工确认并冻结 v1 数据集",
        artifacts={"dataset_manifest": str(_dataset_root(task_id) / "datasets" / "v1" / "manifest.json"), "dataset_samples": str(_dataset_root(task_id) / "datasets" / "v1" / "samples.jsonl"), "dataset_version": "v1"},
    )
    return {"ok": True, "task": updated, "draft": confirmed}


@router.get("/mem/evolution/tasks/{task_id}/candidate")
async def get_mem_evolution_candidate(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    from evaluation.skill_evolution_candidate_editor import load_result
    result = None
    if task.get("artifacts", {}).get("candidate_manifest"):
        manifest_path = Path(task["artifacts"]["candidate_manifest"])
        if manifest_path.exists():
            result = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = {**result, "manifest_path": str(manifest_path), "checks": result.get("checks", {})}
    if result is None:
        result = load_result(_dataset_root(task_id))
    if result and Path(result["skill_path"]).exists():
        result["content"] = Path(result["skill_path"]).read_text(encoding="utf-8")
    return {"ok": True, "candidate": result}


@router.post("/mem/evolution/tasks/{task_id}/candidate/generate")
async def generate_mem_evolution_candidate(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] not in {"dataset_confirmed", "candidate_generated"}:
        raise HTTPException(status_code=409, detail="请先确认并冻结数据集")
    _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="generate_candidate", status="running")
    try:
        from evaluation.skill_evolution_candidate_editor import generate
        result = await generate(_dataset_root(task_id), task)
        task = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="candidate_generated", note=f"已生成 Candidate v{result['version']} 并完成静态检查", artifacts={"candidate_manifest": result["manifest_path"], "candidate_skill": result["skill_path"], "candidate_version": f"v{result['version']}"})
        task = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="generate_candidate", status="succeeded", result={"candidate_id": result["candidate_id"], "static_check_passed": result["checks"]["passed"], "static_check_reasons": result["checks"]["reasons"]})
        result["content"] = Path(result["skill_path"]).read_text(encoding="utf-8")
        return {"ok": True, "task": task, "candidate": result}
    except Exception as exc:
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="generate_candidate", status="failed", error=str(exc))
        logger.warning("mem_evolution_candidate error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/mem/evolution/tasks/{task_id}/candidate/confirm")
async def confirm_mem_evolution_candidate(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "candidate_generated":
        raise HTTPException(status_code=409, detail="当前没有待确认的 Candidate")
    from evaluation.skill_candidate import check_candidate
    skill_path = Path(task.get("artifacts", {}).get("candidate_skill", ""))
    check = check_candidate(skill_path)
    if not check.passed:
        raise HTTPException(status_code=422, detail={"message": "静态检查未通过", "errors": list(check.reasons)})
    version = task.get("artifacts", {}).get("candidate_version", "")
    updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="candidate_confirmed", note=f"人工确认 Candidate {version}".strip())
    return {"ok": True, "task": updated}


@router.get("/mem/evolution/tasks/{task_id}/dev")
async def get_mem_evolution_dev(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    report_path = _latest_report(_dataset_root(task_id), "dev")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return {"ok": True, "report": report}


@router.post("/mem/evolution/tasks/{task_id}/dev/run")
async def run_mem_evolution_dev(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] not in {"candidate_confirmed", "dev_evaluated"}:
        raise HTTPException(status_code=409, detail="请先确认 Candidate")
    job = _start_evaluation_job(task, split="dev")
    return {"ok": True, "task": _evolution_workflow.get(task_id, agent_id), "job": job}


@router.post("/mem/evolution/tasks/{task_id}/dev/confirm")
async def confirm_mem_evolution_dev(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "dev_evaluated":
        raise HTTPException(status_code=409, detail="请先完成开发集评估")
    report_path = _latest_report(_dataset_root(task_id), "dev")
    if not report_path.exists():
        raise HTTPException(status_code=409, detail="开发集报告不存在")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    external = any(float(row.get("external_failures", 0)) > 0 for row in report.get("systems", []))
    if external:
        raise HTTPException(status_code=422, detail="存在外部失败，不能确认开发评估")
    from evaluation.skill_evolution_runner import gate_report
    gate = gate_report(report_path, regression_candidate_passed=None)
    next_stage = "regression_pending" if gate["status"] == "pending_regression" else "revision_pending"
    note = "开发评估达标，进入回归验证" if next_stage == "regression_pending" else "开发评估未达标，进入失败归因与局部优化"
    updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage=next_stage, note=note, artifacts={"dev_gate": str(_dataset_root(task_id) / "reports" / "dev-gate.json")})
    (_dataset_root(task_id) / "reports" / "dev-gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "task": updated, "gate": gate}


def _split_report(task_id: str, split: str) -> Path:
    return _latest_report(_dataset_root(task_id), split)


def _latest_report(root: Path, split: str) -> Path:
    reports_dir = root / "reports"
    reports = [
        *reports_dir.glob(f"{split}-candidate-v*.json"),
        *reports_dir.glob(f"{split}-v*.json"),
    ]
    reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return reports[0] if reports else root / "reports" / f"{split}-v1.json"


def _stop_evaluation_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, text=True, check=False)
    else:
        process.terminate()


def _monitor_evaluation_job(task_id: str, agent_id: str, split: str, process: subprocess.Popen[str], log_handle: Any) -> None:
    operation = f"{split}_evaluation"
    try:
        try:
            exit_code = process.wait(timeout=_evaluation_timeout_seconds)
        except subprocess.TimeoutExpired:
            _stop_evaluation_process(process)
            _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=operation, status="failed", error=f"评估超过 {_evaluation_timeout_seconds} 秒，已终止")
            return
        task = _evolution_workflow.get(task_id, agent_id)
        if task is None or task.get("execution", {}).get("status") == "cancelled":
            return
        if exit_code != 0:
            _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=operation, status="failed", error=f"评估进程退出，返回码 {exit_code}；请查看阶段日志")
            return
        report_path = _latest_report(_dataset_root(task_id), split)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if split == "dev" and task["stage"] == "candidate_confirmed":
            _evolution_workflow.transition(task_id, agent_id=agent_id, stage="dev_evaluated", note="开发集三版本对照评估完成", artifacts={"dev_report": str(report_path)})
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=operation, status="succeeded", result={"systems": report.get("systems", []), "case_count": len(report.get("cases", [])), "report_path": str(report_path)})
    except Exception as exc:
        logger.exception("evaluation job monitor failed")
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=operation, status="failed", error=str(exc))
    finally:
        log_handle.close()
        with _evaluation_jobs_lock:
            if _evaluation_jobs.get(task_id) is process:
                _evaluation_jobs.pop(task_id, None)


def _start_evaluation_job(task: dict[str, Any], *, split: str, manifest_path: Path | None = None) -> dict[str, Any]:
    task_id = str(task["id"])
    agent_id = str(task.get("agent_id") or "main")
    operation = f"{split}_evaluation"
    with _evaluation_jobs_lock:
        stored_execution = task.get("execution", {}) if isinstance(task.get("execution"), dict) else {}
        stored_result = stored_execution.get("result", {}) if isinstance(stored_execution.get("result"), dict) else {}
        stored_pid = int(stored_result.get("pid", 0) or 0)
        if stored_execution.get("status") == "running" and stored_pid:
            try:
                os.kill(stored_pid, 0)
            except OSError:
                _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=str(stored_execution.get("operation") or operation), status="cancelled", error="后端重启后未发现原评估进程，已清除运行状态")
            else:
                raise HTTPException(status_code=409, detail="该进化任务正在执行，请勿重复提交")
        current = _evaluation_jobs.get(task_id)
        if current is not None and current.poll() is None:
            raise HTTPException(status_code=409, detail="该进化任务正在执行，请勿重复提交")
        root = _dataset_root(task_id)
        jobs_dir = root / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        snapshot = jobs_dir / f"{split}-task.json"
        snapshot.write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        version = str(task.get("artifacts", {}).get("candidate_version") or "v1")
        report_name = f"{split}-candidate-{version}.json"
        log_path = jobs_dir / f"{split}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        command = [sys.executable, "-u", "-m", "evaluation.skill_evolution_job", "--root", str(root), "--task", str(snapshot), "--split", split, "--report-name", report_name]
        if manifest_path is not None:
            command.extend(["--manifest", str(manifest_path)])
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process_env = os.environ.copy()
        process_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=log_handle, stderr=subprocess.STDOUT, text=True, creationflags=creationflags, env=process_env)
        _evaluation_jobs[task_id] = process
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=operation, status="running", result={"pid": process.pid, "phase": "starting", "log_path": str(log_path), "report_name": report_name})
        threading.Thread(target=_monitor_evaluation_job, args=(task_id, agent_id, split, process, log_handle), daemon=True).start()
    return {"pid": process.pid, "log_path": str(log_path), "report_name": report_name}


@router.get("/mem/evolution/tasks/{task_id}/execution")
async def get_mem_evolution_execution(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    execution = task.get("execution", {})
    log_path = Path(str((execution.get("result") or {}).get("log_path") or ""))
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:] if log_path.is_file() else []
    log_updated_at = datetime.fromtimestamp(log_path.stat().st_mtime, timezone.utc).isoformat() if log_path.is_file() else None
    pid = int((execution.get("result") or {}).get("pid", 0) or 0)
    process_alive = False
    if execution.get("status") == "running" and pid:
        try:
            os.kill(pid, 0)
        except OSError:
            pass
        else:
            process_alive = True
    if execution.get("status") == "running" and pid and not process_alive:
        report_name = str((execution.get("result") or {}).get("report_name") or "")
        report_path = _dataset_root(task_id) / "reports" / report_name
        if report_name and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if str(execution.get("operation")) == "dev_evaluation" and task.get("stage") == "candidate_confirmed":
                _evolution_workflow.transition(task_id, agent_id=agent_id, stage="dev_evaluated", note="开发集三版本对照评估完成", artifacts={"dev_report": str(report_path)})
            task = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=str(execution.get("operation") or "evaluation"), status="succeeded", result={**(execution.get("result") or {}), "systems": report.get("systems", []), "case_count": len(report.get("cases", [])), "report_path": str(report_path)})
        else:
            task = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=str(execution.get("operation") or "evaluation"), status="failed", error="评估进程已退出且未生成报告，请查看阶段日志")
        execution = task.get("execution", {})
    phase = "等待执行"
    for line in reversed(lines):
        if "报告已生成" in line:
            phase = "生成评估报告"
            break
        if "verifier" in line.lower() or "[5/5]" in line:
            phase = "运行验证器"
            break
        if "Running agent" in line or "[4/5]" in line:
            phase = "执行 Agent"
            break
        if "Starting container" in line or "[2/5]" in line:
            phase = "启动测试容器"
            break
        if "[Phase 1]" in line or "BUILD" in line:
            phase = "构建或复用 Docker 镜像"
            break
        if line.startswith("[PIPIXIA_STAGE] "):
            phase = line.removeprefix("[PIPIXIA_STAGE] ")
            break
    return {"ok": True, "execution": execution, "phase": phase, "logs": lines, "log_updated_at": log_updated_at, "process_alive": process_alive}


@router.post("/mem/evolution/tasks/{task_id}/execution/cancel")
async def cancel_mem_evolution_execution(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    with _evaluation_jobs_lock:
        process = _evaluation_jobs.get(task_id)
    if process is None or process.poll() is not None:
        if task.get("execution", {}).get("status") == "running":
            updated = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=str(task["execution"].get("operation") or "evaluation"), status="cancelled", error="评估进程已不存在，已清除运行状态")
            return {"ok": True, "task": updated}
        raise HTTPException(status_code=409, detail="当前没有正在运行的评估")
    _stop_evaluation_process(process)
    updated = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation=str(task["execution"].get("operation") or "evaluation"), status="cancelled", error="用户取消评估")
    return {"ok": True, "task": updated}


@router.get("/mem/evolution/tasks/{task_id}/validation/{split}")
async def get_mem_evolution_validation(task_id: str, split: Literal["regression", "holdout"], agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    path = _split_report(task_id, split)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {"ok": True, "report": report}


@router.post("/mem/evolution/tasks/{task_id}/validation/{split}/run")
async def run_mem_evolution_validation(task_id: str, split: Literal["regression", "holdout"], agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    required_stage = "regression_pending" if split == "regression" else "holdout_pending"
    if task["stage"] != required_stage:
        raise HTTPException(status_code=409, detail=f"当前阶段不能执行 {split} 验证")
    root = _dataset_root(task_id)
    manifest_path = root / "datasets" / "v1" / "manifest.json"
    if split == "regression":
        revised = root / "datasets" / "v1" / "manifest-with-regression.json"
        if revised.exists():
            manifest_path = revised
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        family = next((row for row in manifest.get("families", []) if row.get("task_family") == task["task_family"]), {})
        if not family.get("splits", {}).get("regression"):
            report_name = f"regression-candidate-{task.get('artifacts', {}).get('candidate_version', 'v1')}.json"
            path = root / "reports" / report_name
            path.parent.mkdir(parents=True, exist_ok=True)
            report = {"schema_version": "1.0", "stage": split, "task_family": task["task_family"], "status": "not_applicable", "reason": "本轮没有历史失败样本", "systems": [], "cases": [], "report_path": str(path)}
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            updated = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="regression_evaluation", status="succeeded", result={"systems": [], "case_count": 0, "report_path": str(path)})
            return {"ok": True, "task": updated, "report": report}
    job = _start_evaluation_job(task, split=split, manifest_path=manifest_path if split == "regression" else None)
    return {"ok": True, "task": _evolution_workflow.get(task_id, agent_id), "job": job}


@router.post("/mem/evolution/tasks/{task_id}/validation/{split}/confirm")
async def confirm_mem_evolution_validation(task_id: str, split: Literal["regression", "holdout"], agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    required_stage = "regression_pending" if split == "regression" else "holdout_pending"
    if task["stage"] != required_stage:
        raise HTTPException(status_code=409, detail=f"当前阶段不能确认 {split} 结果")
    path = _split_report(task_id, split)
    if not path.exists():
        raise HTTPException(status_code=409, detail=f"{split} 报告不存在")
    report = json.loads(path.read_text(encoding="utf-8"))
    candidate = next((row for row in report.get("systems", []) if row.get("system") == "candidate_skill"), None)
    external = any(int(row.get("external_failures", 0) or 0) > 0 for row in report.get("systems", []))
    if external:
        raise HTTPException(status_code=422, detail=f"{split} 验证未通过，需返回失败归因")
    if report.get("status") != "not_applicable" and not candidate:
        raise HTTPException(status_code=422, detail=f"{split} 缺少 Candidate 结果")
    if split == "holdout":
        from evaluation.skill_evolution_runner import gate_report
        gate = gate_report(path, regression_candidate_passed=True)
        if gate["status"] != "accepted":
            raise HTTPException(status_code=422, detail={"message": "留出集未通过发布门禁", "gate": gate})
    next_stage = "regression_verified" if split == "regression" else "holdout_verified"
    updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage=next_stage, note=f"{split} 验证已人工确认", artifacts={f"{split}_report": str(path)})
    if split == "regression":
        updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="holdout_pending", note="进入独立留出集验收")
    return {"ok": True, "task": updated}


@router.post("/mem/evolution/tasks/{task_id}/approve")
async def approve_mem_evolution_task(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "holdout_verified":
        raise HTTPException(status_code=409, detail="请先完成留出集验收")
    updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="approved", note="人工审核批准发布")
    return {"ok": True, "task": updated}


@router.post("/mem/evolution/tasks/{task_id}/publish")
async def publish_mem_evolution_task(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "approved":
        raise HTTPException(status_code=409, detail="请先完成人工审核")
    manifest_path = Path(task.get("artifacts", {}).get("candidate_manifest", ""))
    if not manifest_path.is_file():
        raise HTTPException(status_code=409, detail="Candidate 清单不存在")
    from config import resolve_agent_skills_dir
    from mem.skill_version_store import SkillVersion, SkillVersionStore
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = SkillVersionStore(evolution_root=_dataset_root(task_id), active_root=resolve_agent_skills_dir(agent_id))
    store.publish(SkillVersion(skill_id=str(candidate["skill_id"]), version=int(candidate["version"]), status="accepted", source="offline_evolution", reason="holdout gate and human approval passed"))
    updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="published", note=f"Candidate v{candidate['version']} 已发布")
    return {"ok": True, "task": updated, "skill_id": candidate["skill_id"], "version": candidate["version"]}


@router.get("/mem/evolution/tasks/{task_id}/revision")
async def get_mem_evolution_revision(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    root = _dataset_root(task_id)
    attribution_path = root / "reports" / "failure-attribution.json"
    revision_path = root / "reports" / "revision-result.json"
    iteration_path = root / "reports" / "iteration-state.json"
    return {"ok": True, "attribution": json.loads(attribution_path.read_text(encoding="utf-8")) if attribution_path.exists() else None, "revision": json.loads(revision_path.read_text(encoding="utf-8")) if revision_path.exists() else None, "iteration": json.loads(iteration_path.read_text(encoding="utf-8")) if iteration_path.exists() else None}


@router.post("/mem/evolution/tasks/{task_id}/revision/analyze")
async def analyze_mem_evolution_revision(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "revision_pending":
        raise HTTPException(status_code=409, detail="请先确认开发评估结果")
    from evaluation.skill_evolution_revision_editor import attribution
    try:
        result = attribution(_dataset_root(task_id))
        return {"ok": True, "attribution": result}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/mem/evolution/tasks/{task_id}/revision/run")
async def run_mem_evolution_revision(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "revision_pending":
        raise HTTPException(status_code=409, detail="当前阶段不能执行局部修改")
    _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="local_revision", status="running")
    try:
        from evaluation.skill_evolution_revision_editor import create_revision
        result = await create_revision(_dataset_root(task_id), task)
        updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="candidate_generated", note="已生成局部修改 Candidate，等待重新确认", artifacts={"candidate_manifest": result["candidate_manifest"], "candidate_skill": result["skill_path"], "candidate_version": str(result["version"]), "revision_result": str(_dataset_root(task_id) / "reports" / "revision-result.json")})
        updated = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="local_revision", status="succeeded", result=result)
        return {"ok": True, "task": updated, "revision": result}
    except Exception as exc:
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="local_revision", status="failed", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/mem/evolution/tasks/{task_id}/revision/iterate")
async def iterate_mem_evolution_revision(task_id: str, agent_id: str = Query("main")):
    task = _evolution_workflow.get(task_id, agent_id)
    if task is None:
        raise HTTPException(status_code=404, detail="evolution task not found")
    if task["stage"] != "revision_pending":
        raise HTTPException(status_code=409, detail="请先确认开发评估结果")
    _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="bounded_revision_iteration", status="running")
    try:
        from evaluation.skill_evolution_revision_editor import iterate
        result = await iterate(_dataset_root(task_id), task)
        candidate_manifest = result.get("candidate_manifest")
        artifacts = {"iteration_state": str(_dataset_root(task_id) / "reports" / "iteration-state.json")}
        if candidate_manifest:
            candidate_path = Path(candidate_manifest)
            if candidate_path.exists():
                data = json.loads(candidate_path.read_text(encoding="utf-8"))
                artifacts.update({"candidate_manifest": str(candidate_path), "candidate_skill": str(data.get("skill_path", "")), "candidate_version": str(data.get("version", ""))})
        if result.get("status") == "abandoned":
            updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="abandoned", note="自动迭代达到上限或连续无提升，舍弃本次进化", artifacts=artifacts)
        elif result.get("status") == "stopped" and not result.get("iterations"):
            updated = task
        else:
            updated = _evolution_workflow.transition(task_id, agent_id=agent_id, stage="candidate_generated", note=f"自动迭代结束: {result.get('status')}", artifacts=artifacts)
        updated = _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="bounded_revision_iteration", status="succeeded", result=result)
        return {"ok": True, "task": updated, "iteration": result}
    except Exception as exc:
        _evolution_workflow.record_execution(task_id, agent_id=agent_id, operation="bounded_revision_iteration", status="failed", error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ------------------------------------------------------------------
# GET /api/mem/stats
# ------------------------------------------------------------------

@router.get("/mem/stats")
async def mem_stats(
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "error": "mem system not initialized"}
    try:
        return {"ok": True, **store.get_dashboard_stats()}
    except Exception as e:
        logger.warning("mem_stats error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/tasks
# ------------------------------------------------------------------

@router.get("/mem/tasks")
async def mem_tasks(
    agent_id: str = Query("main"),
    status: str = Query("", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "tasks": [], "total": 0}
    try:
        items, total = store.list_dashboard_tasks(
            status=status, limit=limit, offset=offset
        )
        return {"ok": True, "tasks": items, "total": total}
    except Exception as e:
        logger.warning("mem_tasks error: %s", e)
        return {"ok": False, "tasks": [], "total": 0, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/task/{task_id}
# ------------------------------------------------------------------

@router.get("/mem/task/{task_id}")
async def mem_task_detail(
    task_id: str,
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "error": "not initialized"}
    try:
        task = store.get_task(task_id)
        if not task:
            return {"ok": False, "error": "not found"}
        chunks = store.get_chunks_by_task(task_id, limit=100)
        chunk_items = []
        for c in chunks:
            text = c.content
            if len(text) > 500:
                text = text[:497] + "..."
            chunk_items.append({
                "id": c.id,
                "role": c.role,
                "content": text,
                "summary": c.summary,
                "createdAt": c.created_at,
            })
        return {
            "ok": True,
            "id": task.id,
            "sessionKey": task.session_key,
            "title": task.title,
            "summary": task.summary,
            "status": task.status,
            "startedAt": task.started_at,
            "endedAt": task.ended_at,
            "chunks": chunk_items,
        }
    except Exception as e:
        logger.warning("mem_task_detail error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# Boundary reviews
# ------------------------------------------------------------------

@router.get("/mem/boundary-reviews")
async def mem_boundary_reviews(
    agent_id: str = Query("main"),
    limit: int = Query(100, ge=1, le=200),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "reviews": [], "total": 0, "error": "not initialized"}
    try:
        reviews = store.list_pending_boundary_reviews(agent_id, limit)
        items = []
        for review in reviews:
            task = store.get_task(review.current_task_id)
            turn_chunks = store.get_chunks_by_turn(review.session_key, review.turn_id)
            task_chunks = store.get_chunks_by_task(review.current_task_id, limit=100)
            items.append({
                "id": review.id,
                "sessionKey": review.session_key,
                "currentTaskId": review.current_task_id,
                "currentTaskTitle": task.title if task else "",
                "currentTaskSummary": (
                    (task.boundary_summary or task.summary) if task else ""
                ),
                "turnId": review.turn_id,
                "confidence": review.confidence,
                "reason": review.reason,
                "retryCount": review.retry_count,
                "createdAt": review.created_at,
                "recentContext": [{
                    "id": chunk.id,
                    "role": chunk.role,
                    "content": chunk.content[:500],
                } for chunk in task_chunks[-4:]],
                "pendingChunks": [{
                    "id": chunk.id,
                    "role": chunk.role,
                    "content": chunk.content[:1000],
                } for chunk in turn_chunks],
            })
        return {"ok": True, "reviews": items, "total": len(items)}
    except Exception as e:
        logger.warning("mem_boundary_reviews error: %s", e)
        return {"ok": False, "reviews": [], "total": 0, "error": str(e)}


@router.post("/mem/boundary-reviews/{review_id}/resolve")
async def resolve_mem_boundary_review(
    review_id: str,
    request: BoundaryReviewResolution,
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    processor = getattr(agent_manager, "mem_task_processors", {}).get(agent_id)
    if not processor:
        raise HTTPException(status_code=503, detail="mem task processor not initialized")
    if request.action == "assign_other" and not request.target_task_id:
        raise HTTPException(status_code=422, detail="target_task_id is required")
    try:
        review = await processor.resolve_boundary_review(
            review_id,
            action=request.action,
            target_task_id=request.target_task_id,
            note=request.note,
        )
        return {
            "ok": True,
            "id": review.id,
            "status": review.status,
            "resolution": review.resolution,
            "targetTaskId": review.target_task_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("resolve_mem_boundary_review error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ------------------------------------------------------------------
# GET /api/mem/skills
# ------------------------------------------------------------------

@router.get("/mem/skills")
async def mem_skills(
    agent_id: str = Query("main"),
    status: str = Query("", description="Filter by status"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "skills": []}
    try:
        items = store.list_dashboard_skills(status=status)
        return {"ok": True, "skills": items}
    except Exception as e:
        logger.warning("mem_skills error: %s", e)
        return {"ok": False, "skills": [], "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/skill/{skill_id}
# ------------------------------------------------------------------

@router.get("/mem/skill/{skill_id}")
async def mem_skill_detail(
    skill_id: str,
    agent_id: str = Query("main"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "error": "not initialized"}
    try:
        skill = store.get_skill(skill_id)
        if not skill:
            return {"ok": False, "error": "not found"}
        return {
            "ok": True,
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "version": skill.version,
            "status": skill.status,
            "qualityScore": skill.quality_score,
            "dirPath": skill.dir_path,
            "createdAt": skill.created_at,
            "updatedAt": skill.updated_at,
        }
    except Exception as e:
        logger.warning("mem_skill_detail error: %s", e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/memories
# ------------------------------------------------------------------

@router.get("/mem/memories")
async def mem_memories(
    agent_id: str = Query("main"),
    limit: int = Query(40, ge=1, le=200),
    page: int = Query(1, ge=1),
    session: str = Query("", description="Filter by session_key"),
    role: str = Query("", description="Filter by role"),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store:
        return {"ok": False, "memories": [], "total": 0}
    try:
        offset = (page - 1) * limit
        items, total = store.list_dashboard_memories(
            limit=limit,
            offset=offset,
            session=session,
            role=role,
        )
        return {
            "ok": True,
            "memories": items,
            "total": total,
            "page": page,
            "totalPages": max(1, -(-total // limit)),
        }
    except Exception as e:
        logger.warning("mem_memories error: %s", e)
        return {"ok": False, "memories": [], "total": 0, "error": str(e)}


# ------------------------------------------------------------------
# GET /api/mem/search
# ------------------------------------------------------------------

@router.get("/mem/search")
async def mem_search(
    agent_id: str = Query("main"),
    q: str = Query("", description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    agent_manager: Any = Depends(get_agent_manager),
):
    store = _get_store(agent_id, agent_manager)
    if not store or not q.strip():
        return {"ok": True, "results": [], "query": q}
    try:
        fts_hits = store.fts_search_chunks(q, limit=limit)
        results = []
        for h in fts_hits:
            results.append({
                "id": h.chunk_id,
                "score": round(h.score, 3),
                "role": h.role,
                "summary": h.summary,
                "excerpt": h.content_excerpt[:300],
                "sessionKey": h.session_key,
                "taskId": h.task_id,
                "createdAt": h.created_at,
            })
        return {"ok": True, "results": results, "query": q, "total": len(results)}
    except Exception as e:
        logger.warning("mem_search error: %s", e)
        return {"ok": True, "results": [], "query": q, "error": str(e)}
