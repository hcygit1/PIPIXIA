"""Editable Seed/Dev/Holdout dataset drafts for the offline evolution UI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.skill_evolution_dataset import validate_sample

SPLITS = ("seed", "dev", "holdout", "excluded")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_samples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _suggest(rows: list[dict[str, Any]], *, source: str) -> dict[str, list[str]]:
    rows = sorted(rows, key=lambda row: str(row.get("sample_id", "")))
    valid = [row for row in rows if not validate_sample(row)]
    if source == "skilllearnbench":
        seed = [str(row["sample_id"]) for row in valid if row.get("category") == "seed"][:1]
    else:
        successful = [row for row in valid if row.get("category") == "success" and row.get("trajectory")]
        seed_count = min(len(successful), max(1, round(len(valid) * 0.3))) if successful else 0
        seed = [str(row["sample_id"]) for row in successful[:seed_count]]
    remaining = [row for row in valid if str(row["sample_id"]) not in seed]
    midpoint = (len(remaining) + 1) // 2
    return {
        "seed": seed,
        "dev": [str(row["sample_id"]) for row in remaining[:midpoint]],
        "holdout": [str(row["sample_id"]) for row in remaining[midpoint:]],
        "excluded": [str(row["sample_id"]) for row in rows if validate_sample(row)],
    }


def load_or_create(
    root: Path, task_id: str, task_family: str, source_path: Path,
    *, source: str = "langfuse",
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    draft_path = root / "dataset-draft.json"
    if draft_path.exists():
        return json.loads(draft_path.read_text(encoding="utf-8"))
    rows = _read_samples(source_path)
    draft = {
        "schema_version": "1.0",
        "task_id": task_id,
        "task_family": task_family,
        "source": source,
        "source_path": str(source_path),
        "created_at": _now(),
        "updated_at": _now(),
        "status": "draft",
        "samples": {str(row["sample_id"]): row for row in rows if row.get("sample_id")},
        "assignments": _suggest(rows, source=source),
    }
    save(root, draft)
    return draft


def save(root: Path, draft: dict[str, Any]) -> dict[str, Any]:
    draft["updated_at"] = _now()
    (root / "dataset-draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return draft


def validate_draft(draft: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    samples = draft.get("samples", {})
    assignments = draft.get("assignments", {})
    seen: set[str] = set()
    for split in SPLITS:
        for sample_id in assignments.get(split, []):
            if sample_id not in samples:
                errors.append(f"{split}: 未知样本 {sample_id}")
            if sample_id in seen:
                errors.append(f"样本重复分配: {sample_id}")
            seen.add(sample_id)
    for sample_id, row in samples.items():
        if sample_id not in seen:
            errors.append(f"样本未分配: {sample_id}")
        if sample_id in assignments.get("seed", []):
            if draft.get("source") == "skilllearnbench":
                if row.get("category") != "seed":
                    errors.append(f"SkillLearnBench Seed 必须是任务族第一个实例: {sample_id}")
            elif row.get("category") != "success" or not row.get("trajectory"):
                errors.append(f"Seed 必须是成功且包含完整轨迹的样本: {sample_id}")
    if not assignments.get("seed"):
        errors.append("Seed 至少需要一个生成样本")
    return errors


def confirm(root: Path, draft: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors = validate_draft(draft)
    if errors:
        return draft, errors
    version_dir = root / "datasets" / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    source = str(draft.get("source") or "langfuse")
    leakage_policy = (
        "SkillLearnBench Seed uses instance instruction and observable environment metadata only; "
        "tests, verifier source, solution, and human-authored skills are excluded."
        if source == "skilllearnbench"
        else "Seed may use successful trajectories; Dev/Holdout remain evaluation-only."
    )
    manifest = {"schema_version": "1.0", "dataset": "pipixia-skill-evolution", "source": source, "leakage_policy": leakage_policy, "families": [{"task_family": draft["task_family"], "splits": {split: draft["assignments"].get(split, []) for split in ("seed", "dev", "holdout", "regression")}}]}
    (version_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (version_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id, row in draft["samples"].items():
            handle.write(json.dumps({**row, "split": next((split for split in SPLITS if sample_id in draft["assignments"].get(split, [])), "excluded")}, ensure_ascii=False) + "\n")
    draft["status"] = "confirmed"
    draft["version"] = "v1"
    save(root, draft)
    return draft, []
