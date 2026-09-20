"""Prepare SkillLearnBench instances for the staged evolution workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import DATA_DIR
from evaluation.skilllearnbench_adapter import SkillLearnInstance


DEFAULT_MANIFEST = DATA_DIR / "evaluation" / "skilllearnbench" / "manifest.json"


def load_benchmark_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"SkillLearnBench 清单不存在: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    benchmark_root = Path(str(manifest.get("benchmark_root", "")))
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"SkillLearnBench 仓库不存在: {benchmark_root}")
    return manifest


def list_families(path: Path = DEFAULT_MANIFEST) -> list[dict[str, Any]]:
    manifest = load_benchmark_manifest(path)
    result: list[dict[str, Any]] = []
    for entry in manifest.get("families", []):
        instances = [entry.get("seed_instance"), *entry.get("evaluation_instances", [])]
        rows = []
        for raw in instances:
            if not isinstance(raw, dict):
                continue
            instruction_path = Path(str(raw.get("instruction_path", "")))
            rows.append({
                "instance_id": str(raw.get("instance_id", "")),
                "role": "seed" if raw is instances[0] else "evaluation",
                "instruction_preview": (
                    instruction_path.read_text(encoding="utf-8", errors="replace")[:500]
                    if instruction_path.is_file() else ""
                ),
                "environment_available": (Path(str(raw.get("path", ""))) / "environment").is_dir(),
                "verifier_available": Path(str(raw.get("verifier_path", ""))).is_file(),
            })
        result.append({
            "family": str(entry.get("family", "")),
            "revision": str(manifest.get("revision", "")),
            "instances": rows,
        })
    return result


def _family_entry(manifest: dict[str, Any], family: str) -> dict[str, Any]:
    entry = next(
        (row for row in manifest.get("families", []) if str(row.get("family")) == family),
        None,
    )
    if entry is None:
        raise ValueError(f"SkillLearnBench 任务族不存在: {family}")
    return entry


def _environment_inventory(instance: SkillLearnInstance) -> list[dict[str, Any]]:
    """Expose observable input metadata without leaking tests or solutions."""
    environment = Path(instance.path) / "environment"
    if not environment.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in environment.rglob("*") if item.is_file()):
        relative = path.relative_to(environment)
        if relative.name == "Dockerfile" or relative.parts[0] == "skills":
            continue
        rows.append({"path": str(relative).replace("\\", "/"), "size_bytes": path.stat().st_size})
    return rows


def _sample(instance: SkillLearnInstance, *, role: str) -> dict[str, Any]:
    instruction = Path(instance.instruction_path).read_text(encoding="utf-8", errors="replace")
    return {
        "sample_id": instance.instance_id,
        "task_family": instance.family,
        "category": "seed" if role == "seed" else "evaluation",
        "task_snapshot": {
            "title": instance.instance_id,
            "instruction": instruction,
            "environment_inventory": _environment_inventory(instance),
        },
        "trajectory": [],
        "benchmark": {
            "name": "SkillLearnBench",
            "instance_id": instance.instance_id,
        },
    }


def prepare_family(
    root: Path,
    family: str,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    manifest = load_benchmark_manifest(manifest_path)
    entry = _family_entry(manifest, family)
    seed = SkillLearnInstance(**entry["seed_instance"])
    instances = [seed, *(SkillLearnInstance(**raw) for raw in entry.get("evaluation_instances", []))]
    rows = [
        _sample(instance, role="seed" if instance.instance_id == seed.instance_id else "evaluation")
        for instance in instances
    ]
    output = root / "traces.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "source_path": str(output),
        "sample_count": len(rows),
        "seed_instance_id": seed.instance_id,
        "benchmark_manifest": str(manifest_path),
        "benchmark_revision": str(manifest.get("revision", "")),
    }
