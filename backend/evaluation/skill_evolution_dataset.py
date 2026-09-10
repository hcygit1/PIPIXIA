"""Offline Skill-evolution dataset contract and validation utilities.

The dataset is JSONL (one sample per line).  A companion manifest references
sample IDs, keeping the raw trajectories immutable while allowing Dev and
Regression membership to change between iterations.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
SPLITS = ("seed", "dev", "holdout", "regression")
REQUIRED_FIELDS = ("sample_id", "task_snapshot", "trajectory")


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """Return validation errors for one offline evolution sample."""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in sample:
            errors.append(f"missing field: {field}")
    if not isinstance(sample.get("sample_id"), str) or not sample.get("sample_id", "").strip():
        errors.append("sample_id must be a non-empty string")
    if not isinstance(sample.get("trajectory"), (list, str, dict)):
        errors.append("trajectory must be a list, object, or string")
    split = sample.get("split")
    if split is not None and split not in SPLITS:
        errors.append(f"split must be one of: {', '.join(SPLITS)}")
    return errors


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load and validate JSONL samples, rejecting malformed input early."""
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            sample = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(sample, dict):
            raise ValueError(f"{path}:{line_number}: sample must be an object")
        errors = validate_sample(sample)
        if errors:
            raise ValueError(f"{path}:{line_number}: {'; '.join(errors)}")
        sample_id = str(sample["sample_id"])
        if sample_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        samples.append(sample)
    return samples


def build_manifest(
    samples: Iterable[dict[str, Any]],
    *,
    source: str = "langfuse",
    dataset_name: str = "pipixia-skill-evolution",
) -> dict[str, Any]:
    """Build a deterministic Seed/Dev/Holdout manifest.

    Explicit ``sample.split`` values are preserved.  For unlabelled samples,
    each task family is sorted by sample ID and assigned first to Seed, last
    to Holdout, and the middle to Dev.  Regression starts empty and is filled
    from Dev failures by the later evaluation stage.
    """
    rows = list(samples)
    for sample in rows:
        errors = validate_sample(sample)
        if errors:
            raise ValueError(f"sample {sample.get('sample_id', '<unknown>')}: {'; '.join(errors)}")

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in rows:
        families[str(sample.get("task_family") or "unclassified")].append(sample)

    family_entries: list[dict[str, Any]] = []
    for family in sorted(families):
        family_rows = sorted(families[family], key=lambda row: str(row["sample_id"]))
        assignments: dict[str, list[str]] = {split: [] for split in SPLITS}
        labelled = [row for row in family_rows if row.get("split")]
        if labelled:
            for row in labelled:
                assignments[str(row["split"])].append(str(row["sample_id"]))
            for row in family_rows:
                if not row.get("split"):
                    assignments["dev"].append(str(row["sample_id"]))
        else:
            ids = [str(row["sample_id"]) for row in family_rows]
            if ids:
                assignments["seed"].append(ids[0])
            if len(ids) > 1:
                assignments["holdout"].append(ids[-1])
            if len(ids) > 2:
                assignments["dev"].extend(ids[1:-1])
        family_entries.append({"task_family": family, "splits": assignments})

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "source": source,
        "leakage_policy": "Seed may use successful trajectories; Dev/Holdout/Regression remain evaluation-only.",
        "families": family_entries,
    }


def validate_manifest(manifest: dict[str, Any], sample_ids: set[str] | None = None) -> list[str]:
    """Validate manifest structure and optionally ensure all IDs exist."""
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version: {manifest.get('schema_version')}")
    families = manifest.get("families")
    if not isinstance(families, list) or not families:
        return [*errors, "families must be a non-empty list"]
    referenced: set[str] = set()
    for entry in families:
        if not isinstance(entry, dict) or not entry.get("task_family"):
            errors.append("each family requires task_family")
            continue
        splits = entry.get("splits")
        if not isinstance(splits, dict):
            errors.append(f"{entry['task_family']}: splits must be an object")
            continue
        for split in SPLITS:
            values = splits.get(split, [])
            if not isinstance(values, list):
                errors.append(f"{entry['task_family']}.{split} must be a list")
                continue
            for sample_id in values:
                if sample_id in referenced:
                    errors.append(f"sample appears in multiple splits: {sample_id}")
                referenced.add(sample_id)
    if sample_ids is not None:
        missing = sorted(referenced - sample_ids)
        if missing:
            errors.append(f"manifest references unknown samples: {', '.join(missing)}")
    return errors


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验并生成 Skill 进化离线数据集清单")
    parser.add_argument("--samples", type=Path, required=True, help="JSONL 样本文件")
    parser.add_argument("--manifest", type=Path, required=True, help="输出或校验的 Manifest")
    parser.add_argument("--check", action="store_true", help="只校验已有 Manifest")
    args = parser.parse_args(argv)
    samples = load_samples(args.samples)
    if args.check:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        errors = validate_manifest(manifest, {str(row["sample_id"]) for row in samples})
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Manifest 校验通过：{args.manifest}")
        return 0
    manifest = build_manifest(samples)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 Manifest：{args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
