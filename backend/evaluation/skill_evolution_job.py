"""Isolated process entry point for one offline Skill evaluation stage."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evaluation.skill_evolution_dev_editor import run_dev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "regression", "holdout"), required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    task = json.loads(args.task.read_text(encoding="utf-8"))
    print(f"[PIPIXIA_STAGE] 准备 {args.split} 评估", flush=True)
    print("[PIPIXIA_STAGE] 构建或复用 Docker 镜像", flush=True)
    report = asyncio.run(run_dev(
        args.root,
        task,
        split=args.split,
        report_name=args.report_name,
        dataset_manifest_path=args.manifest,
    ))
    print(f"[PIPIXIA_STAGE] 报告已生成: {report['report_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
