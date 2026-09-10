"""Command-line export of Langfuse traces to Skill-evolution JSONL samples."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.langfuse_exporter import export_langfuse_traces


def _build_client():
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    try:
        from langfuse import Langfuse
    except ImportError as error:
        raise RuntimeError(
            "Langfuse SDK 未安装，请先安装 backend/requirements.txt"
        ) from error
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        raise RuntimeError("未找到 Langfuse 凭据，请配置 backend/.env")
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=os.getenv("LANGFUSE_HOST") or None,
    )


def _build_memory_store():
    from config import resolve_mem_config
    from mem.store import MemStore

    mem_config = resolve_mem_config()
    storage = mem_config.get("storage", {})
    embedding = mem_config.get("embedding", {})
    return MemStore(
        str(storage["db_path"]),
        dimensions=int(embedding.get("dimensions", 1536)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="导出 Langfuse Trace 为 PIPIXIA Skill 进化样本 JSONL"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSONL 输出路径")
    parser.add_argument("--session-id", help="按 Langfuse Session 筛选")
    parser.add_argument("--name", help="按 Trace 名称筛选")
    parser.add_argument("--from-timestamp", help="起始时间（ISO 8601）")
    parser.add_argument("--to-timestamp", help="结束时间（ISO 8601）")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1)
    args = parser.parse_args(argv)

    filters = {
        key: value
        for key, value in {
            "session_id": args.session_id,
            "name": args.name,
            "from_timestamp": args.from_timestamp,
            "to_timestamp": args.to_timestamp,
        }.items()
        if value
    }
    memory_store = _build_memory_store()
    try:
        count = export_langfuse_traces(
            _build_client(),
            args.output,
            limit=max(1, args.limit),
            max_pages=max(1, args.max_pages),
            memory_store=memory_store,
            **filters,
        )
    finally:
        memory_store.close()
    print(f"已导出 {count} 条样本：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
