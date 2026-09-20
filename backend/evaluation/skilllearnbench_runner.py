"""Prepare and distill PIPIXIA skills for SkillLearnBench."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
from contextlib import contextmanager
import json
import os
import shutil
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Any

from evaluation.skilllearnbench_adapter import (
    SkillLearnInstance,
    build_completed_task,
    build_manifest,
)
from mem.models import Chunk, Skill, SkillSearchHit
from mem.skill_evolver import MemSkillEvolver


class _EvalStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.skills: list[Skill] = []

    def get_chunks_by_task(self, task_id: str, limit: int | None = None) -> list[Chunk]:
        rows = [chunk for chunk in self.chunks if chunk.task_id == task_id]
        return rows if limit is None else rows[:limit]

    def fts_search_skills(self, query: str, limit: int = 10, owner: str | None = None) -> list[SkillSearchHit]:
        return []

    def ann_search_skills(self, query_vec: list[float], top_k: int = 5, owner: str | None = None) -> list[SkillSearchHit]:
        return []

    def get_skill(self, skill_id: str) -> Skill | None:
        return next((skill for skill in self.skills if skill.id == skill_id), None)

    def insert_skill(self, skill: Skill) -> None:
        self.skills.append(skill)

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        return None

    def update_skill(self, skill_id: str, **fields: Any) -> None:
        skill = self.get_skill(skill_id)
        if skill:
            for key, value in fields.items():
                setattr(skill, key, value)


class _EvalEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0]


async def distill(
    instance: SkillLearnInstance,
    trajectory: Path,
    output_root: Path,
    *,
    force_generate: bool = False,
    create_candidate: bool = False,
) -> dict[str, Any]:
    task, chunks = build_completed_task(instance, trajectory)
    store = _EvalStore(chunks)
    family_dir = output_root / instance.family
    evolver = MemSkillEvolver(
        store=store,
        embedder=_EvalEmbedder(),
        llm_base_url=os.getenv("PIPIXIA_LLM_BASE_URL", ""),
        llm_api_key=os.getenv("OPENAI_API_KEY", ""),
        llm_model=os.getenv("PIPIXIA_LLM_MODEL", "gpt-4o-mini"),
        skill_store_dir=str(family_dir),
        min_chunks_for_eval=6,
        min_confidence=0.7,
        enabled=True,
    )
    skip_reason = evolver._rule_filter(chunks, task)
    if skip_reason:
        return {"generated": False, "reason": skip_reason}
    eval_result = await evolver._evaluate_create(task)
    evaluation_raw = evolver.last_llm_response
    admitted = eval_result.should_generate and eval_result.confidence >= evolver.min_confidence
    if not admitted and not force_generate:
        return {
            "generated": False,
            "reason": eval_result.reason,
            "confidence": eval_result.confidence,
            "forced_generation": False,
            "evaluation_raw": evaluation_raw[:6000],
            "evaluation_response_present": bool(evaluation_raw),
        }
    skill = (
        await evolver._generate_candidate(task, chunks, eval_result)
        if create_candidate
        else await evolver._generate_skill(task, chunks, eval_result)
    )
    return {
        "generated": skill is not None,
        "reason": eval_result.reason,
        "confidence": eval_result.confidence,
        "forced_generation": force_generate and not admitted,
        "evaluation_should_generate": eval_result.should_generate,
        "evaluation_raw": evaluation_raw[:6000],
        "evaluation_response_present": bool(evaluation_raw),
        "skill": (
            {"name": skill.name, "path": skill.dir_path, "quality_score": skill.quality_score}
            if skill else None
        ),
    }


def _load_seed(manifest: dict[str, Any], family: str) -> SkillLearnInstance:
    entry = next(item for item in manifest["families"] if item["family"] == family)
    return SkillLearnInstance(**entry["seed_instance"])


def _load_official_evaluator(benchmark_root: Path) -> Any:
    root = str(benchmark_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    path = benchmark_root / "evaluate_skills.py"
    spec = importlib.util.spec_from_file_location("skilllearnbench_official_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bailian_agent_config() -> dict[str, Any]:
    source = Path(__file__).with_name("pipixia_bailian_agent.py").read_bytes()
    encoded = base64.b64encode(source).decode("ascii")
    install = (
        "(python3 -m pip install --break-system-packages 'openai>=1,<3' "
        "|| python3 -m pip install 'openai>=1,<3') && "
        "mkdir -p /opt/pipixia && "
        f"printf '%s' '{encoded}' | base64 -d > /opt/pipixia/bailian_agent.py"
    )
    return {
        "name": "PIPIXIA Bailian (OpenAI-compatible)",
        "env": ["OPENAI_API_KEY", "PIPIXIA_LLM_BASE_URL"],
        "runtime_deps": "",
        "install": install,
        "run": (
            "python3 /opt/pipixia/bailian_agent.py "
            "--instruction-file {instruction_file} --model {model} --max-steps {max_steps}"
        ),
        "trajectory_tee": "/logs/agent/pipixia-bailian.txt",
        "default_model": os.getenv("PIPIXIA_LLM_MODEL", "glm-5.2"),
    }


def _normalize_bailian_env() -> None:
    legacy_base_url = os.getenv("PIPIXIA_LLM_BASE_UR", "").strip()
    if not os.getenv("PIPIXIA_LLM_BASE_URL", "").strip() and legacy_base_url:
        os.environ["PIPIXIA_LLM_BASE_URL"] = legacy_base_url


def _require_bailian_env() -> None:
    _normalize_bailian_env()
    missing = [
        name
        for name in ("OPENAI_API_KEY", "PIPIXIA_LLM_BASE_URL")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise ValueError(f"missing Bailian environment variable(s): {', '.join(missing)}")


def _set_bailian_image_scope(official: Any, scope: str) -> None:
    """Make the official stable image tag unique for one benchmark instance."""
    original = getattr(official, "_pipixia_unscoped_stable_tag", None)
    if original is None:
        original = official._stable_tag
        official._pipixia_unscoped_stable_tag = original
    official._stable_tag = lambda task_id: original(  # type: ignore[method-assign]
        f"{task_id}-{scope}-pipixia-bailian"
    )


@contextmanager
def _isolated_instance_task_root(
    tasks_root: Path,
    family: str,
    instance_id: str,
):
    """Expose one instance as family-1 for Docker build and keep its own verifier."""
    source = tasks_root / family / instance_id
    if not source.is_dir():
        raise FileNotFoundError(f"SkillLearnBench instance not found: {source}")

    temp_root = Path(tempfile.mkdtemp(prefix="skilllearnbench_instance_"))
    isolated_tasks = temp_root / "tasks"
    isolated_family = isolated_tasks / family
    actual = isolated_family / instance_id
    try:
        shutil.copytree(source, actual)
        build_alias = isolated_family / f"{family}-1"
        if build_alias != actual:
            build_alias.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / "environment", build_alias / "environment")
        yield isolated_tasks
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _register_bailian_agent(official: Any, image_scope: str | None = None) -> None:
    import agents

    _normalize_bailian_env()
    agents.AGENTS["pipixia-bailian"] = _bailian_agent_config()
    _set_bailian_image_scope(official, image_scope or "shared")


class _Utf8SubprocessProxy:
    """Keep Docker/BuildKit UTF-8 output from being decoded with Windows GBK."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self._output = getattr(sys.stdout, "_real", sys.stdout)
        self._temporary_mounts: list[Path] = []
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        for path in self._temporary_mounts:
            shutil.rmtree(path, ignore_errors=True)
        self._temporary_mounts.clear()

    def _normalize_test_mount(self, command: Any) -> Any:
        if not isinstance(command, (list, tuple)) or len(command) < 2:
            return command
        if str(command[0]).lower() != "docker" or str(command[1]).lower() != "run":
            return command
        normalized = list(command)
        mount_suffix = ":/tests:ro"
        for index, value in enumerate(normalized):
            raw = str(value)
            if not raw.endswith(mount_suffix):
                continue
            source = Path(raw[:-len(mount_suffix)])
            if not source.is_dir():
                continue
            temp_root = Path(tempfile.mkdtemp(prefix="skilllearnbench_tests_"))
            temp_tests = temp_root / "tests"
            shutil.copytree(source, temp_tests)
            for script in temp_tests.rglob("*.sh"):
                script.write_bytes(script.read_bytes().replace(b"\r\n", b"\n"))
            self._temporary_mounts.append(temp_root)
            normalized[index] = f"{temp_tests.resolve()}:/tests:ro"
        return normalized

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            args = (self._normalize_test_mount(args[0]), *args[1:])
        if kwargs.get("text") or kwargs.get("universal_newlines"):
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("errors", "replace")
        command = args[0] if args else kwargs.get("args")
        if (
            isinstance(command, (list, tuple))
            and len(command) >= 2
            and str(command[0]).lower() == "docker"
            and str(command[1]).lower() == "build"
            and kwargs.get("capture_output")
        ):
            return self._run_streaming(command, kwargs)
        return self._delegate.run(*args, **kwargs)

    def _run_streaming(self, command: Any, options: dict[str, Any]) -> Any:
        kwargs = dict(options)
        kwargs.pop("capture_output", None)
        check = bool(kwargs.pop("check", False))
        input_value = kwargs.pop("input", None)
        kwargs["stdout"] = self._delegate.PIPE
        kwargs["stderr"] = self._delegate.STDOUT
        kwargs["stdin"] = self._delegate.PIPE if input_value is not None else None
        if kwargs.get("text") or kwargs.get("universal_newlines"):
            kwargs.setdefault("encoding", "utf-8")
            kwargs.setdefault("errors", "replace")

        process = self._delegate.Popen(command, **kwargs)
        if input_value is not None and process.stdin is not None:
            process.stdin.write(input_value)
            process.stdin.close()
        chunks: list[str] = []
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                chunks.append(line)
                self._output.write(line)
                self._output.flush()
            process.stdout.close()
        returncode = process.wait()
        output = "".join(chunks)
        result = self._delegate.CompletedProcess(command, returncode, stdout=output, stderr="")
        if check and returncode:
            raise self._delegate.CalledProcessError(returncode, command, output=output, stderr="")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _patch_official_subprocess_encoding(official: Any) -> None:
    proxy = _Utf8SubprocessProxy(subprocess)
    official.subprocess = proxy
    core_runner = sys.modules.get("core.eval_runner")
    if core_runner is not None:
        core_runner.subprocess = proxy


def _patch_empty_skill_build_context(official: Any) -> None:
    """Docker omits empty directories, while SkillLearnBench Dockerfiles COPY skills/."""
    original_prepare = official._prepare_base_build_env

    def prepare_with_marker(env_dir: Path) -> Path:
        build_env = original_prepare(env_dir)
        dockerfile = build_env / "Dockerfile"
        if dockerfile.exists():
            dockerfile.write_bytes(
                dockerfile.read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"wget -q \\\n", b"wget --progress=dot:giga \\\n")
            )
        skills_dir = build_env / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / ".pipixia-empty").write_text("", encoding="utf-8")
        for child in skills_dir.iterdir():
            if child.is_dir():
                (child / ".pipixia-empty").write_text("", encoding="utf-8")
        return build_env

    official._prepare_base_build_env = prepare_with_marker  # type: ignore[method-assign]


def evaluate(
    manifest: dict[str, Any],
    *,
    family: str,
    skill_root: Path,
    active_skill_root: Path | None = None,
    human_authored_skill_root: Path | None = None,
    agent: str,
    model: str,
    repeats: int,
    max_steps: int,
    trials_dir: Path,
    dry_run: bool,
    instance_ids: list[str] | None = None,
) -> int:
    benchmark_root = Path(manifest["benchmark_root"])
    skill_root = skill_root.resolve()
    active_skill_root = active_skill_root.resolve() if active_skill_root else None
    human_authored_skill_root = human_authored_skill_root.resolve() if human_authored_skill_root else None
    trials_dir = trials_dir.resolve()
    entry = next(item for item in manifest["families"] if item["family"] == family)
    selected = set(instance_ids or [])
    evaluation_instances = [
        item for item in entry["evaluation_instances"]
        if not selected or str(item["instance_id"]) in selected
    ]
    if selected:
        known = {str(item["instance_id"]) for item in evaluation_instances}
        missing = sorted(selected - known)
        if missing:
            raise ValueError("SkillLearnBench evaluation instances not found: " + ", ".join(missing))
    if not evaluation_instances:
        raise ValueError(f"no SkillLearnBench evaluation instances selected: {family}")
    official = _load_official_evaluator(benchmark_root)
    _patch_official_subprocess_encoding(official)
    _patch_empty_skill_build_context(official)
    if agent == "pipixia-bailian" and not dry_run:
        _require_bailian_env()

    exit_code = 0
    skill_paths = [None]
    if active_skill_root is not None:
        skill_paths.append(active_skill_root)
    if human_authored_skill_root is not None:
        skill_paths.append(human_authored_skill_root)
    skill_paths.append(skill_root)
    for item in evaluation_instances:
        instance_id = item["instance_id"]
        task_id = f"{family}/{instance_id}"
        if agent == "pipixia-bailian":
            _register_bailian_agent(official, image_scope=instance_id)
        with _isolated_instance_task_root(benchmark_root / "tasks", family, instance_id) as isolated_root:
            exit_code = max(exit_code, int(official.hyper_eval(
                [task_id],
                task_root=isolated_root,
                agent_id=agent,
                model=model,
                skill_paths=skill_paths,
                repeats=repeats,
                max_steps=max_steps,
                max_workers=1,
                build_workers=1,
                remove_images=False,
                record=True,
                dry_run=dry_run,
                trials_dir=trials_dir,
            )))
    return exit_code


def evaluate_seed(
    manifest: dict[str, Any],
    *,
    family: str,
    agent: str,
    model: str,
    max_steps: int,
    trials_dir: Path,
    dry_run: bool,
) -> int:
    benchmark_root = Path(manifest["benchmark_root"])
    trials_dir = trials_dir.resolve()
    seed = _load_seed(manifest, family)
    official = _load_official_evaluator(benchmark_root)
    _patch_official_subprocess_encoding(official)
    _patch_empty_skill_build_context(official)
    if agent == "pipixia-bailian":
        if not dry_run:
            _require_bailian_env()
        _register_bailian_agent(official, image_scope=seed.instance_id)
    return int(official.hyper_eval(
        [f"{family}/{seed.instance_id}"],
        task_root=benchmark_root / "tasks",
        agent_id=agent,
        model=model,
        skill_paths=[None],
        repeats=1,
        max_steps=max_steps,
        max_workers=1,
        build_workers=1,
        remove_images=False,
        record=True,
        dry_run=dry_run,
        trials_dir=trials_dir,
    ))


def summarize_trials(trials_dir: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    cases: list[dict[str, Any]] = []
    for path in trials_dir.rglob("result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        config = _normalize_variant_name(result.get("skill_config"))
        groups.setdefault(config, []).append(result)
        raw_passed = result.get("passed")
        passed = raw_passed if isinstance(raw_passed, bool) else None
        relative_trial = str(path.parent.relative_to(trials_dir))
        sample_id = str(
            result.get("task_id") or result.get("instance_id") or result.get("task") or relative_trial
        )
        usage = result.get("token_usage") or {}
        case_tokens = (
            int(usage.get("total_tokens") or 0)
            if usage.get("total_tokens") is not None
            else int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
        )
        cases.append({
            "sample_id": sample_id,
            "variant": config,
            "passed": passed,
            "external_failure": passed is None,
            "failure_type": (
                str(result.get("failure_type") or "external_failure") if passed is None else None
            ),
            "error": str(result.get("error") or "") or None,
            "trial_path": relative_trial,
            "tokens": case_tokens,
            "agent_exit": result.get("agent_exit"),
            "agent_timed_out": bool(result.get("agent_timed_out", False)),
            "verifier_exit": result.get("verifier_exit"),
            "duration_ms": float(result.get("duration_ms") or result.get("duration") or 0),
            "tool_calls": int(result.get("tool_calls") or result.get("tool_call_count") or 0),
        })
    systems = []
    for config, rows in sorted(groups.items()):
        evaluated_rows = [row for row in rows if isinstance(row.get("passed"), bool)]
        passed = sum(1 for row in evaluated_rows if row["passed"] is True)
        token_total = 0
        for row in rows:
            usage = row.get("token_usage") or {}
            if usage.get("total_tokens") is not None:
                token_total += int(usage["total_tokens"] or 0)
            else:
                token_total += int(usage.get("input_tokens") or 0)
                token_total += int(usage.get("output_tokens") or 0)
        systems.append({
            "system": config,
            "total_cases": len(rows),
            "evaluated_cases": len(evaluated_rows),
            "passed": passed,
            "failed": len(evaluated_rows) - passed,
            "external_failures": len(rows) - len(evaluated_rows),
            "pass_rate": passed / len(evaluated_rows) if evaluated_rows else 0.0,
            "total_tokens": token_total,
            "avg_tokens": token_total / len(rows),
            "avg_duration_ms": sum(float(row.get("duration_ms") or 0) for row in rows) / len(rows),
            "avg_tool_calls": sum(int(row.get("tool_calls") or 0) for row in rows) / len(rows),
            "stability_sample_count": len(rows),
        })
    return {
        "trials_dir": str(trials_dir),
        "systems": systems,
        "cases": sorted(cases, key=lambda row: (row["sample_id"], row["variant"], row["trial_path"])),
    }


def _normalize_variant_name(value: Any) -> str:
    """Map evaluator-specific labels to the first-phase comparison names."""
    raw = str(value or "unknown").lower()
    if raw in {"no_skill", "none", "without_skill"} or "no_skill" in raw:
        return "without_skill"
    if "candidate" in raw or "skill-" in raw:
        return "candidate_skill"
    if "human_authored" in raw:
        return "human_authored"
    if "active" in raw:
        return "active_skill"
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="PIPIXIA SkillLearnBench adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--benchmark-root", type=Path, required=True)
    prepare.add_argument("--families", nargs="+", required=True)
    prepare.add_argument("--out", type=Path, required=True)

    generate = sub.add_parser("distill")
    generate.add_argument("--manifest", type=Path, required=True)
    generate.add_argument("--family", required=True)
    generate.add_argument("--trajectory", type=Path, required=True)
    generate.add_argument("--skill-out", type=Path, required=True)
    generate.add_argument("--json-out", type=Path, required=True)
    generate.add_argument(
        "--force-generate",
        action="store_true",
        help="generate a benchmark skill even when the production admission evaluator rejects it",
    )
    generate.add_argument(
        "--candidate",
        action="store_true",
        help="write the generated Skill to the offline candidate store",
    )

    seed = sub.add_parser("seed")
    seed.add_argument("--manifest", type=Path, required=True)
    seed.add_argument("--family", required=True)
    seed.add_argument("--agent", default="pipixia-bailian")
    seed.add_argument("--model", default=os.getenv("PIPIXIA_LLM_MODEL", "glm-5.2"))
    seed.add_argument("--max-steps", type=int, default=100)
    seed.add_argument("--trials-dir", type=Path, required=True)
    seed.add_argument("--dry-run", action="store_true")

    run_eval = sub.add_parser("evaluate")
    run_eval.add_argument("--manifest", type=Path, required=True)
    run_eval.add_argument("--family", required=True)
    run_eval.add_argument("--skill-root", type=Path, required=True)
    run_eval.add_argument("--active-skill-root", type=Path)
    run_eval.add_argument("--agent", default="pipixia-bailian")
    run_eval.add_argument("--model", default=os.getenv("PIPIXIA_LLM_MODEL", "glm-5.2"))
    run_eval.add_argument("--repeats", type=int, default=1)
    run_eval.add_argument("--max-steps", type=int, default=100)
    run_eval.add_argument("--trials-dir", type=Path, required=True)
    run_eval.add_argument("--instances", nargs="*", help="只评估指定实例 ID")
    run_eval.add_argument("--dry-run", action="store_true")

    report = sub.add_parser("report")
    report.add_argument("--trials-dir", type=Path, required=True)
    report.add_argument("--json-out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        report = build_manifest(args.benchmark_root, args.families)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "report":
        report_data = summarize_trials(args.trials_dir)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
        return

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.command == "evaluate":
        raise SystemExit(evaluate(
            manifest,
            family=args.family,
            skill_root=args.skill_root,
            active_skill_root=args.active_skill_root,
            agent=args.agent,
            model=args.model,
            repeats=args.repeats,
            max_steps=args.max_steps,
            trials_dir=args.trials_dir,
            dry_run=args.dry_run,
            instance_ids=args.instances,
        ))

    if args.command == "seed":
        raise SystemExit(evaluate_seed(
            manifest,
            family=args.family,
            agent=args.agent,
            model=args.model,
            max_steps=args.max_steps,
            trials_dir=args.trials_dir,
            dry_run=args.dry_run,
        ))

    seed = _load_seed(manifest, args.family)
    report_data = asyncio.run(distill(
        seed,
        args.trajectory,
        args.skill_out,
        force_generate=args.force_generate,
        create_candidate=args.candidate,
    ))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
