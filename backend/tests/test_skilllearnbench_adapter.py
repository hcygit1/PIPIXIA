from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class SkillLearnBenchAdapterTests(unittest.TestCase):
    def test_discover_family_splits_first_instance_from_evaluation(self) -> None:
        from evaluation.skilllearnbench_adapter import build_manifest

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            for index in (1, 2):
                instance = base / "tasks" / "demo" / f"demo-{index}"
                (instance / "tests").mkdir(parents=True)
                (instance / "instruction.md").write_text("instruction", encoding="utf-8")
                (instance / "tests" / "test_outputs.py").write_text("", encoding="utf-8")
            (base / ".git").mkdir()
            import unittest.mock as mock
            with mock.patch(
                "evaluation.skilllearnbench_adapter.benchmark_revision",
                return_value="abc123",
            ):
                manifest = build_manifest(base, ["demo"])

        family = manifest["families"][0]
        self.assertEqual(family["seed_instance"]["instance_id"], "demo-1")
        self.assertEqual(family["evaluation_instances"][0]["instance_id"], "demo-2")
        self.assertIn("verifier and solution excluded", manifest["leakage_policy"])

    def test_build_completed_task_requires_and_preserves_successful_trajectory(self) -> None:
        from evaluation.skilllearnbench_adapter import SkillLearnInstance, build_completed_task

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            instruction = base / "instruction.md"
            instruction.write_text("Organize the files using a reusable workflow.", encoding="utf-8")
            trajectory = base / "trajectory.jsonl"
            rows = [
                {"role": "assistant", "content": f"Step {index}: inspected and processed files. " + "x" * 500}
                for index in range(6)
            ]
            trajectory.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            instance = SkillLearnInstance(
                family="organize-messy-files", instance_id="organize-messy-files-1",
                path=str(base), instruction_path=str(instruction),
                verifier_path=str(base / "tests.py"),
            )
            task, chunks = build_completed_task(instance, trajectory)

        self.assertEqual(task.status, "completed")
        self.assertGreaterEqual(len(chunks), 6)
        self.assertEqual(chunks[0].role, "user")
        self.assertTrue(any(chunk.role == "assistant" for chunk in chunks))

    def test_report_summarizes_pass_rate_and_tokens(self) -> None:
        from evaluation.skilllearnbench_runner import summarize_trials

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            for index, passed in enumerate((True, False)):
                trial = base / "pipixia-generated" / f"trial-{index}"
                trial.mkdir(parents=True)
                (trial / "result.json").write_text(json.dumps({
                    "skill_config": "pipixia-generated",
                    "passed": passed,
                    "token_usage": {"input_tokens": 10, "output_tokens": 5},
                }), encoding="utf-8")
            report = summarize_trials(base)

        system = report["systems"][0]
        self.assertEqual(system["total_cases"], 2)
        self.assertEqual(system["pass_rate"], 0.5)
        self.assertEqual(system["total_tokens"], 30)

    def test_report_does_not_double_count_total_tokens(self) -> None:
        from evaluation.skilllearnbench_runner import summarize_trials

        with tempfile.TemporaryDirectory() as root:
            trial = Path(root) / "no_skill" / "trial"
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps({
                "skill_config": "no_skill",
                "passed": True,
                "token_usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            }), encoding="utf-8")
            report = summarize_trials(Path(root))

        self.assertEqual(report["systems"][0]["total_tokens"], 15)

    def test_report_counts_results_without_passed_as_external_failures(self) -> None:
        from evaluation.skilllearnbench_runner import summarize_trials

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            results = (
                {"passed": True, "token_usage": {"total_tokens": 10}},
                {"passed": False, "token_usage": {"total_tokens": 20}},
                {"status": "error", "error": "model request timed out"},
            )
            for index, result in enumerate(results):
                trial = base / "candidate" / f"trial-{index}"
                trial.mkdir(parents=True)
                (trial / "result.json").write_text(json.dumps({
                    "skill_config": "candidate",
                    **result,
                }), encoding="utf-8")

            report = summarize_trials(base)

        system = report["systems"][0]
        self.assertEqual(system["total_cases"], 3)
        self.assertEqual(system["evaluated_cases"], 2)
        self.assertEqual(system["passed"], 1)
        self.assertEqual(system["failed"], 1)
        self.assertEqual(system["external_failures"], 1)
        self.assertEqual(system["pass_rate"], 0.5)

    def test_bailian_agent_config_embeds_runtime(self) -> None:
        from evaluation.skilllearnbench_runner import _bailian_agent_config

        config = _bailian_agent_config()

        self.assertEqual(
            config["env"],
            ["OPENAI_API_KEY", "PIPIXIA_LLM_BASE_URL"],
        )
        self.assertIn("bailian_agent.py", config["install"])
        self.assertIn("--max-steps {max_steps}", config["run"])

    def test_bailian_env_accepts_legacy_base_url_name(self) -> None:
        from unittest.mock import patch
        from evaluation.skilllearnbench_runner import _require_bailian_env

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "PIPIXIA_LLM_BASE_UR": "https://example.test/v1",
        }, clear=True):
            _require_bailian_env()
            self.assertEqual(
                os.environ["PIPIXIA_LLM_BASE_URL"],
                "https://example.test/v1",
            )

    def test_bailian_runtime_loads_each_skill_once(self) -> None:
        from evaluation.pipixia_bailian_agent import load_skills

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            first = base / "first" / "demo"
            second = base / "second" / "demo"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            content = "# Demo\n\nUse the reusable workflow."
            (first / "SKILL.md").write_text(content, encoding="utf-8")
            (second / "SKILL.md").write_text(content, encoding="utf-8")

            skills = load_skills((base / "first", base / "second"))

        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0][1], content)

    def test_bailian_glm_requests_disable_thinking_and_limit_output(self) -> None:
        from evaluation.pipixia_bailian_agent import _completion_options

        options = _completion_options("glm-5.2")

        self.assertEqual(options["max_tokens"], 2048)
        self.assertEqual(options["extra_body"], {"enable_thinking": False})

    def test_bailian_instruction_reader_falls_back_to_gb18030(self) -> None:
        from evaluation.pipixia_bailian_agent import _read_instruction

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "instruction.md"
            path.write_bytes("整理这些文件".encode("gb18030"))

            self.assertEqual(_read_instruction(path), "整理这些文件")

    def test_docker_test_mount_is_normalized_without_mutating_source(self) -> None:
        from evaluation.skilllearnbench_runner import _Utf8SubprocessProxy

        class FakeSubprocess:
            def run(self, command, **_kwargs):
                self.command = command
                return type("Result", (), {"returncode": 0})()

        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "tests"
            source.mkdir()
            script = source / "test.sh"
            script.write_bytes(b"#!/bin/bash\r\necho ok\r\n")
            fake = FakeSubprocess()
            proxy = _Utf8SubprocessProxy(fake)
            proxy.run(["docker", "run", f"{source}:/tests:ro"])

            mounted = Path(fake.command[2].split(":/tests:ro")[0])
            self.assertNotEqual(mounted.resolve(), source.resolve())
            self.assertEqual((mounted / "test.sh").read_bytes(), b"#!/bin/bash\necho ok\n")
            self.assertEqual(script.read_bytes(), b"#!/bin/bash\r\necho ok\r\n")

    def test_docker_build_output_is_streamed_and_preserved(self) -> None:
        from evaluation.skilllearnbench_runner import _Utf8SubprocessProxy

        class Output:
            def __init__(self):
                self.lines = iter(["step 1\n", "step 2\n", ""])

            def readline(self):
                return next(self.lines)

            def close(self):
                pass

        class Process:
            stdin = None
            stdout = Output()

            @staticmethod
            def wait():
                return 0

        proxy = _Utf8SubprocessProxy(subprocess)
        streamed = io.StringIO()
        proxy._output = streamed
        with patch.object(subprocess, "Popen", return_value=Process()):
            result = proxy.run(["docker", "build", "."], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "step 1\nstep 2\n")
        self.assertEqual(streamed.getvalue(), result.stdout)

    def test_bailian_image_scope_is_unique_per_instance(self) -> None:
        from evaluation.skilllearnbench_runner import _set_bailian_image_scope

        class FakeOfficial:
            @staticmethod
            def _stable_tag(task_id: str) -> str:
                return f"eval-{task_id}"

        official = FakeOfficial()
        _set_bailian_image_scope(official, "organize-messy-files-2")
        second = official._stable_tag("organize-messy-files")
        _set_bailian_image_scope(official, "organize-messy-files-3")
        third = official._stable_tag("organize-messy-files")

        self.assertNotEqual(second, third)
        self.assertIn("organize-messy-files-2", second)
        self.assertIn("organize-messy-files-3", third)

    def test_instance_task_root_uses_selected_environment_and_verifier(self) -> None:
        from evaluation.skilllearnbench_runner import _isolated_instance_task_root

        with tempfile.TemporaryDirectory() as root:
            tasks = Path(root) / "tasks"
            source = tasks / "demo" / "demo-2"
            (source / "environment").mkdir(parents=True)
            (source / "tests").mkdir()
            (source / "environment" / "marker.txt").write_text("env-2", encoding="utf-8")
            (source / "tests" / "marker.txt").write_text("tests-2", encoding="utf-8")

            with _isolated_instance_task_root(tasks, "demo", "demo-2") as isolated:
                build_marker = isolated / "demo" / "demo-1" / "environment" / "marker.txt"
                verifier_marker = isolated / "demo" / "demo-2" / "tests" / "marker.txt"
                self.assertEqual(build_marker.read_text(encoding="utf-8"), "env-2")
                self.assertEqual(verifier_marker.read_text(encoding="utf-8"), "tests-2")


class SkillLearnBenchDistillTests(unittest.IsolatedAsyncioTestCase):
    async def test_force_generate_bypasses_only_admission_decision(self) -> None:
        from evaluation.skilllearnbench_adapter import SkillLearnInstance
        from evaluation.skilllearnbench_runner import distill
        from mem.models import Skill
        from mem.skill_evaluation import CreateEvalResult
        from mem.skill_evolver import MemSkillEvolver

        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            instruction = base / "instruction.md"
            instruction.write_text("Organize files by subject.", encoding="utf-8")
            trajectory = base / "trajectory.jsonl"
            trajectory.write_text("\n".join(
                json.dumps({"role": "assistant", "content": "Processed files. " + "x" * 500})
                for _ in range(6)
            ), encoding="utf-8")
            instance = SkillLearnInstance(
                family="organize-messy-files",
                instance_id="organize-messy-files-1",
                path=str(base),
                instruction_path=str(instruction),
                verifier_path=str(base / "tests.py"),
            )
            evaluation = CreateEvalResult(
                should_generate=False,
                reason="production admission rejected this task",
                suggested_name="organizing-files-by-subject",
                confidence=0.9,
            )
            skill = Skill(
                id="skill-1",
                name="organizing-files-by-subject",
                description="Reusable file organization workflow",
                dir_path=str(base / "skills" / "organizing-files-by-subject"),
                quality_score=0.8,
            )

            with (
                patch.object(MemSkillEvolver, "_rule_filter", return_value=None),
                patch.object(MemSkillEvolver, "_evaluate_create", AsyncMock(return_value=evaluation)),
                patch.object(MemSkillEvolver, "_generate_skill", AsyncMock(return_value=skill)) as generate,
            ):
                result = await distill(
                    instance,
                    trajectory,
                    base / "skills",
                    force_generate=True,
                )

        generate.assert_awaited_once()
        self.assertTrue(result["generated"])
        self.assertTrue(result["forced_generation"])
        self.assertFalse(result["evaluation_should_generate"])
        self.assertEqual(result["skill"]["name"], "organizing-files-by-subject")


if __name__ == "__main__":
    unittest.main()
