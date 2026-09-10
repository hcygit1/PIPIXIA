from __future__ import annotations

import json
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skill_candidate import (
    build_candidate_record,
    check_candidate,
    repair_candidate_content,
    validate_candidate_for_evaluation,
)


class SkillCandidateTests(unittest.TestCase):
    def test_valid_candidate_is_ready_for_dev(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "SKILL.md"
            path.write_text(
                '---\nname: demo\ndescription: "Do demo workflow"\n---\n\n'
                "# Demo\n\n## Steps\n1. Do the verified steps.\n",
                encoding="utf-8",
            )
            record = build_candidate_record(
                family="demo",
                source_samples=[{"sample_id": "seed-1"}],
                skill_id="demo",
                version=1,
                skill_path=path,
            )
            self.assertTrue(record.checks.passed)
            self.assertEqual(record.status, "ready_for_dev")

    def test_static_check_rejects_secret_and_missing_section(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "SKILL.md"
            path.write_text(
                '---\nname: demo\ndescription: "x"\n---\n'
                "api_key: sk-secret\n" + "x" * 100,
                encoding="utf-8",
            )
            result = check_candidate(path)
            self.assertFalse(result.passed)
            self.assertTrue(any("secret" in reason for reason in result.reasons))

    def test_long_candidate_is_warning_but_not_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "SKILL.md"
            path.write_text(
                '---\nname: demo\ndescription: "Do demo workflow"\n---\n\n'
                "## Steps\n" + "1. Do the verified step.\n" * 401,
                encoding="utf-8",
            )
            result = check_candidate(path)
            self.assertTrue(result.passed)
            self.assertTrue(any("400 lines" in warning for warning in result.warnings))

    def test_failed_candidate_cannot_enter_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "candidate.json"
            manifest.write_text(json.dumps({
                "candidate_id": "c-1", "skill_id": "demo", "version": 1,
                "task_family": "demo", "source_sample_ids": ["seed-1"],
                "skill_path": str(Path(root) / "SKILL.md"),
                "status": "static_check_failed",
                "checks": {"passed": False, "reasons": ["bad format"]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "static checks"):
                validate_candidate_for_evaluation(manifest)

    def test_repairable_format_failure_gets_two_bounded_attempts(self) -> None:
        calls: list[str] = []

        async def repair(prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return "still invalid"
            return (
                '---\nname: demo\ndescription: "Do demo workflow"\n---\n\n'
                "## Steps\n1. Do the verified steps safely.\n"
            )

        content, check, attempts = asyncio.run(repair_candidate_content(
            "invalid", llm_call=repair, max_attempts=2,
        ))
        self.assertTrue(check.passed)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(calls), 2)
        self.assertIn("name: demo", content)

    def test_security_failure_is_abandoned_without_llm_repair(self) -> None:
        calls = 0

        def repair(_prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "unused"

        _, check, attempts = asyncio.run(repair_candidate_content(
            '---\nname: demo\ndescription: "Demo"\n---\n\n## Steps\napi_key: sk-secret\n',
            llm_call=repair,
        ))
        self.assertFalse(check.passed)
        self.assertFalse(check.repairable)
        self.assertEqual(attempts, 0)
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
