from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skilllearnbench_dataset_importer import list_families, prepare_family


class SkillLearnBenchDatasetImporterTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        benchmark = root / "benchmark"
        benchmark.mkdir()
        instances = []
        for index in range(1, 4):
            instance = benchmark / "tasks" / "demo" / f"demo-{index}"
            (instance / "environment").mkdir(parents=True)
            (instance / "tests").mkdir()
            (instance / "instruction.md").write_text(f"完成任务 {index}", encoding="utf-8")
            (instance / "tests" / "test_outputs.py").write_text("SECRET_VERIFIER", encoding="utf-8")
            instances.append({
                "family": "demo", "instance_id": f"demo-{index}",
                "path": str(instance),
                "instruction_path": str(instance / "instruction.md"),
                "verifier_path": str(instance / "tests" / "test_outputs.py"),
            })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "benchmark": "SkillLearnBench", "benchmark_root": str(benchmark),
            "revision": "abc123", "families": [{
                "family": "demo", "seed_instance": instances[0],
                "evaluation_instances": instances[1:],
            }],
        }), encoding="utf-8")
        return manifest

    def test_lists_instances_without_exposing_verifier_content(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            manifest = self._fixture(Path(value))
            rows = list_families(manifest)
            self.assertEqual(rows[0]["family"], "demo")
            self.assertEqual(len(rows[0]["instances"]), 3)
            self.assertTrue(rows[0]["instances"][0]["verifier_available"])
            self.assertNotIn("SECRET_VERIFIER", json.dumps(rows))

    def test_prepares_seed_spec_and_evaluation_only_samples_without_running_agent(self) -> None:
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            manifest = self._fixture(root)

            result = prepare_family(
                root / "evolution", "demo", manifest_path=manifest,
            )
            samples = [
                json.loads(line) for line in Path(result["source_path"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(samples[0]["category"], "seed")
            self.assertEqual(samples[0]["trajectory"], [])
            self.assertEqual(samples[1]["category"], "evaluation")
            self.assertEqual(samples[1]["trajectory"], [])
            self.assertNotIn("SECRET_VERIFIER", Path(result["source_path"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
