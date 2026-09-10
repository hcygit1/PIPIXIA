from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
import sys

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evaluation.skill_evolution_dataset import (
    build_manifest,
    load_samples,
    validate_manifest,
    validate_sample,
)


def _sample(sample_id: str, family: str = "demo", split: str | None = None) -> dict:
    row = {
        "schema_version": "1.0",
        "sample_id": sample_id,
        "task_family": family,
        "task_snapshot": {"goal": "完成任务"},
        "trajectory": [{"role": "user", "content": "完成任务"}],
    }
    if split:
        row["split"] = split
    return row


class SkillEvolutionDatasetTests(unittest.TestCase):
    def test_build_manifest_assigns_seed_dev_holdout_and_empty_regression(self) -> None:
        manifest = build_manifest([_sample(f"s{i}") for i in range(1, 5)])
        splits = manifest["families"][0]["splits"]
        self.assertEqual(splits["seed"], ["s1"])
        self.assertEqual(splits["dev"], ["s2", "s3"])
        self.assertEqual(splits["holdout"], ["s4"])
        self.assertEqual(splits["regression"], [])
        self.assertEqual(validate_manifest(manifest, {f"s{i}" for i in range(1, 5)}), [])

    def test_explicit_split_is_preserved_and_unlabelled_goes_to_dev(self) -> None:
        manifest = build_manifest([
            _sample("seed", split="seed"),
            _sample("holdout", split="holdout"),
            _sample("extra"),
        ])
        splits = manifest["families"][0]["splits"]
        self.assertEqual(splits["seed"], ["seed"])
        self.assertEqual(splits["holdout"], ["holdout"])
        self.assertEqual(splits["dev"], ["extra"])

    def test_load_samples_rejects_duplicate_ids_and_bad_split(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "samples.jsonl"
            path.write_text(
                json.dumps(_sample("same")) + "\n" + json.dumps(_sample("same")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                load_samples(path)
        self.assertIn("split must be one of", validate_sample({**_sample("x"), "split": "bad"})[0])

    def test_manifest_rejects_duplicate_and_unknown_references(self) -> None:
        manifest = {
            "schema_version": "1.0",
            "families": [{
                "task_family": "demo",
                "splits": {"seed": ["x"], "dev": ["x", "missing"]},
            }],
        }
        errors = validate_manifest(manifest, {"x"})
        self.assertTrue(any("multiple splits" in error for error in errors))
        self.assertTrue(any("unknown samples" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
