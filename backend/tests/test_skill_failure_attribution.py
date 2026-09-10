from __future__ import annotations

import unittest

from evaluation.skill_failure_attribution import attribute_failure
from evaluation.skill_release_gate import ReleaseDecision


class SkillFailureAttributionTests(unittest.TestCase):
    def test_candidate_failures_route_to_local_skill_revision(self) -> None:
        result = attribute_failure(
            {
                "systems": [
                    {"system": "without_skill", "external_failures": 0},
                    {"system": "candidate_skill", "external_failures": 0},
                ],
                "cases": [
                    {"sample_id": "dev-1", "variant": "without_skill", "passed": True},
                    {"sample_id": "dev-1", "variant": "candidate_skill", "passed": False},
                ],
            },
            ReleaseDecision("rejected", ("candidate did not outperform without_skill",)),
        )
        self.assertEqual(result.classification, "skill_defect")
        self.assertEqual(result.next_action, "create_local_candidate_revision")
        self.assertEqual(result.evidence_sample_ids, ("dev-1",))

    def test_shared_task_failure_does_not_trigger_a_skill_patch(self) -> None:
        result = attribute_failure(
            {
                "systems": [
                    {"system": "without_skill", "external_failures": 0},
                    {"system": "candidate_skill", "external_failures": 0},
                ],
                "cases": [
                    {"sample_id": "dev-1", "variant": "without_skill", "passed": False},
                    {"sample_id": "dev-1", "variant": "candidate_skill", "passed": False},
                ],
            },
            ReleaseDecision("rejected", ("candidate did not outperform without_skill",)),
        )
        self.assertEqual(result.classification, "inconclusive")
        self.assertEqual(result.next_action, "manual_failure_analysis")

    def test_timeout_routes_to_environment_repair_not_skill_patch(self) -> None:
        result = attribute_failure(
            {
                "systems": [
                    {"system": "without_skill", "external_failures": 1},
                    {"system": "candidate_skill", "external_failures": 0},
                ],
                "cases": [{
                    "sample_id": "dev-1", "variant": "without_skill", "passed": None,
                    "external_failure": True, "failure_type": "timeout",
                }],
            },
            ReleaseDecision("external_failure", ("one or more runs had external failures",)),
        )
        self.assertEqual(result.classification, "external_failure")
        self.assertEqual(result.next_action, "repair_environment_and_rerun")

    def test_verifier_error_routes_to_evaluation_repair(self) -> None:
        result = attribute_failure(
            {
                "systems": [
                    {"system": "without_skill", "external_failures": 0},
                    {"system": "candidate_skill", "external_failures": 1},
                ],
                "cases": [{
                    "sample_id": "dev-1", "variant": "candidate_skill", "passed": None,
                    "external_failure": True, "failure_type": "verifier_error",
                }],
            },
            ReleaseDecision("external_failure", ("one or more runs had external failures",)),
        )
        self.assertEqual(result.classification, "evaluation_defect")
        self.assertEqual(result.next_action, "repair_evaluation_and_rerun")

    def test_accepted_candidate_waits_for_holdout_and_review(self) -> None:
        result = attribute_failure({}, ReleaseDecision("accepted", ()))
        self.assertEqual(result.classification, "ready_for_review")
        self.assertEqual(result.next_action, "run_holdout_and_request_review")


if __name__ == "__main__":
    unittest.main()
