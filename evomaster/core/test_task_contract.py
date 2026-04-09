"""Tests for framework-level task contract parsing."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from evomaster.core.task_contract import (
    TASK_CONTRACT_FILE,
    default_task_contract,
    parse_task_description_with_contract,
    write_task_contract,
)


class TestTaskContract(unittest.TestCase):
    def test_plain_task_uses_default_contract(self) -> None:
        bundle = parse_task_description_with_contract("# 任务\n\n请发现方程。")
        self.assertEqual(bundle.task_body, "# 任务\n\n请发现方程。")
        self.assertEqual(bundle.contract["protocol"]["evidence_policy"], "advisory")
        self.assertEqual(bundle.contract["protocol"]["review_focus"], [])
        self.assertEqual(bundle.contract["protocol"]["required_evidence"], {})
        self.assertEqual(bundle.contract["protocol"]["evaluation_profile"], "auto")
        self.assertEqual(bundle.contract["protocol"]["evaluation_metrics"], [])
        self.assertEqual(bundle.contract["protocol"]["paper_rubric_sources"], [])
        self.assertFalse(bundle.contract["meta"]["has_front_matter"])

    def test_valid_front_matter_is_normalized(self) -> None:
        bundle = parse_task_description_with_contract(
            "---\n"
            "protocol:\n"
            "  evidence_policy: blocking\n"
            "  review_focus: [support_set, physics_consistency]\n"
            "  required_evidence:\n"
            "    fit: [script, result, trace_metrics]\n"
            "    falsification: trace_falsification\n"
            "  evaluation_profile: dynamics_identification\n"
            "  evaluation_metrics: [support_set_stability, rollout_stability]\n"
            "  paper_rubric_sources: [paper/README_CN.md]\n"
            "---\n"
            "# 任务\n\n请发现方程。\n"
        )
        self.assertEqual(bundle.task_body, "# 任务\n\n请发现方程。")
        self.assertEqual(bundle.contract["protocol"]["evidence_policy"], "blocking")
        self.assertEqual(
            bundle.contract["protocol"]["required_evidence"]["fit"],
            ["script", "result", "trace_metrics"],
        )
        self.assertEqual(
            bundle.contract["protocol"]["required_evidence"]["falsification"],
            ["trace_falsification"],
        )
        self.assertEqual(bundle.contract["protocol"]["evaluation_profile"], "dynamics_identification")
        self.assertEqual(
            bundle.contract["protocol"]["evaluation_metrics"],
            ["support_set_stability", "rollout_stability"],
        )
        self.assertEqual(bundle.contract["protocol"]["paper_rubric_sources"], ["paper/README_CN.md"])
        self.assertTrue(bundle.contract["meta"]["has_front_matter"])
        self.assertEqual(bundle.contract["meta"]["parse_errors"], [])

    def test_invalid_front_matter_falls_back_to_default_contract(self) -> None:
        bundle = parse_task_description_with_contract(
            "---\n"
            "protocol: [\n"
            "---\n"
            "# 任务\n\n请继续。\n"
        )
        self.assertEqual(bundle.task_body, "# 任务\n\n请继续。")
        self.assertEqual(bundle.contract["protocol"], default_task_contract()["protocol"])
        self.assertTrue(bundle.contract["meta"]["has_front_matter"])
        self.assertTrue(bundle.contract["meta"]["parse_errors"])

    def test_write_task_contract_materializes_json(self) -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="task_contract_test_"))
        try:
            contract_path = write_task_contract(tmpdir, default_task_contract())
            self.assertEqual(contract_path.name, TASK_CONTRACT_FILE)
            self.assertTrue(contract_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
