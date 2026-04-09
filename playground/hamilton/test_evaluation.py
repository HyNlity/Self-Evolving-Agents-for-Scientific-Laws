"""Tests for Hamilton evaluation rubric materialization."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from playground.hamilton.core.evaluation import (
    build_evaluation_context,
    infer_evaluation_profile,
    materialize_evaluation_context,
)


class TestEvaluationRubric(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="hamilton_eval_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_profile_is_inferred_from_dynamics_task(self) -> None:
        profile = infer_evaluation_profile(
            "Discover a VIV dynamics equation with trajectory and limit-cycle validation.",
            {"protocol": {"evaluation_profile": "auto"}},
        )
        self.assertEqual(profile, "dynamics_identification")

    def test_context_merges_explicit_metrics_and_paper_hints(self) -> None:
        paper_path = self.tmpdir / "rubric_notes.md"
        paper_path.write_text(
            "# Notes\n\n"
            "We care about support set stability, OOD generalization, amplitude, phase, and monotonic physics consistency.\n",
            encoding="utf-8",
        )
        context = build_evaluation_context(
            task_description="Symbolic regression for VIV with OOD validation",
            task_contract={
                "protocol": {
                    "evaluation_profile": "sr_structure_discovery",
                    "evaluation_metrics": ["dimension_consistency"],
                    "paper_rubric_sources": [str(paper_path)],
                }
            },
            project_root=self.tmpdir,
        )
        metric_names = {metric["name"] for metric in context["metrics"]}
        self.assertIn("support_set_stability", metric_names)
        self.assertIn("ood_slice_consistency", metric_names)
        self.assertIn("amplitude_error", metric_names)
        self.assertIn("phase_error", metric_names)
        self.assertIn("monotonicity_consistency", metric_names)
        self.assertIn("dimension_consistency", metric_names)

    def test_materialize_writes_json_and_markdown(self) -> None:
        context = build_evaluation_context(
            task_description="Basic symbolic regression",
            task_contract={"protocol": {}},
            project_root=self.tmpdir,
        )
        json_path, md_path = materialize_evaluation_context(self.tmpdir, context)
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())


if __name__ == "__main__":
    unittest.main()
