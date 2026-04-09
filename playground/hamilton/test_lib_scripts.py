"""Tests for fixed Hamilton lib scripts."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIB_DIR = PROJECT_ROOT / "playground" / "hamilton" / "workspace" / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

import fit_viv_analytic  # type: ignore  # noqa: E402
import support_ablation  # type: ignore  # noqa: E402
import validate_viv_rollout  # type: ignore  # noqa: E402
import viv_common  # type: ignore  # noqa: E402


def _synthetic_frame_rows():
    rows = ["t,x,v,a"]
    for idx in range(21):
        t = idx * 0.1
        x = 1.0 if idx % 2 == 0 else -1.0
        v = 0.0
        a = -1.0 * x
        rows.append(f"{t:.1f},{x:.1f},{v:.1f},{a:.1f}")
    return "\n".join(rows) + "\n"


class TestHamiltonLibScripts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="hamilton_lib_scripts_"))
        self.input_dir = self.tmpdir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        dataset_text = _synthetic_frame_rows()
        for split in ("train", "test"):
            (self.input_dir / f"U248_{split}.csv").write_text(dataset_text, encoding="utf-8")

        self.results_dir = self.tmpdir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.coef_table = self.results_dir / "coef_table.csv"

        argv = [
            "fit_viv_analytic.py",
            "--input-dir",
            str(self.input_dir),
            "--output",
            str(self.coef_table),
            "--summary-json",
            str(self.results_dir / "fit_summary.json"),
            "--terms",
            "x,v",
        ]
        with mock.patch.object(sys, "argv", argv):
            fit_viv_analytic.main()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_integrate_rollout_respects_requested_small_step(self) -> None:
        coeff_row = {
            "c_x": -1.0,
            "c_x3": 0.0,
            "c_v": 0.0,
            "c_v3": 0.0,
            "c_v5": 0.0,
        }
        times, xs = viv_common.integrate_rollout(
            coeff_row,
            1.0,
            0.0,
            duration=1.0,
            step=0.002,
            method="Radau",
            max_step=0.002,
        )
        self.assertAlmostEqual(float(times[0]), 0.0)
        self.assertAlmostEqual(float(times[-1]), 1.0)
        self.assertEqual(len(times), 501)
        self.assertEqual(len(xs), 501)

    def test_validate_rollout_accepts_aliases_and_writes_outputs(self) -> None:
        missing_alias_path = self.results_dir / "coeff_table.csv"
        output_dir = self.results_dir / "rollout_alias"

        argv = [
            "validate_viv_rollout.py",
            "--input-dir",
            str(self.input_dir),
            "--coeff_table",
            str(missing_alias_path),
            "--output-dir",
            str(output_dir),
            "--duration",
            "2",
            "--step",
            "0.05",
            "--method",
            "Radau",
            "--max-step",
            "0.05",
        ]
        with mock.patch.object(sys, "argv", argv):
            validate_viv_rollout.main()

        json_path = output_dir / "amplitude_error.json"
        csv_path = output_dir / "rollout_metrics.csv"
        self.assertTrue(json_path.exists())
        self.assertTrue(csv_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["coef_table"], str(self.coef_table))
        self.assertIn(payload["status"], {"success", "partial_success", "failed"})
        self.assertGreaterEqual(payload["num_successful_cases"], 1)
        csv_text = csv_path.read_text(encoding="utf-8")
        self.assertIn("status", csv_text.splitlines()[0])

    def test_validate_rollout_accepts_support_ablation_json_and_invalid_method_falls_back(self) -> None:
        support_json = self.results_dir / "support_ablation.json"
        with mock.patch.object(
            sys,
            "argv",
            [
                "support_ablation.py",
                "--input-dir",
                str(self.input_dir),
                "--output",
                str(support_json),
                "--variant",
                "no_v3v5",
            ],
        ):
            support_ablation.main()

        output_dir = self.results_dir / "ablation_rollout"
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_viv_rollout.py",
                "--input-dir",
                str(self.input_dir),
                "--coef-table",
                str(support_json),
                "--output-dir",
                str(output_dir),
                "--duration",
                "2",
                "--step",
                "0.05",
                "--method",
                "no_v3v5",
            ],
        ):
            validate_viv_rollout.main()

        payload = json.loads((output_dir / "amplitude_error.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["coef_table"], str(support_json))
        self.assertEqual(payload["method_requested"], "no_v3v5")
        self.assertEqual(payload["method"], "Radau")
        self.assertGreaterEqual(payload["num_successful_cases"], 1)

    def test_validate_rollout_accepts_light_mode_and_output_json_only(self) -> None:
        support_json = self.results_dir / "support_ablation_for_light.json"
        with mock.patch.object(
            sys,
            "argv",
            [
                "support_ablation.py",
                "--input-dir",
                str(self.input_dir),
                "--output",
                str(support_json),
                "--variant",
                "no_v3v5",
            ],
        ):
            support_ablation.main()

        output_json = self.results_dir / "ablation_amplitude_error.json"
        with mock.patch.object(
            sys,
            "argv",
            [
                "validate_viv_rollout.py",
                "--input-dir",
                str(self.input_dir),
                "--coef-table",
                str(support_json),
                "--output-json",
                str(output_json),
                "--light-mode",
                "--method",
                "ablation",
            ],
        ):
            validate_viv_rollout.main()

        output_csv = self.results_dir / "ablation_rollout_metrics.csv"
        self.assertTrue(output_json.exists())
        self.assertTrue(output_csv.exists())
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["coef_table"], str(support_json))
        self.assertEqual(payload["method_requested"], "ablation")
        self.assertEqual(payload["method"], "Radau")
        self.assertTrue(payload["light_mode"])
        self.assertLessEqual(float(payload["duration"]), 60.0)

    def test_support_ablation_accepts_light_mode_and_coef_alias(self) -> None:
        output_path = self.results_dir / "support_ablation.json"
        argv = [
            "support_ablation.py",
            "--input-dir",
            str(self.input_dir),
            "--output",
            str(output_path),
            "--variant",
            "no_v3v5",
            "--coeff_table",
            str(self.coef_table),
            "--light_mode",
        ]
        with mock.patch.object(sys, "argv", argv):
            support_ablation.main()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["variant"], "no_v3v5")
        self.assertTrue(payload["light_mode"])
        self.assertEqual(payload["coef_table"], str(self.coef_table))

    def test_support_ablation_accepts_input_and_short_aliases(self) -> None:
        output_path = self.results_dir / "support_ablation_short.json"
        argv = [
            "support_ablation.py",
            "--input-dir",
            str(self.input_dir),
            "--output",
            str(output_path),
            "--variant",
            "no_v3v5",
            "--input",
            str(self.coef_table),
            "--short",
        ]
        with mock.patch.object(sys, "argv", argv):
            support_ablation.main()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["variant"], "no_v3v5")
        self.assertTrue(payload["light_mode"])
        self.assertEqual(payload["coef_table"], str(self.coef_table))


if __name__ == "__main__":
    unittest.main()
