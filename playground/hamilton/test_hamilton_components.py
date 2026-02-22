"""Hamilton components smoke tests (stdlib-only).

These tests focus on:
- experiment.json initialization/robustness
- PySRTool output parsing & recording format

They do NOT require PySR to be installed.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path so we can import playground/evomaster modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from playground.hamilton.core.exp import RoundExp
from playground.hamilton.tools.pysr_tool import PySRTool, PySRToolParams


class TestHamiltonComponents(unittest.TestCase):
    def test_roundexp_init_round_files_always_ensures_experiment_json(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)

            # analysis.md already contains this round, should not early-return
            (run_dir / "analysis.md").write_text("## Round 1\n", encoding="utf-8")

            exp = RoundExp(hamilton_agent=None, eureka_agent=None, config={}, round_num=1)
            exp.set_run_dir(run_dir)

            exp._init_round_files()

            experiment_file = run_dir / "experiment.json"
            self.assertTrue(experiment_file.exists())
            data = json.loads(experiment_file.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict)
            self.assertIn("rounds", data)
            self.assertIsInstance(data["rounds"], dict)

            # No duplicated header written
            analysis = (run_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertEqual(analysis.count("## Round 1"), 1)

    def test_pysrtool_parse_json_block(self):
        tool = PySRTool()
        stdout = "\n".join(
            [
                "hello",
                "===EVO_PYSR_RESULTS_JSON_BEGIN===",
                json.dumps([{"rank": 1, "equation": "x0+1", "mse": 0.1, "complexity": 3}]),
                "===EVO_PYSR_RESULTS_JSON_END===",
                "bye",
            ]
        )
        results = tool._parse_pysr_results(stdout)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["equation"], "x0+1")

    def test_pysrtool_parse_legacy_stdout(self):
        tool = PySRTool()
        stdout = "\n".join(
            [
                "Rank 1:",
                "  Expression: x0 + 1",
                "  MSE: 0.1",
                "  Complexity: 3",
                "-" * 30,
                "Rank 2:",
                "  Expression: x0 + x1",
                "  MSE: 0.2",
                "  Complexity: 5",
                "-" * 30,
            ]
        )
        results = tool._parse_pysr_output(stdout)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[1]["rank"], 2)

    def test_pysrtool_record_to_experiment_json_recovers_from_invalid_json(self):
        with tempfile.TemporaryDirectory() as td:
            workspace = Path(td)
            (workspace / "experiment.json").write_text("not json", encoding="utf-8")

            tool = PySRTool()
            params = PySRToolParams(
                y="y",
                expression_spec={"expressions": ["f"], "variable_names": ["x1"], "combine": "f(x1)"},
            )

            os.environ["HAMILTON_ROUND"] = "2"
            tool._record_to_experiment_json(
                workspace=str(workspace),
                params=params,
                results=[{"rank": 1, "equation": "x0+1", "mse": 0.1, "complexity": 3}],
                exit_code=0,
            )

            data = json.loads((workspace / "experiment.json").read_text(encoding="utf-8"))
            self.assertIn("rounds", data)
            self.assertIn("2", data["rounds"])
            self.assertEqual(data["rounds"]["2"]["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()

