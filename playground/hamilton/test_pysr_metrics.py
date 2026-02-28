import json
import os
import tempfile
import unittest
from pathlib import Path

from playground.hamilton.tools.pysr_tool import PySRTool


class _MockSession:
    def __init__(self, workspace_path: str):
        self.config = type("Config", (), {"workspace_path": workspace_path})
        ws = Path(workspace_path)
        ws.mkdir(parents=True, exist_ok=True)
        ws.joinpath("data.csv").write_text(
            "x1,x2,y\n"
            "1,2,3\n"
            "2,1,3\n"
            "3,4,7\n",
            encoding="utf-8",
        )
        ws.joinpath("data_ood.csv").write_text(
            "x1,x2,y\n"
            "4,5,9\n"
            "5,6,11\n",
            encoding="utf-8",
        )

    def exec_bash(self, command, timeout=300, is_input=False):
        if "mkdir -p" in command:
            rel = command.split("mkdir -p", 1)[1].strip().split()[0].strip("'\"")
            Path(self.config.workspace_path, rel).mkdir(parents=True, exist_ok=True)
            return {"stdout": "", "stderr": "", "exit_code": 0}

        if "pysr_round" in command and ".py" in command:
            stdout = (
                "===EVO_PYSR_RESULTS_JSON_BEGIN===\n"
                '[{"rank":1,"equation":"#1 + #2","mse":0.0,"complexity":5}]\n'
                "===EVO_PYSR_RESULTS_JSON_END===\n"
            )
            return {"stdout": stdout, "stderr": "", "exit_code": 0}

        return {"stdout": "", "stderr": "", "exit_code": 0}

    def write_file(self, path, content, encoding="utf-8"):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)


class TestPySRToolMetrics(unittest.TestCase):
    def test_records_train_ood_metrics_and_max_complexity(self):
        os.environ["HAMILTON_ROUND"] = "1"
        tool = PySRTool()
        params = {
            "y": "y",
            "expression_spec": {
                "expressions": ["f"],
                "variable_names": ["x1", "x2"],
                "combine": "f(x1, x2)",
            },
            "niterations": 10,
            "max_evals": 1000,
            "max_complexity": 20,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["sin", "cos"],
            "operator_rationale": "test",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            session = _MockSession(tmpdir)
            _, info = tool.execute(session, json.dumps(params))
            self.assertEqual(info.get("exit_code"), 0)

            payload = json.loads(Path(tmpdir, "experiment.json").read_text(encoding="utf-8"))
            round_data = payload["rounds"]["1"]
            self.assertEqual(round_data["pysr_config"]["max_complexity"], 20)

            result0 = round_data["results"][0]
            self.assertEqual(result0["equation_eval"], "x1 + x2")
            self.assertIsInstance(result0["train_mse"], float)
            self.assertIsInstance(result0["ood_mse"], float)
            self.assertIsInstance(result0["ood_gap"], float)
            self.assertEqual(result0["mse_source"], "recomputed")

            evaluation = round_data["evaluation"]
            self.assertTrue(evaluation["enabled"])
            self.assertTrue(evaluation["train_available"])
            self.assertTrue(evaluation["ood_available"])
            self.assertEqual(evaluation["best_rank"], 1)


if __name__ == "__main__":
    unittest.main()
