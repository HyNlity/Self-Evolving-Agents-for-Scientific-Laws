import json
import tempfile
import unittest
from pathlib import Path

from playground.hamilton.core.exp import RoundExp


class TestRoundExpCurrentBestGate(unittest.TestCase):
    def _build_exp(self, run_dir: Path) -> RoundExp:
        exp = RoundExp.__new__(RoundExp)
        exp.run_dir = run_dir
        exp.round_num = 1
        return exp

    def test_accepts_exit_code_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            payload = {
                "rounds": {
                    "1": {
                        "exit_code": 0,
                        "results": [{"equation": "x1 + x2", "mse": 0.1}],
                    }
                }
            }
            (run_dir / "experiment.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            exp = self._build_exp(run_dir)
            ok, reason = exp._has_effective_round_results(1)
            self.assertTrue(ok)
            self.assertEqual(reason, "ok")

    def test_rejects_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            payload = {
                "rounds": {
                    "1": {
                        "exit_code": 1,
                        "results": [{"equation": "x1 + x2", "mse": 0.1}],
                    }
                }
            }
            (run_dir / "experiment.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            exp = self._build_exp(run_dir)
            ok, reason = exp._has_effective_round_results(1)
            self.assertFalse(ok)
            self.assertEqual(reason, "round_exit_code_1")


if __name__ == "__main__":
    unittest.main()
