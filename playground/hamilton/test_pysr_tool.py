#!/usr/bin/env python3
"""Smoke-test PySRTool input/output contract (mocked PySR run).

This script does NOT require PySR to be installed. It intercepts the generated
`python3 history/roundN/scripts/pysr_roundN.py` execution and returns a mocked
stdout containing the JSON marker block.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add repo root + playground/ to sys.path for `from hamilton...` imports.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "playground"))

# 设置环境变量
os.environ["HAMILTON_ROUND"] = "1"

from hamilton.tools.pysr_tool import PySRTool


class MockSession:
    """模拟 Session 对象"""

    def __init__(self, workspace_path: str):
        self.config = type("Config", (), {"workspace_path": workspace_path})
        ws = Path(workspace_path)
        ws.mkdir(parents=True, exist_ok=True)
        # Provide train/OOD datasets so PySRTool can write structured train/ood metrics.
        ws.joinpath("data.csv").write_text(
            "x1,x2,x3,x4,y\n"
            "1,2,0,0,3\n"
            "2,1,0,0,3\n"
            "3,4,0,0,7\n",
            encoding="utf-8",
        )
        ws.joinpath("data_ood.csv").write_text(
            "x1,x2,x3,x4,y\n"
            "4,5,0,0,9\n"
            "5,6,0,0,11\n",
            encoding="utf-8",
        )

    def exec_bash(self, command, timeout=300, is_input=False):
        """模拟 exec_bash 执行"""
        print(f"\n[MockSession.exec_bash] Executing: {command[:120]}...")

        # 检查是否是创建目录的命令
        if "mkdir -p" in command:
            # Example: cd /tmp/ws && mkdir -p history/round1/scripts
            try:
                rel = command.split("mkdir -p", 1)[1].strip().split()[0].strip("'\"")
                Path(self.config.workspace_path, rel).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return {"stdout": "", "stderr": "", "exit_code": 0}

        # 检查是否是运行 pysr 脚本的命令 - 模拟成功但不真正运行PySR
        if "pysr_round" in command and ".py" in command:
            # 模拟 PySR 输出
            mock_output = """Running PySR with niterations=10, max_evals=1000
Binary operators: ["+", "-", "*", "/"]
Unary operators: ["sin", "cos", "exp", "log"]
Expression template: f(x1, x2, x3)
--------------------------------------------------
Population: 0/20
Equation: x1 + x2 + x3
MSE: 0.0
Complexity: 3
--------------------------------------------------
Population: 1/20
Equation: x1 * x2 + x3
MSE: 0.5
Complexity: 4
--------------------------------------------------
===EVO_PYSR_RESULTS_JSON_BEGIN===
[{"rank": 1, "equation": "x1 + x2", "mse": 0.0, "complexity": 3}, {"rank": 2, "equation": "x1 * x2", "mse": 0.5, "complexity": 4}]
===EVO_PYSR_RESULTS_JSON_END===
"""
            return {"stdout": mock_output, "stderr": "", "exit_code": 0}

        return {"stdout": "", "stderr": "", "exit_code": 0}

    def write_file(self, path, content, encoding="utf-8"):
        """模拟 write_file"""
        print(f"\n[MockSession.write_file] Writing to: {path}")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)


def test_all_params():
    """测试工具参数接口与 experiment.json 落盘"""
    print("=" * 60)
    print("Testing PySRTool - All Input Parameters")
    print("=" * 60)

    # 1. 测试工具实例创建
    tool = PySRTool()
    print(f"\n[OK] Tool created: {tool.name}")
    print(f"     Params class: {tool.params_class.__name__}")

    # 2. 测试所有参数
    test_cases = [
        # 最小参数（显式传入必需运算符）
        {
            "name": "Minimal params",
            "params": {
                "y": "y",
                "expression_spec": {
                    "expressions": ["f"],
                    "variable_names": ["x1", "x2"],
                    "combine": "f(x1, x2)"
                },
                "binary_operators": ["+", "-", "*", "/"],
                "unary_operators": ["sin", "cos", "exp", "log"],
                "operator_rationale": "baseline: keep low-arity smooth operators first",
            }
        },
        # 完整参数
        {
            "name": "Full params",
            "params": {
                "y": "target",
                "expression_spec": {
                    "expressions": ["f", "g"],
                    "variable_names": ["x1", "x2", "x3", "x4"],
                    "combine": "f(x1, x2) + g(x3, x4)"
                },
                "niterations": 100,
                "max_evals": 50000,
                "binary_operators": ["+", "-", "*", "/", "^"],
                "unary_operators": ["sin", "cos", "tan", "exp", "log", "sqrt"],
                "operator_rationale": "nonlinear trend detected, add tan/sqrt and power",
            }
        },
        # 自定义运算符
        {
            "name": "Custom operators",
            "params": {
                "y": "y",
                "expression_spec": {
                    "expressions": ["f"],
                    "variable_names": ["a", "b"],
                    "combine": "f(a, b)"
                },
                "binary_operators": ["+", "-"],
                "unary_operators": ["sin", "cos"],
                "operator_rationale": "small-operator search for efficiency",
            }
        },
    ]

    all_passed = True
    with tempfile.TemporaryDirectory() as tmpdir:
        session = MockSession(workspace_path=tmpdir)

        for tc in test_cases:
            print(f"\n{'='*50}")
            print(f"Test: {tc['name']}")
            print(f"{'='*50}")

            try:
                params_json = json.dumps(tc["params"])
                print(f"Params: {params_json[:120]}...")

                observation, info = tool.execute(session, params_json)

                print(f"\n[OK] Execution completed")
                print(f"  - Exit code: {info.get('exit_code')}")
                print(f"  - Script: {info.get('script')}")
                print(f"  - Round: {info.get('round')}")
                print(f"  - Recorded: {info.get('recorded')}")

                # 检查 experiment.json
                exp_file = Path(session.config.workspace_path) / "experiment.json"
                if exp_file.exists():
                    exp_data = json.loads(exp_file.read_text(encoding="utf-8"))
                    round_data = (exp_data.get("rounds", {}) or {}).get(str(info.get("round")), {}) or {}
                    print(f"  - Config recorded: {bool(round_data.get('pysr_config'))}")
                    print(f"  - Results recorded: {bool(round_data.get('results'))}")
                    results = round_data.get("results", []) if isinstance(round_data, dict) else []
                    if results and isinstance(results[0], dict):
                        print(f"  - train_mse recorded: {'train_mse' in results[0]}")
                        print(f"  - ood_mse recorded: {'ood_mse' in results[0]}")
                    print(f"  - Round evaluation recorded: {bool(round_data.get('evaluation'))}")

                # 检查输出
                if "EVO_PYSR_RESULTS_JSON_BEGIN" in observation:
                    print("  - Output format: OK (JSON found)")
                else:
                    print(f"  - Output format: {observation[:200]}...")

            except Exception as e:
                print(f"\n[FAIL] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_all_params()
    sys.exit(0 if success else 1)
