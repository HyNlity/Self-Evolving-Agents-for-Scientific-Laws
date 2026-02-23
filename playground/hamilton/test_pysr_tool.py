#!/usr/bin/env python3
"""测试 PySRTool 所有输入参数接口"""

import os
import sys
import json

# 切换到项目根目录
os.chdir("/opt/EvoMaster")

# 添加项目路径
sys.path.insert(0, "/opt/EvoMaster")
sys.path.insert(0, "/opt/EvoMaster/playground")

# 设置环境变量
os.environ["HAMILTON_ROUND"] = "1"

# 使用相对导入
from hamilton.tools.pysr_tool import PySRTool, PySRToolParams


class MockSession:
    """模拟 Session 对象"""

    class MockConfig:
        workspace_path = "/opt/EvoMaster/playground/hamilton/test_workspace"

    config = MockConfig()

    def exec_bash(self, command, timeout=300, is_input=False):
        """模拟 exec_bash 执行"""
        print(f"\n[MockSession.exec_bash] Executing: {command[:100]}...")

        # 检查是否是创建目录的命令
        if "mkdir" in command:
            os.system(command)
            return {"stdout": "", "stderr": "", "exit_code": 0}

        # 检查是否是运行 python 脚本的命令 - 模拟成功但不真正运行PySR
        if "python3" in command and "pysr_round" in command:
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
[{"rank": 1, "equation": "x1 + x2 + x3", "mse": 0.0, "complexity": 3}, {"rank": 2, "equation": "x1 * x2 + x3", "mse": 0.5, "complexity": 4}]
===EVO_PYSR_RESULTS_JSON_END===
"""
            return {"stdout": mock_output, "stderr": "", "exit_code": 0}

        return {"stdout": "", "stderr": "", "exit_code": 0}

    def write_file(self, path, content, encoding="utf-8"):
        """模拟 write_file"""
        print(f"\n[MockSession.write_file] Writing to: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)


def test_all_params():
    """测试所有输入参数"""
    print("=" * 60)
    print("Testing PySRTool - All Input Parameters")
    print("=" * 60)

    # 1. 测试工具实例创建
    tool = PySRTool()
    print(f"\n[OK] Tool created: {tool.name}")
    print(f"     Params class: {tool.params_class.__name__}")

    # 2. 测试所有参数
    test_cases = [
        # 最小参数（使用默认值）
        {
            "name": "Minimal params",
            "params": {
                "y": "y",
                "expression_spec": {
                    "expressions": ["f"],
                    "variable_names": ["x1", "x2"],
                    "combine": "f(x1, x2)"
                }
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
                "timeout_in_seconds": 600,
                "binary_operators": ["+", "-", "*", "/", "^"],
                "unary_operators": ["sin", "cos", "tan", "exp", "log", "sqrt"],
                "constraints": {"*": (3, 3), "+": (5, 5)},
                "nested_constraints": {"sin": {"cos": 0}, "exp": {"exp": 0}},
                "select_k": 5,
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
            }
        },
    ]

    session = MockSession()
    all_passed = True

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {tc['name']}")
        print(f"{'='*50}")

        try:
            params_json = json.dumps(tc["params"])
            print(f"Params: {params_json[:100]}...")

            observation, info = tool.execute(session, params_json)

            print(f"\n[OK] Execution completed")
            print(f"  - Exit code: {info.get('exit_code')}")
            print(f"  - Script: {info.get('script')}")
            print(f"  - Round: {info.get('round')}")
            print(f"  - Recorded: {info.get('recorded')}")

            # 检查 experiment.json
            exp_file = os.path.join(session.config.workspace_path, "experiment.json")
            if os.path.exists(exp_file):
                with open(exp_file) as f:
                    exp_data = json.load(f)
                    round_data = exp_data.get("rounds", {}).get(str(info.get("round")), {})
                    print(f"  - Config recorded: {bool(round_data.get('pysr_config'))}")
                    print(f"  - Results recorded: {bool(round_data.get('results'))}")

            # 检查输出
            if "EVO_PYSR_RESULTS_JSON_BEGIN" in observation:
                print(f"  - Output format: OK (JSON found)")
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
