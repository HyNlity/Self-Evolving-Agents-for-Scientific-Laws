"""
PySR Tool - 符号回归工具

封装PySR的关键参数，让Agent专注于变量选择和模板设计。
支持自动记录参数和结果到 experiment.json
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any, ClassVar
from pydantic import BaseModel, Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ExpressionSpec(BaseModel):
    """表达式模板规格

    用于定义期望的方程结构，加速PySR搜索。

    Example:
    {
        "expressions": ["f", "g"],           # 占位符子表达式
        "variable_names": ["x1", "x2", "x3"], # 数据中的特征名
        "combine": "sin(f(x1)) * g(x2, x3)"   # 期望的方程骨架
    }
    """
    expressions: list[str] = Field(
        description="占位符子表达式列表，由PySR自动拟合"
    )
    variable_names: list[str] = Field(
        description="数据中的特征变量名列表"
    )
    combine: str = Field(
        description="最终方程骨架，使用expressions中的占位符"
    )


class PySRToolParams(BaseToolParams):
    """PySR 符号回归工具参数"""

    name: ClassVar[str] = "pysr_symbolic_regression"

    # ===== 核心参数 =====

    y: str = Field(
        description="目标变量名（数据列名）"
    )

    expression_spec: ExpressionSpec = Field(
        description="""表达式模板规格，包含:
- expressions: 占位符列表，如 ["f", "g"]
- variable_names: 特征变量名列表
- combine: 方程骨架，如 "f(x1) + g(x2)"
"""
    )

    # ===== 搜索控制 =====

    niterations: int = Field(
        default=100,
        description="PySR迭代次数"
    )

    max_evals: int = Field(
        default=100000,
        description="最大评估次数"
    )

    timeout_in_seconds: int = Field(
        default=300,
        description="超时时间（秒）"
    )

    # ===== 运算符 =====

    binary_operators: list[str] = Field(
        default=["+", "-", "*", "/"],
        description="二元运算符列表"
    )

    unary_operators: list[str] = Field(
        default=["sin", "cos", "exp", "log"],
        description="一元运算符列表，如 ['sin', 'cos', 'exp', 'log']"
    )


class PySRTool(BaseTool):
    """PySR 符号回归工具"""

    name: ClassVar[str] = "pysr_symbolic_regression"
    params_class: ClassVar[type[BaseToolParams]] = PySRToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行PySR符号回归"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数解析错误: {str(e)}", {"error": str(e)}

        assert isinstance(params, PySRToolParams)

        # 获取当前轮次 - 从环境变量或默认为 1
        round_num = int(os.environ.get("HAMILTON_ROUND", "1"))

        # 构建PySR调用代码（写入脚本文件，避免 python -c 的引号/换行转义脆弱性）
        code = self._build_pysr_code(params)

        workspace = session.config.workspace_path
        workspace_q = shlex.quote(workspace)

        script_rel_path = f"history/round{round_num}/scripts/pysr_round{round_num}.py"
        script_dir_rel = os.path.dirname(script_rel_path)

        # 确保脚本目录存在
        session.exec_bash(f"cd {workspace_q} && mkdir -p {shlex.quote(script_dir_rel)}")

        # 写入脚本
        script_abs_path = os.path.join(workspace, script_rel_path)
        session.write_file(script_abs_path, code, encoding="utf-8")

        # 执行脚本
        result = session.exec_bash(
            f"cd {workspace_q} && python3 {shlex.quote(script_rel_path)}",
            timeout=params.timeout_in_seconds,
        )

        stdout = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""
        exit_code = int(result.get("exit_code", 0) or 0)

        output = stdout
        if stderr:
            output = output + ("\n" if output else "") + stderr

        # 提取结果并记录到 experiment.json
        pysr_results = self._parse_pysr_results(stdout)
        if not pysr_results:
            # 兼容旧格式：从 stdout 文本里解析
            pysr_results = self._parse_pysr_output(stdout)

        self._record_to_experiment_json(
            workspace=workspace,
            params=params,
            results=pysr_results,
            exit_code=exit_code,
        )

        return output, {"exit_code": exit_code, "recorded": True, "round": round_num, "script": script_rel_path}

    def _record_to_experiment_json(
        self,
        workspace: str,
        params: PySRToolParams,
        results: list[dict],
        exit_code: int
    ):
        """记录到 experiment.json"""
        round_num = int(os.environ.get("HAMILTON_ROUND", "1"))

        experiment_file = os.path.join(workspace, "experiment.json")

        # 读取现有记录或创建新记录
        experiment_data: dict[str, Any] = {"rounds": {}}
        if os.path.exists(experiment_file):
            try:
                with open(experiment_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    experiment_data = loaded
            except json.JSONDecodeError:
                # 文件损坏则从空结构开始（避免整个流程崩溃）
                experiment_data = {"rounds": {}}

        # 构建本轮记录
        normalized_results = []
        for item in results:
            if not isinstance(item, dict):
                continue
            equation = item.get("equation")
            if not isinstance(equation, str):
                continue
            normalized_results.append({
                "equation": equation,
                "mse": item.get("mse", item.get("loss")),
                "complexity": item.get("complexity"),
            })

        round_record = {
            "round": round_num,
            "pysr_config": {
                "y": params.y,
                "variable_names": params.expression_spec.variable_names,
                "expression_spec": params.expression_spec.model_dump(),
                "niterations": params.niterations,
                "max_evals": params.max_evals,
                "timeout_in_seconds": params.timeout_in_seconds,
                "binary_operators": params.binary_operators,
                "unary_operators": params.unary_operators,
            },
            "results": normalized_results,
            "exit_code": exit_code,
        }

        # 写入记录（确保 rounds 结构存在）
        rounds = experiment_data.get("rounds")
        if not isinstance(rounds, dict):
            rounds = {}
            experiment_data["rounds"] = rounds
        rounds[str(round_num)] = round_record

        with open(experiment_file, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

    def _parse_pysr_results(self, stdout: str) -> list[dict]:
        """从 stdout 中提取 JSON 结果块（优先；更鲁棒）"""
        begin = "===EVO_PYSR_RESULTS_JSON_BEGIN==="
        end = "===EVO_PYSR_RESULTS_JSON_END==="
        if begin not in stdout or end not in stdout:
            return []

        try:
            payload = stdout.split(begin, 1)[1].split(end, 1)[0].strip()
            data = json.loads(payload) if payload else []
            if not isinstance(data, list):
                return []
            return [x for x in data if isinstance(x, dict)]
        except Exception:
            return []

    def _parse_pysr_output(self, stdout: str) -> list[dict]:
        """解析 PySR 输出，提取结果表"""
        results = []

        # PySR 输出格式解析 - 基于 _build_pysr_code 的输出格式
        lines = stdout.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # 检测结果行格式: Rank N:
            if line.startswith("Rank "):
                try:
                    rank = int(line.split("Rank ")[1].split(":")[0])
                except (ValueError, IndexError):
                    rank = len(results) + 1

                expr = ""
                mse = None
                complexity = None

                # 往下读取表达式和指标
                i += 1
                while i < len(lines):
                    detail_line = lines[i].strip()
                    if detail_line.startswith("Expression:"):
                        expr = detail_line.replace("Expression:", "").strip()
                    elif detail_line.startswith("MSE:"):
                        try:
                            mse = float(detail_line.replace("MSE:", "").strip())
                        except ValueError:
                            mse = None
                    elif detail_line.startswith("Complexity:"):
                        try:
                            complexity = int(detail_line.replace("Complexity:", "").strip())
                        except ValueError:
                            complexity = None

                    # 结果块结束标志
                    if detail_line.startswith("Rank ") or detail_line.startswith("=") or detail_line.startswith("-"):
                        break
                    i += 1

                if expr:
                    results.append({
                        "rank": rank,
                        "equation": expr,
                        "mse": mse,
                        "complexity": complexity,
                    })
                # 如果是遇到下一个 Rank 行导致 break，不要跳过该行
                if i < len(lines) and lines[i].strip().startswith("Rank "):
                    continue

            i += 1

        return results

    def _build_pysr_code(self, params: PySRToolParams) -> str:
        """构建PySR调用代码"""
        spec = params.expression_spec.model_dump()

        # 将列表转为Python代码中的列表表示
        var_names = spec.get("variable_names", [])
        expressions = spec.get("expressions", [])
        combine = spec.get("combine", "")

        # 序列化参数
        binary_ops_json = json.dumps(params.binary_operators)
        unary_ops_json = json.dumps(params.unary_operators)

        # 使用模板字符串并替换变量
        code = """
import pandas as pd
import numpy as np
import json
import sys

try:
    from pysr import pysr
except ImportError:
    print("Error: PySR not installed. Run: pip install pysr")
    sys.exit(1)

# 加载数据
df = pd.read_csv('data.csv')

# 提取特征和目标
var_names = VARIABLE_NAMES_PLACEHOLDER
X = df[var_names].values
y_col = Y_COL_PLACEHOLDER
y = df[y_col].values

# 表达式模板配置
expressions = EXPRESSIONS_PLACEHOLDER
combine = COMBINE_PLACEHOLDER

# PySR调用
print("Running PySR with niterations=NITERATIONS_PLACEHOLDER, max_evals=MAX_EVALS_PLACEHOLDER")
print("Binary operators: " + BINARY_OPS_PLACEHOLDER)
print("Unary operators: " + UNARY_OPS_PLACEHOLDER)
print("Expression template: " + COMBINE_PLACEHOLDER)
print("-" * 50)

equations = pysr(
    X,
    y,
    niterations=NITERATIONS_PLACEHOLDER,
    max_evals=MAX_EVALS_PLACEHOLDER,
    binary_operators=BINARY_OPS_PLACEHOLDER,
    unary_operators=UNARY_OPS_PLACEHOLDER,
    expression_selection=expressions,
    combine=combine,
    timeout=TIMEOUT_PLACEHOLDER,
    verbose=True,
    n_jobs=1,
    populations=20,
)

# 输出最优表达式
print("=" * 50)
print("PySR Results (Top 1):")
print("=" * 50)
results = []
try:
    selected = equations.select_k(1)
except Exception:
    selected = []

for i, eq in enumerate(selected):
    expr = getattr(eq, "expr", None)
    loss = getattr(eq, "loss", None)
    complexity = getattr(eq, "complexity", None)
    expr_s = str(expr) if expr is not None else str(eq)
    try:
        loss_v = float(loss) if loss is not None else None
    except Exception:
        loss_v = None
    try:
        complexity_v = int(complexity) if complexity is not None else None
    except Exception:
        complexity_v = None

    print("Rank " + str(i+1) + ":")
    print("  Expression: " + expr_s)
    print("  MSE: " + str(loss_v))
    print("  Complexity: " + str(complexity_v))
    print("-" * 30)

    results.append({
        "rank": i + 1,
        "equation": expr_s,
        "mse": loss_v,
        "complexity": complexity_v,
    })

print("===EVO_PYSR_RESULTS_JSON_BEGIN===")
print(json.dumps(results, ensure_ascii=False))
print("===EVO_PYSR_RESULTS_JSON_END===")
"""
        # 替换占位符（注意：长的字符串要先替换，避免子字符串冲突）
        code = code.replace("VARIABLE_NAMES_PLACEHOLDER", json.dumps(var_names))
        code = code.replace("Y_COL_PLACEHOLDER", json.dumps(params.y))
        code = code.replace("EXPRESSIONS_PLACEHOLDER", json.dumps(expressions))
        code = code.replace("COMBINE_PLACEHOLDER", json.dumps(combine))
        code = code.replace("NITERATIONS_PLACEHOLDER", str(params.niterations))
        code = code.replace("MAX_EVALS_PLACEHOLDER", str(params.max_evals))
        code = code.replace("BINARY_OPS_PLACEHOLDER", binary_ops_json)
        code = code.replace("UNARY_OPS_PLACEHOLDER", unary_ops_json)
        code = code.replace("TIMEOUT_PLACEHOLDER", str(params.timeout_in_seconds))

        return code
