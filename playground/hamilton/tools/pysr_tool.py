"""
PySR Tool - 符号回归工具

封装 PySR 的关键参数，让 Agent 专注于变量选择和模板设计。
支持自动记录参数和结果到 experiment.json。

v2: 使用 PySRRegressor API，支持自定义数据文件、更多搜索参数、
    可选 expression_spec、可选 OOD 评估。
"""

from __future__ import annotations

import json
import os
import shlex
from typing import Any, ClassVar
from pydantic import BaseModel, Field

from evomaster.agent.tools.base import BaseTool, BaseToolParams


class ExpressionSpec(BaseModel):
    """表达式模板规格（高级模式）

    用于定义期望的方程结构，加速 PySR 搜索。

    Example:
    {
        "expressions": ["f", "g"],
        "variable_names": ["x1", "x2", "x3"],
        "combine": "sin(f(x1)) * g(x2, x3)"
    }
    """
    expressions: list[str] = Field(
        description="占位符子表达式列表，由 PySR 自动拟合"
    )
    variable_names: list[str] = Field(
        description="数据中的特征变量名列表"
    )
    combine: str = Field(
        description="最终方程骨架，使用 expressions 中的占位符"
    )


class PySRToolParams(BaseToolParams):
    """PySR 符号回归工具参数"""

    name: ClassVar[str] = "pysr_symbolic_regression"

    # ===== 核心参数 =====

    y: str = Field(
        description="目标变量名（数据列名）"
    )

    variable_names: list[str] | None = Field(
        default=None,
        description="特征变量名列表。若省略且未指定 expression_spec，则使用 data 中除 y 之外的所有列"
    )

    data_file: str = Field(
        default="data.csv",
        description="数据文件路径（相对于 workspace），默认 data.csv。Agent 可指定派生数据文件"
    )

    # ===== 高级模式：表达式模板（可选） =====

    expression_spec: ExpressionSpec | None = Field(
        default=None,
        description="可选的表达式模板规格（高级模式）。若指定，则使用模板化搜索。若省略，则使用标准 model.fit(X, y) 自由搜索"
    )

    # ===== 搜索控制 =====

    niterations: int = Field(
        default=100,
        description="PySR 迭代次数"
    )

    maxsize: int = Field(
        default=20,
        description="表达式最大节点数（控制复杂度上限）"
    )

    parsimony: float = Field(
        default=0.0032,
        description="复杂度惩罚系数（越大越偏好简单表达式）"
    )

    populations: int = Field(
        default=20,
        description="进化种群数量"
    )

    timeout_in_seconds: int | None = Field(
        default=None,
        description="PySR 超时时间（秒）。若不指定则不设超时"
    )

    # ===== 运算符 =====

    binary_operators: list[str] = Field(
        default=["+", "-", "*", "/"],
        description="二元运算符列表"
    )

    unary_operators: list[str] = Field(
        default=["sin", "cos", "exp", "log"],
        description="一元运算符列表"
    )

    # ===== OOD 评估 =====

    ood_data_file: str | None = Field(
        default=None,
        description="可选的 OOD 数据文件路径（相对于 workspace）。若指定则自动计算 OOD MSE"
    )


class PySRTool(BaseTool):
    """PySR 符号回归工具"""

    name: ClassVar[str] = "pysr_symbolic_regression"
    params_class: ClassVar[type[BaseToolParams]] = PySRToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行 PySR 符号回归"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"参数解析错误: {str(e)}", {"error": str(e)}

        assert isinstance(params, PySRToolParams)

        # 获取当前轮次
        round_num = int(os.environ.get("HAMILTON_ROUND", "1"))

        # 构建 PySR 调用代码
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

        experiment_data: dict[str, Any] = {"rounds": {}}
        if os.path.exists(experiment_file):
            try:
                with open(experiment_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    experiment_data = loaded
            except json.JSONDecodeError:
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
                "rank": item.get("rank"),
                "equation": equation,
                "mse": item.get("mse", item.get("loss")),
                "complexity": item.get("complexity"),
                "ood_mse": item.get("ood_mse"),
            })

        # 确定记录的 variable_names
        var_names = params.variable_names
        if var_names is None and params.expression_spec is not None:
            var_names = params.expression_spec.variable_names

        pysr_config = {
            "y": params.y,
            "data_file": params.data_file,
            "variable_names": var_names,
            "niterations": params.niterations,
            "maxsize": params.maxsize,
            "parsimony": params.parsimony,
            "populations": params.populations,
            "binary_operators": params.binary_operators,
            "unary_operators": params.unary_operators,
        }
        if params.expression_spec is not None:
            pysr_config["expression_spec"] = params.expression_spec.model_dump()
        if params.timeout_in_seconds is not None:
            pysr_config["timeout_in_seconds"] = params.timeout_in_seconds
        if params.ood_data_file is not None:
            pysr_config["ood_data_file"] = params.ood_data_file

        round_record = {
            "round": round_num,
            "pysr_config": pysr_config,
            "results": normalized_results,
            "exit_code": exit_code,
        }

        rounds = experiment_data.get("rounds")
        if not isinstance(rounds, dict):
            rounds = {}
            experiment_data["rounds"] = rounds
        rounds[str(round_num)] = round_record

        with open(experiment_file, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)

    def _parse_pysr_results(self, stdout: str) -> list[dict]:
        """从 stdout 中提取 JSON 结果块"""
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
        """解析 PySR 输出，提取结果表（兼容旧格式）"""
        results = []
        lines = stdout.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Rank "):
                try:
                    rank = int(line.split("Rank ")[1].split(":")[0])
                except (ValueError, IndexError):
                    rank = len(results) + 1

                expr = ""
                mse = None
                complexity = None
                ood_mse = None

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
                    elif detail_line.startswith("OOD_MSE:"):
                        try:
                            ood_mse = float(detail_line.replace("OOD_MSE:", "").strip())
                        except ValueError:
                            ood_mse = None
                    elif detail_line.startswith("Complexity:"):
                        try:
                            complexity = int(detail_line.replace("Complexity:", "").strip())
                        except ValueError:
                            complexity = None
                    if detail_line.startswith("Rank ") or detail_line.startswith("=") or detail_line.startswith("-"):
                        break
                    i += 1

                if expr:
                    results.append({
                        "rank": rank,
                        "equation": expr,
                        "mse": mse,
                        "complexity": complexity,
                        "ood_mse": ood_mse,
                    })
                if i < len(lines) and lines[i].strip().startswith("Rank "):
                    continue
            i += 1
        return results

    def _build_pysr_code(self, params: PySRToolParams) -> str:
        """构建 PySR 调用代码（使用 PySRRegressor API）"""

        data_file = json.dumps(params.data_file)
        y_col = json.dumps(params.y)
        binary_ops = json.dumps(params.binary_operators)
        unary_ops = json.dumps(params.unary_operators)

        # Determine variable_names source
        if params.expression_spec is not None:
            var_names = json.dumps(params.expression_spec.variable_names)
            expressions = json.dumps(params.expression_spec.expressions)
            combine = json.dumps(params.expression_spec.combine)
            use_template = True
        else:
            var_names = json.dumps(params.variable_names) if params.variable_names else "None"
            use_template = False

        # Build timeout arg
        timeout_arg = ""
        if params.timeout_in_seconds is not None:
            timeout_arg = f"    timeout_in_seconds={params.timeout_in_seconds},\n"

        # OOD evaluation
        has_ood = params.ood_data_file is not None
        ood_file = json.dumps(params.ood_data_file) if has_ood else "None"

        # --- Build the script ---
        lines = [
            "import pandas as pd",
            "import numpy as np",
            "import json",
            "import sys",
            "import os",
            "",
            "try:",
            "    from pysr import PySRRegressor",
            "except ImportError:",
            '    print("Error: PySR not installed. Run: pip install pysr")',
            "    sys.exit(1)",
            "",
            f"# Load data",
            f"data_file = {data_file}",
            f"df = pd.read_csv(data_file)",
            f"print(f'Loaded {{data_file}}: {{df.shape[0]}} rows, {{df.shape[1]}} columns')",
            "",
            f"# Target variable",
            f"y_col = {y_col}",
            f"y = df[y_col].values",
            "",
        ]

        if use_template:
            # Template mode (expression_spec)
            lines += [
                f"# Feature variables (from expression_spec)",
                f"var_names = {var_names}",
                f"X = df[var_names].values",
                "",
                f"# Expression template",
                f"expressions = {expressions}",
                f"combine = {combine}",
                "",
            ]
        else:
            # Standard mode
            lines += [
                f"# Feature variables",
                f"var_names = {var_names}",
                "if var_names is None:",
                f"    var_names = [c for c in df.columns if c != y_col]",
                "X = df[var_names].values",
                "",
            ]

        # Print config summary
        lines += [
            f'print("=" * 50)',
            f'print("PySR Configuration:")',
            f'print(f"  Target: {{y_col}}")',
            f'print(f"  Features: {{var_names}}")',
            f'print(f"  niterations: {params.niterations}")',
            f'print(f"  maxsize: {params.maxsize}")',
            f'print(f"  parsimony: {params.parsimony}")',
            f'print(f"  populations: {params.populations}")',
            f'print(f"  binary_operators: {binary_ops}")',
            f'print(f"  unary_operators: {unary_ops}")',
        ]
        if use_template:
            lines.append(f'print(f"  expression_template: {{{combine}}}")')
        if params.timeout_in_seconds is not None:
            lines.append(f'print(f"  timeout: {params.timeout_in_seconds}s")')
        lines += [
            f'print("=" * 50)',
            "",
        ]

        # Build PySRRegressor
        lines += [
            "# Create PySRRegressor",
            "model = PySRRegressor(",
            f"    niterations={params.niterations},",
            f"    binary_operators={binary_ops},",
            f"    unary_operators={unary_ops},",
            f"    maxsize={params.maxsize},",
            f"    parsimony={params.parsimony},",
            f"    populations={params.populations},",
        ]
        if timeout_arg:
            lines.append(timeout_arg.rstrip())
        if use_template:
            lines.append(f"    expression_spec={{")
            lines.append(f"        'expressions': {expressions},")
            lines.append(f"        'combine': {combine},")
            lines.append(f"    }},")

        lines += [
            "    verbose=1,",
            "    progress=True,",
            ")",
            "",
            "# Fit model",
            "model.fit(X, y, variable_names=var_names)",
            "",
        ]

        # Extract results
        lines += [
            "# Extract results",
            'print("=" * 50)',
            'print("PySR Results:")',
            'print("=" * 50)',
            "",
            "results = []",
            "try:",
            "    eqs = model.equations_",
            "    if hasattr(eqs, 'iterrows'):",
            "        # DataFrame format",
            "        for idx, row in eqs.iterrows():",
            "            eq_str = str(row.get('equation', row.get('sympy_format', '')))",
            "            loss = row.get('loss', None)",
            "            complexity = row.get('complexity', None)",
            "            try:",
            "                loss_v = float(loss) if loss is not None else None",
            "            except Exception:",
            "                loss_v = None",
            "            try:",
            "                complexity_v = int(complexity) if complexity is not None else None",
            "            except Exception:",
            "                complexity_v = None",
            "            results.append({",
            '                "rank": len(results) + 1,',
            '                "equation": eq_str,',
            '                "mse": loss_v,',
            '                "complexity": complexity_v,',
            "            })",
            "    else:",
            "        # Fallback: try as list",
            "        for i, eq in enumerate(eqs):",
            "            eq_str = str(eq)",
            "            results.append({",
            '                "rank": i + 1,',
            '                "equation": eq_str,',
            '                "mse": None,',
            '                "complexity": None,',
            "            })",
            "except Exception as e:",
            '    print(f"Warning: could not parse equations: {e}")',
            "",
            "# Also try best equation",
            "try:",
            "    best = model.get_best()",
            "    best_eq = str(best.get('equation', best.get('sympy_format', ''))) if isinstance(best, dict) else str(best)",
            "    best_loss = None",
            "    if isinstance(best, dict):",
            "        best_loss = best.get('loss')",
            "    elif hasattr(best, 'loss'):",
            "        best_loss = best.loss",
            "    try:",
            "        best_loss = float(best_loss) if best_loss is not None else None",
            "    except Exception:",
            "        best_loss = None",
            "    # Ensure best is in results",
            "    best_in_results = any(r['equation'] == best_eq for r in results)",
            "    if not best_in_results and best_eq:",
            "        results.insert(0, {",
            '            "rank": 0,',
            '            "equation": best_eq,',
            '            "mse": best_loss,',
            '            "complexity": None,',
            "        })",
            "except Exception:",
            "    pass",
            "",
            "# Sort by MSE (best first)",
            "results_with_mse = [r for r in results if r['mse'] is not None]",
            "results_without_mse = [r for r in results if r['mse'] is None]",
            "results_with_mse.sort(key=lambda x: x['mse'])",
            "results = results_with_mse + results_without_mse",
            "for i, r in enumerate(results):",
            "    r['rank'] = i + 1",
            "",
            "# Print top results",
            "for r in results[:10]:",
            '    print(f"Rank {r[\'rank\']}:")',
            '    print(f"  Expression: {r[\'equation\']}")',
            '    print(f"  MSE: {r[\'mse\']}")',
            '    print(f"  Complexity: {r[\'complexity\']}")',
            '    print("-" * 30)',
            "",
        ]

        # OOD evaluation
        lines += [
            "# OOD evaluation",
            f"ood_file = {ood_file}",
            "if ood_file is None:",
            "    # Auto-detect data_ood.csv",
            "    if os.path.exists('data_ood.csv'):",
            "        ood_file = 'data_ood.csv'",
            "",
            "if ood_file is not None and os.path.exists(ood_file):",
            "    print(f'\\nOOD Evaluation on {ood_file}:')",
            "    try:",
            "        df_ood = pd.read_csv(ood_file)",
            "        X_ood = df_ood[var_names].values",
            "        y_ood = df_ood[y_col].values",
            "        y_pred_ood = model.predict(X_ood)",
            "        ood_mse = float(np.mean((y_ood - y_pred_ood) ** 2))",
            "        print(f'  OOD MSE (best model): {ood_mse}')",
            "        # Add OOD MSE to best result",
            "        if results:",
            "            results[0]['ood_mse'] = ood_mse",
            "    except Exception as e:",
            "        print(f'  OOD evaluation failed: {e}')",
            "else:",
            "    print('\\nNo OOD data file found. Skipping OOD evaluation.')",
            "",
        ]

        # JSON output block
        lines += [
            '# Machine-readable output',
            'print("===EVO_PYSR_RESULTS_JSON_BEGIN===")',
            'print(json.dumps(results, ensure_ascii=False))',
            'print("===EVO_PYSR_RESULTS_JSON_END===")',
        ]

        return "\n".join(lines) + "\n"
