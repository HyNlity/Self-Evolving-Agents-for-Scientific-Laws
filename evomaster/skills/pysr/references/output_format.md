# PySR 脚本输出格式规范

## 目的

当你编写 PySR 脚本通过 `run_pysr` 工具执行时，脚本的 stdout **必须** 包含以下格式的结果块，工具才能自动解析并记录到 `experiment.json`。

## 必须遵循的输出格式

在脚本末尾，打印搜索结果为 JSON 数组，用 marker 包围：

```python
import json

# ... PySR 搜索完成后 ...

results = []
for idx, row in model.equations_.iterrows():
    results.append({
        "rank": idx + 1,
        "equation": str(row.get("equation", "")),
        "mse": float(row["loss"]) if row["loss"] is not None else None,
        "complexity": int(row["complexity"]) if row["complexity"] is not None else None,
    })

# 按 MSE 排序
results.sort(key=lambda x: (x["mse"] is None, x["mse"] or 0))
for i, r in enumerate(results):
    r["rank"] = i + 1

# === 必须的 marker 输出 ===
print("===EVO_PYSR_RESULTS_JSON_BEGIN===")
print(json.dumps(results, ensure_ascii=False))
print("===EVO_PYSR_RESULTS_JSON_END===")
```

## JSON 结构

```json
[
    {
        "rank": 1,
        "equation": "x + 2.3*v",
        "mse": 0.0012,
        "complexity": 5,
        "ood_mse": 0.0015
    },
    {
        "rank": 2,
        "equation": "sin(x) + v",
        "mse": 0.0045,
        "complexity": 7
    }
]
```

### 字段说明

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `rank` | int | 是 | 排名（1-based） |
| `equation` | str | 是 | 表达式字符串 |
| `mse` | float\|null | 是 | 损失值（通常是 MSE） |
| `complexity` | int\|null | 否 | 表达式复杂度 |
| `ood_mse` | float\|null | 否 | OOD 数据上的 MSE |

## Marker 常量

```
===EVO_PYSR_RESULTS_JSON_BEGIN===
===EVO_PYSR_RESULTS_JSON_END===
```

必须精确匹配，独占一行。

## 完整脚本模板

```python
"""PySR 脚本"""
import json
import sys
import numpy as np
import pandas as pd
from pysr import PySRRegressor

# === 数据 ===
df = pd.read_csv("data.csv")
y = df["target_col"].values
X = df[["x", "v"]].values
var_names = ["x", "v"]

# === 模型 ===
model = PySRRegressor(
    niterations=5000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos"],
    maxsize=20,
    parsimony=0.005,
    populations=31,
    early_stop_condition="f(loss, complexity) = (loss < 1e-4) && (complexity < 15)",
)
model.fit(X, y, variable_names=var_names)

# === 提取结果 ===
results = []
eqs = model.equations_
if hasattr(eqs, "iterrows"):
    for idx, row in eqs.iterrows():
        results.append({
            "rank": len(results) + 1,
            "equation": str(row.get("equation", "")),
            "mse": float(row["loss"]) if row.get("loss") is not None else None,
            "complexity": int(row["complexity"]) if row.get("complexity") is not None else None,
        })

# 排序
results_with_mse = [r for r in results if r["mse"] is not None]
results_without = [r for r in results if r["mse"] is None]
results_with_mse.sort(key=lambda x: x["mse"])
results = results_with_mse + results_without
for i, r in enumerate(results):
    r["rank"] = i + 1

# === OOD 评估（可选）===
import os
if os.path.exists("data_ood.csv"):
    df_ood = pd.read_csv("data_ood.csv")
    X_ood = df_ood[var_names].values
    y_ood = df_ood["target_col"].values
    y_pred = model.predict(X_ood)
    ood_mse = float(np.mean((y_ood - y_pred) ** 2))
    if results:
        results[0]["ood_mse"] = ood_mse
    print(f"OOD MSE: {ood_mse}")

# === 输出（必须） ===
print("===EVO_PYSR_RESULTS_JSON_BEGIN===")
print(json.dumps(results, ensure_ascii=False))
print("===EVO_PYSR_RESULTS_JSON_END===")
```

## 注意事项

1. **Marker 独占一行**：前后不要有多余空格
2. **JSON 必须是数组**：即使只有一个结果也用 `[{...}]`
3. **equation 字段必须是字符串**：不要传 SymPy 对象
4. **允许其他输出**：marker 之外可以自由 print，工具只提取 marker 之间的内容
5. **TemplateExpressionSpec 模式**：`equations_` 的 `equation` 列是 Julia 格式字符串，照样放入即可
