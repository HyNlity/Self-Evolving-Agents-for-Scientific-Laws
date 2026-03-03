---
name: pysr
description: "PySR symbolic regression API guide. Covers PySRRegressor parameters, TemplateExpressionSpec for constrained search, differential operators, custom loss functions, and output format. Use get_info for quick reference, references for detailed guides."
license: null
---

# PySR Skill

PySR (v1.5.9) 符号回归工具的完整 API 指南。让你充分利用 PySR 的所有能力。

## Quick Reference

### 基本用法
```python
from pysr import PySRRegressor
model = PySRRegressor(niterations=100, binary_operators=["+","-","*","/"])
model.fit(X, y, variable_names=["x","v"])
print(model.equations_)  # DataFrame: equation, loss, complexity, score
best = model.get_best()  # 最优方程
```

### 模板搜索（约束结构）
```python
from pysr import PySRRegressor, TemplateExpressionSpec
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x", "v"],
    combine="f(x, v) + g(x)",
)
model = PySRRegressor(expression_spec=spec, ...)
model.fit(X, y)
```

### 微分算子（ODE 发现）
```python
spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x", "t"],
    combine="D(f, 1)(x, t)",  # df/dx
)
```

## Resources

| Resource | Path | Purpose |
|----------|------|---------|
| API 完整参考 | `references/api_reference.md` | 全部 ~80 个参数 + 方法 |
| 模板表达式指南 | `references/template_guide.md` | TemplateExpressionSpec 详解 |
| 输出格式规范 | `references/output_format.md` | 脚本必须遵循的 marker 输出格式 |

## 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `niterations` | 100 | 搜索迭代次数 |
| `maxsize` | 30 | 表达式最大节点数 |
| `parsimony` | 0.0 | 复杂度惩罚（越大越偏好简洁） |
| `populations` | 31 | 进化种群数 |
| `timeout_in_seconds` | None | 超时（秒） |
| `binary_operators` | ["+","-","*","/"] | 二元运算符 |
| `unary_operators` | None | 一元运算符 |
| `expression_spec` | ExpressionSpec() | 表达式类型（模板/标准） |
| `elementwise_loss` | "L2DistLoss()" | 损失函数 |
| `batching` | False | 大数据集启用 batch |
| `batch_size` | 50 | batch 大小 |
| `early_stop_condition` | None | 提前终止条件 |
| `constraints` | None | 运算符参数约束 |
| `nested_constraints` | None | 嵌套运算符约束 |
| `complexity_of_operators` | None | 运算符复杂度权重 |
| `warm_start` | False | 从上次结果继续搜索 |
| `denoise` | False | GP 去噪 |
| `select_k_features` | None | 随机森林特征预选 |
