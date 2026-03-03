# TemplateExpressionSpec 详解

## 概述

`TemplateExpressionSpec` 让你约束搜索空间的结构。你定义一个组合公式 `combine`，其中包含若干子表达式符号（如 `f`, `g`），PySR 分别搜索每个子表达式。

**适用场景**：
- 你有先验知识（如 "a = 弹性力 + 阻尼力"）
- ODE 发现：需要微分算子 `D(f, n)`
- 多尺度分解：不同变量组合方式已知

## 基本用法

```python
from pysr import PySRRegressor, TemplateExpressionSpec

# 约束结构: a = f(x, v) + g(x)
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x", "v"],
    combine="f(x, v) + g(x)",
)

model = PySRRegressor(
    expression_spec=spec,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp"],
    niterations=100,
    maxsize=20,
)
model.fit(X, y)
```

## 参数说明

```python
TemplateExpressionSpec(
    combine: str,                          # 组合公式（Julia 语法）
    *,
    expressions: list[str],                # 子表达式符号名列表
    variable_names: list[str],             # 变量名列表
    parameters: dict[str, int] | None,     # 可优化参数（名称 → 向量长度）
)
```

### combine

Julia 函数字符串，定义子表达式如何组合：
- 子表达式作为函数调用：`f(x, v)` 表示 f 接受 x 和 v 作为输入
- 支持所有 Julia 运算：`+`, `-`, `*`, `/`, `^`, `sin()`, ...
- 支持微分算子：`D(f, n)` 对 f 的第 n 个参数求导

### expressions

子表达式的符号名。每个都会被 PySR 独立搜索：
```python
expressions=["f", "g"]  # PySR 同时搜索 f 和 g 的最优形式
```

### variable_names

combine 公式中使用的变量名。必须与训练数据 X 的列一一对应：
```python
variable_names=["x", "v"]  # X 必须有 2 列
```

### parameters

可优化的外部参数，PySR 会在搜索中同时优化：
```python
parameters={"p": 3}  # p 是长度为 3 的向量参数
# combine 中可用 p[1], p[2], p[3]
```

## 微分算子

`D(f, n)(args...)` — 对子表达式 f 的第 n 个参数求偏导，然后在 args 处求值。

### 示例：发现 ODE dx/dt = f(x, t)

```python
# 数据: t, x（时序数据）
# 目标: 用 PySR 发现 dx/dt 的表达式

spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x", "t"],
    combine="D(f, 2)(x, t)",  # df/dt — 对第 2 个参数 t 求导
)

# 目标 y 设为 0（因为 dx/dt - f(x,t) = 0 在 PySR 内部处理）
# 或更常见：先数值估计 dx/dt，直接拟合 f(x, t) ≈ dx/dt
```

### 示例：二阶 ODE d²x/dt² = f(x, dx/dt)

```python
spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x", "v"],  # v = dx/dt
    combine="f(x, v)",
)
# 用 a = d²x/dt² 作为 y，直接拟合
```

### 示例：混合力模型 a = f(x) + g(v)

```python
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x", "v"],
    combine="f(x) + g(v)",
)
```

## 进阶示例

### 带参数的模板

```python
spec = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["x"],
    combine="p[1] * f(x) + p[2]",
    parameters={"p": 2},
)
# PySR 搜索 f(x) 的形式，同时优化系数 p[1], p[2]
```

### 复杂组合

```python
# 两个独立项的乘积
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x", "v", "t"],
    combine="f(x, v) * g(t)",
)

# 嵌套 + 三角函数
spec = TemplateExpressionSpec(
    expressions=["f", "g"],
    variable_names=["x", "v"],
    combine="sin(f(x, v)) + g(x)^2",
)
```

### 自定义 loss（配合模板）

```python
model = PySRRegressor(
    expression_spec=spec,
    loss_function_expression="""
    function eval_loss(expression, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(expression, dataset.X, options)
        if !flag; return L(Inf); end
        return sum((prediction .- dataset.y) .^ 2) / dataset.n
    end
    """,
)
```

## 重要限制

1. **不支持 SymPy 导出**：`model.sympy()` 不可用
2. **不支持 LaTeX 导出**：`model.latex()` 不可用
3. **不支持 JAX/PyTorch 导出**
4. **结果格式不同**：`equations_` 中有 `julia_expression` 列，`equation` 列是 Julia 字符串格式
5. **预测在 Julia 中执行**：`predict()` 调用 Julia，不是 NumPy

## 最佳实践

1. **从简单开始**：先用标准 `ExpressionSpec` 跑一轮，了解数据结构
2. **逐步约束**：根据第一轮结果，设计合理的 combine 模板
3. **变量对齐**：`variable_names` 的顺序必须与 X 的列顺序一致
4. **适当 maxsize**：模板模式下每个子表达式的 maxsize 独立生效
5. **ODE 发现推荐**：先用有限差分估计导数，直接拟合 f(x,v) ≈ a，而非用 D 算子（更稳定）
