# PySR API Reference (v1.5.9)

## PySRRegressor

scikit-learn 兼容的符号回归模型。

### Constructor

```python
PySRRegressor(
    model_selection="best",  # "best" | "accuracy" | "score"
    *,
    # --- 运算符 ---
    binary_operators=["+", "-", "*", "/"],
    unary_operators=None,              # e.g. ["sin", "cos", "exp", "log"]
    expression_spec=None,              # ExpressionSpec() | TemplateExpressionSpec(...)

    # --- 搜索控制 ---
    niterations=100,                   # 搜索迭代次数
    populations=31,                    # 进化种群数
    population_size=27,                # 每个种群大小
    max_evals=None,                    # 总评估预算
    ncycles_per_iteration=380,         # 每轮变异循环数

    # --- 表达式大小 ---
    maxsize=30,                        # 最大节点数
    maxdepth=None,                     # 最大树深度
    warmup_maxsize_by=None,            # 逐步放大 maxsize 的比例

    # --- 提前退出 ---
    timeout_in_seconds=None,
    early_stop_condition=None,         # Julia 表达式，如 "f(loss, complexity) = (loss < 0.01) && (complexity < 10)"

    # --- 约束 ---
    constraints=None,                  # {"^": (-1, 1), "sin": 9} 限制子树大小
    nested_constraints=None,           # {"sin": {"cos": 0}} 限制嵌套

    # --- 损失函数 ---
    elementwise_loss=None,             # 默认 "L2DistLoss()"
    loss_function=None,                # 完整 Julia loss 函数
    loss_function_expression=None,     # 用于 TemplateExpressionSpec 的 loss
    loss_scale="log",                  # "log" | "linear"

    # --- 复杂度 ---
    complexity_of_operators=None,      # {"sin": 2, "^": 3}
    complexity_of_constants=None,      # 默认 1
    complexity_of_variables=None,      # 默认 1，或 per-variable list

    # --- 简洁性 ---
    parsimony=0.0,                     # 全局复杂度惩罚
    adaptive_parsimony_scaling=1040.0, # 自适应惩罚缩放
    use_frequency=True,
    use_frequency_in_tournament=True,

    # --- 退火 ---
    alpha=3.17,                        # 初始温度
    annealing=False,

    # --- 变异权重 ---
    weight_add_node=2.47,
    weight_insert_node=0.0112,
    weight_delete_node=0.870,
    weight_do_nothing=0.273,
    weight_mutate_constant=0.0346,
    weight_mutate_operator=0.293,
    weight_swap_operands=0.198,
    weight_rotate_tree=4.26,
    weight_randomize=0.000502,
    weight_simplify=0.00209,
    weight_optimize=0.0,

    # --- 交叉 ---
    crossover_probability=0.0259,
    skip_mutation_failures=True,

    # --- 迁移 ---
    migration=True,
    hof_migration=True,
    fraction_replaced=0.00036,
    fraction_replaced_hof=0.0614,
    topn=12,

    # --- 常数优化 ---
    should_simplify=True,
    should_optimize_constants=True,
    optimizer_algorithm="BFGS",        # "BFGS" | "NelderMead"
    optimizer_nrestarts=2,
    optimizer_f_calls_limit=None,      # 默认 10000
    optimize_probability=0.14,
    optimizer_iterations=8,
    perturbation_factor=0.129,
    probability_negate_constant=0.00743,

    # --- 锦标赛 ---
    tournament_selection_n=15,
    tournament_selection_p=0.982,

    # --- 并行 ---
    parallelism=None,                  # "serial" | "multithreading" | "multiprocessing"
    procs=None,                        # 默认 cpu_count()
    cluster_manager=None,              # "slurm" | "pbs" | ...

    # --- 批处理（大数据集必用） ---
    batching=False,
    batch_size=50,

    # --- 性能 ---
    turbo=False,                       # 实验性 LoopVectorization
    bumper=False,                      # 实验性 Bumper.jl
    precision=32,                      # 16 | 32 | 64
    autodiff_backend=None,             # "Zygote" for reverse-mode AD

    # --- 可复现 ---
    random_state=None,
    deterministic=False,
    warm_start=False,                  # 从上次 fit 继续

    # --- 输出 ---
    verbosity=1,
    progress=True,
    print_precision=5,
    logger_spec=None,                  # TensorBoardLoggerSpec(...)

    # --- 文件 ---
    run_id=None,
    output_directory=None,             # 默认 "outputs"
    temp_equation_file=False,
    delete_tempfiles=True,

    # --- 导出 ---
    output_jax_format=False,
    output_torch_format=False,
    extra_sympy_mappings=None,
    extra_torch_mappings=None,
    extra_jax_mappings=None,

    # --- 预处理 ---
    denoise=False,                     # GP 去噪
    select_k_features=None,            # 随机森林特征预选

    # --- 物理量纲 ---
    dimensional_constraint_penalty=None,  # 默认 1000.0
    dimensionless_constants_only=False,
)
```

### fit()

```python
model.fit(
    X,                          # ndarray | DataFrame, (n_samples, n_features)
    y,                          # ndarray | DataFrame, (n_samples,) 或 (n_samples, n_targets)
    *,
    weights=None,               # 样本权重，与 y 同形
    variable_names=None,        # 变量名列表（或用 DataFrame 列名）
    complexity_of_variables=None,
    X_units=None,               # 物理单位，如 ["m", "kg", "s^-1"]
    y_units=None,
    category=None,              # ParametricExpressionSpec 用
)
```

> **大数据集提示**：超过 10000 行时建议启用 `batching=True, batch_size=50`。
>
> **搜索预算提示**：默认参数（`niterations=100`, `ncycles_per_iteration=380`）仅适合简单问题。对于复杂非线性系统（如含交叉项、自激机制的 ODE），需要显著增加搜索预算，否则只能发现平凡线性解。关键原则：
> - **数据预处理**：万行以上数据应先降采样（保留动力学特征）再搜索，全量数据留给验证
> - **迭代次数**：与问题复杂度成正比。如果 Pareto 前沿还在变化，说明搜索未收敛
> - **表达式大小**：`maxsize` 需要容纳目标方程的所有项，含交叉项的方程通常需要 25+
> - **时间预算**：`timeout_in_seconds` 应留出充足时间（小时量级），过早截断会丢失好解。复杂非线性系统的搜索通常需要数小时
> - **收敛判断**：观察 `equations_` 的 Pareto 前沿是否稳定，而非仅看 loss 数值

### equations_ (结果 DataFrame)

| 列名 | 类型 | 说明 |
|------|------|------|
| `equation` | str | 表达式字符串 |
| `loss` | float | 损失值 |
| `complexity` | int | 复杂度（节点数） |
| `score` | float | 得分 = -log(loss_i/loss_{i-1}) / (complexity_i - complexity_{i-1}) |
| `sympy_format` | sympy.Expr | SymPy 表达式（仅标准模式） |
| `lambda_format` | callable | 可调用函数 f(X) -> ndarray |

### get_best()

```python
best = model.get_best(index=None)  # pd.Series
print(best["equation"], best["loss"], best["complexity"])
```

`model_selection` 策略：
- `"accuracy"` — 最低 loss
- `"score"` — 最高 score
- `"best"`（默认）— 最高 score，限制 loss < 1.5x 最优

### predict()

```python
y_pred = model.predict(X, index=None)
```

### sympy() / latex()

```python
expr = model.sympy()       # sympy 表达式（仅标准模式）
tex = model.latex()        # LaTeX 字符串
table = model.latex_table() # LaTeX 表格
```

### 保存/加载

```python
# 保存（自动 checkpoint）
model.fit(X, y)  # 产生 outputs/{run_id}/

# 加载
model = PySRRegressor.from_file(run_directory="outputs/my_run/")
model.refresh()  # 重新读取方程
```

### 多目标回归

当 y 形状为 (n_samples, n_targets) 时：
- `equations_` 变为 `list[DataFrame]`
- `get_best()`, `predict()` 等返回列表

## 自定义损失函数

### elementwise_loss（最常用）

```python
# 内置
PySRRegressor(elementwise_loss="L2DistLoss()")   # MSE（默认）
PySRRegressor(elementwise_loss="L1DistLoss()")   # MAE
PySRRegressor(elementwise_loss="HuberLoss(1.0)") # Huber

# 自定义
PySRRegressor(elementwise_loss="myloss(x, y) = abs(x - y)^1.5")
PySRRegressor(elementwise_loss="myloss(x, y, w) = w * (x - y)^2")  # 加权
```

### loss_function（完整控制）

```python
PySRRegressor(loss_function="""
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag; return L(Inf); end
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end
""")
```

## 自定义运算符

```python
PySRRegressor(
    binary_operators=["+", "-", "*", "/", "myop(x, y) = x^2 + y"],
    unary_operators=["sin", "cos", "inv(x) = 1/x"],
    extra_sympy_mappings={
        "inv": lambda x: 1/x,
        "myop": lambda x, y: x**2 + y,
    },
)
```

## 约束

### constraints — 限制子树大小

```python
PySRRegressor(
    constraints={
        "^": (-1, 1),     # 幂运算：左子树无限制，右子树最多 1 个节点
        "sin": 9,          # sin 的参数最多 9 个节点
    }
)
```

### nested_constraints — 限制运算符嵌套

```python
PySRRegressor(
    nested_constraints={
        "sin": {"sin": 0, "cos": 0},  # sin 内部不允许 sin/cos
        "cos": {"sin": 0, "cos": 0},
    }
)
```

## 物理量纲分析

```python
model = PySRRegressor(
    dimensional_constraint_penalty=1000.0,
    dimensionless_constants_only=True,
)
model.fit(
    X, y,
    X_units=["m", "m/s"],
    y_units="m/s^2",
)
```

## TensorBoard 日志

```python
from pysr import PySRRegressor, TensorBoardLoggerSpec

model = PySRRegressor(
    logger_spec=TensorBoardLoggerSpec(log_dir="logs/run1"),
)
```
