# NewtonBench 交互式方程发现任务（Hamilton 单 Agent）

## 任务目标

在 NewtonBench 环境中完成“实验 -> 候选 -> 评测 -> 迭代 -> 收敛”的闭环，产出可执行的 `discovered_law(...)`。

你不知道标准答案，必须依赖实验数据和评测反馈逐轮逼近。

## 运行输入

运行时任务通常包含：

```yaml
profile: newtonbench
module: m10_be_distribution
system: complex_system
difficulty: hard
law_version: v2
noise: 0.0
code_assisted: false
```

最终函数签名以 `generate_task_prompt.py` 返回的 `function_signature` 为准。

## 本轮最小流程

1. 调用 `generate_task_prompt.py`（拿签名与参数说明）。
2. 调用 `run_experiment.py`（至少 1 组新参数，建议 2~6 组）。
3. `runtime.search_mode=pysr_assisted` 时必须调用 `fit_pysr_candidates.py` 生成候选（原版 Hamilton 主流程）。
4. 调用 `evaluate_submission.py` 比较候选。
5. 更新 `findings.md` 与 `plan.md`。
6. 调用 `finish(...)`。

## 协议闭环要求（必须）

1. 至少一次成功 `run_experiment.py`。
2. 至少一次成功 `evaluate_submission.py`。
3. `finish.message` 包含：

```text
<final_law>
def discovered_law(...):
    ...
</final_law>
```
4. `runtime.search_mode=pysr_assisted` 时，至少一次成功 `fit_pysr_candidates.py`。

如果本轮没有明显改进，`task_completed="false"`，并在 `plan.md` 写下一轮动作。

## 迭代策略（轻量）

1. 示例公式只用于格式演示，不是答案。
2. 若某结构连续 2 轮无改进，切换到不同结构族。
3. 新候选退化时，回滚到 `plan.md` 中“当前最优”。

## findings.md 记录格式（必须保留标记）

请保留并使用以下标记位，通过 `str_replace` 在标记前追加内容：

```markdown
<!-- APPEND_FINDINGS -->
<!-- APPEND_RESULTS -->
<!-- APPEND_NEXT -->
```

基础表头必须存在：

```markdown
## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
```

每轮至少追加：
- 一条“关键洞察”；
- 一条实验结果表格行（包含 `rmsle/exact_accuracy/symbolic_equivalent`）；
- 一条下一轮计划。

## findings.md 深度内容（建议）

对本轮最有希望的方程，补充：
1. 方程表达式及项解释；
2. 系数稳定性说明（哪些稳定、哪些随条件变化）；
3. 物理洞察（主导项、极限行为、尺度关系）；
4. 简要消融（去掉关键项后的影响）。
