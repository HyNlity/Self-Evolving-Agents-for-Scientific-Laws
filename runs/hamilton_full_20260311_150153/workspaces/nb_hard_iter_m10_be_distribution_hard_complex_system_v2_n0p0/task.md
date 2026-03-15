# NewtonBench 交互式方程发现任务（Hamilton 单 Agent）

## 背景

你正在 NewtonBench 的交互式科学定律发现环境中工作。  
任务不是“写一个看起来合理的公式”，而是通过**实验-分析-评测-迭代**闭环，得到可验证、可解释、可复现的定律。

请始终假设：你不知道真实方程，必须通过数据和评测反馈逐步逼近。

## 任务输入（运行时由 `--task` 提供）

任务描述通常包含以下字段：

```yaml
profile: newtonbench
module: m0_gravity
system: vanilla_equation
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false
```

其中：
- `module/system/difficulty/law_version/noise` 决定数据生成与评测条件；
- 最终函数签名必须以 `generate_task_prompt.py` 返回的 `function_signature` 为准。

## 核心目标

1. 发现满足签名约束的 `discovered_law(...)`；
2. 在评测中达到尽可能好的 `rmsle / exact_accuracy / symbolic_equivalent`；
3. 形成有物理解释的结论，并把迭代过程沉淀到 `findings.md` 与 `plan.md`。

## 反锚定与回滚要求（重点）

1. `generate_task_prompt.py` 中的示例方程只用于格式演示，不是正确答案提示。
2. 每轮至少维护 2 个候选家族（family），其中必须包含至少 1 个“非示例家族”。
3. “非示例家族”必须与示例家族核心结构不同，不能只是改常数或加一个幂次乘子。
4. 例如示例是 `1/(exp(c*omega/T)-1)`，则 `omega^k/(exp(...)-1)` 仍算同核，不满足反锚定要求。
5. 如果同一家族连续 2 轮没有实质提升（如 `symbolic_equivalent=false` 且 `rmsle` 无明显下降），必须在 `plan.md` 标记为禁用并切换家族。
6. 每轮必须读取并维护 `plan.md` 中的 `当前最优`：
   - 只有候选更优时才更新；
   - 若候选退化/失败，必须回滚到 `当前最优`，下一轮从最优解继续。

## 强制执行顺序（每一轮）

1. 先调用 `generate_task_prompt.py`，拿到函数签名和参数说明；
2. 调用 `run_experiment.py`（必须传 `--inputs-json`，且不能空）获取实验数据；
3. 基于数据提出候选方程，并在结束前调用 `evaluate_submission.py`；
4. 根据评测结果更新 `findings.md`（记录失败点/改进点）和 `plan.md`（下一轮新参数）；
5. 最后调用 `finish(...)`。

## 完成标准

只有同时满足以下条件，才允许 `finish(task_completed="true")`：

1. 至少一次成功的 `run_experiment.py`（`exit_code=0`）；
2. 至少一次成功的 `evaluate_submission.py`（`exit_code=0`）；
3. `finish.message` 中包含（且与评测使用的 law-text 完全一致）：

```text
<final_law>
def discovered_law(...):
    ...
</final_law>
```

4. 已完成 `findings.md` 和 `plan.md` 的本轮更新（含下一轮计划或最优更新说明）。

若未满足，则必须 `finish(task_completed="false")`，并明确：
- 本轮失败原因（含最近评测指标与错误）；
- 下一轮计划（至少 1 组新的 `--inputs-json`，不得原样重复）。

## 实验设计要求

1. 每轮至少给出 1 组新实验参数，建议 3 组以上覆盖不同数量级；
2. 参数采样要有“探索性”：避免总在同一局部范围取值；
3. 若上一轮失败，本轮必须体现新假设（不是只改常数位数）；
4. 所有结论必须能追溯到实验结果或评测输出。

## 评测指标解读（用于决策）

- `rmsle`：数值误差（越小越好）；
- `exact_accuracy`：精确匹配度（越高越好）；
- `symbolic_equivalent`：是否符号等价（`true/false`）；
- `error`：评测失败或执行异常信息（必须记录并处理）。

## 发现记录要求（重点）

在 `findings.md` 中，对每个“候选/最终方程”，必须包含：

1. **方程表达式**及各项的物理解释；
2. **系数表**（跨实验条件），标注哪些系数基本不变（结构属性）、哪些随条件变化（情景属性）；
3. **物理洞察**：该方程揭示了什么规律（例如尺度关系、主导项、极限行为）；
4. **消融分析**：逐个去掉关键项或固定系数，说明该项是否必要。

为保证系统自动回填兼容，`findings.md` 必须保留以下表头（逐轮追加）：

```markdown
## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
```

同时必须保留以下标记位，所有“新增内容”都通过 `str_replace_editor` 在标记位前追加：

```markdown
<!-- APPEND_RESULTS -->
<!-- APPEND_FINDINGS -->
<!-- APPEND_NEXT -->
```

推荐追加方式：
- `old_str = "<!-- APPEND_RESULTS -->"`
- `new_str = "- Round N: ...\\n<!-- APPEND_RESULTS -->"`
- 必须使用 `str_replace_editor` 的 `str_replace`（不要用 `insert`）来更新标记位，保证标记唯一。

建议按下列骨架组织 `findings.md`：

```markdown
# 研究发现

## 关键洞察
<!-- APPEND_FINDINGS -->
- 本轮主要发现 ...

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
<!-- APPEND_RESULTS -->
| Round N | hypothesis_x | ... | - | - | rmsle=..., exact=..., symbolic=... |

## Worth Trying Next
<!-- APPEND_NEXT -->
- Round N+1 参数草案：...

## 候选方程解析（Round N）
### 1) 方程与物理解释
- 方程：...
- 项解释：...

### 2) 系数表（跨实验条件）
| 系数 | 数值/范围 | 稳定性标注 | 物理解释 |
|------|-----------|------------|----------|
| c1 | ... | 结构属性(基本不变) | ... |
| c2 | ... | 情景属性(随条件变化) | ... |

### 3) 物理洞察
- ...

### 4) 消融分析
- 去掉项 A：...
- 去掉项 B：...

## 最优方程演化
- Round 1: ...
- Round 2: ...
```

## Baseline 参照（NewtonBench 协议基线）

最低合格基线是：
1. 协议完整执行（prompt -> experiment -> evaluation -> finish）；
2. 有可执行且签名正确的 `final_law`；
3. 有可复查的 `findings.md` 记录（不是口头描述）。

“比基线更好”通常意味着一项或多项提升：
1. `rmsle` 更低，`exact_accuracy`/`symbolic_equivalent` 更高；
2. 方程更简洁（更少冗余项）；
3. 跨输入区间更稳健（不依赖局部参数凑巧）；
4. 物理解释更清晰，消融证据更充分。
