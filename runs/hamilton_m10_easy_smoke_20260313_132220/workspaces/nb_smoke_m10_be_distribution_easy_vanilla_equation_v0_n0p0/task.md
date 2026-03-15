# NewtonBench 方程发现任务（Hamilton 单 Agent）

## 背景

NewtonBench 是交互式科学定律发现基准。你无法直接访问真实方程，只能通过实验接口采样，再用评测接口验证候选方程。

每个任务由 `module/system/difficulty/law_version/noise` 定义，代表不同物理情境与观测接口。你的目标不是“写一个看起来像的公式”，而是完成可复查的闭环研究过程：

实验 -> 候选 -> 评测 -> 迭代 -> 收敛。

## 任务目标

1. 产出满足签名约束的 `discovered_law(...)`。
2. 在可执行评测上尽量优化 `rmsle / exact_accuracy / symbolic_equivalent`。
3. 给出清晰的物理解释与失败分析，让 `findings.md` 成为可复现研究记录。

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

## 本轮最小流程（必须）

1. 调用 `generate_task_prompt.py` 获取签名、参数说明、任务提示。
2. 调用 `run_experiment.py` 采样（使用 `--inputs-json/--inputs-file`，不要用 `--num-samples`）。
3. 在 `runtime.search_mode=pysr_assisted` 下，调用 `fit_pysr_candidates.py` 生成候选：
   - 若此前没有成功 fit，本轮首次 fit 必须带 `--inputs-json` 且 >=8 组唯一输入。
   - 若返回 `Not enough valid samples`，本轮继续补样并重试 fit，不要空参重复调用。
   - 若观测是积分型 `total_power`（如 `m10_be_distribution/complex_system`），优先采窄带样本：`bandwidth / center_frequency <= 0.05`。
   - 对这些窄带样本，可用 `total_power / (bandwidth * center_frequency^3)` 作为 `n(omega, T)` 的工作 proxy，再交给 PySR 搜索结构。
4. 调用 `evaluate_submission.py` 评测候选（使用 `--law-text` / `--law-file`，不要传 `--candidate/--submission/--system/--noise`）。
5. 更新 `findings.md` 与 `plan.md`（包含本轮决策、指标、下一步）。
6. 调用 `finish(...)` 结束本轮。

## 工具调用约束（必须）

1. 运行 NewtonBench 脚本时，必须使用 `use_skill(action="run_script", skill_name="newtonbench", ...)`。
2. 不要用 `execute_bash` 直接执行 `python3 scripts/*.py`，因为当前 workspace 下通常不存在该相对路径。
3. `execute_bash` 只用于通用 shell 操作（查看文件、统计、排障），不用于调用 NewtonBench 脚本。

## 环境约束（必须）

1. 禁止执行系统级安装命令（`sudo`、`apt-get`、`yum`、`dnf`、`pacman`）。
2. 禁止请求 root 权限或系统管理员密码。
3. 依赖缺失时，只允许工作目录内用户态修复（`.julia-bin`、`.julia_depot`、`.venv`、环境变量）。
4. 如果用户态修复失败，记录失败并继续可执行实验，不要发起系统级安装。

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
5. `<final_law>` 必须与“最后一次成功 `evaluate_submission.py` 的候选”一致；不要在评测后私自改式子再 finish。

如果本轮没有明显改进，`task_completed="false"`，并在 `plan.md` 给出下一轮可执行动作。

## 评估口径（NewtonBench 定制）

1. **可执行性优先**：候选必须可通过 `evaluate_submission.py` 正常执行，无语法/数值崩溃。
2. **泛化指标**：重点看 `rmsle`（越小越好）、`exact_accuracy`（越高越好）、`symbolic_equivalent`（尽量为 true）。
3. **稳定性**：若出现 `math range error` / `overflow`，先修复数值稳定性（如指数输入截断）再继续。
4. **迭代质量**：新候选退化时必须回滚到当前最优，不可盲目覆盖。

## 迭代策略（轻量）

1. 示例公式只用于格式演示，不是答案。
2. 若同结构连续 2 轮无改进，切换结构族或变量组合。
3. 先保证协议闭环，再追求指标上限。
4. 每次评测后维护“当前最优候选”；若新候选退化，必须回滚，不得用退化候选作为最终输出。
5. 在 `pysr_assisted` 模式下，优先以最近一次 PySR top 候选为主线；手写先验公式仅可作为对照，且必须成功评测并优于当前最优后才能替换。

## findings.md 记录规范（必须）

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

注意：即使 NewtonBench 不直接给出 MSE，也要在该表中填写可比指标（如 `RMSLE=...`、`Exact=...`），保持表结构稳定。

每轮至少追加 3 类信息：

1. 一条关键洞察（实验观察或失败归因）。
2. 一行实验结果（含 `rmsle/exact_accuracy/symbolic_equivalent/error`）。
3. 一条下一轮动作（具体到参数或结构，不要写空泛描述）。

## findings.md 深度分析规范（NewtonBench 定制）

对每个重点候选（至少当前最优）必须包含：

1. **方程表达式与物理解释**：逐项说明其可能物理意义。
2. **参数/系数敏感性表**：标注哪些参数在不同输入尺度下稳定，哪些会导致明显退化。
3. **物理洞察**：方程揭示了什么关系（尺度律、极限行为、主导项切换等）。
4. **消融分析**：去掉关键项后，评测指标如何变化，验证每个项必要性。

推荐使用如下段落标题（便于自动审阅）：

```markdown
## 候选方程解析（Round N）
### 1) 方程与物理解释
### 2) 参数/系数敏感性
### 3) 物理洞察
### 4) 消融分析
```

## Baseline 与改进定义（NewtonBench 定制）

本任务中，“更好”至少满足以下之一：

1. `rmsle` 明显下降；
2. `exact_accuracy` 或 `symbolic_equivalent` 提升；
3. 在指标相近时，方程更简洁且更稳定（更少数值异常）；
4. 同等指标下，物理解释更完整、消融证据更充分。
