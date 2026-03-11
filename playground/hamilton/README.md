# Hamilton Playground (NewtonBench-Ready)

Hamilton 是 EvoMaster 中用于“科学规律发现”的单 Agent playground。  
当前版本重点支持 **NewtonBench**：通过 `prompt + skill` 驱动交互实验、规律假设、评测闭环，而不是把 benchmark 逻辑硬编码到 core。

## 1. 目标与设计原则

### 目标
1. 在不分叉 playground 的前提下，用 Hamilton 主线跑 NewtonBench。
2. 让 Agent 在任务内完成“假设 -> 实验 -> 评测 -> 收敛”的闭环。
3. 产出可汇总、可复盘、可比较的结果（协议合规 + 质量指标）。

### 设计原则
1. **单 Agent**：尽量减少多 Agent 协调开销。
2. **Skill 驱动**：环境交互和评测通过 `newtonbench` skill 脚本完成。
3. **最小系统职责**：系统做编排和护栏，不做任务语义推理。
4. **可追踪**：运行日志、轨迹、结构化 summary 全部落盘。

## 2. 架构概览

```text
run.py
  -> HamiltonPlayground
       -> RoundExp
            -> Agent (LLM + tools + skills)
                 -> use_skill(run_script)
                      -> newtonbench scripts
                           - generate_task_prompt.py
                           - run_experiment.py
                           - evaluate_submission.py
```

### 组件职责
- `run.py`
  - 任务入口（`--task` / `--task-file`）
  - 运行目录管理（`runs/<agent>_<timestamp>/...`）
- `HamiltonPlayground`
  - workspace 初始化
  - 多轮编排（NewtonBench profile 当前默认 `max_rounds: 1`）
  - 记录 `playground/hamilton/records/experiment_*.json`
- `RoundExp`
  - 单轮执行
  - 从 `finish(task_completed=...)` 解析停止信号
  - 执行 NewtonBench 协议与质量护栏（如签名校验、RMSLE 阈值）
- `newtonbench` skill
  - 对 NewtonBench 环境的统一桥接，不侵入 core

## 3. NewtonBench 任务数据流

### Step A: 任务输入
任务描述建议包含：

```yaml
profile: newtonbench
module: m0_gravity
system: vanilla_equation
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false
```

### Step B: Agent 执行动作
推荐动作序列：
1. `generate_task_prompt.py` 获取 `function_signature` 与任务提示。
2. `run_experiment.py` 发起实验（每次最多 20 组输入）。
3. 形成候选方程后调用 `evaluate_submission.py` 获取评测。
4. 在 `finish.message` 中输出 `<final_law>...</final_law>` 并设置 `task_completed`。

### Step C: 系统护栏判定
`RoundExp` 会在 `task_completed="true"` 前做协议检查，核心包含：
1. 至少一次成功的 `run_experiment.py`。
2. 至少一次成功的 `evaluate_submission.py`。
3. `finish.message` 包含 `<final_law>def discovered_law(...)</final_law>`。
4. 可选质量护栏：签名匹配、RMSLE 有限、RMSLE 不超过阈值。

如果违规，系统会把 `satisfied` 改为 `false`，并在 `signal.protocol.violations` 记录原因。

## 4. 目录结构（当前关键文件）

```text
playground/hamilton/
├── core/
│   ├── playground.py                    # HamiltonPlayground
│   └── exp.py                           # RoundExp + protocol guard
├── prompts/
│   ├── hamilton_newtonbench_system.txt # NewtonBench 系统提示词
│   └── hamilton_newtonbench_user.txt   # NewtonBench 用户提示词
├── workspace_newtonbench/
│   └── task.md                          # NewtonBench 任务模板
├── tasks/
│   └── *.json                           # 批量 task-file（可选）
└── records/
    └── experiment_*.json                # 每次实验记录

evomaster/skills/newtonbench/
├── scripts/
│   ├── generate_task_prompt.py
│   ├── run_experiment.py
│   └── evaluate_submission.py
└── references/protocol.md

scripts/newtonbench/
├── generate_hamilton_tasks.py           # 生成批量任务
└── summarize_hamilton_run.py            # 汇总 run 结果
```

## 5. 快速开始

### 5.1 环境变量

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_BASE_URL="https://llm.dp.tech"   # 如使用兼容网关
```

### 5.2 单任务运行

```bash
python run.py --agent hamilton \
  --config configs/hamilton/newtonbench.yaml \
  --task "profile: newtonbench
module: m0_gravity
system: vanilla_equation
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false"
```

### 5.3 批量任务（推荐先跑 easy36）

1. 生成任务文件

```bash
python scripts/newtonbench/generate_hamilton_tasks.py \
  --output playground/hamilton/tasks/newtonbench_easy36.json \
  --modules all \
  --systems vanilla_equation,simple_system,complex_system \
  --difficulties easy \
  --law-versions v0 \
  --noise-levels 0.0
```

2. 执行批量运行

```bash
python run.py --agent hamilton \
  --config configs/hamilton/newtonbench.yaml \
  --task-file playground/hamilton/tasks/newtonbench_easy36.json
```

3. 汇总结果

```bash
RUN_DIR=$(ls -1dt runs/hamilton_* | head -n1)
python scripts/newtonbench/summarize_hamilton_run.py \
  --run-dir "$RUN_DIR" \
  --task-file playground/hamilton/tasks/newtonbench_easy36.json \
  --auto-evaluate
```

### 5.4 单个 Hard 任务（迭代展示）

用于展示“通过多轮迭代不断优化”的效果（单题 + `max_rounds=10`）：

```bash
RUN_DIR="runs/hamilton_hard_iter_$(date +%Y%m%d_%H%M%S)"
python run.py --agent hamilton \
  --config configs/hamilton/newtonbench_single_hard_iter.yaml \
  --task-file playground/hamilton/tasks/newtonbench_single_hard_iter.json \
  --run-dir "$RUN_DIR"

python scripts/newtonbench/summarize_hamilton_run.py \
  --run-dir "$RUN_DIR" \
  --task-file playground/hamilton/tasks/newtonbench_single_hard_iter.json \
  --auto-evaluate
```

## 6. 输出产物说明

运行后会生成：

```text
runs/hamilton_YYYYMMDD_HHMMSS/
├── config.yaml
├── logs/
│   └── <task_id>.log
├── trajectories/
│   └── <task_id>/trajectory.json
├── workspaces/
│   └── <task_id>/...
├── newtonbench_summary.json
└── newtonbench_trials.csv
```

`newtonbench_summary.json` 关键字段：
- `total_tasks`, `completed_tasks`
- `task_completed_true`, `task_completed_false`
- `with_run_experiment_success`, `with_evaluate_success`
- `protocol_core_ok`, `protocol_full_ok`
- `avg_exact_accuracy`, `avg_rmsle`, `avg_total_tokens`

## 7. 当前已实现能力

1. NewtonBench skill 全链路（生成任务提示 / 运行实验 / 提交评测）。
2. 非标准 function-calling 网关兼容（文本 JSON 回收 tool call）。
3. 协议护栏落地（完成前必须满足实验与评测约束）。
4. 批量任务生成与结果汇总脚本。
5. 质量护栏验证样本已跑通（低质量样本会被阻断为 `task_completed=false`）。

## 8. 已知限制

1. 当前 `max_rounds` 在 NewtonBench profile 下为 `1`，复杂题探索深度有限。
2. easy36 baseline 中仍存在高 RMSLE 与函数签名不匹配样本。
3. `playground/hamilton/records/` 与 run 目录会快速增长，建议定期归档。

## 9. 开发建议

1. 继续 Phase 8 第二轮改造：基于失败样本强化提示词约束（尤其是签名与评测复述）。
2. 增加“失败重试 + 断点续跑”批量执行层，支撑 324 题 full benchmark。
3. 补充结构化工件落盘（每轮实验输入、输出、最后一次评测 JSON），减少后处理对日志解析的依赖。

---

如需快速定位整体计划与实验结论，请优先查看：
- `task_plan.md`
- `findings.md`
- `progress.md`
