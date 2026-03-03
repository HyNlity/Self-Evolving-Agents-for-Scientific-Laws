# Hamilton - 符号回归 Agent

Hamilton 是基于 EvoMaster 框架的符号回归（Symbolic Regression）Agent，专门用于在**过完备变量**环境下发现数学方程。

## 架构

单 Agent + HCC（Hierarchical Cognitive Caching）分层记忆，四阶段闭环迭代。

```
┌─────────────────────────────────────────────────────┐
│                HamiltonPlayground                    │
│              (多轮循环编排 + L2 post-check)           │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                     RoundExp                         │
│  系统: 重置 L1 → Agent 执行 → 解析 signal → post-check │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                  单 Agent 闭环                        │
│  Discovery → Verification → Promotion → Finish       │
│  (变量分析/PySR/拟合) (残差/OOD) (写L2) (signal)      │
└─────────────────────────────────────────────────────┘
```

### 每轮流程

```
Round N 开始
    │
    ├─ 系统: 重置 execution_trace.md（L1 工作记忆）
    ├─ 系统: 快照 L2 文件 mtime（用于 post-check）
    │
    ├─ Agent 执行（四阶段闭环）
    │     ├─ Phase 1 Discovery: 读 L2 → 变量分析 → 拟合/PySR
    │     ├─ Phase 2 Verification: 残差分析 → OOD 验证
    │     ├─ Phase 3 Promotion: 提炼结论到 findings.md + plan.md
    │     └─ Phase 4 Finish: 发出 satisfied 信号
    │
    ├─ 系统: 解析 satisfied 信号
    ├─ 系统: L2 post-check（检测 findings.md / plan.md 是否更新）
    │
Round N 结束 → satisfied=true ? 停止 : 进入 Round N+1
```

### HCC 分层记忆

| 层级 | 文件 | 生命周期 | 内容 |
|------|------|----------|------|
| **L1** | `execution_trace.md` | 每轮重置（锯齿形） | 当前轮的操作记录、指标、工作笔记 |
| **L2** | `plan.md` | 持久积累（阶梯形） | 战略计划、当前最优、策略队列、失败方法 |
| **L2** | `findings.md` | 持久积累（阶梯形） | 验证结论、实验结果表、最优方程演化 |

L2 文件驱动跨轮知识传递：Agent 每轮读取 L2 → 基于历史做决策 → 将新发现提炼回 L2。

---

## 文件结构

```
playground/hamilton/
├── core/
│   ├── playground.py      # HamiltonPlayground: 多轮编排 + workspace 初始化
│   ├── exp.py             # RoundExp: 单轮执行 + signal 解析 + L2 post-check
│   └── constants.py       # Signal markers、字段定义
├── prompts/
│   ├── hamilton_system.txt # Agent 系统提示（四阶段协议 + HCC 规范）
│   └── hamilton_user.txt   # Agent 用户提示（任务注入）
├── benchmarks/
│   └── viv/               # VIV 多风速基准数据 + 任务描述
├── workspace/             # 模板目录（自动 seed 到 run workspace）
│   ├── task.md            # 任务描述（含数据路径和评估标准）
│   └── input/             # 数据文件（CSV）
├── README.md
└── TODO.md
```

### Run Workspace（运行时）

```
{run_dir}/workspace/
├── task.md                # 任务描述（只读）
├── plan.md                # L2 战略（当前最优 + 策略队列 + 失败方法）
├── findings.md            # L2 知识（验证结论 + 实验结果 + 最优方程演化）
├── execution_trace.md     # L1 工作记忆（每轮重置）
├── input/                 # 数据文件（只读）
└── history/
    └── round{N}/
        ├── scripts/       # Agent 写的脚本
        └── results/       # 每轮结果 + 派生数据
```

---

## 核心组件

### PySR Skill (`evomaster/skills/pysr/`)
- PySR API 速查和模板指南
- Agent 通过 `use_skill pysr get_info` / `get_reference` 按需加载

### Evo Protocol Skill (`evomaster/skills/evo-protocol/`)
- 科学迭代协议（假设 → 实验 → 记录 → 迭代）
- plan 模板（含 Current Best markers）、完整规则、收敛指南

### Signal 机制
- Agent 调用 `finish(message="...", task_completed="true"/"false")` 结束本轮
- 系统从 `task_completed` 判断是否停止迭代（`"true"` = 停止，`"false"` = 继续）
- 如果 Agent 未调用 finish，系统默认继续迭代并输出 warning

---

## 使用方法

```bash
# 准备数据：将 CSV 放入 workspace/input/
cp your_data.csv playground/hamilton/workspace/input/

# 编写任务描述
vim playground/hamilton/workspace/task.md

# 运行
python run.py --agent hamilton --task "发现数据中的方程"

# 指定 run 目录
python run.py --agent hamilton --task "task" --run-dir runs/my_experiment
```

### 配置

修改 `configs/hamilton/config.yaml`：

```yaml
agent:
  max_turns: 100      # 单轮最大工具调用次数

experiment:
  max_rounds: 10      # 最大迭代轮数
```

---

## 设计理念

### 外部化记忆（HCC）
不依赖 Agent 内部 memory，用文件作为持久化知识库：
- L1 每轮重置，避免上下文膨胀
- L2 持久积累，确保知识不丢失
- 人类可阅读、检查和干预

### 单 Agent 闭环
一个 Agent 完成发现 → 验证 → 提炼全流程，避免多 Agent 间信息损耗。

### 系统最小职责
系统只做三件事：重置 L1、解析 satisfied 信号、L2 post-check。所有语义决策由 Agent 自主完成。

### 可调试性
- 每轮脚本保存到 `history/round{N}/scripts/`
- L2 文件记录完整实验演化过程
- L2 post-check 提前发现 Agent 跳过 Promotion 的问题
