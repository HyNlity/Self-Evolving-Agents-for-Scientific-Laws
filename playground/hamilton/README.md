# Hamilton - 冗余变量感知的符号回归 Agent

Hamilton 是基于 EvoMaster 的符号回归 playground，目标是在**过完备变量**场景下做支持集恢复、方程结构发现和证伪驱动的科学迭代。

当前版本采用：

- **HCC 三层记忆**：L1 / L2 / L3
- **双 Agent 对抗编排**：Hamilton proposer + Critic challenger
- **任务后 promotion**：仅把可迁移经验从 L2 提炼到 L3

## 架构

```
┌────────────────────────────────────────────────────────────┐
│                    HamiltonPlayground                      │
│     任务级 orchestration + L3 retrieval/promotion          │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                         RoundExp                           │
│   Hamilton proposer → Critic challenger → system gate      │
└────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────┐
│                 HCC Memory + Runtime Artifacts             │
│ L1: trace / reports   L2: findings/plan/state   L3: cards  │
└────────────────────────────────────────────────────────────┘
```

### 轮级协议

```
Round N 开始
    │
    ├─ 系统: 初始化 history/round{N}/trace.md / critic_report.*
    ├─ 系统: 读取 L2 + L3（l3_context.md, critic_context.md, debate_state.json）
    │
    ├─ Hamilton proposer
    │     ├─ 读 task/L2/L3
    │     ├─ 变量角色分析 / 路由 / 搜索 / 证伪
    │     └─ 更新 findings.md / plan.md / machine-readable state
    │
    ├─ Critic challenger
    │     ├─ 审查本轮结论
    │     ├─ 输出 critic_report.md / critic_report.json
    │     └─ 决定 approved / blocking
    │
    ├─ 系统: 汇总 gate
    └─ Round N 结束 → proposer satisfied 且 critic approved ? 停止 : 下一轮
```

## HCC 分层记忆

| 层级 | 载体 | 生命周期 | 作用 |
|------|------|----------|------|
| **L1** | `history/round{N}/trace.md`、`critic_report.*` | 每轮独立 | 当前轮操作记录、challenge、局部观察 |
| **L2** | `plan.md`、`findings.md`、`variable_memory.json`、`routing_state.json`、`hypothesis_archive.jsonl`、`falsification_log.jsonl` | 当前任务持续存在 | 任务内累积知识与可审计状态 |
| **L3** | `runs/hamilton_l3/` 下的结构化 cards | 跨任务持久 | 支持集先验、失败模式、验证 rubric、工具路由经验 |

### L3 设计原则

- 不存原始对话，不把长轨迹直接塞进长期记忆
- 只保留可迁移的 distilled experience
- 默认存六类 card：
  - `domain_prior`
  - `support_set_prior`
  - `operator_motif`
  - `failure_card`
  - `validation_rubric`
  - `tool_recipe`

## 运行时工作空间

```
{run_dir}/workspace/
├── task.md
├── plan.md
├── findings.md
├── variable_memory.json
├── routing_state.json
├── hypothesis_archive.jsonl
├── falsification_log.jsonl
├── l3_context.md
├── critic_context.md
├── debate_state.json
├── task_signature.json
├── l3_hits.json
├── lib/
└── history/
    └── round{N}/
        ├── trace.md
        ├── critic_report.md
        ├── critic_report.json
        ├── scripts/
        └── results/
```

## 配置

`configs/hamilton/config.yaml` 和 `config_no_pysr.yaml` 现在都包含：

- `agents.hamilton`
- `agents.critic`
- `memory.l3`
- `critic_policy`
- `completion_policy`

关键配置项：

```yaml
memory:
  l3:
    enabled: true
    root: "./runs/hamilton_l3"
    top_k: 6
    retrieval_mode: "hybrid"
    promotion_policy: "task_end_only"

completion_policy:
  require_critic_approval: true
```

## Prompt 角色分工

### Hamilton proposer

- 主动发现方程、支持集和变量角色
- 更新 L2 与实验产物
- 解决 Critic 留下的 blocker

### Critic challenger

- 不负责代写答案
- 只做结构化审查和 challenge 提出
- 使用固定 taxonomy：
  - `support_set_attack`
  - `structure_attack`
  - `ood_generalization_attack`
  - `physics_consistency_attack`
  - `numerical_stability_attack`
  - `evidence_gap_attack`

## 使用方法

```bash
# no-pysr 模式 smoke
python run.py --agent hamilton --config configs/hamilton/config_no_pysr.yaml --task "发现数据中的方程"

# 默认模式
python run.py --agent hamilton --config configs/hamilton/config.yaml --task playground/hamilton/workspace/task.md

# 指定 run 目录
python run.py --agent hamilton --config configs/hamilton/config.yaml --task "task" --run-dir runs/my_experiment
```

## 当前验证状态

- L3 memory 的 promotion / retrieval 已有单测
- proposer-critic gate 已有单测
- 真实端到端运行仍依赖外部 LLM API 可用性
