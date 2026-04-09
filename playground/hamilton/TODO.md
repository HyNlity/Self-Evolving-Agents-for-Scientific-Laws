# Hamilton Agent

## 架构

GAN 启发的双 Agent 对抗系统 + HCC（Hierarchical Cognitive Caching）三层记忆。

### 模式

- `--agent hamilton`：原始单 Agent 模式（Solver 独立工作）
- `--agent hamilton-gan`：GAN 对抗双 Agent 模式（Solver + Critic）

### GAN 对抗架构

```
┌───────────────────────────────────────────────────────┐
│              GANHamiltonPlayground                      │
│    循环编排 + Critic 触发条件判断 + L3 经验管理         │
└───────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼ (条件触发)
  ┌──────────────────┐        ┌──────────────────┐
  │   Solver (生成器)  │        │  Critic (鉴别器)  │
  │   RoundExp        │──────→│  CriticExp        │
  │   发现→验证→提炼   │ 结果  │  干预实验→裁定     │
  └──────────────────┘        └──────────────────┘
            │                           │
            │   ┌───────────────────┐   │
            └──→│  共享 HCC 记忆     │←──┘
                │  L1: trace.md     │
                │  L2: plan/findings│
                │  L3: experience   │
                │  (+/- 标签)       │
                └───────────────────┘
```

### Critic 触发条件

Critic 不是每轮都运行（节省 token），触发条件（OR 逻辑）：

1. **Solver 声称完成**：`task_completed="true"` → 必须经过 Critic 审查
2. **周期触发**：每 N 轮（默认 5 轮）至少触发一次
3. **MSE 平台期**：连续 M 轮改进 < 阈值 → 触发 Critic 检查是否陷入局部最优

### Critic 特色：主动干预

Critic **不是被动评审员**，而是 **主动攻击者**：
- 设计极端值实验（x→0, x→∞）检查方程边界行为
- 逐项消融测试，证明每个非线性项的必要性
- 交叉验证对抗（跨风速、跨初始条件）
- 长时间积分稳定性攻击
- 物理一致性拷问（量纲、能量、退化行为）

### HCC 三层记忆

| 层级 | 文件 | 生命周期 | 内容 | 写入者 |
|------|------|----------|------|--------|
| **L1** | `history/round{N}/trace.md` | 每轮独立 | 操作记录、指标、工作笔记 | Solver |
| **L2** | `plan.md`, `findings.md` | 跨轮持久 | 战略计划、验证结论、最优方程 | Solver + Critic |
| **L3** | `experience.md` | 跨任务持久 | [POSITIVE] 有效策略 / [NEGATIVE] 应避免做法 | Solver + Critic |

L3 存储在项目级目录 `playground/hamilton/experience/`，通过 symlink 映射到 workspace。

### 每轮流程

```
Round N 开始
    │
    ├─ 系统: 创建 history/round{N}/trace.md（L1 工作记忆）
    ├─ 系统: 快照 L2/L3 文件 mtime
    │
    ├─ Solver 执行（五阶段闭环）
    │     ├─ Phase 1 规划: 读 L2 + L3 → 确定改进方向
    │     ├─ Phase 2 实验: 执行 PySR / 拟合
    │     ├─ Phase 3 验证: 振幅误差、Z2S/B2S、结构一致性
    │     ├─ Phase 4 记录: 更新 findings.md（逐轮格式）+ plan.md + experience.md
    │     └─ Phase 5 结束: finish(task_completed)
    │
    ├─ 系统: 解析 satisfied 信号
    ├─ 系统: 检查 Critic 触发条件
    │     ├─ 条件满足 → Critic 执行对抗审查
    │     │     ├─ 设计 2-3 个干预实验
    │     │     ├─ 执行脚本 (critic_*.py)
    │     │     ├─ 更新 findings.md [CRITIC] + experience.md [POSITIVE/NEGATIVE]
    │     │     └─ finish(approved/rejected)
    │     └─ 条件不满足 → 跳过 Critic
    │
    ├─ Critic approved + Solver done → 停止
    ├─ Critic rejected → 反馈注入下一轮 Solver
    │
Round N 结束 → 进入 Round N+1
```

## 目录

```
workspace/
├── task.md                    # 任务描述（只读，含数据路径和评估标准）
├── plan.md                    # L2 战略（当前最优 + 策略队列 + 失败方法）
├── findings.md                # L2 知识（逐轮公式+物理解释+多维指标）
├── experience.md              # L3 跨任务经验（symlink → project level）
├── input/                     # 数据文件（只读）
├── lib/                       # 可复用脚本（跨轮持久）
│   └── README.md              # 脚本索引
└── history/
    └── round{N}/
        ├── trace.md           # L1 工作记忆（每轮独立）
        ├── scripts/           # Solver/Critic 脚本（Critic 用 critic_ 前缀）
        └── results/           # 每轮结果 + 派生数据
```

## 核心组件

### PySR Skill (`evomaster/skills/pysr/`) — 知识层
- SKILL.md: PySR API 速查
- references/: API 参考、模板指南、输出格式
- Agent 通过 `use_skill pysr get_info` / `get_reference` 按需加载

### Evo Protocol Skill (`evomaster/skills/evo-protocol/`) — 方法论
- 科学迭代协议（假设→实验→记录→迭代）
- plan 模板（含当前最优 markers）、完整规则、收敛指南

### Constants (`playground/hamilton/core/constants.py`)
- CURRENT_BEST_BEGIN/END, STRATEGY_QUEUE_BEGIN/END
- EXPERIENCE_POSITIVE_BEGIN/END, EXPERIENCE_NEGATIVE_BEGIN/END
- CRITIC_ROUND_INTERVAL, CRITIC_MSE_PLATEAU_THRESHOLD/WINDOW

## 使用方法

```bash
# 单 Agent 模式（原始 Hamilton）
python run.py --agent hamilton --task "发现数据中的方程"

# GAN 对抗模式（Solver + Critic）
python run.py --agent hamilton-gan --task "发现数据中的方程"

# 指定配置和 run 目录
python run.py --agent hamilton-gan --config configs/hamilton/config.yaml --run-dir runs/my_experiment
```

## 已完成

- [x] HCC 重构：Agent 自主化 + 分层记忆
- [x] 删除 run_pysr 工具和 experiment.json
- [x] 系统简化：只保留 L1 重置 + satisfied 信号解析
- [x] VIV 多风速基准（5 风速 × 2 组 + 3 bonus OOD）
- [x] task.md 机制（任务描述从 prompt 分离到文件）
- [x] 移除复用工具库（eurekatool + workspace/tools/）
- [x] evo-protocol 中文化
- [x] 双 Agent → 单 Agent 重构
- [x] 修复双→单 Agent 重构残留
- [x] 添加 L2 post-check
- [x] README.md 重写
- [x] L1 trace.md 移入 round 目录
- [x] **L3 跨任务经验记忆**（experience.md，[POSITIVE]/[NEGATIVE] 标签）
- [x] **GAN 对抗双 Agent 架构**（GANHamiltonPlayground + CriticExp）
- [x] **Critic 触发条件**（周期/平台期/完成声明）
- [x] **多维评估指标体系**（超越 MSE，振幅误差、结构一致性等）
- [x] **findings.md 逐轮格式**（公式 + 物理解释 + 多维指标表）
- [x] **Solver prompt 升级**（L3 + 多维评估 + 逐轮记录格式）
- [x] **Critic prompt**（五维审查框架 + 干预实验 + L3 经验积累）
- [x] **GAN 模式端到端测试**（5 轮完整流程验证，框架逻辑通过）
- [x] **修复 Agent 不执行 bash 的问题**（Prompt 添加 execute_bash 说明 + 执行保护 + 幻觉检测）

## TODO

- [ ] 修复后再次运行 GAN 模式端到端测试（验证 Agent 实际执行 bash）
- [ ] 验证 PySR Skill 在 Agent 对话中的实际效果
- [ ] Critic 干预实验效果评估
- [ ] L3 经验积累效果验证（跨任务复用）
