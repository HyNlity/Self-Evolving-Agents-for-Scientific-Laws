# Hamilton Agent

## 架构

单 Agent 迭代式方程发现系统，使用 HCC（Hierarchical Cognitive Caching）分层记忆。

```
每轮:
  系统 → 重置 execution_trace.md（L1 工作记忆）
  Agent → 读 L2 → 发现方程 → 验证 → 提炼到 L2（findings.md + plan.md）→ finish(satisfied)
  系统 → 解析 satisfied 信号，决定是否继续
```

记忆分层：
- **L1 (execution_trace.md)**：每轮重置的工作记忆（锯齿形）
- **L2 (plan.md, findings.md)**：只增不减的知识积累（阶梯形）

## 目录

```
workspace/
├── task.md                    # 任务描述（只读，含数据路径和评估标准）
├── plan.md                    # L2 战略（当前最优 + 策略队列 + 失败方法）
├── findings.md                # L2 知识（验证结论 + 实验结果 + 建议）
├── execution_trace.md         # L1 工作记忆（每轮重置）
├── input/                     # 数据文件（只读）
├── skills/
│   └── pysr/                  # PySR API 文档（symlink，只读）
└── history/
    └── round{N}/
        ├── scripts/           # Agent 写的脚本
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
- EUREKA_SIGNAL_BEGIN/END
- CURRENT_BEST_BEGIN/END, STRATEGY_QUEUE_BEGIN/END

## 已完成

- [x] HCC 重构：Agent 自主化 + 分层记忆
- [x] 删除 run_pysr 工具和 experiment.json
- [x] 系统简化：只保留 L1 重置 + satisfied 信号解析
- [x] VIV 多风速基准（5 风速 × 2 组 + 3 bonus OOD）
- [x] task.md 机制（任务描述从 prompt 分离到文件）
- [x] 移除复用工具库（eurekatool + workspace/tools/）
- [x] evo-protocol 中文化
- [x] 双 Agent → 单 Agent 重构：合并 Hamilton + Eureka 为单 Agent，每轮完成发现→验证→提炼闭环
  - config.yaml: `agents:` → `agent:` 单 agent 模式
  - exp.py: 去掉 eureka_agent，单次 agent.run()
  - playground.py: 使用 BasePlayground 的 self.agent
  - 删除 eureka_system.txt / eureka_user.txt
  - hamilton_system.txt: 合并四阶段（Discovery → Verification → Promotion → Finish）
- [x] 修复双→单 Agent 重构残留（"Eureka 维护"文本、过时常量字段）
- [x] 添加 L2 post-check（检测 Agent 是否完成 Promotion，warning 级别）
- [x] README.md 重写（单 Agent + HCC 架构）

## TODO

- [ ] 集成测试：完整多轮迭代验证
- [ ] 验证 PySR Skill 在 Agent 对话中的实际效果
