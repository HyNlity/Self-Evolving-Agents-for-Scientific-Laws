# Hamilton Agent

## 架构

```
每轮:
  HamiltonAgent → analysis.md (自由书写) + plan.md (更新知识/失败记录)
  系统自动 → experiment.json (PySR 参数和结果)
  Eureka Agent → insight.md (每轮验证结论 + Current Best) + plan.md (Strategy Queue)
  系统自动 → plan.md (同步 Eureka 建议)
```

## 目录

```
workspace/
├── data.csv
├── plan.md            # 研究计划（Evo Protocol — Hamilton/Eureka 共同维护）
├── analysis.md        # 分析历史（格式: ## Round N，自由书写）
├── insight.md         # 靠谱发现（每轮追加；顶部 Current Best 自动维护）
├── experiment.json    # PySR 参数和结果（系统自动）
└── history/
    └── round{N}/
        ├── scripts/   # 每轮代码
        └── results/   # 每轮结果 + 派生数据
```

## 核心组件

### Evo Protocol Skill (`evomaster/skills/evo-protocol/`)
- SKILL.md: 方法论概览（~500 token）
- references/: plan 模板、完整规则、收敛指南
- scripts/: init_plan.py, check_progress.py, failure_report.py

### PySR Tool (`playground/hamilton/tools/pysr_tool.py`)
- PySRRegressor API（现代 PySR）
- 支持 `data_file` 参数（派生数据）
- 支持 `maxsize`, `parsimony`, `populations`, `timeout_in_seconds`
- `expression_spec` 可选（高级模板模式 vs 自由搜索）
- 自动 OOD 评估（`data_ood.csv` 或 `ood_data_file`）

### plan.md 机制
- 系统在首轮自动创建（使用 evo-protocol 模板）
- Hamilton 每轮读写：Confirmed Knowledge, Failed Approaches, Strategy Queue
- Eureka 每轮建议自动同步到 Strategy Queue
- 结构：Task / Data Overview / Hypotheses / Knowledge / Strategy / Failed

## 文件格式

### plan.md (Hamilton + Eureka + 系统 共同维护)
```markdown
# Research Plan

## Task
...

## Data Overview
...

## Current Hypotheses
...

## Confirmed Knowledge
- Best equation: ...
- Best MSE: ...

## Strategy Queue
(Hamilton 和 Eureka 的建议)

## Failed Approaches
| Round | Strategy | Variables | Template/Params | MSE | Why Failed |
```

### analysis.md (HamiltonAgent 写)
```markdown
## Round N

自由书写分析过程...
```

### experiment.json (系统自动)
```json
{
  "rounds": {
    "1": {
      "pysr_config": {"data_file": "...", "maxsize": 20, ...},
      "results": [...]
    }
  }
}
```

### insight.md (Eureka Agent 写)
```markdown
<!-- EVO_CURRENT_BEST_BEGIN -->
## Current Best (auto-updated)
- Round: ...
- Equation: ...
- MSE: ...
<!-- EVO_CURRENT_BEST_END -->

## Round N
...
```

## 迭代模式

- **plan.md** = 策略协调中心（Hamilton + Eureka 共同维护）
- **analysis.md** = Hamilton 的实验笔记
- **insight.md** = Eureka 的验证发现
- **experiment.json** = 程序可读的 PySR 记录
- Evo Protocol 通过 skill 提供方法论（渐进式披露）

## 可复用工具（Eureka）

- 可复用分析能力维护在：`evomaster/skills/eurekatool/`（stdlib-only）
- 在 Hamilton workspace 中会映射为：`skills/eurekatool/`
- 代码复用：在脚本中直接 `from skills.eurekatool import tool`

## 已完成

- [x] Evo Protocol Skill 创建 (`evomaster/skills/evo-protocol/`)
- [x] PySR Tool 升级（PySRRegressor API + data_file + 更多参数 + OOD）
- [x] plan.md 机制（自动创建 + prompt 引导）
- [x] Eureka 协作升级（plan.md 双向同步）
- [x] 通用化改造（prompts 通用化 + 数据自主 + 动力学支持）

## TODO

- [ ] 集成测试：完整 5 轮迭代验证
- [ ] 验证 evo-protocol skill 被 SkillRegistry 正确发现
- [ ] 验证 PySR 新参数在实际数据上的效果
- [ ] 验证 plan.md 同步在多轮中正常工作
- [ ] 考虑：是否需要 evo-protocol skill 的 workspace symlink
