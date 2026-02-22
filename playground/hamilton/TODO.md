# Hamilton Agent

## 架构

```
每轮:
  HamiltonAgent → analysis.md (自由书写)
  系统自动 → experiment.json (PySR 参数和结果)
  Eureka Agent → insight.md (BestEq/MSE/AltEqs/Notes/Recommendations)
```

## 目录

```
workspace/
├── data.csv
├── analysis.md    # 分析历史（格式: ## Round N，自由书写）
├── insight.md         # 靠谱发现（格式: ## Round N）
├── experiment.json     # PySR 参数和结果（系统自动）
└── history/
    └── round{N}/
        ├── scripts/   # 每轮代码
        └── results/
```

## 文件格式

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
      "pysr_config": {...},
      "results": [...]
    }
  }
}
```

### insight.md (Eureka Agent 写)

```markdown
## Round N

BestEq: ...
MSE: ...
AltEqs: ... ; ... ; ...
Notes: ...

Recommendations:
- ...
```

## 迭代

- **analysis.md** = 唯一的知识库（自由书写）
- **experiment.json** = 程序可读的 PySR 记录
- 科学家看笔记的方式：需要什么方法，看历史记录，自己实现

## 可复用工具（Eureka）

- 可复用分析能力维护在 Operator Skill: `evomaster/skills/eurekatool/`
- 在 Hamilton workspace 中会映射为：`skills/eurekatool/`
- 说明书分层加载：`use_skill(get_info)` / `use_skill(get_reference)`
- 代码复用通过 Python import（不需要用 use_skill 调用函数）
