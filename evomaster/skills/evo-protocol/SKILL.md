---
name: evo-protocol
description: "科学迭代协议：为自主研究 Agent 提供计划结构、注意力管理、失败追踪和收敛分析。"
license: null
---

# 演化协议（Evo Protocol）

轻量级科学迭代方法论。核心循环：

**假设 → 实验 → 记录 → 迭代**

## 快速开始

1. 运行 `scripts/init_plan.py` 从任务描述生成 `plan.md`
2. 每次实验前：读 `plan.md`，避免重复失败策略
3. 每次实验后：更新 `plan.md`，记录结果和新假设
4. 用 `scripts/check_progress.py` 获取进度摘要
5. 用 `scripts/failure_report.py` 提取失败模式

## 核心规则

### 注意力规则："行动前先阅读"
- 每次实验前读 `plan.md`
- 检查「失败方法」表避免重复
- 检查「已确认知识」在已知基础上推进

### 记录规则："失败也是知识"
- 每次失败实验都必须记入「失败方法」表
- 包含：尝试了什么、结果如何、为什么失败
- 防止策略循环，加速收敛

### 变异规则："不要重复"
- 每次新策略必须与之前所有尝试有实质性差异
- 至少改变一项：变量、算子、表达式结构、参数
- 如果卡住，尝试彻底不同的方法

## 资源

| 资源 | 路径 | 用途 |
|------|------|------|
| 计划模板 | `references/plan_template.md` | 推荐的 plan.md 结构 |
| 完整规则 | `references/evo_rules.md` | 完整的演化协议方法论 |
| 收敛指南 | `references/convergence_guide.md` | 何时停止迭代 |
| 初始化计划 | `scripts/init_plan.py` | 从任务描述生成 plan.md |
| 进度检查 | `scripts/check_progress.py` | 分析迭代进度 |
| 失败报告 | `scripts/failure_report.py` | 提取失败模式 |

## Usage Patterns

```python
# 在工作空间中运行：
# 生成初始计划
python scripts/init_plan.py --task "从数据中发现控制方程" --output plan.md

# 实验中途检查进度
python scripts/check_progress.py --plan plan.md

# 多轮后分析失败模式
python scripts/failure_report.py --plan plan.md
```
