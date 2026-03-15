# 研究计划

## 任务
profile: newtonbench
module: m10_be_distribution
system: complex_system
difficulty: hard
law_version: v2
noise: 0.0
code_assisted: false

<!-- EVO_CURRENT_BEST_BEGIN -->
## 当前最优
- 轮次：0
- 方程：无
- MSE：未知
- 更新时间：待定
<!-- EVO_CURRENT_BEST_END -->

## 最优解维护规则
- 仅当新候选在 `symbolic_equivalent / exact_accuracy / rmsle` 上优于当前最优时，才更新 `当前最优`。
- 若新候选退化或评测失败，保留当前最优并在下一轮回滚使用。
- 禁止连续多轮重复同一失败家族；若连续 2 轮无改进，写入下方“禁用家族”。

## 禁用家族
- （示例）family_be_exp_over_t: 连续 2 轮无改进，暂不再尝试

## 数据概览
（首轮 EDA 后填写：变量列表、基本统计、初步观察）

- 列：待定
- 目标变量：待定
- 时间列：待定（如有，考虑动力学）
- 数据规模：待定行 × 待定列
- 缺失值：待定
- 显著模式：待定

## 当前假设
（Agent 基于数据探索填写——可能存在什么方程/关系？）

1. 待定

## 已确认知识
- 相关变量：待定
- 排除变量：待定
- 已发现的关键关系：无

## 策略队列
<!-- EVO_STRATEGY_QUEUE_BEGIN -->
（Agent 自行制定）
<!-- EVO_STRATEGY_QUEUE_END -->

## 失败方法
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
|------|------|------|-----------|-----|----------|

<!-- HAM_SYS_BACKFILL_PLAN_ROUND_1 -->
## 系统回填 Round 1
- task_completed: false
- satisfied: False
- 本轮协议问题: rmsle_above_threshold
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。

<!-- HAM_SYS_BACKFILL_PLAN_ROUND_2 -->
## 系统回填 Round 2
- task_completed: false
- satisfied: False
- 本轮协议问题: rmsle_above_threshold
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。
