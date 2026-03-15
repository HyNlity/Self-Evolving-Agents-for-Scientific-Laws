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
- 轮次：6
- 方程：1 / (math.exp(C * omega / T) - 1)
- MSE：未知
- 更新时间：系统自动维护（symbolic_equivalent=false, exact_accuracy=0, rmsle=31.7882344374）
<!-- EVO_CURRENT_BEST_META: {"round": 6, "symbolic_equivalent": false, "exact_accuracy": 0.0, "rmsle": 31.788234437432575, "equation": "1 / (math.exp(C * omega / T) - 1)", "law_code": "def discovered_law(omega, T):\n    import math\n    C = 1.5e-21\n    return 1 / (math.exp(C * omega / T) - 1)"} -->
<!-- EVO_CURRENT_BEST_END -->
## Round 5 最优候选
- 方程：def discovered_law(omega, T):
    import math
    c = 1.2e-21
    return 1.0/(math.exp(c*omega/T)-1)
- RMSLE=32.0258, Exact=0.0（略差于当前最优）

## 最优解维护规则
- 仅当新候选明显优于当前最优时，才更新 `当前最优`。
- 新候选退化或评测失败时，保留当前最优并回滚。
- 若同一结构连续无改进，下一轮切换结构族。

## 禁用家族
- （可选）记录暂时不再尝试的结构族

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