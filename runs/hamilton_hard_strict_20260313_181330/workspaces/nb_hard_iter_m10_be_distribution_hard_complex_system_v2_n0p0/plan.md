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
- 轮次：20
- 方程：(5.133649382716048)*(3.1659591617788225e-6*math.sqrt(0.7996435919785698*T + 0.7996435919785698*omega)/(T - 2.9700715329968843e-5*math.sqr...
- MSE：未知
- 更新时间：系统自动维护（symbolic_equivalent=false, exact_accuracy=0, rmsle=0.0135144205218）
<!-- EVO_CURRENT_BEST_META: {"round": 20, "symbolic_equivalent": false, "exact_accuracy": 0.0, "rmsle": 0.013514420521811557, "equation": "(5.133649382716048)*(3.1659591617788225e-6*math.sqrt(0.7996435919785698*T + 0.7996435919785698*omega)/(T - 2.9700715329968843e-5*math.sqr...", "law_code": "def discovered_law(omega, T):\n    import math\n    return (5.133649382716048)*(3.1659591617788225e-6*math.sqrt(0.7996435919785698*T + 0.7996435919785698*omega)/(T - 2.9700715329968843e-5*math.sqrt(omega)) + 0.013703417537053625)"} -->
<!-- EVO_CURRENT_BEST_END -->

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
### Round 11 -> Next
目标: 降低 RMSLE 至门槛以下并验证系数物理一致性。
动作: 补充低 ω/T 样本和高 ω/T 窄带样本，探索指数型与分式混合结构。
验收: RMSLE < 当前协议 max_rmsle，exact_accuracy 提升。

## 失败方法
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
|------|------|------|-----------|-----|----------|