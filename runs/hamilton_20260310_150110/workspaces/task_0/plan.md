# 研究计划
Round 3 plan: Try law with distance^3 denominator. Run experiments with varying q1,q2 signs and distances in range [0.05, 2.0]. Example --inputs-json: '[{"q1":1e-6,"m1":1e3,"q2":2e-6,"m2":1,"distance":0.05,"duration":5.0,"time_step":0.1},{"q1":1e-6,"m1":1e3,"q2":2e-6,"m2":1,"distance":2.0,"duration":5.0,"time_step":0.1},{"q1":-1e-6,"m1":1e3,"q2":2e-6,"m2":1,"distance":1.0,"duration":5.0,"time_step":0.1}]'
Next round plan: Test Coulomb-like law with exponent 3 instead of 2. Use varying q1, q2, distance to confirm scaling.
Proposed inputs-json: [{"q1": 1.0, "m1": 1000.0, "q2": 1.0, "m2": 1.0, "distance": 1.0, "duration": 1.0, "time_step": 0.1}, {"q1": 1.0, "m1": 1000.0, "q2": 1.0, "m2": 1.0, "distance": 2.0, "duration": 1.0, "time_step": 0.1}, {"q1": 2.0, "m1": 1000.0, "q2": 2.0, "m2": 1.0, "distance": 1.0, "duration": 1.0, "time_step": 0.1}]

## 任务
profile: newtonbench
module: m1_coulomb_force
system: complex_system
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false
Next round plan:
- Hypothesize force law ~ q1*q2 / distance**n. Prior result shows n likely=3.
- Run experiments with varied distances to fit exponent n.
- Example inputs:
  --inputs-json '[{"q1": 1.0, "m1": 1000.0, "q2": 1.0, "m2": 1.0, "distance": 1.5, "duration": 1.0, "time_step": 0.1}, {"q1": 1.0, "m1": 1000.0, "q2": 1.0, "m2": 1.0, "distance": 3.0, "duration": 1.0, "time_step": 0.1}]'

这是一次验证 run：如果本轮评测不达标，请按协议 finish(task_completed="false")，并在下一轮继续改进。

<!-- EVO_CURRENT_BEST_BEGIN -->
## 当前最优
- 轮次：0
- 方程：无
- MSE：未知
- 更新时间：待定
<!-- EVO_CURRENT_BEST_END -->

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