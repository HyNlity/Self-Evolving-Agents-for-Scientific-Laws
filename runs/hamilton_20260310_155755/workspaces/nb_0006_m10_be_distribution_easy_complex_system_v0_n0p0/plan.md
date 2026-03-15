# 研究计划
Next round plan:
- Test Fermi-Dirac form directly: n = 1 / (exp(C * omega / T) + 1)
- Explore constants near hidden constant scale by adjusting guess C
- Use new experimental parameters to sample transitional regime:
  [{"temperature": 1000, "center_frequency": 5e11, "bandwidth": 1e9}, {"temperature": 10000, "center_frequency": 2e10, "bandwidth": 5e9}, {"temperature": 200, "center_frequency": 5e13, "bandwidth": 5e12}]

Next round plan: Explore broader omega/T ranges to capture behavior with '+1' denominator. Propose tests with small and large omega/T ratios.
New experiments:
--inputs-json '[{"temperature":1e2,"center_frequency":1e9,"bandwidth":1e3},{"temperature":1e6,"center_frequency":1e9,"bandwidth":1e3},{"temperature":1e2,"center_frequency":1e12,"bandwidth":1e6}]'

## 任务
profile: newtonbench
module: m10_be_distribution
system: complex_system
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false

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