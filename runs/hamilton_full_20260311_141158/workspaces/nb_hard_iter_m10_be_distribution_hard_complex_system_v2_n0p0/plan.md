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
下一轮策略：基于 Round9 的BE分布指数结构，引入 omega 与 T 的额外幂律修正因子 (omega**a * T**b)，并考虑在指数项中添加非线性缩放系数或偏移，以模拟复杂系统中可能存在的修正项。同时扩大实验范围覆盖低频低温与高频高温极端情况，选取更窄的 bandwidth 以近似光谱辐射瞬时值。
拟议实验参数：
--inputs-json '[{"temperature":2e1,"center_frequency":5e8,"bandwidth":1e4},{"temperature":8e3,"center_frequency":8e10,"bandwidth":5e5},{"temperature":1e5,"center_frequency":2e12,"bandwidth":1e6}]'
<!-- EVO_STRATEGY_QUEUE_END -->

## 失败方法
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
|------|------|------|-----------|-----|----------|
<!-- HAM_SYS_BACKFILL_PLAN_ROUND_2 -->
## 系统回填 Round 2
- task_completed: false
- satisfied: False
- 本轮协议问题: missing_final_law_block, final_law_missing_discovered_law_signature, rmsle_above_threshold
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。

<!-- HAM_SYS_BACKFILL_PLAN_ROUND_6 -->
## 系统回填 Round 6
- task_completed: false
- satisfied: False
- 本轮协议问题: missing_successful_evaluate_submission, missing_final_law_block, final_law_missing_discovered_law_signature
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。

<!-- HAM_SYS_BACKFILL_PLAN_ROUND_10 -->
## 系统回填 Round 10
- task_completed: false
- satisfied: False
- 本轮协议问题: missing_final_law_block, final_law_missing_discovered_law_signature, missing_final_law_signature, non_finite_rmsle, rmsle_above_threshold
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。
