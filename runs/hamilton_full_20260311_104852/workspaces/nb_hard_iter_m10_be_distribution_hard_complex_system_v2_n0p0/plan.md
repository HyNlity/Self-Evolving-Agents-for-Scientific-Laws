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
- 下一轮策略：引入 ω 和 T 的额外幂次或多项式修正，采样涵盖低温高频、极高温极低频等极端组合，重点探测 ω/T 在 1e-6 至 1e6 的过渡行为，减少积分近似误差的同时丰富中间区域数据。
- 实验参数示例：
  --inputs-json '[{"temperature":1e2, "center_frequency":1e12, "bandwidth":1e4}, {"temperature":1e7, "center_frequency":1e8, "bandwidth":1e5}, {"temperature":5e3, "center_frequency":5e9, "bandwidth":1e3}]' --tag round4
<!-- EVO_STRATEGY_QUEUE_END -->
- 下一轮策略补充：针对Round 4中指数参数幅度不足的问题，计划在下一轮引入 ω**p / T 形式，p在0.5至2之间拟合，采样参数覆盖 ω/T 在 1e-4 至 1e8 范围。
- 实验参数示例（新组合）：
  --inputs-json '[{"temperature":2e2, "center_frequency":2e10, "bandwidth":5e5}, {"temperature":8e5, "center_frequency":5e9, "bandwidth":5e4}, {"temperature":5e3, "center_frequency":1e11, "bandwidth":2e6}]' --tag round5

- 下一轮策略补充：针对Round 5指数形式无法覆盖全尺度的问题，计划引入 ω^p / T^q 的双幂次形式，其中 p、q 通过实验数据拟合获得。
- 新实验参数示例（含极端组合）：
  --inputs-json '[{"temperature":5e1, "center_frequency":1e9, "bandwidth":1e4}, {"temperature":1e6, "center_frequency":2e8, "bandwidth":1e5}, {"temperature":2e3, "center_frequency":1e12, "bandwidth":2e6}]' --tag round6
## 失败方法
| 6 | 指数型 BE 分布常数选取过小 | ω, T | C=1e-21 | - | float division by zero 错误，数值稳定性差 |

- 下一轮策略补充：为避免除零并改善跨尺度精度，计划引入 ω**p / T^q 的双幂次组合，同时在指数内添加平移项以缓冲低 ω/T 区域的数值问题。p、q 将在实验中调优。
- 新实验参数示例（新组合）：
  --inputs-json '[{"temperature":1e4, "center_frequency":5e8, "bandwidth":1e3}, {"temperature":2e2, "center_frequency":1e11, "bandwidth":5e5}, {"temperature":5e6, "center_frequency":2e9, "bandwidth":2e4}]' --tag round7
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
- 下一轮策略补充：针对Round 7 RMSLE较高的问题，将在指数项中引入 ω/T 的非线性缩放（如 (ω/T)^p + α），并测试多组 p、α 的组合，以增强中频区拟合能力。
- 新实验参数示例（新组合）:
  --inputs-json '[{"temperature":5e2, "center_frequency":5e10, "bandwidth":2e6}, {"temperature":1e4, "center_frequency":2e9, "bandwidth":5e5}, {"temperature":3e3, "center_frequency":8e11, "bandwidth":1e7}]' --tag round8
|------|------|------|-----------|-----|----------|
<!-- HAM_SYS_BACKFILL_PLAN_ROUND_8 -->
## 系统回填 Round 8
- task_completed: false
- satisfied: False
- 本轮协议问题: missing_successful_evaluate_submission, missing_final_law_block, final_law_missing_discovered_law_signature, missing_final_law_signature
- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），并写明与上一轮不同的假设。
- 下一轮策略（Round 10）: 针对Round 9评测RMSLE高、exact=0的失败，计划在指数项中引入 ω/T 的幂次修正并添加平移项，例如形式 1 / (exp(a * (ω/T)^p + b) - 1)，其中 p 在0.8~1.2之间调优，b为小正数避免除零，并结合宽窄带混合采样覆盖中频过渡区。
- 新实验参数示例（与Round 9不同）:
  --inputs-json '[{"temperature":5e2, "center_frequency":2e13, "bandwidth":1e8}, {"temperature":2e4, "center_frequency":5e9, "bandwidth":5e5}, {"temperature":1e6, "center_frequency":1e10, "bandwidth":1e6}]' --tag round10
- 下一轮策略补充：针对Round 10高 RMSLE 的失败，本轮 BE 分布常数虽调整但全域拟合依然偏差大，计划在指数项内部引入 (ω/T)^p 的幂次修正与加性平移项 b，并拟合 a,p,b 三个参数，期望改善中频过渡区拟合。
- 新实验参数示例（新组合）:
  --inputs-json '[{"temperature":1e3, "center_frequency":1e13, "bandwidth":1e5}, {"temperature":5e4, "center_frequency":5e10, "bandwidth":1e4}, {"temperature":1e6, "center_frequency":2e9, "bandwidth":1e6}]' --tag round11