# 研究计划
## 下一轮计划（Round 9）
- 失败原因：Round 8 中使用的常数 C=1.438e-23，导致某些输入参数下指数计算产生零除错误，评测 NaN，exact_accuracy=0.0，symbolic_equivalent=False。
- 改进方案：拟合常数 C 以避免数值溢出，使用 math.expm1 提高数值稳定性；纳入 ω^3 因子以匹配功率增长趋势；探索不同带宽以逼近谱值。
- 下一轮实验参数：
  --inputs-json '[{"temperature":50,"center_frequency":1e8,"bandwidth":1e4},{"temperature":500,"center_frequency":1e10,"bandwidth":1e4},{"temperature":5000,"center_frequency":1e12,"bandwidth":1e4},{"temperature":1000,"center_frequency":5e9,"bandwidth":5e4}]'
# 下一轮实验计划
- 失败原因：RMSLE仍为7.78，exact_accuracy=0.0，符号等价性False，说明常数C和方程形式未能完全捕捉真实规律；可能缺失多项式因子或 ω 与 T 的复合依赖。
- 下一轮计划：
    1. 在 occupation number 公式中引入 ω^p/T^q 的多项式因子，探索 p,q 在 1~3 范围内的拟合。
    2. 扩展实验范围：T 取 1e1, 1e3, 1e5；ω 取 1e8, 1e10, 1e12；带宽取 1e6, 1e8。
    3. 通过 run_experiment.py 收集这些条件下的 total_power，用窄带宽近似R(ω)并反算 n(ω)。
    4. 调整指数常数C以适配扩展区间拟合。

下一轮 inputs-json 草案：
[
{"temperature": 1e1, "center_frequency": 1e8, "bandwidth": 1e6},
{"temperature": 1e3, "center_frequency": 1e10, "bandwidth": 1e8},
{"temperature": 1e5, "center_frequency": 1e12, "bandwidth": 1e6}
]

## 任务
profile: newtonbench
module: m10_be_distribution
system: complex_system
### 下一轮计划（Round 8）
- 失败原因：RMSLE=7.117，拟合不足；exact_accuracy=0.0，symbolic_equivalent=False；高频低温区出现NaN，数值溢出；常数C未优化，导致整体偏离。
- 实验策略：调整C的尺度，并在高频低温区使用对数或指数稳定算法。增加 Δω 与 ω 同阶变化以考察积分效应。
- 新实验参数草案：
  --inputs-json '[{"temperature": 1e2, "center_frequency": 5e8, "bandwidth": 1.0}, {"temperature": 1e4, "center_frequency": 5e10, "bandwidth": 100.0}, {"temperature": 1e6, "center_frequency": 5e12, "bandwidth": 1.0}]'
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
下一轮计划：将带宽缩小到极小值以近似测得单频谱辐射，从而更精确反推占据数公式，并调节 h_over_k 常数。
实验参数草案：
--inputs-json '[{"temperature":1e2,"center_frequency":1e10,"bandwidth":1e2},{"temperature":1e3,"center_frequency":1e12,"bandwidth":1e2},{"temperature":5e2,"center_frequency":5e11,"bandwidth":1e2}]' --tag round2_fine_bw
<!-- EVO_STRATEGY_QUEUE_END -->
下一轮计划：尝试缩小带宽到极小(1e2)并跨多尺度调整 omega 与 T，确保探测到从指数支配到幂律支配的过渡区。
实验参数草案：
--inputs-json '[{"temperature":50,"center_frequency":1e9,"bandwidth":1e2},{"temperature":500,"center_frequency":1e11,"bandwidth":1e2},{"temperature":5000,"center_frequency":1e13,"bandwidth":1e2}]' --tag round3_bw_scan
下一轮计划：将bandwidth缩小到1e2并同时调节omega与T跨越多数量级，尝试引入ω^3依赖项以匹配谱辐射积分结构。
实验参数草案：
--inputs-json '[{"temperature":20,"center_frequency":1e8,"bandwidth":1e2},{"temperature":2000,"center_frequency":1e12,"bandwidth":1e2},{"temperature":5e3,"center_frequency":5e13,"bandwidth":1e2}]' --tag round4_bw_powerlaw

下一轮计划：将bandwidth缩小到1e2并同时调节omega与T跨多数量级，大幅扩大对幂律与指数混合区的采样范围，引入ω^3依赖项以匹配谱辐射积分结构。
实验参数草案：
--inputs-json '[{"temperature":10,"center_frequency":1e8,"bandwidth":1e2},{"temperature":1000,"center_frequency":1e12,"bandwidth":1e2},{"temperature":8000,"center_frequency":1e14,"bandwidth":1e2}]' --tag round5_bw_w3
## 失败方法
- 失败原因：本轮 math range error，RMSLE=NaN，exact_accuracy=0.0，符号等价性False；常数选择不当导致溢出，且未引入 ω³ 因子及带宽修正。
- 下一轮计划：
    1. 在玻色–爱因斯坦分布指数中减小常数量级（~1e-12），并采用 math.expm1 保证数值稳定。
    2. 在分母外引入 ω^3 因子并结合 Δω 估计，拟合总功率积分对应的 n。
    3. 使用窄带宽 Δω=1e2 对多个 ω、T 组合采样，覆盖指数主导区与幂律区过渡。

下一轮 inputs-json 草案：
[
{"temperature": 1e2, "center_frequency": 1e9, "bandwidth": 1e2},
{"temperature": 5e3, "center_frequency": 5e11, "bandwidth": 1e2},
{"temperature": 1e4, "center_frequency": 1e13, "bandwidth": 1e2}
]
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
|------|------|------|-----------|-----|----------|