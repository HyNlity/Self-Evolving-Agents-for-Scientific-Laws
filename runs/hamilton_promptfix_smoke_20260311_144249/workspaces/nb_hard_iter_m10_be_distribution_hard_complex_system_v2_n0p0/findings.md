# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

<!-- APPEND_FINDINGS -->
## 候选方程解析（Round 2）
### 1) 方程与物理解释
family_A: `1 / (exp(c * omega / T) - 1)` （Bose-Einstein 控制公式）
family_B: `omega**3 / (exp(c * omega / T) - 1)` （引入 ω³ 以匹配光谱辐射比例）
### 2) 系数表（跨实验条件）
c=1e-10  在三个不同条件下测试
### 3) 物理洞察
实验数据表明功率随 ω 急剧增长，符合 ω³ 项放大效应，但仪器无法直接捕捉 n(ω) 过渡区特性
### 4) 消融分析
去掉 ω³ 项回到 family_A，功率缩小数个量级；保留 ω³ 项但调整 c 值对 RMSLE 未改善明显
## 候选方程解析（Round 1）
### 1) 方程与物理解释
family_A: Bose-Einstein form 1/(exp(c omega/T)-1)
family_B: omega^2 / (exp(c omega/T)-1) 引入频率平方增强项
### 2) 系数表（跨实验条件）
c=1e-10 测试条件跨 1e2-1e3 温度与 1e9-1e10 频率
### 3) 物理洞察
family_B 在高频条件下产出较大功率预测，未捕捉到低频过渡行为
### 4) 消融分析
去掉 omega^2 项将回到 family_A，对比可见平方项在本数据集下偏离真实
- Round 2: family_B candidate (omega**3 / (exp(c omega/T)-1)), RMSLE=69.07, symbolic_equivalent=False
<!-- APPEND_FINDINGS -->
下一轮计划：切换到 family_A 并尝试引入对 bandwidth 的显式依赖，inputs-json 例如 [{"temperature":1e2,"center_frequency":1e8,"bandwidth":1e6},{"temperature":1e4,"center_frequency":1e12,"bandwidth":1e9}]
<!-- APPEND_NEXT -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| Round 1 | system_backfill | `omega**2 / (math.exp(c * omega / T) - 1)` | - | - | 未完成; RMSLE=48.2789377782; exact=0; symbolic=false |
| Round 2 | system_backfill | `omega**3 / (math.exp(c * omega / T) - 1)` | - | - | 未完成; RMSLE=69.0670054688; exact=0; symbolic=false |

<!-- APPEND_RESULTS -->
- Round 1: family_B candidate (omega^2 / (exp(c omega/T)-1)), RMSLE=48.28, symbolic_equivalent=False
<!-- APPEND_RESULTS -->

## Worth Trying Next
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
