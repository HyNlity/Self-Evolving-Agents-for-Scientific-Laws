# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

## 候选方程解析（Round 1）
### 1) 方程与物理解释
- family_A: 经典B-E核结构，结果拟合差。
- family_B: 对数核结构，解释为非平衡下 occupation number 对温度与频率的次方比值的对数依赖。
### 2) 系数表（跨实验条件）
- K=2.5e5（family_B）
### 3) 物理洞察
- 实验表明 occupation number 随 T 增强、随 ω² 抑制，对数依赖可以捕捉全量级行为。
### 4) 消融分析
- 移除 ω² 项后拟合劣化明显。
<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| Round 1 | system_backfill | `math.log(1 + K * T / (omega**2))` | - | - | 完成; RMSLE=0.0838765976254; exact=0; symbolic=false |

- Round 1: family_A (示例核) → rmsle=31.83; family_B (log核) → rmsle=0.0839，优胜家族=family_B
<!-- APPEND_RESULTS -->

## Worth Trying Next
- 下一轮计划：验证family_B在更极端ω/T比值下的稳定性，探索双曲核结构作为新的非示例候选。
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
