# 研究发现
Round 2 failure:
- RMSLE remained high at 7.4863
- exact_accuracy = 0.0, symbolic_equivalent = False
- Used Bose-Einstein (-1 in denominator) form, but ground truth uses Fermi-Dirac (+1)
- constants likely mismatched hidden constant

Round 1 failure: RMSLE=7.4575, exact_accuracy=0.0, symbolic_equivalent=False. Mis-match likely due to incorrect sign or constant in exponential, ground truth uses '+1' in denominator instead of '-1'.

## 关键洞察
（经验证的数据观察和物理关系）

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|

## 最优方程演化
（记录最优方程在各轮中的变化过程）