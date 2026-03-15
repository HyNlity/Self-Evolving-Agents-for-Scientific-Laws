# 研究发现
Round 2 failure: Evaluated law used distance^2, but ground truth uses distance^3. RMSLE=20.6223, exact_accuracy=0.0, symbolic_equivalent=False.
Round1 Failure: Evaluated law gave very high RMSLE=20.63, exact_accuracy=0.0, symbolic_equivalent=False. Ground truth appears to be CONSTANT * q1 * q2 / distance**3, our submission used **2** exponent.

## 关键洞察
（经验证的数据观察和物理关系）

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|

## 最优方程演化
（记录最优方程在各轮中的变化过程）
Round3 Failure: Evaluated law (Coulomb-like with distance**-2) mismatched ground truth. RMSLE=20.65, exact_accuracy=0.0, symbolic_equivalent=false. Ground truth uses distance**-3 dependency.