# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

<!-- APPEND_FINDINGS -->
轮次1：首次闭环成功，但fit_pysr_candidates未生成候选，改用手动构造数值稳定候选公式进行评测。评测RMSLE≈27.51，结构接近玻色-爱因斯坦分布但常数尚需优化。

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|

| 1 | 手动候选公式 | def discovered_law(omega, T):\n    import math\n    const = 1.0e-21\n    x = max(min(const * omega / T, 60), -60)\n    return 1 / (math.exp(x) - 1 + 1e-12) | RMSLE≈27.51 | RMSLE≈27.51 | 数值稳定但性能一般 |
| 2 | ω³修饰候选公式 | def discovered_law(omega, T):\n    import math\n    const = 1.0e-21\n    x = max(min(const * omega / T, 60), -60)\n    return (omega**3) / (math.exp(x) - 1 + 1e-12) | RMSLE≈89.73 | RMSLE≈89.73 | 性能退化，无改进 |
<!-- APPEND_RESULTS -->

## Worth Trying Next
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）