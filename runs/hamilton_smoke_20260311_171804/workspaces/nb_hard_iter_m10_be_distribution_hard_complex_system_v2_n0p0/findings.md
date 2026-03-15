# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|

| 1 | direct  | 1/(exp(a*omega/T)-1), a=1e-11 | 9.345 | N/A | First trial, poor fit |
| 2 | direct  | 1/(exp(a*omega**3/T)-1), a=1e-10 | NaN | N/A | math range error |
<!-- APPEND_RESULTS -->

## Worth Trying Next
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
