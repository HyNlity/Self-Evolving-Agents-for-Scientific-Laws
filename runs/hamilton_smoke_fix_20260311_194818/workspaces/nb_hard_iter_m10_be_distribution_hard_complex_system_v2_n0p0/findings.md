# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

本轮实验遇到 fit_pysr_candidates 数据不足问题，多次尝试 run_experiment 但有效记录仅为 4，未能生成候选。
<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|

| 1 | pysr_assisted | omega**3/(exp(omega/T)-1) | math range error | math range error | 溢出需修正 |
<!-- APPEND_RESULTS -->

## Worth Trying Next
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
