# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 机制判读：出现指数项，说明模型在尝试表达热激发机制，但无量纲比值结构（omega/T）仍不完整。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

<!-- APPEND_FINDINGS -->
- 该候选的 sqrt(sqrt(exp(T))+omega/T) 结构暗示了温度驱动激发和频率比效应的混合机制；指数项对应热激发增长（但系数不同于经典B-E分布），sqrt嵌套可能是探测器或代理公式中积分到窄带的数学残留。数值稳定且跨尺度表现一致。
<!-- APPEND_FINDINGS -->
### Round 1 -> Next
目标: 尝试有理式+指数混合结构，验证是否可在保持RMSLE低的前提下简化复杂度
动作: 设计包含(omega/T)有理项和exp项的候选，重新fit并评测
验收: 新候选RMSLE较0.045改善>5%，或Exact Accuracy提升显著
<!-- APPEND_NEXT -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(0.30901259999999997)*(0.0017844287664398673*math.sqrt(math.sqrt(math.exp(0.02636736573827715...` | RMSLE=0.0450342988122 | Exact=0 | 完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->

## Worth Trying Next
### Round 1 -> Next
- 目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。
- 动作：重写为 `omega/T` 无量纲输入，并重新做常数扫描。
- 验收：新候选需优于当前最优（RMSLE < 0.0450343），且 `evaluate_submission` 无 `math range error`。

<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
