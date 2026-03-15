# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

### Round 2
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

### Round 3
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

### Round 4
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

### Round 5
- 机制判读：出现指数项，说明模型在尝试表达热激发机制，但无量纲比值结构（omega/T）仍不完整。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(3.595242e+41)*(omega*(1.134119810762599e-23*omega*((1.1965644467333898e-9 - 1.16943041617549...` | RMSLE=87.7551994687 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `(1.215779e+49)*(1.26588215648957e-46*T*omega**2*(T + 3766.4993112067277)*(math.sqrt(omega) + ...` | RMSLE=101.656482534 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 3 | protocol_eval | `(1.215779e+49)*(omega*(omega*9.914857059839108e-53*omega*omega - 0.9999999999999992) + omega)` | RMSLE=98.88678882 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 4 | protocol_eval | `(1.215779e+49)*(omega*(-1.0670532361618422e-28*omega + 1.0670532361618422e-28*math.sqrt(omega...` | RMSLE=84.3865222968 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 5 | protocol_eval | `1.0 / (math.exp(x) - 1)` | RMSLE=0.0832749827756 | Exact=0 | 完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->

## Worth Trying Next
### Round 1 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.083275），且 `evaluate_submission` 无 `math range error`。

### Round 2 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.083275），且 `evaluate_submission` 无 `math range error`。

### Round 3 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.083275），且 `evaluate_submission` 无 `math range error`。

### Round 4 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.083275），且 `evaluate_submission` 无 `math range error`。

### Round 5 -> Next
- 目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。
- 动作：重写为 `omega/T` 无量纲输入，并重新做常数扫描。
- 验收：新候选需优于当前最优（RMSLE < 0.083275），且 `evaluate_submission` 无 `math range error`。

<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
## 候选方程解析（Round 1）
### 1) 方程与物理解释
该方程为多项式与平方根混合结构，包含 T 的倒数线性项，试图拟合 total_power 与 omega、T 的关系，但缺乏指数型依赖特征，无法反映光子占据数的物理机制。
### 2) 参数/系数敏感性
系数级别跨越数十数量级，方程对 omega 的高阶项敏感，对 T 的倒数项变化亦敏感，但数值上易在高频区发散。
### 3) 物理洞察
缺乏与 Bose-Einstein 分布一致的 exp(omega/T) 依赖，推测是因结构族限制导致。
### 4) 消融分析
移除 sqrt(omega) 项或 T 倒数项会显著改变输出，说明它们目前在拟合中占据重要位置，但可能是对物理信号的误拟合，需换结构验证。
