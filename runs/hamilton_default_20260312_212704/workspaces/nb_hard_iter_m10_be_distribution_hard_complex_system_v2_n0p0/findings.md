# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

### Round 2
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 3
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

### Round 4
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 5
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 6
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(301391.1)*((omega*omega*(T*2.9732123988720336e-9 - 9.625140623322043e-7) - 3.383442146031981...` | RMSLE=60.6555415555 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `1/(math.exp(c*omega/T)-1)` | RMSLE=32.0144540607 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 3 | protocol_eval | `(5.509393e+47)*(0.006866680831164841*(-4.960710990451404 + 16861.600485995008/T)**(1/4)/(-5.6...` | RMSLE=101.081278509 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 4 | protocol_eval | `1.0/(math.exp(c*omega/T)-1.0)` | RMSLE=32.040780828 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 5 | protocol_eval | `1.0/(math.exp(c*omega/T)-1)` | RMSLE=32.0258090925 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 6 | protocol_eval | `1 / (math.exp(C * omega / T) - 1)` | RMSLE=31.7882344374 | Exact=0 | 完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->

## Worth Trying Next
### Round 1 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

### Round 2 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

### Round 3 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

### Round 4 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

### Round 5 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

### Round 6 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 31.7882），且 `evaluate_submission` 无 `math range error`。

<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
