# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

### Round 2
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

### Round 3
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 4
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 5
- 机制判读：当前主要是经验拟合项（多项式/对数/根号），更偏插值形态，物理机制解释较弱。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：存在 log/sqrt/高阶多项式等修正项，经典文献中通常不是主导项，需额外证据支持其物理真实性。

### Round 6
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 7
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 8
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 9
- 机制判读：方程包含 `exp(omega/T)` 型核心结构，反映了热激发下占据数随能量比上升而衰减的统计机制。
- 情境关联：任务为黑体腔+带通滤波+量热测功率，`omega/T` 的无量纲耦合与该物理情境一致。
- 文献对照：与经典 Bose-Einstein/Planck 家族同型；主要差异在常数标定，尚未出现明显“新项”。

### Round 10
- 机制判读：出现指数项，说明模型在尝试表达热激发机制，但无量纲比值结构（omega/T）仍不完整。
- 情境关联：当前形式未显式体现 `omega/T` 主导关系，与黑体辐射情境的关联仍偏弱。
- 文献对照：与经典 `1/(exp(c*omega/T)-1)` 仍有结构差距，暂不支持“新规律”结论。

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(1.68957e+59)*(omega*(1.003734820727337e-34*T - 5.4653256093945001e-30)*math.sqrt(T*(-T**2 + ...` | RMSLE=102.284394119 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `(1.68957e+59)*((2.0030865230377035e-7 - 2.0350889923328664e-22*omega)*(0.005137767274235255 -...` | RMSLE=119.657425698 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 3 | protocol_eval | `1 / (math.exp(c * omega / T) - 1)` | RMSLE=11.5531209725 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 4 | protocol_eval | `1 / (math.exp(C * omega / T) - 1)` | RMSLE=31.8400019986 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 5 | protocol_eval | `(1.68957e+59)*(0.6456657110798717*T*math.sqrt(omega)*(1.594705313220391e-27*T**2 + 2.31033133...` | RMSLE=117.05921606 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 6 | protocol_eval | `1/(math.exp(const * omega / T) - 1)` | RMSLE=31.9017934767 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 7 | protocol_eval | `1 / (math.exp(a * omega / T) - b)` | RMSLE=0.306350918125 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 8 | protocol_eval | `1 / (math.exp(a * omega / T) - b)` | RMSLE=0.306721212593 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 9 | protocol_eval | `1/(math.exp(c*omega/T)-1)` | RMSLE=32.1158218286 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 10 | protocol_eval | `1/(math.exp(x)-1)` | RMSLE=0.0840764809129 | Exact=0 | 完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->
## Worth Trying Next
### Round 1 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 2 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 3 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 4 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 5 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：切换到 `1/(exp(c*omega/T)-k)` 结构族，避免继续纯多项式/对数经验拟合。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 6 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 7 -> Next
- 目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 8 -> Next
- 目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 9 -> Next
- 目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。
- 动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

### Round 10 -> Next
- 目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。
- 动作：重写为 `omega/T` 无量纲输入，并重新做常数扫描。
- 验收：新候选需优于当前最优（RMSLE < 0.0840765），且 `evaluate_submission` 无 `math range error`。

<!-- APPEND_NEXT -->
## 候选方程解析（Round 8）
### 1) 方程与物理解释
该候选方程为PySR拟合的高次多项式+对数混合结构，包含omega²与log(omega)项乘以T的线性组合，物理意义上与量子占据数公式偏离较大，难以解释其物理对应关系。
### 2) 参数/系数敏感性
系数夹带10^-32级别因子，整体输出跨数量级变化会导致结果极端不稳定，且对log(omega)输入极为敏感。
### 3) 物理洞察
模型结构缺乏指数退火形态，导致对高温或高频区拟合差异巨大，未能体现Bose-Einstein分布特征。
### 4) 消融分析
移除log项或调整系数会显著改变输出，但并不能恢复物理占据数行为，说明整体结构与真实物理规律偏离。

## 最优方程演化
（记录最优方程在各轮中的变化过程）
