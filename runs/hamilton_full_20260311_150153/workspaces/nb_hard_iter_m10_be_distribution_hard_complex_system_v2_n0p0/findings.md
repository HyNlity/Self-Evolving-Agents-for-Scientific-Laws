# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

## 候选方程解析（Round 1）
### 1) 方程与物理解释
family_A: Bose-Einstein相关核结构 1/(exp(c*omega/T)-1)，出现除零问题
family_B: 对数结构 log(1+c*T/omega)，无除零，捕捉占据数的非指数型趋势
### 2) 系数表（跨实验条件）
- family_A: c=1.4e-23
- family_B: c=5e3
### 3) 物理洞察
数据呈对数渐近性质，指数核未能稳定拟合。
### 4) 消融分析
去除带宽影响，使用 bandwidth->0 近似，family_B 拟合较稳。
## 候选方程解析（Round 2）
### 1) 方程与物理解释
family_A: 1/(exp(C*omega/T)-1) 核结构，作为示例对照；family_C: A*T^p/(omega^q + B) ，有理式核结构，避免了上一轮的对数核零除问题。
### 2) 系数表（跨实验条件）
- family_A: C=1.4e-23（未实际评测）
- family_C: A=1e5, p=2.0, q=3.0, B=1e4
### 3) 物理洞察
family_C通过引入omega的幂次分母和T的幂次分子，能模拟功率随T和omega的非线性变化，避免高频或低温下的极端值。
### 4) 消融分析
去除B项会导致高频omega时分母过小，功率暴增；降低p会削弱温度依赖。
## 候选方程解析（Round 3）
### 1) 方程与物理解释
family_A: Bose-Einstein核结构 1/(exp(C*omega/T)-1)，出现除零问题；family_B: 假设为多项式有理式 T^a/(omega^b+c) 避免在小T或大omega时极值发散。
### 2) 系数表（跨实验条件）
- family_A: C=1.4e-23
- family_B: a=2.0, b=3.0, c=1e4
### 3) 物理洞察
family_B通过调整分母参数可平衡频率和温度依赖，避免极端条件下功率暴增。
### 4) 消融分析
去除c项会导致大omega时功率暴增；改变a会改变温度依赖强度。
## 候选方程解析（Round 4）
### 1) 方程与物理解释
family_A: 改进Bose-Einstein形式 1/(exp(C*omega/T)-1+eps) 避免float除零；family_B: 多项式有理式 T^a/(omega^b+c)
### 2) 系数表（跨实验条件）
- family_A: C=1.4e-23, eps=1e-9
- family_B: a=2.0, b=3.0, c=1e4
### 3) 物理洞察
family_A仍在高rmsle区间表现不佳，可能原因是指数核对输入尺度过敏；family_B有潜力通过调节指数与平移改善拟合。
### 4) 消融分析
删除eps将导致除零，中小T无法稳定；调整a/b会改变温度依赖与频率依赖的强度，应在下一轮优化。
## 候选方程解析（Round 5）
### 1) 方程与物理解释
family_A: Bose-Einstein 核结构 1/(exp(C*omega/T)-1) 仅作对照；family_B: 多项式有理式形式 (T**2)/(omega**3 + C)，避免除零并稳定捕捉低温与高频行为。
### 2) 系数表（跨实验条件）
- family_A: C=1.4e-23
- family_B: C=1.4e-23
### 3) 物理洞察
family_B 通过 omega 的三次幂分母与 T 的平方依赖实现功率随频率/温度的非线性响应，避免了 BE 核结构在高频下发散的问题。
### 4) 消融分析
移除 C 会导致 omega 接近零时分母过小造成功率暴增；改变指数将改变温度依赖强度，建议在下一轮进一步调优。
## 候选方程解析（Round 7）
### 1) 方程与物理解释
family_A: BE 核改 eps，避免除零；family_B: 倒数平方根核结构，假设占据数随 omega/T 的平方根下降。
### 2) 系数表（跨实验条件）
- family_A: C=1.4e-23, eps=1e-9
- family_B: a=1e-9, b=1.0
### 3) 物理洞察
family_B 对输入尺度不敏感，但精度远低于当前最优；family_A 作为对照未评测。
### 4) 消融分析
去掉 b 会导致小 omega/T 时发散；改变 a 会影响频率与温度的衰减强度。
## 候选方程解析（Round 8）
### 1) 方程与物理解释
family_A: BE 核结构仅作对照，未参与评测；family_B: 多项式有理式核结构 (T**b)/(omega**b + c)，避免除零并捕捉低温高频下的非线性功率响应。
### 2) 系数表（跨实验条件）
- family_B: a=1e-21, b=3.0, c=1e-9
### 3) 物理洞察
分母引入 omega**b + c 稳定高频区响应，分子 T**b 强化温度依赖，整体形态模仿功率随温度/频率变化的缓变曲线。
### 4) 消融分析
去除 c 将导致 omega->0 时功率暴增；改变 b 将影响温度频率依赖强度。
<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| Round 1 | system_backfill | `math.log(1 + c * T / (omega + 1e-9))` | - | - | 未完成; RMSLE=0.0825271705282; exact=0; symbolic=false |
| Round 2 | system_backfill | `A * (T**p) / ( (omega**q) + B )` | - | - | 未完成; RMSLE=0.0835220253751; exact=0; symbolic=false |
| Round 3 | system_backfill | `1 / (math.exp(C * omega / T) - 1)` | - | - | 未完成; RMSLE=None; exact=0; symbolic=false |
| Round 4 | system_backfill | `1 / (math.exp(C * omega / T) - 1 + eps)` | - | - | 未完成; RMSLE=20.6453982061; exact=0; symbolic=false |
| Round 5 | system_backfill | `(T**2) / ((omega**3) + C)` | - | - | 未完成; RMSLE=0.0832676116104; exact=0; symbolic=false |
| Round 6 | system_backfill | `1 / math.log(1 + a * omega / T)` | - | - | 未完成; RMSLE=None; exact=None; symbolic=None |
| Round 7 | system_backfill | `1 / math.sqrt(a * omega / T + b)` | - | - | 未完成; RMSLE=0.606863111603; exact=0; symbolic=false |
| Round 8 | system_backfill | `(T**b) / (omega**b + c)` | - | - | 完成; RMSLE=0.0832048192724; exact=0; symbolic=false |

- Round 1: family_A failed (float division by zero, symbolic_equivalent=false, exact_accuracy=0.0, rmsle=NaN), family_B succeeded partially (symbolic_equivalent=false, exact_accuracy=0.0, rmsle=0.0825)
- Round 2: family_A (示例核结构)测试失败(未评测), family_C (多项式有理式)评测成功(symbolic_equivalent=false, exact_accuracy=0.0, rmsle=0.0835220253751)
- Round 3: family_A failed (float division by zero, symbolic_equivalent=false, exact_accuracy=0.0, rmsle=NaN), family_B not yet tested fully.
- Round 4: family_A (改进Bose-Einstein核+eps) symbolic_equivalent=false, exact_accuracy=0.0, rmsle=20.645; family_B 暂未评测，需下轮验证
- Round 5: family_B (多项式有理式核结构) symbolic_equivalent=false, exact_accuracy=0.0, rmsle=0.0832676116104297；family_A 未测试，本轮未达标。
- Round 7: 实验成功运行 3 组条件 (symbolic_equivalent=false, exact_accuracy=待评测, rmsle=待评测), family_A=BE核改eps, family_B=倒数平方根核结构。
- Round 8: family_B 多项式有理式核结构成功运行实验 (rmsle=0.083204819272, exact_accuracy=0.0, symbolic_equivalent=false)，优于上一轮失败的倒数平方根核结构，保留作为下一轮优化基线；family_A 仅作对照未评测。
<!-- APPEND_RESULTS -->

## Worth Trying Next
- 下一轮计划：尝试新核结构 family_C （例如多项式有理式形式 T^a/(omega^b + c)）以替代失败的 family_A，保留 family_B 作为对照，并探索更宽/更窄带宽组合
- 下一轮计划：切换到family_B多项式有理式核结构并调整指数和平移项，探索更宽/更窄带宽组合，避免family_A除零失败。
- 下一轮计划：调整 family_B 中 T、omega 的指数参数，探索更宽频率范围，并引入 bandwidth 的极值测试以观察积分近似影响；禁用 family_A（BE 核结构）以避免高 rmsle。
- 下一轮计划：禁用 family_B 倒数平方根核结构；改用 family_C 多项式有理式核结构并优化指数和平移项，探索更宽温度/频率范围。新参数示例: [{"temperature":2e2,"center_frequency":5e9,"bandwidth":2e7},{"temperature":5e3,"center_frequency":1e13,"bandwidth":1e9}]
- 下一轮将继续优化 family_B 多项式有理式核结构，通过调整指数 b 和常数 c 探索更大温度/频率范围内的拟合稳定性，避免 family_A 高 rmsle 问题；新增实验参数组合: [{"temperature":1e2,"center_frequency":1e9,"bandwidth":1e6},{"temperature":1e5,"center_frequency":1e14,"bandwidth":1e10}]
<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）

<!-- HAM_SYS_BACKFILL_ROUND_6 -->
## 系统回填 Round 6
- task_completed: false
- satisfied: False
- run_experiment_success_calls: 0
- evaluate_submission_success_calls: 0
- 评测指标: rmsle=None, exact_accuracy=None, symbolic_equivalent=None
- protocol_violations: missing_successful_run_experiment, missing_successful_evaluate_submission
- 最终方程:
```python
def discovered_law(omega, T):
    import math
    a = 1e-9
    return 1 / math.log(1 + a * omega / T)
```

<!-- HAM_FINDINGS_TEMPLATE_ROUND_6 -->
## 候选方程解析（Round 6）
### 1) 方程与物理解释
- 方程：`1 / math.log(1 + a * omega / T)`
- 项解释：待补充（请逐项解释变量、算子与物理机制）。

### 2) 系数表（跨实验条件）
| 系数 | 数值/范围 | 稳定性标注 | 物理解释 |
|------|-----------|------------|----------|
| 待补充 | 待补充 | 结构属性/情景属性 | 待补充 |

### 3) 物理洞察
- 当前评测：rmsle=None, exact_accuracy=None, symbolic_equivalent=None
- 规律解释：待补充（如尺度关系、主导项、极限行为）。

### 4) 消融分析
- 去掉关键项 A：待补充
- 去掉关键项 B：待补充

- 本轮候选函数：
```python
def discovered_law(omega, T):
    import math
    a = 1e-9
    return 1 / math.log(1 + a * omega / T)
```
