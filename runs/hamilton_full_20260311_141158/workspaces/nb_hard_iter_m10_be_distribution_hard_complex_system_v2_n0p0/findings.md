# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| Round 1 | system_backfill | `N/A` | - | - | 未完成; RMSLE=None; exact=0; symbolic=false |
| Round 6 | system_backfill | `N/A` | - | - | 未完成; RMSLE=None; exact=None; symbolic=None |
| Round 7 | hypothesis_BE_form | n = 1/(exp(1.2e-21*omega/T)-1) | - | - | 评测失败; RMSLE=32.05; exact=0; symbolic=false |

| Round 10 | BE_distribution_simple_constant | n = 1/(exp(1.4388e-21*omega/T)-1) | - | - | 评测失败; RMSLE=NaN; exact=0; symbolic=false; code syntax error |

## 候选方程解析（Round 10）
### 1) 方程与物理解释
本轮继续基于Bose-Einstein分布假设，尝试使用常数c=1.4388e-21，保持omega/T的指数依赖关系，旨在测试不同常数是否能改善拟合效果。常量取值参考了物理常数比值量级。实验测得不同温度和频率下总功率，但本轮提交因语法错误未被正常评测。
### 2) 系数表（跨实验条件）
| 条件 | 系数 c |
|------|--------|
| Round10实验 | 1.4388e-21 |
### 3) 物理洞察
评测返回语法错误导致无法获得RMSLE指标，exact=0，symbolic=false。说明提交的<final_law>代码块存在格式问题，需严格保证函数定义与缩进符合Python语法，且<final_law>标签内无额外字符。
### 4) 消融分析
去掉T会失去温度依赖性，去掉omega会失去频率依赖性。不同带宽条件对积分结果的影响在本轮未被验证。下一轮应修复语法问题，并尝试在指数项外侧引入幂律修正（如omega^m*T^n）探测复杂系统的真实规律。
| Round 8 | BE_distribution_refined | n = 1/(exp(hbar*omega/(kB*T))-1) | - | - | 评测失败; RMSLE=9.64; exact=0; symbolic=false |

| Round 9 | BE_distribution_with_constants | n = 1/(exp(hbar*omega/(kB*T))-1) | - | - | 评测失败; RMSLE=9.60; exact=0; symbolic=false |

## 候选方程解析（Round 9）
### 1) 方程与物理解释
本轮采用标准物理常数形式的 Bose-Einstein 分布：使用普朗克常数 ħ 与玻尔兹曼常数 kB 直接构造指数因子 hbar*omega/(kB*T)，返回光子平均占据数 n。原假设是复杂系统的行为依然可以由理想 BE 分布描述。
### 2) 系数表（跨实验条件）
| 条件 | ħ | k_B |
|------|-------------|-------------|
| 物理常数 | 1.054e-34 | 1.38e-23 |
### 3) 物理洞察
评测结果 RMSLE≈9.60，exact=0，symbolic=false，与上一轮基本相同，表明直接套用物理常数形式不足以捕捉复杂系统的非线性特征，可能需要在指数项外引入额外多项式或幂律因子修正。
### 4) 消融分析
去掉 T 会失去温度依赖，去掉 omega 则失去频率依赖；单参数 BE 分布不足以拟合全域。不同 bandwidth 实验下功率差异仍与拟合结果存在较大偏差，下一轮应尝试 omega^m*T^n 等修正形式，探索跨尺度匹配能力。
## 候选方程解析（Round 8）
### 1) 方程与物理解释
本轮重新采用类 Bose-Einstein 分布模型，但常数直接使用物理常数 ħ 与 k_B 的比值，通过 omega/T 的比值生成指数项，旨在与复杂系统的频率和温度关系保持物理一致性。
### 2) 系数表（跨实验条件）
| 条件 | ħ/k_B |
|------|--------|
| 物理常数 | 1.054e-34 / 1.381e-23 |
### 3) 物理洞察
评测结果 RMSLE≈9.64，exact=0，symbolic=false，虽然相比上一轮有显著改善，但符号等价判定依旧未通过。这说明简单的物理常数比值形式虽能降低误差，但不足以捕捉复杂系统中额外的非线性依赖。
### 4) 消融分析
移除 T 会失去温度依赖性，移除 omega 会失去频率依赖性。本轮在不同 bandwidth 下功率变化与理论曲线存在偏差，建议下一轮引入额外的多项式修正（如 omega^2 或 T^m 项）或非指数因子以改进拟合。
| Round 5 | system_backfill | n = 1/(exp(1.4e-21*omega/T)-1) | - | - | 评测失败; RMSLE=NaN; exact=0; symbolic=false; code error |

## 候选方程解析（Round 5）
### 1) 方程与物理解释
本轮基于BE分布假设，尝试使用常数c=1.4e-21并保持omega/T的指数依赖关系，意在探测常数调整对拟合的影响。
### 2) 系数表（跨实验条件）
| 条件 | 系数 c |
|------|--------|
| Round5实验 | 1.4e-21 |
### 3) 物理洞察
评测过程中出现语法错误导致代码解析失败（unexpected character after line continuation），因此无法获得有效的RMSLE。说明需要在生成<final_law>代码块时确保函数定义严格符合Python语法，同时计算需保证数值稳定性。
### 4) 消融分析
从实验结果看，不同温度和频率下总功率的变化明显，但当前方程未经过有效验证。未来需修正代码格式并考虑引入频率或温度的幂律因子，以提升符号等价的可能性。
| Round 4 | system_backfill | n = 1/(exp(h_over_k*omega/T)-1) | - | - | 评测失败; RMSLE=7.84; exact=0; symbolic=false |

## 候选方程解析（Round 4）
### 1) 方程与物理解释
本轮继续基于 Bose-Einstein 分布假设，系数 h_over_k = 4.799e-11 设定为有效常数，平均占据数 n 与 omega/T 呈指数衰减形式，反映了频率与温度的比值在调节光子占据数中的作用。
### 2) 系数表（跨实验条件）
| 条件 | 系数 h_over_k |
|------|---------------|
| Round4实验 | 4.799e-11 |
### 3) 物理洞察
RMSLE≈7.84，exact=0，symbolic=false，表明方程形式未能符号等价复杂系统真实规律。与Round3相比，几乎无指标改善，暗示需要引入附加项或修正常数。
### 4) 消融分析
移除 T 导致失去温度敏感性，移除 omega 导致失去频率依赖。本轮在不同 bandwidth 下输出差异仍存在，但与真实系统的积分峰值匹配不足，需尝试结合多项式或幂律因子改善拟合。
| Round 3 | system_backfill | n = 1/(exp(h_k*omega/T)-1) | - | - | 评测失败; RMSLE=7.81; exact=0; symbolic=false |

## 候选方程解析（Round 3）
### 1) 方程与物理解释
采用类 BE 分布形式，系数 h_k=4.8e-11 作为有效常数，假设平均占据数 n 与 omega/T 呈指数衰减，反映频率与温度的相对比值对光子占据数的调节作用。
### 2) 系数表（跨实验条件）
| 条件 | 系数 h_k |
|------|----------|
| Round3实验 | 4.8e-11 |
### 3) 物理洞察
RMSLE≈7.81，较前一轮有明显下降，但符号等价判定未通过。模型能捕捉部分规模变化，但仍未完全匹配复杂系统的输出，可能需引入额外因子或修正指数项。
### 4) 消融分析
移除 T 将丧失温度敏感性，移除 omega 则失去频率依赖，单纯指数项仍不足以拟合全域数据。在不同 bandwidth 条件下输出差异表明积分近似对指标有影响，需进一步测试更窄频段或调整常数以改善拟合。
| Round 2 | system_backfill | n = 1/(exp(c*omega/T)-1) | - | - | 评测失败; RMSLE=29.86; exact=0; symbolic=false |

## 候选方程解析（Round 2）
### 1) 方程与物理解释
继续采用类普朗克分布形式，但调整常数 c = 1.1e-20 以避免除零错误，假设平均占据数 n 与 omega/T 呈指数衰减。
### 2) 系数表（跨实验条件）
| 条件 | 系数 c |
|------|--------|
| Round2实验 | 1.1e-20 |
### 3) 物理洞察
数据与模型预测差异较大（RMSLE≈29.86），说明所选常数及函数形式可能未能捕捉复杂系统的真实频率与温度依赖关系，需探索更复杂的项或不同的函数依赖。
### 4) 消融分析
若移除 T，则模型缺乏温度敏感性；若移除 omega，则缺乏频率位移效应，两者缺一不可。但单纯使用指数项不足以拟合复杂系统输出，需要引入额外修正项或改变函数形态以降低误差。
| 1 | 初始猜测 | n = 1/(exp(c*omega/T)-1) | NaN | NaN | 评测失败，出现float division by zero |

## 候选方程解析（Round 1）
### 1) 方程与物理解释
采用类普朗克分布的形式，假设平均占据数 n 随 omega/T 呈指数衰减，常数 c 为试探值。
### 2) 系数表（跨实验条件）
| 条件 | 系数 c |
|------|--------|
| 基础 | 1.2e-22 |
### 3) 物理洞察
实验数据跨度大，推测谱辐射依赖于 omega 和 T 的比值，但当前常数估计导致不稳定（分母为零）。
### 4) 消融分析
去掉 T 影响或仅保留 omega 参数时，输出行为异常，表明必须同时考虑两者。
## 最优方程演化
（记录最优方程在各轮中的变化过程）

<!-- HAM_SYS_BACKFILL_ROUND_6 -->
## 系统回填 Round 6
- task_completed: false
- satisfied: False
- run_experiment_success_calls: 1
- evaluate_submission_success_calls: 0
- 评测指标: rmsle=None, exact_accuracy=None, symbolic_equivalent=None
- protocol_violations: missing_successful_evaluate_submission, missing_final_law_block, final_law_missing_discovered_law_signature

<!-- HAM_FINDINGS_TEMPLATE_ROUND_6 -->
## 候选方程解析（Round 7)
### 1) 方程与物理解释
本轮采用类 Bose-Einstein 分布形式，常数 C=1.2e-21，平均占据数 n 与 omega/T 呈指数依赖，旨在探测复杂系统中频率与温度的关系。
### 2) 系数表（跨实验条件）
| 条件 | 系数 C |
|------|--------|
| Round7实验 | 1.2e-21 |
### 3) 物理洞察
评测结果 RMSLE≈32.05，exact=0，symbolic=false，距离符号等价较远，说明仅使用单纯 omega/T 的指数项不足以捕捉全域。
### 4) 消融分析
去掉 T 导致失去温度依赖性，去掉 omega 导致失去频率依赖。本轮测试 bandwidth 对结果有明显影响，但积分峰值与真实系统匹配不足，需探索引入幂律修正项、频率平方项等结构以改进拟合。
## 候选方程解析（Round 6）
### 1) 方程与物理解释
- 方程：`N/A`
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