# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）
本轮PySR输出的top-1候选在训练集与OOD上误差极高，说明符号搜索进入非物理相关区域，需调整输入采样范围与搜索结构以逼近真实规律。
洞察：PySR第一候选复杂度高但在测试集表现严重退化，可能陷入局部最优或过拟合，需尝试指数/分数型结构并调整输入覆盖范围。
本轮PySR候选包含log(omega)和sqrt(omega)项，体现对频率依赖的探索，但未能捕捉到明确的occupation number形态，性能较差。
本轮尝试通过 PySR 拟合获得候选公式，引入了 sqrt 与 log 混合项，较前轮更复杂，但在 RMSLE 上稍有改善，Exact Accuracy 依旧为 0。
本轮PySR的最佳候选在训练集上表现较好（loss≈0.0036），但在评测中RMSLE大幅退化至101，Exact Accuracy为0，表明在OOD数据上存在严重偏差。需优化搜索空间结构并加强跨数量级的输入覆盖。
- 新候选通过PySR发现，公式中包含log和sqrt(omega)的组合，但实际RMSLE较上一轮上升，说明当前结构对数据拟合不佳，需要探索其他非线性形式。
本轮 PySR 候选通过复杂系数与ω、T的线性组合拟合了 total_power，损失值较低，但 RMSLE 高于预期，Exact 为 0。需要探索更符合物理先验的指数或分段形式来降低 RMSLE。
关键洞察：PySR 生成的 rank1 候选公式在本轮数据上拟合误差较低，但在全局评测 RMSLE 高于当前最优，说明公式结构在 OOD 范围内泛化不足，需调整分母与 log(omega) 项的组合。

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(3.444598e+45)*(T*(8.142002341883547e-14 - 2.0992900833979092e-12/math.log(omega))*(math.sqrt...` | RMSLE=85.9640193235 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `(4.129727e+45)*((4.483732908356811e-25*math.sqrt(omega) - 4.483732908356811e-25*omega)*(omega...` | RMSLE=97.9714000733 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 3 | protocol_eval | `(4.129727e+45)*((T - math.log(omega))*(-13.2022055350093*T/omega + 1.0020224891175376e-5 - 1....` | RMSLE=100.566921705 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 4 | protocol_eval | `(4.129727e+45)*((2.3878083771959095e-9*math.sqrt(omega) - 0.0010938474009335424)*(1.106483311...` | RMSLE=100.234289707 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 5 | protocol_eval | `(4.129727e+45)*(-0.02954237615904198 + 31.53596612035581*math.log(0.1423870720418131*T)/(1355...` | RMSLE=101.181118741 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 6 | protocol_eval | `(2.011977e+51)*(0.12916028016773815/((T - 923.6172292778409)*(-244.8404480785782*T**2/omega -...` | RMSLE=111.537093892 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 7 | protocol_eval | `(2.011977e+51)*(0.0016850376548828614 + 0.004401924005539781/math.log(44.378094842089603*T*ma...` | RMSLE=111.856632399 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 8 | protocol_eval | `(1.062421e+52)*(0.0029886708485411147 - 0.03727963032519568*math.log(0.25742358465667725*math...` | RMSLE=113.808135192 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 9 | protocol_eval | `(1.062421e+52)*(omega*(-2.716763228739602e-20 + 2.250417429759434e-21/((omega*(-(-7.243973602...` | RMSLE=104.972023797 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 10 | protocol_eval | `(1.062421e+52)*(3.9792645882527131/((T + 3.0893709470011146*T/omega)*(T - 11.245656889195*mat...` | RMSLE=111.617488396 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->

## 候选方程解析（Round 5）
### 1) 方程与物理解释
该候选公式引入了对温度的对数依赖项以及对频率的平方根依赖，尝试通过分母线性组合捕捉occupation number随ω和T变化的非线性响应。
### 2) 参数/系数敏感性
系数31.53对log(T)的放大作用明显，但分母中13550常数与√ω的线性组合使得对频率响应较弱。
### 3) 物理洞察
公式暗示了某种限制机制（分母形式）对辐射功率的调制，可能对应滤波或能级间隙效应，但未能捕捉到真实的B-E分布指数形态。
### 4) 消融分析
去掉常数偏移项（-0.0295）几乎不影响训练loss，说明该项贡献不大；改变放大系数或分母结构可能显著影响结果，应重点优化分母结构与指数项组合。

## Worth Trying Next
调整PySR搜索空间与参数组合，扩大输入采样的数量级覆盖，并引入指数或分式结构以捕捉occupation number的非线性变化。
下一步：扩大输入采样范围，覆盖更多数量级，引入指数和分式结构复合项进行PySR搜索；尝试降低复杂度以减少过拟合。
引入变换后的指数项，在指数计算前对输入进行夹取避免溢出，并尝试跨数量级的输入采样以探索occupation number过渡区域。
- 下一轮将切换到包含exp和多项式混合的结构族，控制指数部分数值稳定在[-60,60]，并扩大输入跨数量级覆盖，避免出现nan值。
下一步将尝试引入物理先验中常见的指数项，结合系数截断提高数值稳定性，并扩大输入覆盖范围以改善 OOD 指标。
下一步计划：尝试不同结构族（如指数型或分段函数）并引入物理先验，扩展 PySR 搜索空间，增强跨数量级输入覆盖以改善 OOD 表现；必要时手动构造候选进行对照评测。

<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
