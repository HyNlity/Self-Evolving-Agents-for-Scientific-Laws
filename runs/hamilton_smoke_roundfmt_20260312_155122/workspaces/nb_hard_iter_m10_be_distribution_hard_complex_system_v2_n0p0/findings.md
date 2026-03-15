# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 指标：RMSLE=83.7650216495；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 2
- 指标：RMSLE=84.4247022753；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(2.028636e+41)*(1.72951225778193e-15*T*(omega*math.log(T)/(T - 49.966170019931816) + 0.843300...` | RMSLE=83.7650216495 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `(8.750127e+47)*(-3.048497025276545e-22/((1.7278079324701907/(T - 1*4027.7591240210295))*((-11...` | RMSLE=84.4247022753 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->

## Worth Trying Next
下一轮计划：扩大 T 与 omega 的输入范围，尝试不同结构族（引入 exp 截断避免 overflow），以改善 RMSLE。
扩大 T 与 omega 的输入范围，添加极值样本并引入新结构族如含截断指数项的表达式，测试数值稳定性，力争改善 RMSLE。

<!-- APPEND_NEXT -->

## 最优方程演化
（记录最优方程在各轮中的变化过程）
## 候选方程解析（Round 2）
### 1) 方程与物理解释
该表达式为多项式与复杂分式组合，将 T 和 omega 置于多层乘除结构中，暗示物理量间存在非线性耦合。
### 2) 参数/系数敏感性
对 T 接近 4027.76 处的值敏感，分母包含 T*T*omega 项，频率和温度的变化会显著改变输出量级。
### 3) 物理洞察
复杂结构可能拟合积分关系中的局部行为，但泛化能力不足，尤其在极端参数下不稳定。
### 4) 消融分析
移除高阶 T*omega 项或简化分母结构会显著改变输出，表明这些项对当前拟合有主导作用。
