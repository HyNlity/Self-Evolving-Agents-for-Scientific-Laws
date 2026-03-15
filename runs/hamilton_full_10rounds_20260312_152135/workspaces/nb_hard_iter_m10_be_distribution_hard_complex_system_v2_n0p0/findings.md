# 研究发现

## 关键洞察
（经验证的数据观察和物理关系）

### Round 1
- 指标：RMSLE=86.8904437126；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 2
- 指标：RMSLE=78.8730929201；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 3
- 指标：RMSLE=84.566158725；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 4
- 指标：RMSLE=89.7272646123；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 5
- 指标：RMSLE=95.4260294516；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 6
- 指标：RMSLE=95.2769960062；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 7
- 指标：RMSLE=99.0432335673；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 8
- 指标：RMSLE=97.4811638965；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 9
- 指标：RMSLE=95.7633864627；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

### Round 10
- 指标：RMSLE=95.644731432；Exact=0
- 结论：未完成; symbolic=false; source=protocol_eval

<!-- APPEND_FINDINGS -->

## 实验结果
| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |
|------|------|------|-----------|-----------|------|
| 1 | protocol_eval | `(4.118995e+44)*(T*T*(T - 0.5582268302996053)*(T*1.5525401360258228e-14 + 1.804397091377231e-1...` | RMSLE=86.8904437126 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 2 | protocol_eval | `(4.118995e+44)*((-T + T*T)*T*(2.3963455910560154 - (-0.0005117832564098537)*(T + 1.8451697586...` | RMSLE=78.8730929201 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 3 | protocol_eval | `(4.118995e+44)*((T - 107.51419836213866)**2*(9.863715831515978e-21*T*(T + math.log(omega/T)) ...` | RMSLE=84.566158725 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 4 | protocol_eval | `(4.118995e+44)*(omega*(omega*omega*(omega*5.165770413939921e-7*(T - 1*1.3166714850286587) - o...` | RMSLE=89.7272646123 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 5 | protocol_eval | `(4.118995e+44)*(omega*(-7.727822032299534e-18*T - 3.863908625804662e-18*math.sqrt(omega) + 4....` | RMSLE=95.4260294516 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 6 | protocol_eval | `(4.118995e+44)*(T*(-T - 6.291423882743758e-6 - (-1)*1.0000000000661338/(omega/(T*omega))) - o...` | RMSLE=95.2769960062 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 7 | protocol_eval | `(2.880351e+45)*((omega*(-4.605564209756787e-8) - (T + omega*(-1.831169761563026e-19)*omega))*...` | RMSLE=99.0432335673 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 8 | protocol_eval | `(2.880351e+45)*(-9.948317955694439e-12*omega*(8.94469729387428e-7*T - 1.278275265352842e-13*o...` | RMSLE=97.4811638965 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 9 | protocol_eval | `(2.880351e+45)*(-3.9026773425164025e-14*math.sqrt(omega) + 1.1708032027549207e-13*omega - 3.9...` | RMSLE=95.7633864627 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |
| 10 | protocol_eval | `(2.880351e+45)*((-7.7899167664443838e-11*T*omega**(0.25) + 8.046954131857384e-15*omega)*(math...` | RMSLE=95.644731432 | Exact=0 | 未完成; symbolic=false; source=protocol_eval |

<!-- APPEND_RESULTS -->
## Worth Trying Next
下一步：尝试引入 omega 与 T 的组合项（比值或积）并测试指数族与有理多项式混合结构，以提升 exact_accuracy。
下一步：回滚到 Round 2 最优候选，并探索包含 omega 与 T 乘积的指数型模型，以改善 RMSLE 和 exact_accuracy。
下一步：尝试基础结构中加入物理常数与 omega/T 的指数项，并进行截断以防溢出，探索指数+多项式混合结构是否能提升 RMSLE 和 exact_accuracy。
下一步：回滚到 Round 2 最优候选，并探索包含 omega/T 指数项与多项式组合的新结构，控制指数输入截断以防溢出，并使用跨数量级样本进行 PySR 拟合。
下一步：切换到包含omega/T指数项与多项式混合的新结构族，同时在evaluate中对指数输入做截断，防止溢出，并使用跨数量级覆盖的样本再次PySR拟合以改善RMSLE与Exact。
下一步：针对当前候选性能不佳情况，回滚到 Round 2 最优结构族，设计包含 omega/T 指数项并结合多项式或对数项的模型，利用跨数量级样本重新拟合。
## 候选方程解析（Round 10）
### 1) 方程与物理解释
该方程为多项式与线性组合结构，捕捉了T的平方根和线性项，以及omega的1/4次幂与线性项，反映了频率与温度的复合影响。
### 2) 参数/系数敏感性
对T的变化高度敏感，尤其是平方根和线性项系数；omega的次幂项影响在低频时较显著，高频时由线性项主导。
### 3) 物理洞察
结构暗示occupation number受T和omega共同调制，可能对应某种复合模式的光谱形状而非单一的指数衰减或纯多项式增长。
### 4) 消融分析
去除omega的1/4次幂项会降低拟合精度，说明该非线性成分对捕捉实验数据至关重要。
在下一轮中，尝试构建包含 omega/T 的指数项与多项式组合的新结构，并在拟合中应用指数输入截断（例如 [-60, 60]），以防止溢出；同时继续使用跨数量级样本进行 PySR 拟合，期待提高 exact_accuracy。
下一步：探索包含指数衰减项并对 omega/T 比值进行归一化，尝试改善 OOD 表现，同时避免 math range error。
下一步：切换到包含omega/T指数项与多项式混合的结构族，并在evaluate_submission中对指数输入[-60,60]截断，继续用跨数量级样本进行PySR拟合，提升RMSLE和Exact。

<!-- APPEND_NEXT -->
## 最优方程演化
（记录最优方程在各轮中的变化过程）
