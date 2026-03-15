评测结果显示拟合公式与真实规律不同：真实规律是 CONSTANT*q1*q2/distance**3，当前公式是 /distance**2，导致 symbolic_equivalent=false 且 exact_accuracy=0. 下一轮计划：在实验中改变 distance 值，以更精确推断指数。

失败原因已记录到 findings.md，新的实验参数草案已写入 plan.md。
