# NewtonBench Task Template

在运行时，请通过 `--task` 提供具体任务描述。建议包含如下字段：

```yaml
profile: newtonbench
module: m0_gravity
system: vanilla_equation
difficulty: easy
law_version: v0
noise: 0.0
code_assisted: false
```

并附上任务目标与约束（例如最多实验轮次、最终输出函数签名）。

