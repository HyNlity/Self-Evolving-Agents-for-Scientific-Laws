# Hamilton 任务（第2轮）

## 数据与变量
- 数据文件：
  - `playground/hamilton/workspace/data.csv`
  - `playground/hamilton/workspace/data_ood.csv`
- 变量：`x1..x10`（共 10 个）
- 真值结构是 `x1, x4, x7` 驱动，`x2, x3, x5, x6, x8, x9, x10` 可视为冗余变量

## 目标（最小版）
直接对 `x1..x10` 进行一次 PySR 基线，输出 top-k、train/ood MSE。 
