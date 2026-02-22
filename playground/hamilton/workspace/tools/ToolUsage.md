# Hamilton Eureka Tools

## 函数索引

（暂无函数，添加函数后请更新此表）

| 函数名 | 功能 | 输入 | 输出 |
|--------|------|------|------|
| | | | |

## 查看函数详情

```bash
# 查看所有函数
python -c "from tools import tool; print([x for x in dir(tool) if not x.startswith('_')])"

# 查看单个函数详情
python -c "from tools import tool; help(tool.func_name)"
```

## 使用示例

```python
from tools import tool

# 查看可用函数
print([x for x in dir(tool) if not x.startswith('_')])
```
