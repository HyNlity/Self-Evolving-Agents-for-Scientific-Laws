# Hamilton - 符号回归 Agent

Hamilton 是一个基于 EvoMaster 框架的符号回归（Symbolic Regression）Agent，专门用于在**过完备变量**环境下发现数学方程。

## 核心能力

- **多轮迭代分析**：通过多轮迭代逐步发现数据中的潜在方程
- **变量分析**：自动分析变量相关性、重要性，筛选关键特征
- **PySR 集成**：调用 PySR 进行符号回归，自动记录参数和结果
- **残差分析**：验证发现方程的可靠性
- **物理含义推理**：解释发现方程的物理意义

---

## 架构

### 三层架构

```
Playground (编排) → Exp (执行) → Agent (执行)
```

- **HamiltonPlayground**: 整体编排，管理多轮迭代
- **RoundExp**: 单轮执行，协调两个 Agent
- **Agent**: 具体执行分析任务

### 双 Agent 协作

```
┌─────────────────────────────────────────────────────────────┐
│                    HamiltonPlayground                       │
│                    (多轮循环编排)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       RoundExp                              │
│  ┌──────────────────┐    ┌─────────────────────────────┐ │
│  │ HamiltonAgent    │    │ Eureka Agent           │ │
│  │ - 变量分析        │    │ - 评估 PySR 结果             │ │
│  │ - 方法选择        │    │ - 残差分析                  │ │
│  │ - PySR 调用       │    │ - 物理含义推理               │ │
│  └──────────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 复用 EvoMaster 能力

Hamilton 完全基于 EvoMaster 框架构建，复用了大量基础设施：

| 复用模块 | 用途 |
|----------|------|
| `BasePlayground` | 编排生命周期管理 |
| `BaseExp` | 实验执行基类 |
| `ToolRegistry` | 工具注册与管理 |
| `BaseTool` | 自定义工具基类 |
| `@register_playground` | 装饰器注册 |
| LLM 配置与调用 | OpenAI/Anthropic |
| Session 管理 | 本地执行环境 |

---

## 文件结构

```
playground/hamilton/
├── core/
│   ├── playground.py      # HamiltonPlayground，多轮迭代编排
│   └── exp.py             # RoundExp，单轮执行逻辑
├── tools/
│   └── pysr_tool.py       # PySR 工具封装
├── prompts/
│   ├── hamilton_system.txt
│   ├── hamilton_user.txt
│   ├── data_analysis_system.txt
│   └── data_analysis_user.txt
└── TODO.md
```

### Workspace 文件

```
workspace/
├── data.csv              # 输入数据
├── analysis.md       # 分析历史（HamiltonAgent 书写）
├── insight.md            # 靠谱发现（Eureka Agent 书写）
├── experiment.json       # PySR 参数与结果（系统自动记录）
└── history/
    └── round{N}/
        ├── scripts/      # 每轮代码
        └── results/      # 每轮结果
```

> 注意：`run.py` 默认会为每次任务创建全新的 run workspace（`runs/<agent>_<timestamp>/workspace/`）。
> 建议把你的 `data.csv` 放在 `playground/hamilton/workspace/data.csv` 作为模板，Hamilton 会在每次运行时自动拷贝到新 workspace。

---

## 迭代机制

### 历史信息复用

通过三个文件实现跨轮次信息传递：

| 文件 | 写入者 | 内容 | 格式 |
|------|--------|------|------|
| **analysis.md** | HamiltonAgent | 分析过程、方法、决策 | `## Round N` + 自由书写 |
| **experiment.json** | 系统自动 | PySR 参数与结果表 | JSON |
| **insight.md** | Eureka Agent | BestEq/MSE/AltEqs/Notes/Recommendations | `## Round N` + 最小模板 |

### 迭代流程

```
Round N 开始
    │
    ├─ 系统自动初始化
    │     ├─ analysis.md 添加 ## Round N 头部
    │     └─ experiment.json 初始化结构
    │
    ├─ HamiltonAgent 执行
    │     ├─ 读取 analysis.md（历史）
    │     ├─ 变量分析、方法选择
    │     ├─ 调用 PySR
    │     │     └─ 系统自动记录参数和结果到 experiment.json
    │     └─ 追加分析过程到 analysis.md
    │
    ├─ Eureka Agent 执行
    │     ├─ 读取 experiment.json（PySR 结果）
    │     ├─ 残差分析、物理含义推理
    │     └─ 写入 insight.md
    │
Round N 结束 → 进入 Round N+1
```

---

## 可拓展性

### 1. 新增 Agent

在 `HamiltonPlayground.setup()` 中添加新 Agent：

```python
# playground.py
def setup(self):
    # 现有 Agent
    self.hamilton_agent = self._create_agent(...)
    self.eureka_agent = self._create_agent(...)

    # 新增 Agent
    agents_config = self.config.agents
    new_agent_config = agents_config['new_agent']
    self.new_agent = self._create_agent(
        name="new_agent",
        agent_config=new_agent_config,
        ...
    )
```

配置文件中添加：

```yaml
agents:
  new_agent:
    llm: "openai"
    max_turns: 50
    enable_tools: true
    system_prompt_file: "prompts/new_agent_system.txt"
    user_prompt_file: "prompts/new_agent_user.txt"
```

### 2. 新增工具

继承 `BaseTool` 创建新工具：

```python
# tools/new_tool.py
from evomaster.agent.tools.base import BaseTool, BaseToolParams

class NewToolParams(BaseToolParams):
    name: ClassVar[str] = "new_tool"
    param1: str = Field(description="参数1")
    param2: int = Field(default=10, description="参数2")

class NewTool(BaseTool):
    name: ClassVar[str] = "new_tool"
    params_class: ClassVar[type[BaseToolParams]] = NewToolParams

    def execute(self, session, args_json: str) -> tuple[str, dict]:
        params = self.parse_params(args_json)
        # 实现逻辑
        return result, metadata
```

在 `create_hamilton_registry()` 中注册：

```python
def create_hamilton_registry() -> ToolRegistry:
    registry = ToolRegistry()
    tools = [BashTool(), EditorTool(), ThinkTool(), FinishTool(), NewTool()]
    for tool in tools:
        registry.register(tool)
    return registry
```

### 3. 新增分析算法

在 prompt 中引导 Agent 调用新算法：

```python
# hamilton_user.txt
## 执行

1. 读取分析历史
2. 执行分析（可使用：相关性分析、PCA、特征重要性、互信息等）
3. 追加到 analysis.md
```

### 4. 拓展 RoundExp 流程

在 `RoundExp.run()` 中添加新步骤：

```python
def run(self, task_description: str, task_id: str = "exp_001") -> dict:
    # ... 现有步骤 ...

    # 新增：步骤3
    self.logger.info(f"[Round {self.round_num}] Running NewAgent...")
    new_task = TaskInstance(...)
    new_trajectory = self.new_agent.run(new_task)
    new_result = self._extract_agent_response(new_trajectory)

    # ... 返回结果 ...
```

### 5. 新增数据记录

在 `experiment.json` 中添加新字段：

```python
# pysr_tool.py 或 exp.py
def _record_extra(self, workspace, extra_data):
    # 读取现有 JSON
    # 添加新字段
    # 写入
```

---

## 使用方法

### 快速开始

```bash
# 准备数据
echo "x1,x2,x3,y" > workspace/data.csv
echo "1,2,3,5" >> workspace/data.csv
# ... 更多数据

# 运行
python run.py --agent hamilton --task "发现数据中的方程"
```

### 配置

修改 `configs/hamilton/config.yaml`：

```yaml
agents:
  hamilton:
    max_turns: 100  # 增加迭代次数

experiment:
  max_rounds: 10   # 增加轮数

experiment:
  pysr:
    niterations: 200  # 增加 PySR 迭代
```

---

## 设计理念

### 外部化记忆

不依赖 Agent 内部 memory，而是用文件作为持久化知识库：
- Agent 可自由读取历史
- 人类可阅读和检查
- 便于调试和审计

### 职责分离

- **Agent**: 负责语义分析、推理、决策
- **系统**: 负责结构化记录、流程控制
- **文件**: 负责信息传递、持久化

### 可调试性

- 每轮代码保存到 `history/round{N}/scripts/`
- PySR 参数和结果自动记录到 JSON
- 可回溯任何一轮的分析过程
