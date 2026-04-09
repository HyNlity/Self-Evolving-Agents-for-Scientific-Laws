# Hamilton — GAN 对抗式符号回归 Agent

<p align="center">
  <b>基于 EvoMaster 框架的自演化科学定律发现系统</b>
</p>

<p align="center">
  <a href="#architecture">架构</a> •
  <a href="#quickstart">快速开始</a> •
  <a href="#hcc">HCC 记忆系统</a> •
  <a href="#gan">GAN 对抗机制</a> •
  <a href="#evaluation">评估体系</a> •
  <a href="#benchmark">基准测试</a>
</p>

---

## 概述

Hamilton 是一个基于 LLM 的自主符号回归 Agent，能够从实验数据中自动发现物理定律（常微分方程）。

**核心创新**：

1. **HCC（Hierarchical Cognitive Caching）三层记忆系统** — L1 每轮工作记忆 / L2 跨轮知识积累 / L3 跨任务长期经验
2. **GAN 启发的对抗 Agent 结构** — Solver（生成器）提出方程，Critic（鉴别器）主动设计干预实验来拆穿伪规律
3. **多维评估体系** — 超越 MSE，从极限环复现、结构一致性、物理可解释性等多维度评估方程质量

---

<h2 id="architecture">架构</h2>

### 双模式运行

| 模式 | 命令 | 说明 |
|------|------|------|
| 单 Agent | `--agent hamilton` | Solver 独立工作，适合快速探索 |
| GAN 对抗 | `--agent hamilton-gan` | Solver + Critic 对抗，适合高质量发现 |

### GAN 对抗架构

```
┌──────────────────────────────────────────────────┐
│            GANHamiltonPlayground                  │
│    多轮编排 + Critic 触发判断 + L3 经验管理       │
└──────────────────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         ▼                            ▼ (条件触发)
┌─────────────────┐          ┌─────────────────┐
│  Solver (生成器)  │          │  Critic (鉴别器) │
│  发现→验证→提炼   │────────→│  干预实验→裁定   │
│  RoundExp        │  提交结果 │  CriticExp      │
└─────────────────┘          └─────────────────┘
         │                            │
         │   ┌────────────────────┐   │
         └──→│    共享 HCC 记忆    │←──┘
             │  L1: trace.md      │
             │  L2: plan/findings │
             │  L3: experience.md │
             │  [+] / [-] 标签    │
             └────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
     Critic approved     Critic rejected
     + Solver done       → 反馈注入下轮
     → 停止                Solver
```

### 三层架构（EvoMaster 标准）

```
Playground → Exp → Agent
```

- **Playground**：工作流编排、多 Agent 协调、L3 经验管理
- **Exp**：单轮执行、信号解析、L2 post-check
- **Agent**：LLM + Tools + Memory，执行工具调用循环

---

<h2 id="quickstart">快速开始</h2>

### 安装

```bash
git clone https://github.com/HyNlity/Self-Evolving-Agents-for-Scientific-Laws.git
cd Self-Evolving-Agents-for-Scientific-Laws

# 安装依赖
pip install -r requirements.txt
# 或使用 uv（更快）
uv sync
```

### 配置 API

编辑 `configs/hamilton/config.yaml`，设置 LLM API：

```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-5"
    api_key: "your-api-key"
    base_url: "https://your-api-endpoint"
```

### 运行

```bash
# 单 Agent 模式
python run.py --agent hamilton --task playground/hamilton/workspace/task.md

# GAN 对抗模式（推荐）
python run.py --agent hamilton-gan --task playground/hamilton/workspace/task.md

# 指定运行目录
python run.py --agent hamilton-gan --task playground/hamilton/workspace/task.md --run-dir runs/my_experiment

# 不使用 PySR（纯 scipy/sklearn）
python run.py --agent hamilton --config configs/hamilton/config_no_pysr.yaml --task playground/hamilton/workspace/task.md

# NewtonBench 批量测试
python run_newton.py
python run_newton.py --module m0_gravity --no-pysr
```

---

<h2 id="hcc">HCC 三层记忆系统</h2>

Hierarchical Cognitive Caching（分层认知缓存）是 Hamilton 的核心记忆架构：

```
┌─────────────────────────────────────────────┐
│  L3: experience.md                          │
│  跨任务持久 · 项目级存储 · [+]/[-] 标签     │
│  "这类任务用模板搜索比暴力搜索快 5x"         │
├─────────────────────────────────────────────┤
│  L2: plan.md + findings.md                  │
│  跨轮持久 · 阶梯式积累 · 只增不减            │
│  "当前最优: x''+163x+c₁x³+c₂x'=0, MSE=0.3" │
├─────────────────────────────────────────────┤
│  L1: history/round{N}/trace.md              │
│  每轮独立 · 系统重置 · Agent 填写            │
│  "尝试了 5 阶多项式，R²=0.92 但积分发散"     │
└─────────────────────────────────────────────┘
```

| 层级 | 文件 | 生命周期 | 内容 | 写入者 |
|------|------|----------|------|--------|
| **L1** | `history/round{N}/trace.md` | 每轮重置 | 操作记录、指标、工作笔记 | Solver |
| **L2** | `plan.md`, `findings.md` | 跨轮积累 | 战略计划、验证结论、最优方程 | Solver + Critic |
| **L3** | `experience.md` | 跨任务持久 | 有效策略 / 应避免做法 | Solver + Critic |

### L3 经验标签

```markdown
### [POSITIVE] 模板搜索优于暴力搜索
- **任务类型**: VIV-ODE
- **策略**: 基于物理假设设计 TemplateExpressionSpec
- **效果**: MSE 降低 3 个数量级，搜索时间减少 80%
- **适用条件**: 有明确物理先验的 ODE 发现任务

### [NEGATIVE] 不要用 timeout_in_seconds 控制 PySR
- **任务类型**: 所有符号回归
- **错误做法**: 设置 timeout=120s 让 PySR 自动停止
- **后果**: 搜索不充分，Pareto 前沿未收敛
- **正确做法**: 用 niterations 控制，观察收敛后手动停止
```

---

<h2 id="gan">GAN 对抗机制</h2>

### Critic 不是被动评审，是主动攻击者

传统双 Agent 系统的 Critic 只是"看结果打分"。Hamilton 的 Critic 是 **GAN 意义上的鉴别器**：

| 传统 Critic | Hamilton Critic |
|------------|-----------------|
| 读取结果 → 评分 | 设计实验 → 执行 → 用证据说话 |
| "MSE 太高，不通过" | "我用 x=50mm 测试了你的方程，积分 500s 后发散了" |
| 被动评审 | 主动攻击 |

### 五维审查框架

1. **极端值干预** — 把变量推到边界，检查方程是否产生非物理行为
2. **消融实验** — 逐项移除非线性项，证明每项不可或缺
3. **交叉验证对抗** — 跨风速预测、反向初始条件、OOD 退化
4. **数值稳定性攻击** — 长时间积分、初始条件微扰、步长敏感性
5. **物理一致性拷问** — 量纲分析、能量守恒、与经典理论对比

### Critic 触发条件（轻量执行）

Critic 不是每轮都跑（控制 token 消耗），触发条件（OR 逻辑）：

| 条件 | 说明 | 默认值 |
|------|------|--------|
| Solver 完成声明 | `task_completed="true"` 时必须审查 | 强制 |
| 周期触发 | 每 N 轮至少触发一次 | N=5 |
| MSE 平台期 | 连续 M 轮改进 < 阈值 | M=3, 阈值=1% |

配置方式（`config.yaml`）：

```yaml
experiment:
  max_rounds: 10
  critic:
    round_interval: 5
    mse_plateau_threshold: 0.01
    mse_plateau_window: 3
```

---

<h2 id="evaluation">多维评估体系</h2>

Hamilton 的评估 **超越 MSE**，根据任务特点选择合适的评估组合：

| 指标 | 优先级 | 适用场景 |
|------|--------|----------|
| **极限环振幅误差** | ★★★ | ODE/动力系统 — `\|A_pred-A_exp\|/A_exp` |
| **Z2S/B2S 双向收敛** | ★★★ | 极限环问题 — 不同 IC 必须收敛到同一稳态 |
| **跨工况结构一致性** | ★★☆ | 多工况数据 — 同一函数形式，仅系数变化 |
| **系数趋势合理性** | ★★☆ | 参数化方程 — 系数随控制参数单调变化 |
| **方程简洁性** | ★☆☆ | Occam's Razor — 更少的项更好 |
| **残差分析** | ★☆☆ | 所有任务 — 系统性模式说明缺少项 |
| MSE / R² | 诊断 | 仅作参考，不作主要优化目标 |

### findings.md 逐轮报告格式

每轮结束后，Solver 必须在 `findings.md` 中追加完整报告：

```markdown
### 第 3 轮

**方程**: x'' + 163.2x + 0.015x³ + (-1.23)x' + 0.0034x'³ = 0

**各项物理解释**:
- ω²x (163.2): 线性恢复力，弹簧效应，ω≈12.8 rad/s → f≈2.03 Hz
- c₁x³ (0.015): 硬化弹簧非线性，大位移时恢复力增强
- c₂x' (-1.23): 线性负阻尼，气动自激项，驱动振幅增长
- c₃x'³ (0.0034): 非线性正阻尼，限制振幅增长

**评估指标**:
| 风速 | 振幅误差(%) | MSE(训练) | Z2S收敛 | B2S收敛 |
|------|------------|-----------|---------|---------|
| U248 | 3.2        | 1.2e-1    | ✓       | ✓       |
| U254 | 4.1        | 1.5e-1    | ✓       | ✓       |
| ...  | ...        | ...       | ...     | ...     |
```

---

<h2 id="benchmark">基准测试</h2>

### VIV 涡激振动（默认任务）

从 5 组风速的振动时序数据中发现 ODE 方程族 `x'' = f(x, x'; U)`。

- **数据**：5 风速 × 2 初始条件（Z2S/B2S）+ 3 OOD 工况 = 13 CSV 文件
- **Baseline**：EvLOWN 论文 — `x'' + ω²x + c₁x³ + c₂x' + c₃x'³ + c₄x'⁵ = 0`
- **目标**：在振幅精度、泛化性、简洁性上超越 EvLOWN

### NewtonBench（批量测试）

12 个物理模块的定律发现基准：

```bash
python run_newton.py                              # 全部模块
python run_newton.py --module m0_gravity           # 单个模块
python run_newton.py --no-pysr                     # 不使用 PySR
python run_newton.py --dry-run                     # 仅生成任务列表
```

---

## 项目结构

```
Self-Evolving-Agents-for-Scientific-Laws/
├── run.py                              # 通用入口
├── run_newton.py                       # NewtonBench 批量测试
├── configs/hamilton/
│   ├── config.yaml                     # 主配置（含 Solver + Critic Agent）
│   ├── config_no_pysr.yaml             # 无 PySR 配置
│   └── prompts/
│       ├── hamilton_system.txt          # Solver 系统提示
│       ├── hamilton_user.txt            # Solver 用户提示
│       ├── critic_system.txt           # Critic 系统提示
│       └── critic_user.txt             # Critic 用户提示
├── playground/hamilton/
│   ├── core/
│   │   ├── playground.py               # HamiltonPlayground + GANHamiltonPlayground
│   │   ├── exp.py                      # RoundExp + CriticExp
│   │   └── constants.py                # HCC 标记 + Critic 触发常量
│   ├── prompts/                        # 提示词（与 configs 同步）
│   ├── workspace/                      # 工作空间模板
│   │   ├── task.md                     # VIV 任务描述
│   │   └── input/                      # 数据文件（13 CSV）
│   ├── experience/                     # L3 经验库（跨任务持久）
│   ├── README.md
│   └── TODO.md
├── evomaster/                          # 核心框架
│   ├── agent/agent.py                  # BaseAgent + Agent
│   ├── core/playground.py              # BasePlayground
│   ├── core/exp.py                     # BaseExp
│   └── skills/
│       ├── pysr/                       # PySR 知识技能
│       └── evo-protocol/               # 科学迭代方法论
└── docs/                               # 框架文档
```

### 运行时目录结构

```
runs/hamilton-gan_20260409_123456/
├── config.yaml                         # 运行配置副本
├── logs/evomaster.log                  # 运行日志
├── trajectories/trajectory.json        # 完整对话轨迹
├── records/experiment_*.json           # 实验总结
└── workspace/
    ├── task.md                         # 任务描述（只读）
    ├── plan.md                         # L2 战略计划
    ├── findings.md                     # L2 逐轮发现报告
    ├── experience.md                   # L3 经验库（symlink）
    ├── input/                          # 数据文件
    ├── lib/                            # 可复用脚本
    └── history/
        └── round{N}/
            ├── trace.md                # L1 本轮执行日志
            ├── scripts/                # Solver/Critic 脚本
            └── results/                # 本轮输出
```

---

## 配置参考

### 完整 config.yaml 说明

```yaml
llm:
  openai:
    provider: "openai"
    model: "gpt-5"                    # LLM 模型
    api_key: "sk-..."                 # API Key
    base_url: "https://..."           # API 端点
    temperature: 0.7                  # 采样温度
    max_tokens: 128000                # 最大输出 token

agents:
  hamilton:                            # Solver Agent
    max_turns: 120                     # 每轮最大工具调用次数
    skills: [pysr, evo-protocol]       # 可用技能
    tools:
      builtin: ["*"]                   # 全部工具（含 bash）

  critic:                              # Critic Agent（GAN 模式）
    max_turns: 40                      # 轻量执行
    skills: [evo-protocol]             # 不需要 PySR
    tools:
      builtin: ["*"]                   # 含 bash（用于运行干预实验）

experiment:
  max_rounds: 10                       # 最大轮次
  critic:
    round_interval: 5                  # Critic 触发间隔
    mse_plateau_threshold: 0.01        # 平台期阈值
    mse_plateau_window: 3              # 平台期检测窗口
```

---

## 引用

如果本项目对你的研究有帮助，欢迎引用：

```bibtex
@misc{hamilton2026,
  title={Hamilton: GAN-Adversarial Symbolic Regression Agent with Hierarchical Cognitive Caching},
  year={2026},
  url={https://github.com/HyNlity/Self-Evolving-Agents-for-Scientific-Laws}
}
```

## 致谢

Hamilton 基于 [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) 框架构建，感谢 SciMaster 团队的开源贡献。
