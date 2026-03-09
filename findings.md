# Findings & Decisions

## Requirements
- 理解 EvoMaster 框架（核心抽象、运行方式、扩展点）
- 理解符号回归任务 Hamilton agent（算法/回合/工具/产物）
- 全面审查代码：不优雅之处、bug、逻辑问题、框架复用程度、提示词质量、框架细节
- 输出清单与可执行建议（必要时给出补丁）

## Research Findings
- 仓库 README（中英）描述了 EvoMaster 的核心抽象：`evomaster/`（agent/core/env/skills/utils）、`playground/`（多种示例）、`configs/` 与 `docs/`。
- 当前工作区存在 `playground/hamilton/`，结构为：`core/`（exp/playground）、`prompts/`、`tools/`（pysr_tool）、以及 `workspace/tools/`（自定义工具接口与用法文档）。
- 当前工作区非常“脏”：`git status --porcelain` 显示大量新增/删除/修改文件（包括多个 `playground/*` 与 `configs/*` 被删除），需要在最终合并前拆分/清理提交，否则会显著影响可审查性与可维护性。
- 运行入口为 `run.py`：通过 `--agent <name>` 从注册表加载对应 Playground；Hamilton 已通过 `@register_playground("hamilton")` 注册，文档示例为 `python run.py --agent hamilton --task "..."`。
- `playground/hamilton/README.md` 明确 Hamilton 的多轮迭代机制与双 Agent（hamilton/eureka）分工，并约定跨轮次通过 `workspace/analysis.md`、`workspace/experiment.json`、`workspace/insight.md` 传递信息。
- `playground/hamilton/core/playground.py` 中存在一些“潜在不优雅/风险点”（待进一步确认）：
  - 通过 `sys.path` 注入项目根目录以导入 `evomaster`（通常应由包结构/安装方式解决）。
  - `max_rounds` 的读取方式形如 `getattr(self.config, 'experiment', {}).get('max_rounds', 5)`，如果 `self.config.experiment` 不是 dict 而是 Pydantic 模型/对象，可能会抛异常或读不到值。
- `evomaster/core/playground.py`（`BasePlayground`）实现了“配置加载→run_dir/workspace 管理→Session/LLM/Tool/Skill 初始化→运行→清理”的通用生命周期，但当前实现里出现了“同一份 config 既当 dict 又当 Pydantic 模型”两套兼容分支：
  - `_update_workspace_path()` 同时处理 dict 与 “Pydantic 模型”两类结构；
  - 但 `_setup_session()` 直接调用 `self.config.session.get(...)`，这要求 `self.config.session` 必须是 dict（否则会 AttributeError）。
  - 这类“动态鸭子类型”会让下游 playground（如 Hamilton）在读取 config 时出现隐蔽 bug，需要进一步追踪 `ConfigManager.load()` 的返回类型来统一约束。
- 进一步确认：`ConfigManager.load()` 返回的是 Pydantic 模型 `EvoMasterConfig`，其中 `session/llm/agent` 明确是 `dict` 字段，而 `logging/env/skill` 等是子模型字段；同时 `BaseConfig.Config.extra="allow"` 允许 playground 自定义字段（如 `agents`、`experiment`）直接以“原始 dict”挂到 config 上。
- `BasePlayground._create_agent()` 的提示词路径解析已从脆弱的字符串替换（`configs` -> `playground`）改为“显式推断 `configs/<subdir>` 对应 `playground/<subdir>`”，并优先支持 `config_dir` 相对路径（更稳定/更可移植）。
- 结合 `evomaster/agent/agent.py:load_prompt_from_file()` 可见：Agent 解析相对 prompt 路径默认是“相对于 config_dir”；而 prompts 实际放在 `playground/<name>/prompts/`，因此目前只能在 `BasePlayground._create_agent()` 里做路径重写才能工作。这暴露了一个框架层面的路径约定不一致：
  - 方案A：约定 prompts 跟随 configs（`configs/<agent>/prompts/...`），避免重写；
  - 方案B：引入 `prompt_root`/`playground_dir` 配置项或 `ConfigManager` 提供 `resolve_playground_path()`，由统一方法解析；
  - 方案C：Agent 的 `config_dir` 语义改为“playground 根目录”，而非“configs 目录”（需要梳理兼容性）。
- `BasePlayground.setup()` 的“多 agent 模式”已改为维护 `self.agents: dict[str, Agent]`（并保留 `self.agent` 作为向后兼容的 single-agent 默认），避免“循环覆盖只剩最后一个 agent”的隐性 bug。
- `playground/hamilton/core/exp.py`（`RoundExp`）的关键点与潜在问题：
  - `_init_round_files()` 在检测到 `analysis.md` 已包含 `## Round N` 时直接 `return`，会连带跳过 `experiment.json` 的初始化/结构修复；更合理的是“跳过写 header 但仍确保 experiment.json 存在且结构完整”。
  - 单轮返回结构里只保留了 `hamilton_trajectory`（字段名 `trajectory`），Eureka 的轨迹未返回/未记录（如果后续要诊断 eureka 的行为，会缺关键证据）。
  - `RoundExp` 自己实现了 `_extract_agent_response()`，但当前实现只检查 `trajectory.response/trajectory.result`。根据 `evomaster/utils/types.py`，`Trajectory.result` 是一个 dict（默认空 dict），不会自动包含“最终回答文本”；因此当前实现大概率只能拿到 `{}`。框架里的 `BaseExp._extract_agent_response()` 已经能从 `trajectory.dialogs[-1].messages` 提取最后的 assistant content，更应复用该实现。
- `playground/hamilton/tools/pysr_tool.py`（`PySRTool`）潜在风险点（待结合 `_build_pysr_code` 与真实 stdout 进一步确认）：
  - 通过 `python3 -c "{code}"` 拼接执行，存在引号/换行转义脆弱点，且对 LLM 生成内容的“代码注入面”较大。
  - `experiment.json` 读写缺少健壮性：JSON 损坏会直接抛异常；缺少 `rounds` key 时会 KeyError；初始化结构与 `RoundExp` 的初始化结构不一致（是否包含 `task` 字段等）。
  - `_parse_pysr_output()` 的 while/索引推进方式对“以 `Rank ...` 作为块边界”的输出不鲁棒，存在跳过下一块的风险（若没有明确分隔线）。
- 已确认：`_build_pysr_code()` 生成的代码本身包含大量双引号（`print("...")` 等），而 `execute()` 使用 `python3 -c \"{code}\"` 直接拼接到 shell 命令中，几乎必然导致引号未转义而执行失败（或被意外截断/注入）。建议改为“写入临时 .py 文件再执行”，并对 `combine`/列名等字符串使用 `json.dumps` 作为 Python 字面量注入，避免破坏语法与注入风险。
- `playground/hamilton/workspace/tools/` 目前仅有占位文档与空 `tool.py`（无任何函数实现）。如果 prompts/agent 逻辑依赖“可复用分析函数库”，当前状态会造成认知负担或运行期失败；建议要么补齐实现并在任务中实际调用，要么移除该目录及相关提示，避免“看起来支持但实际上不可用”。
- `evomaster/skills/eurekatool/` 已实现了一套相对完整的“可复用结果分析工具”（stdlib-only），包含：
  - `load_experiment()/get_round_record()/pick_best_equation()/format_alt_eqs()` 等用于稳定生成 BestEq/AltEqs 的函数；
  - `safe_eval_expr()/residual_summary()` 等用于做残差复算与异常点摘要（具备 AST 白名单，避免直接 eval 的风险）。
  这与 Eureka agent 的职责高度契合；相比之下，`workspace/tools/tool.py` 仍为空，建议 prompts 侧优先引导使用 `skills/eurekatool`，或至少说明两者分工，避免重复体系。
- `evomaster/skills/eurekatool/scripts/round_report.py` 会打印一个 `insight.md` block，其中包含 `Recommendations:` 小节；这与当前 Eureka prompts 的模板字段要求需要保持一致。建议把“insight 模板”作为单一事实来源（例如：以 round_report 输出为准），并同步修正 `eureka_system.txt`/`eureka_user.txt` 的模板约束。
- prompts 初步审查（Hamilton/Eureka）：
  - `hamilton_system.txt` / `hamilton_user.txt`：目标与交付物清晰，强调“每轮 delta + 实验定义 + 落盘到 analysis.md”，整体质量较好。
  - `eureka_system.txt` / `eureka_user.txt`：需要避免“自然语言关键字触发早停”这类脆弱机制，优先让 Eureka 输出结构化信号供系统判断是否早停/是否更新 Current Best。
- `configs/hamilton/config.yaml` 显示已尝试在配置层解决“工作区限制 vs 复用能力”的冲突：
  - `session.local.working_dir` 固定为 `./playground/hamilton/workspace`（即把 workspace 放在仓库内的固定目录，保证 `tools/`、历史文件等相对路径可用）。
  - `session.local.symlinks` 把 `evomaster/skills/eurekatool` 映射进 workspace 的 `skills/eurekatool`，从而允许 Eureka 在“不切换目录”的约束下维护可复用能力（这是一个对齐系统提示词约束的好设计）。
  - 但需要进一步确认：当 `run.py` 自动创建 `runs/<agent>_.../workspace` 并动态覆盖 workspace_path 时，`tools/`（workspace 模板）是否还能被正确带入；否则 prompts 中对 `tools/tool.py` 的依赖仍可能在实际运行时断裂。
- 已确认：`run.py` 的默认行为是**始终**为每次任务创建 `runs/<agent>_<timestamp>/workspaces/<task_id>/`（或单任务 `runs/.../workspace/`），并调用 `playground.set_run_dir()` 动态覆盖 `config.session.local.workspace_path/working_dir`。这对 Hamilton 的影响：
  - Hamilton 配置里把 workspace 设为 `./playground/hamilton/workspace`（意图固定工作区、内含 `tools/` 等模板），但会被 `run.py` 覆盖成新的空目录；
  - 结果：`tools/tool.py` 以及任何预置文件不会自动出现在新 workspace（除非用 symlinks/copy 显式带入）；`data.csv` 也需要额外机制注入，否则 PySR 必然失败。
- `evomaster/core/playground.py` 的轨迹机制是“所有 Agent 实例共享同一个 trajectory.json 文件路径（类变量）”，由 `BasePlayground._setup_trajectory_file()` 统一设置；对 Hamilton 这种双 agent 工作流是可行的，但需要确保写入条目里包含 agent 标识（当前 `_append_trajectory_entry()` 确实包含 `agent_name`），否则后续分析会混淆。

## Updates (Hamilton fixes applied)
- 已修复/加固（Hamilton 范围内）：
  - PySRTool：改为写脚本文件执行，避免 `python -c` 引号/注入脆弱性；stdout 输出 JSON 结果块优先解析；experiment.json 读写容错。
  - RoundExp：解析 Eureka 结构化信号（用于早停/更新 Current Best）；返回 `hamilton_trajectory/eureka_trajectory`（不再混用 `trajectory` 别名）。
  - HamiltonPlayground：workspace seed（tools/ + 可选 data.csv）；强校验 data.csv 必须存在；experiment_record 只保存摘要；setup 复用 BasePlayground（含 MCP/Skills/多 agent 初始化）并对 Eureka 工具集做 scope（不允许 PySRTool）。
  - prompts：Eureka 输出结构化信号，避免 finish 文本关键字误触发“完成”早停。
- `evomaster/env/local.py` 的 symlink 实现是“把源目录下的每个条目逐个 symlink 到 workspace/目标目录”，而不是把整个目录做一个 symlink：
  - 优点：目标目录始终是普通目录，结构稳定；编辑目标文件会直接改到源文件（因为是 symlink）。
  - 风险：每次 setup 会在目标路径存在且非 symlink 时 `shutil.rmtree(target_path)`，可能删除目标目录中“非源目录带入”的额外文件（如果用户在 workspace 里临时加了内容，会被清理）。
- Skills 机制（`evomaster/skills/base.py`）整体可用，但有一些“约定/实现不一致”的迹象：
  - `SkillRegistry` 的注释说 skills_root 下有 `knowledge/` 与 `operator/`，但实现是：knowledge skills 从 `skills_root/knowledge` 扫描，operator skills 则直接扫描 `skills_root` 的一级子目录（相当于没有 `operator/` 这一层）。
  - `SKILL.md` 的 YAML frontmatter 解析是“逐行 key:value 的轻量解析”，对复杂 YAML（引号、嵌套、冒号转义等）不鲁棒；若后续 skill 生态丰富，建议使用 `yaml.safe_load` 解析 frontmatter，或明确限制格式并在校验阶段报错提示。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 审查优先级：运行入口/数据流 → 明确 bug → 架构/提示词优化 | 先确保正确性与可运行，再谈优雅与可扩展 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 工作区存在大量未提交改动/新增删除文件 | 在报告中标注“以当前工作区为准”，并建议在 merge 前先清理/拆分提交 |

## Resources
- `README.md`, `README-zh.md`
- `evomaster/`（框架实现）
- `playground/hamilton/`（本次重点）

## Visual/Browser Findings
- N/A（本任务不涉及图片/浏览器）

## NewtonBench 调研（2026-03-09）

### 论文与基准关键信息
- NEWTONBENCH 共 324 个任务：108 条 shifted laws × 3 类系统设置（Vanilla / Simple / Complex），覆盖 12 个物理域。
- 任务从“静态表格拟合”改为“交互式模型探索”：Agent 需主动发起实验，收集 I/O，再推断隐藏 target law。
- 交互协议（Appendix C）核心约束：
  - 每轮只允许一个动作：`<run_experiment>` 或 `<python>` 或 `<final_law>`（无 code-assist 时不含 `<python>`）。
  - 最多 10 轮实验。
  - 每次实验最多 20 组输入参数。
  - 最终答案必须是 `<final_law>def discovered_law(...): ...</final_law>`。
- 评测指标：
  - Symbolic Accuracy（SA）：与 ground-truth 结构等价（常数值可忽略），论文采用 LLM-as-a-judge（98.3% human agreement）。
  - RMSLE：在独立 5000 点采样集上评估数据拟合质量。
- 论文附录给出：
  - 系统提示模板（Vanilla / Code-Assisted）
  - 各域采样分布（用于 RMSLE）
  - 每题预算（10 rounds、每轮最多 20 组输入）

### 与当前 Hamilton 架构的对照
- 可直接复用：
  - `run.py` 的单任务/批任务调度与 run_dir 隔离工作区机制
  - `BasePlayground` 的 agent+session+trajectory 生命周期
  - `RoundExp` 的轮次循环框架与 `finish` 信号停机机制
- 关键不匹配：
  - Hamilton 当前是“给定 CSV 静态拟合（PySR 主导）”，NewtonBench 需要“黑盒交互实验工具 + 增量采样”。
  - 当前无 `run_experiment` 语义工具，只有 `execute_bash`/`finish` 等通用工具。
  - 当前评测链路没有 SA 判等和 RMSLE（5000 点、按域采样）的基准实现。
  - 当前提示词强调多阶段研究记录（HCC/L1-L2），与论文 Appendix C 的严格 tag 协议不一致。

### 落地策略初稿（已废弃）
- 早期方案曾考虑独立 `newtonbench` playground，但已被用户约束否决。
- 当前生效路线：仅在 `hamilton` playground 内扩展 profile + prompt + skill，不再新增平行 playground。

### 用户约束更新（2026-03-09）
- 用户明确要求：**不能抛弃或旁路 Hamilton playground**；必须在现有“单 Agent + 提示词/skill 驱动”的架构上扩展 NewtonBench 能力。
- 这意味着后续方案应优先：
  - 复用 `playground/hamilton/core/playground.py` 和 `RoundExp` 的单 Agent 多轮框架；
  - 将 NewtonBench 的任务协议能力下沉为“可配置任务 profile + skill 工具链”，而非新建平行 playground 分叉。

### 第一批代码改造完成（2026-03-09）
- `playground/hamilton/core/playground.py` 已改为 profile 化 workspace 初始化：
  - 新增可配置项：`workspace_template`、`seed_input_csv`、`require_input_csv`；
  - 默认行为兼容旧 VIV 流程（默认仍 seed CSV 且校验 CSV）；
  - NewtonBench profile 可关闭 CSV 强依赖。
- 新增 `evomaster/skills/newtonbench/` skill 骨架：
  - `scripts/generate_task_prompt.py`
  - `scripts/run_experiment.py`
  - `scripts/evaluate_submission.py`
  - `references/protocol.md`
- 新增 Hamilton NewtonBench 配置与提示词：
  - `configs/hamilton/newtonbench.yaml`（仍是 `--agent hamilton`）
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt`
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt`
  - `playground/hamilton/workspace_newtonbench/task.md`
- README 已补充 NewtonBench profile 启动命令（同 playground，不分叉）。

### 已知验证限制
- 依赖缺口已补齐（见下方 smoke test）；
- 当前端到端唯一阻断为未配置 `OPENAI_API_KEY`（或可用兼容 provider 的 API key/base_url）。

### 依赖补齐与 Smoke Test（2026-03-09）
- 已创建项目虚拟环境：`./.venv`（Python 3.12.3）。
- 已安装主项目依赖：
  - `pip install -r requirements.txt`
  - `pip install -e .`
- 已拉取 NewtonBench 源码：`third_party/NewtonBench`。
- 已安装 NewtonBench 依赖：`pip install -r third_party/NewtonBench/requirements.txt`。
- NewtonBench skill 脚本 smoke test 通过：
  - `generate_task_prompt.py --module m0_gravity ...` 成功返回 `function_signature/param_description/task_prompt`
  - `run_experiment.py --module m0_gravity --inputs-json '[{"mass1":...}]'` 成功返回数值结果
  - `evaluate_submission.py --module m0_gravity --law-text '<final_law>...</final_law>'` 成功返回评测 JSON（在无 judge API key 时 `symbolic_equivalent=false`，并打印重试告警）
- Hamilton profile 单任务 smoke test（`configs/hamilton/newtonbench.yaml`）验证结果：
  - 成功完成：playground 自动导入、run_dir/workspace 初始化、local session 启动、`newtonbench/evo-protocol` skills 加载、symlink 建立
  - 失败点：Agent 初始化 LLM 阶段抛错 `ValueError: OpenAI API key must be provided in config`
  - 结论：框架接线已通，当前只差可用 LLM 凭据即可进入真正对话轮次。

### 接力验证更新（2026-03-09，用户提供 key + gpt-5-chat）
- 已将 `configs/hamilton/newtonbench.yaml` 的模型从 `gpt-5` 切换为 `gpt-5-chat`。
- 重新运行 Hamilton 单任务 smoke test：
  - 在默认沙箱下首先报 `httpcore.ConnectError: [Errno 1] Operation not permitted`（网络受限）；
  - 提权后可连 OpenAI API，但返回 `401 invalid_api_key`，错误为 `Incorrect API key provided`。
- 当前真实阻断从“缺失 key”收敛为“当前 key 无效（或与目标服务不匹配）”。

## NewtonBench 纠偏更新（2026-03-09，晚）

### 已确认可用的运行方式
- Hamilton 使用 `gpt-5-chat` 时可完成单任务端到端链路。
- 最新成功运行记录：
  - `runs/hamilton_20260309_143656/logs/task_0.log`
  - `playground/hamilton/records/experiment_20260309_143703.json`

### 已落地的关键修复（与链路成功直接相关）
- Agent 增加“文本 JSON → tool_calls”回退恢复（兼容网关不返回原生 function-calling 的情况）。
- `newtonbench` skill 脚本补了 repo-root fallback 路径解析（避免 run workspace 下找不到 `third_party/NewtonBench`）。
- `run_experiment.py` 兼容 `--tag value` 形式，避免 agent 输出参数稍有偏差就失败。
- Skill 脚本执行统一改为 `sys.executable`，避免 `.venv` 与系统 Python 混用导致依赖缺失。

### 当前最关键缺口（决定下一阶段改造优先级）
- 协议合规缺口：
  - Agent 仍可能在证据不足时直接 `finish(task_completed="true")`。
  - `finish.message` 中的实验结论可与真实 `<experiment_output>` 不一致（已有记录出现数值幻觉）。
- 评测闭环缺口：
  - 当前没有强制“提交前调用 `evaluate_submission.py`”。
  - 没有统一保存每题 SA / RMSLE 到可聚合结果表。
- 可规模化缺口：
  - 缺少 Hamilton 侧任务生成与批量执行/断点续跑流水线。
  - 324 tasks 的结果还无法一键汇总对比。
- 记忆执行缺口：
  - L2 post-check 持续告警（`findings.md` / `plan.md` 可能未更新），会影响多轮质量和可解释性。

### 结论（用于驱动后续开发）
- “能跑”问题已基本解决，当前核心从连通性转向“可靠性 + 可评测 + 可规模化”。
- 下一步应该优先做协议护栏与评测流水线，再做 pilot benchmark，再扩展到全量 324 tasks。

### 架构改进假设（待 pilot 验证）
- H1：在 `task_completed="true"` 前增加协议护栏（至少一次有效实验 + 合法 `<final_law>`）可显著降低“伪完成”。
- H2：在 round 结束自动执行一次 `evaluate_submission` 并回填指标，可降低高 RMSLE 但误判完成的比例。
- H3：将实验输入/输出、最终方程、评测结果结构化落盘后，可显著提升失败模式定位效率，减少 prompt 盲改。

## Phase 7 第一批实现（2026-03-09）

### 新增能力
- 已新增 `scripts/newtonbench/generate_hamilton_tasks.py`：
  - 可按 module/system/difficulty/law_version/noise 生成 Hamilton `--task-file`。
  - 已验证生成 easy-36 任务文件：`playground/hamilton/tasks/newtonbench_easy36.json`。
- 已新增 `scripts/newtonbench/summarize_hamilton_run.py`：
  - 可汇总每个 task 的状态、实验调用次数、`<final_law>` 是否存在、L2 告警、token 用量。
  - 支持 `--auto-evaluate` 自动调用 `evaluate_submission.py`，回填 SA/RMSLE（受 judge 可用性影响）。

### 验证结果
- `generate_hamilton_tasks.py` 生成结果符合预期（12 modules × 3 systems × easy × v0 × noise0 = 36）。
- `summarize_hamilton_run.py` 已对 `runs/hamilton_20260309_143656` 成功输出：
  - `newtonbench_summary.json`
  - `newtonbench_trials.csv`
- 汇总脚本已支持从 task log 回填任务元数据（即便未提供 `--task-file`）。

### 当前剩余差距
- 批量入口目前依赖 `run.py --task-file` 现有能力，尚未补“失败重试 + 断点续跑”编排层。
- round 级结构化工件（每轮实验输入输出、提交评测原始结果）尚未标准化落盘。

## Phase 7 第二批实现（2026-03-09）

### 协议护栏落地
- `playground/hamilton/core/exp.py` 已新增 NewtonBench 完成护栏：
  - 仅当 `finish(task_completed="true")` 且满足全部条件时才判定 `satisfied=true`：
    - 至少一次成功的 `run_experiment.py`
    - 至少一次 `evaluate_submission.py` 调用
    - `finish.message` 含 `<final_law>def discovered_law(...)</final_law>`
  - 若不满足，`signal` 会写入 `protocol.violations` 并设置 `protocol_guard_blocked=true`，防止伪完成被当作“已收敛”。

### 评测汇总稳健性修复
- `scripts/newtonbench/summarize_hamilton_run.py` 已修复 `invalid_evaluation_json`：
  - 现在会从混合 stdout 中提取“最后一个 JSON 对象”，兼容 `evaluate_submission.py` 前置重试日志。
- 汇总指标新增 `protocol_full_ok`（core 协议 + 至少一次 `evaluate_submission` 调用），用于区分“仅有 final_law”与“完成终态评测”的任务。

### 实测回归（用户 run 目录）
- 在 `runs/hamilton_20260309_154651` 上复跑 `--auto-evaluate` 后：
  - `auto_evaluated_tasks`: `0 -> 1`
  - `evaluation_error`: `invalid_evaluation_json -> null`
  - 新增 `protocol_full_ok: 0`，准确暴露该次运行“未调用 evaluate_submission”的协议缺口。
- 通过离线构造 trajectory 验证 `RoundExp` 新护栏逻辑：
  - 缺少 `evaluate_submission.py` 时：`satisfied=False` 且 `protocol_guard_blocked=true`
  - 具备 `run_experiment.py + evaluate_submission.py + final_law` 时：`satisfied=True`

## Phase 8 Pilot Findings（2026-03-09）

### 稳定性修复结论
- `evaluate_submission.py` 增加超时（`--eval-timeout-sec`）后，easy36 再未出现“评测子进程长时间无返回”的硬卡死。
- `call_llm_api.py` 支持 `OPENAI_BASE_URL`/`GPT_BASE_URL` 后，judge 请求可走与主 Agent 一致的 provider，避免了此前的模型/网关漂移。
- 模型别名归一化后，`gpt-5-mini` / `gpt-5-chat` / `gpt5mini` 混用不会直接触发 `api_source_mapping` 错误。

### easy36 Pilot 总体结果（run: `runs/hamilton_20260309_180705`）
- 汇总口径：`summarize_hamilton_run.py --auto-evaluate`（已改为从 trajectory 精确统计调用）。
- 核心指标：
  - `total_tasks=36`
  - `completed_tasks=36`
  - `protocol_core_ok=35`
  - `protocol_full_ok=34`
  - `with_run_experiment_success=35`
  - `with_evaluate_success=35`
  - `avg_exact_accuracy=0.0`
  - `rmsle_tasks=26/36`
  - `avg_rmsle=1.5485`（仅在有限值 RMSLE 样本上计算）
  - `avg_total_tokens≈31595.7`

### 失败模式（从 `newtonbench_trials.csv`）
- 协议失败样本：
  - `nb_0001_m0_gravity_easy_vanilla_equation_v0_n0p0` 缺少成功 `evaluate_submission`（`evaluate_success_calls=0`）。
- 指标覆盖缺口：
  - 10 个任务 `rmsle` 非有限值，主要由提交函数签名不匹配导致：
    - `Invalid function signature` / `Must be def discovered_law(...)`
    - 典型集中在 `m0/m10/m1/m2/m3/m4` 的部分 system。
  - 另有 1 个样本为 `math domain error`。
- 质量表现：
  - `symbolic_equivalent_true=0`，`exact_accuracy=0.0`（全部样本）；
  - RMSLE 最差样本：`nb_0012_m1_coulomb_force_easy_complex_system_v0_n0p0`（`20.6496`）。
- 行为层面：
  - `l2_not_updated_warning=36/36`，说明 Agent 基本未在每题内落实 L2 更新。

### 关键技术发现
- 旧版汇总按日志文本计数会被“后续任务日志混入当前 task log”污染，导致调用次数虚高。
- 改为 trajectory 口径后，协议计数可信度显著提升；此后 protocol 指标应以 trajectory 为准。
- 自动评测原先会吞掉 `evaluation.error`，导致 `rmsle` 缺失却无错误信息；现已回填到 `evaluation_error`。

### 对下一轮改造的直接启发
- 优先解决“签名不一致”：
  - 将 `generate_task_prompt.py` 返回的 `function_signature` 做强绑定校验（finish 前硬校验）。
- 降低“完成但低质量”的概率：
  - 在 `task_completed=true` 前加入可配置质量阈值（至少 `rmsle` 有限 + 非灾难性上界）。
- 强化结果可诊断性：
  - 每题落地最后一次 `evaluate_submission` 原始 JSON 到结构化文件，避免仅靠日志二次解析。

## Phase 8.1 护栏验证与汇总纠偏（2026-03-09，夜）

### 高风险任务验证结论
- 验证 run：`runs/hamilton_20260309_220308`（`m1_coulomb_force + complex_system`）。
- 日志与实验记录一致显示：
  - 最后一次评测 `rmsle≈20.64`、`exact_accuracy=0.0`；
  - Agent 执行了 `finish(task_completed="false")`，并明确“评测未达标，不能结束任务”；
  - `playground/hamilton/records/experiment_20260309_220346.json` 中 `signal.protocol.violations` 为：
    - `missing_final_law_block`
    - `final_law_missing_discovered_law_signature`
    - `rmsle_above_threshold`
- 这证明 RoundExp 新增的 quality guard（`max_rmsle=2.0`）在高风险样本上可触发拦截。

### 新发现的统计偏差与修复
- 偏差现象：`summarize_hamilton_run.py` 曾错误给出 `with_final_law=1 / protocol_full_ok=1`。
- 根因：脚本从整份 log 文本扫描 `<final_law>`，误命中 system prompt 示例块，而非真实 `finish.message`。
- 修复后行为：
  - `final_law` 仅从 `finish` 消息提取（trajectory 或 `📝 Finish Tool Arguments` 段）；
  - 新增 `task_completed_true/task_completed_false` 统计；
  - 新增“从日志回填最后一次 evaluation”能力（无需额外 auto-evaluate 也能拿到 `rmsle/exact_accuracy/symbolic`）。

### 修复后回归结果
- `runs/hamilton_20260309_220308`：
  - `task_completed_false=1`
  - `with_final_law=0`
  - `protocol_core_ok=0`，`protocol_full_ok=0`
  - `avg_rmsle=20.639647469042792`
- `runs/hamilton_20260309_215738`（2-task guard smoke）：
  - `task_completed_true=2`
  - `protocol_full_ok=2`
  - `avg_exact_accuracy=1.0`，`avg_rmsle=0.00020744057060517894`

### 下一步优先级（已收敛）
- 先做“失败模式驱动改造”第 2 轮：优先处理高 RMSLE 但能完成的问题（不是连通性问题）。
- 最小改造目标：
  - 在系统提示里强化“若评测不达标必须继续探索，不可输出最终完成答案”；
  - 让 Agent 在提交前显式引用最后一次 `evaluate_submission` 指标（减少自说自话）。
