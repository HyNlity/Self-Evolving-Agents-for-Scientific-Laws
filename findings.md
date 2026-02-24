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
