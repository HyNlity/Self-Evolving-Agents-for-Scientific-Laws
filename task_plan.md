# Hamilton Agent 开发计划（NewtonBench）

## Goal
在不分叉 Hamilton playground 的前提下，用现有“单 Agent + prompt + skill”架构完成 NewtonBench 验证（先 pilot，再 324 tasks），并输出可量化的架构改进方案。

## Current Phase
Phase 8（Pilot Benchmark：easy36 已跑通，进入失败模式驱动改造）

## Success Criteria
1. 单任务链路稳定：Hamilton 可在 NewtonBench profile 下完成实验与提交（已达成）。
2. 小规模 pilot（>=36 tasks）可复现，产出 SA / RMSLE / 完成率 / 协议合规率。
3. 全量 324 tasks 可断点续跑并汇总。
4. 形成“改造前 vs 改造后”的对比与改进建议。

## Phases

### Phase 1: 上游合并 (EvoMaster v0.0.2) — `complete`
- [x] rsync 上游 `evomaster/` 到本仓库
- [x] EnvConfig 字段加默认值（解决 pydantic ValidationError）
- [x] 删除 `create_default_registry()`，统一用 `create_registry()`
- [x] config 迁移到 v0.0.2 格式

### Phase 2: 基础可靠性修复 — `complete`
- [x] finish 语义改为 round-scoped
- [x] no-tool-call 死循环提示修复
- [x] `task_completed/satisfied` 语义统一

### Phase 3: HCC 记忆与单 Agent 架构收敛 — `complete`
- [x] L2 post-check（`findings.md/plan.md` mtime）
- [x] `lib/` 持久化复用目录
- [x] 单 Agent 闭环运行（Discovery → Verification → Promotion → Finish）

### Phase 4: NewtonBench 第一批接入 — `complete`
- [x] 论文与协议梳理（round budget / action discipline / final law 格式）
- [x] `configs/hamilton/newtonbench.yaml`
- [x] `evomaster/skills/newtonbench`（prompt/experiment/evaluate 三脚本）
- [x] Hamilton workspace profile 化（支持非 CSV 任务）
- [x] 单任务 smoke test 跑通（`gpt-5-chat` + `OPENAI_BASE_URL=https://llm.dp.tech`）

### Phase 5: 网关兼容与执行链路修复 — `complete`
- [x] 修复“文本 JSON 而非 tool_calls”导致的卡死（agent tool-call recovery fallback）
- [x] 修复 NewtonBench 根路径解析（支持 repo-root fallback）
- [x] 修复 skill 脚本解释器漂移（统一 `sys.executable`）
- [x] 修复 `run_experiment.py --tag <value>` 兼容性

### Phase 6: 最小可跑路径确认 — `complete`
- [x] 在 Hamilton 内跑通 1 个 NewtonBench task（非平行 playground）
- [x] 记录已知质量问题与后续改造入口

### Phase 7: 协议对齐 + 评测流水线（当前） — `in_progress`
- [x] 新增 NewtonBench task 生成器（覆盖 module/system/difficulty/law_version/noise）
- [ ] 新增 Hamilton 批量运行入口（支持 `--task-file`、失败重试、断点续跑）
- [x] 新增结果聚合脚本（SA/RMSLE/完成率/平均轮数/token）
- [x] 汇总统计改为 trajectory 口径（避免 task 日志串扰导致调用计数污染）
- [ ] 在 round 结束时落地结构化工件（实验输入、实验输出、`final_law`、评测 JSON）
- [ ] 增加协议合规检查：
  - [x] `task_completed="true"` 前至少一次 `run_experiment` 成功
  - [x] `finish.message` 必含 `<final_law>def discovered_law(...)</final_law>`
  - [x] `task_completed="true"` 前至少一次 `evaluate_submission` 调用
  - [x] 汇总脚本支持区分 `protocol_core_ok` vs `protocol_full_ok`

### Phase 8: Pilot Benchmark（36~72 tasks）— `in_progress`
- [x] 跑 easy + 12 modules + 3 systems（最小 36 tasks）
- [x] 输出 baseline 指标与失败样本清单
- [x] 用高风险样本验证质量护栏是否实际拦截（`m1 complex`，`rmsle>20` 时不允许完成）
- [x] 修复汇总脚本“误识别 final_law”偏差（改为仅解析真实 finish 消息）
- [ ] 根据失败模式做第 2 轮 prompt/skill 改造

#### Phase 8 下一步执行清单（第 2 轮改造）
- [ ] 提示词强化：明确“只有当最后一次 `evaluate_submission` 达标时才可 `task_completed=true`”
- [ ] 在回合内增加“评测结果复述模板”约束（输出最后一次 `rmsle/exact_accuracy/symbolic`）
- [ ] 选取 12 个失败代表样本（覆盖不同 module/system）做快速回归
- [ ] 与 easy36 baseline 对比，评估 `protocol_full_ok` 与 `avg_rmsle` 是否改善

### Phase 9: Full Benchmark（324 tasks）— `pending`
- [ ] 全量任务执行（支持中断恢复）
- [ ] 产出总表与分组表（按 domain/system/difficulty）

### Phase 10: 架构改进方案沉淀 — `pending`
- [ ] 基于数据输出“问题-证据-改进”三联表
- [ ] 给出可执行改造包（prompt、tool、orchestration、evaluation）
- [ ] 形成最终报告（含下一阶段路线）

## Key Files

| 文件 | 作用 |
|------|------|
| `configs/hamilton/newtonbench.yaml` | Hamilton 的 NewtonBench profile 配置 |
| `playground/hamilton/core/playground.py` | 多轮编排 + workspace 初始化 |
| `playground/hamilton/core/exp.py` | 单轮执行 + `task_completed` 解析 |
| `playground/hamilton/prompts/hamilton_newtonbench_system.txt` | NewtonBench 系统提示词 |
| `playground/hamilton/prompts/hamilton_newtonbench_user.txt` | NewtonBench 用户提示词 |
| `evomaster/agent/agent.py` | tool-call recovery fallback（兼容网关文本 JSON） |
| `evomaster/agent/tools/skill.py` | skill 脚本执行（固定使用当前解释器） |
| `evomaster/skills/newtonbench/scripts/run_experiment.py` | 交互实验桥接脚本 |
| `evomaster/skills/newtonbench/scripts/evaluate_submission.py` | 提交评测桥接脚本 |
| `scripts/newtonbench/generate_hamilton_tasks.py` | 生成 Hamilton `--task-file`（可直接跑 pilot/full） |
| `scripts/newtonbench/summarize_hamilton_run.py` | 汇总 run 目录协议合规率与评测指标（可选自动评测） |
| `playground/hamilton/records/experiment_*.json` | 实验记录（用于失败模式分析） |

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 不新建平行 playground | 满足用户约束，保留 Hamilton 单 Agent 主线 |
| NewtonBench 能力尽量下沉到 skill 脚本 | 减少 core 硬编码，保持“prompt+skill”理念 |
| 先跑通链路，再补协议护栏与评测流水线 | 先确保可执行，再提升可靠性和可比性 |
| 先做 pilot 再全量 324 tasks | 控制成本，先验证改造方向有效 |

---

## Phase 11: Hamilton 基线流程对齐（PySR 回归）— `pending`

### 背景判断
- 当前 NewtonBench 流程可跑通协议，但搜索主力已偏向 LLM 手写候选，和 Hamilton 原始“PySR 主搜索器”路线不一致。
- 结果表现为：候选表达式跨轮跳跃较大、局部变好后退化、缺少稳定的“空间内连续搜索”。

### 目标
在不丢失 NewtonBench 交互协议的前提下，恢复 Hamilton 的核心范式：
- LLM 负责假设与搜索空间设计；
- PySR 负责候选表达式密集搜索；
- 评测脚本负责闭环打分；
- plan/findings 负责跨轮记忆与回滚。

### 工作包（建议顺序）

#### WP1: 运行模式双轨化（`llm_direct` / `pysr_assisted`）— `pending`
- [x] 在 NewtonBench 配置中增加 `search_mode`，默认 hard 任务使用 `pysr_assisted`。
- [x] 保留 `llm_direct` 作为回退路径，避免一次性替换导致不可运行。

#### WP2: Skill 组合恢复 PySR 主线 — `pending`
- [x] NewtonBench 配置同时加载 `newtonbench + pysr + evo-protocol`。
- [x] session symlink 同时挂载 `evomaster/skills/pysr` 到 workspace。
- [x] 提前做 PySR 环境健康检查（Julia/依赖）。

#### WP3: 数据集缓存层（实验数据 -> 回归数据）— `pending`
- [x] 新增数据采样与落盘规范（按 module/system/difficulty/law_version/noise 分桶）。
- [x] 去重并累计跨轮实验点，避免每轮从零开始。
- [x] 输出标准训练表（`X,y`）供 PySR 使用。

#### WP4: PySR 搜索脚本化 — `pending`
- [x] 新增 `newtonbench` skill 脚本：`fit_pysr_candidates.py`（输入 dataset + operator/template，输出 top-k 候选）。
- [x] 支持模块级 operator/template 配置（避免单一算子空间覆盖全部物理域）。
- [x] 输出可直接转成 `def discovered_law(...)` 的候选代码片段。

#### WP5: 候选自动评测与最优回滚 — `pending`
- [ ] 每轮先评测 top-k 候选，再让 Agent做最终选择与解释。
- [ ] 强化 CURRENT_BEST 绑定：`finish(task_completed=true)` 只能提交当轮已评测最优解。
- [ ] 退化时自动回滚上轮最优，防止表达式漂移。

#### WP6: Prompt 回归“PySR 为主”语义 — `pending`
- [ ] 在 `hamilton_newtonbench_system/user` 中恢复“LLM 设计空间，PySR 密集搜索”的工作流约束。
- [ ] 保留现有反锚定、APPEND 标记位、协议闭环要求。
- [ ] 降低“频繁手写候选 + 频繁 finish”的行为倾向。

#### WP7: 指标与验收标准 — `pending`
- [ ] 新增过程指标：`pysr_runs`, `candidate_count`, `topk_evaluated_count`。
- [ ] 对比指标：`protocol_full_ok`, `symbolic_equivalent`, `exact_accuracy`, `rmsle`, `best_so_far_stability`。
- [ ] 先在单 hard 任务做 A/B（llm_direct vs pysr_assisted），再扩展 small batch。

### 风险清单
- PySR 依赖与耗时风险（Julia 环境/长时搜索）。
- 某些模块的目标表达式对 operator/template 敏感，需模块化配置。
- 过强协议门槛可能导致“永不完成”，需和 `max_rounds` 协同调参。

### Phase 11.2: P0 轻量回归（单 Agent 优先）— `complete`
- [x] 关闭 NewtonBench 默认 system backfill，避免系统过度介入 findings/plan。
- [x] 协议护栏收敛到核心闭环（run_experiment + evaluate_submission + final_law）。
- [x] 默认搜索模式切回 `llm_direct`，PySR 作为可选能力保留。
- [x] 简化上一轮反馈注入与 NewtonBench prompts，减少硬约束措辞。
- [x] 重写 NewtonBench task 模板，保留 APPEND 标记与最小记录规范。
