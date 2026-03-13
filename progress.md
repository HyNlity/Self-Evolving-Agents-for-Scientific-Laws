# Progress Log

## Session: 2026-02-21

### Phase 1: 范围确认 & 仓库摸底
- **Status:** complete
- **Started:** 2026-02-21
- Actions taken:
  - 运行 `session-catchup.py`，发现上一会话有未同步上下文提示
  - 检查 `git diff --stat` 与 `git status --porcelain`，确认工作区存在大量新增/删除/修改文件
  - 创建 `task_plan.md` / `findings.md` / `progress.md`
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Phase 2: 架构与数据流审查（EvoMaster Core）
- **Status:** in_progress
- **Started:** 2026-02-21
- Actions taken:
  - 梳理 `BasePlayground` 生命周期（run_dir/workspace/trajectory/session/tools/skills）
  - 梳理 Hamilton 的双 Agent 编排与配置（configs/hamilton/config.yaml）
  - 记录框架与 Hamilton 的关键不一致点（workspace seed、prompt path、multi-agent）
  - 按用户要求撤回对 `evomaster/core/playground.py` 的框架层改动（避免在审查阶段引入全局行为变化）
  - 修复 Hamilton 关键运行阻断点（PySRTool 执行方式、workspace 模板注入、experiment.json 初始化/落盘、结果提取）
  - 补充 Hamilton stdlib-only 单元测试，并通过 `python -m unittest`
- Files created/modified:
  - `findings.md` (updated)
  - `task_plan.md` (updated)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| NewtonBench task prompt 生成 | `generate_task_prompt.py --module m0_gravity ...` | 输出 task package JSON | 成功输出 `function_signature/param_description/task_prompt` | ✅ |
| NewtonBench 单次实验调用 | `run_experiment.py --module m0_gravity --inputs-json ...` | 返回 experiment output | 返回数值（`3.8859756999373156e+30`） | ✅ |
| NewtonBench 提交评测 | `evaluate_submission.py --module m0_gravity --law-text ...` | 返回 evaluation JSON | 返回 RMSLE/SA 结构化结果 | ✅ |
| Hamilton NewtonBench 单任务（无提权） | `run.py --agent hamilton --config ...` | 至少进入对话轮次 | 到 LLM 请求后报 `Operation not permitted`（沙箱网络限制） | ⚠️ |
| Hamilton NewtonBench 单任务（提权后） | 同上（带网络提权） | 至少进入对话轮次 | 可连 OpenAI 但返回 `401 invalid_api_key` | ⚠️ |
| Hamilton NewtonBench 单任务（`gpt-5-chat` + `https://llm.dp.tech`） | `run.py --agent hamilton --config configs/hamilton/newtonbench.yaml --task "...m0_gravity..."` | 端到端完成一轮并调用 finish | 成功完成，状态 `completed`，记录见 `runs/hamilton_20260309_143656/` | ✅ |
| 批量任务文件生成（easy36） | `generate_hamilton_tasks.py --modules all --systems ... --difficulties easy --law-versions v0` | 输出 36 个可运行任务 | 成功生成 `playground/hamilton/tasks/newtonbench_easy36.json` | ✅ |
| run 目录汇总（含 auto-evaluate） | `summarize_hamilton_run.py --run-dir runs/hamilton_20260309_143656 --auto-evaluate` | 输出 summary JSON/CSV 与核心指标 | 成功输出 `newtonbench_summary.json/newtonbench_trials.csv` | ✅ |
| run 目录汇总回归（修复 mixed stdout JSON 解析） | `summarize_hamilton_run.py --run-dir runs/hamilton_20260309_154651 --auto-evaluate` | `invalid_evaluation_json` 消失并写入评测指标 | 成功：`auto_evaluated_tasks=1`，`evaluation_error=null`，新增 `protocol_full_ok` 字段 | ✅ |
| RoundExp 协议护栏单元级验证（离线） | 构造 Trajectory 调用 `_parse_signal`（含/不含 `evaluate_submission.py`） | 无 evaluate 时拦截完成；有 evaluate 时放行 | 结果：`CASE_A satisfied=False`，`CASE_B satisfied=True` | ✅ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-21 | N/A | 1 | N/A |
| 2026-03-09 | OpenAI API key must be provided in config | 1 | 需在项目根 `.env` 或环境变量中配置 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`） |
| 2026-03-09 | `httpcore.ConnectError: [Errno 1] Operation not permitted` | 1 | 以提权方式重跑，验证非代码问题而是沙箱网络限制 |
| 2026-03-09 | `401 invalid_api_key` | 1 | 需替换为有效 OpenAI key（或匹配 base_url 的有效 key） |
| 2026-03-09 | 文本 JSON 未进入 `tool_calls` 导致 max_turns | 1 | 在 `agent.py` 增加 tool-call recovery fallback，恢复 `use_skill/finish` |
| 2026-03-09 | `run_experiment.py --tag round1_test` 参数解析失败 | 1 | `--tag` 改为可选值参数（`nargs='?'`） |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 7（NewtonBench 协议对齐与评测流水线） |
| Where am I going? | Phase 8–10（pilot → full benchmark → 架构改进方案） |
| What's the goal? | 用 NewtonBench 验证 Hamilton 架构并给出数据驱动的改进方案 |
| What have I learned? | See findings.md |
| What have I done? | See above |

## Session: 2026-03-09

### Phase 6: NewtonBench 接入方案调研
- **Status:** in_progress
- **Started:** 2026-03-09
- Actions taken:
  - 读取 `paper/NewtonBench Benchmarking Generalizable Scientific Law Discovery in LLM Agents.pdf`
  - 解析并提取 Appendix C prompt 模板、交互协议（run_experiment/python/final_law）、回合预算
  - 提取评测关键口径（SA + RMSLE，RMSLE 使用 5000 点独立采样）
  - 对照 `playground/hamilton` 当前架构，识别可复用模块与主要冲突点
  - 根据用户最新要求，确认必须在 Hamilton playground 内扩展（不新建平行 playground）
  - 广泛阅读 Hamilton 目录（README/TODO/core/prompts/workspace/records）及 Agent/Skill 执行链路
  - 完成第一批代码改造（Hamilton workspace profile 化 + newtonbench skill 骨架 + newtonbench 配置/提示词）
  - 完成静态校验：`compileall` 与 skill 脚本 `--help`
  - 端到端校验受阻：本地缺失 `pydantic`（导入 Hamilton 失败）与 NewtonBench 依赖 `numpy`
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 6.1: 依赖补齐 + 单任务 Smoke Test
- **Status:** in_progress
- **Started:** 2026-03-09
- Actions taken:
  - 创建项目虚拟环境 `./.venv`
  - 安装主项目依赖：`pip install -r requirements.txt` + `pip install -e .`
  - 克隆 NewtonBench 到 `third_party/NewtonBench`
  - 安装 NewtonBench 依赖：`pip install -r third_party/NewtonBench/requirements.txt`
  - 运行 NewtonBench skill 脚本最小验证：
    - `generate_task_prompt.py`（m0_gravity/easy/vanilla/v0）成功
    - `run_experiment.py` 成功返回实验数值
    - `evaluate_submission.py` 成功返回评测 JSON
  - 运行 Hamilton 单任务命令：
    - `python run.py --agent hamilton --config configs/hamilton/newtonbench.yaml --task "...m0_gravity..."`
    - 初始化到 skills 加载均成功，默认沙箱下在 LLM 请求时报网络受限
    - 提权后可达 OpenAI API，但返回 `401 invalid_api_key`
  - 按用户要求将模型改为 `gpt-5-chat`（`configs/hamilton/newtonbench.yaml`）
- Files created/modified:
  - `.venv/` (created)
  - `third_party/NewtonBench/` (created via git clone)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 6.2: 接力验证成功 + 下一步路线收敛
- **Status:** complete
- **Started:** 2026-03-09
- Actions taken:
  - 根据同事代码确认 provider 兼容配置：`OPENAI_BASE_URL=https://llm.dp.tech`、`model=gpt-5-chat`
  - 完成单任务端到端成功运行，验证 `use_skill -> run_experiment -> finish` 闭环
  - 复盘 `playground/hamilton/records/experiment_*.json` 与 run log，识别当前主要缺口：
    - 协议合规不足（可能“证据不足也完成”）
    - 实验结果引用不严谨（可能出现输出值幻觉）
    - 缺少批量任务与指标聚合流水线
  - 阅读 `third_party/NewtonBench` 批量执行脚本（`run_master.py`/`run_all_evaluations.py`）用于对齐 324 tasks 组织方式
  - 将下一阶段执行路线同步到 `task_plan.md`/`findings.md`/`progress.md`
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 7: 协议对齐与评测流水线（next）
- **Status:** in_progress
- **Planned actions:**
  - 生成 NewtonBench 批量 task-file（先 easy 36 tasks pilot）
  - 增加结果结构化落盘与聚合脚本（SA/RMSLE/完成率/轮次）
  - 增加完成前协议护栏（有效实验 + `<final_law>` 格式校验）

### Phase 7.1: 首批批量化工具落地
- **Status:** complete
- **Started:** 2026-03-09
- Actions taken:
  - 新增 `scripts/newtonbench/generate_hamilton_tasks.py`，支持按维度生成 Hamilton `--task-file`
  - 新增 `scripts/newtonbench/summarize_hamilton_run.py`，支持 run 目录汇总与可选自动评测
  - 产出 pilot 任务文件：`playground/hamilton/tasks/newtonbench_easy36.json`（36 tasks）
  - 在历史 run 目录上验证汇总脚本：成功输出 JSON/CSV 及协议合规统计
  - 更新 `playground/hamilton/README.md` 增加 batch pilot 使用说明
- Files created/modified:
  - `scripts/newtonbench/generate_hamilton_tasks.py` (created)
  - `scripts/newtonbench/summarize_hamilton_run.py` (created)
  - `playground/hamilton/README.md` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 7.2: 协议护栏 + 汇总稳健性修复
- **Status:** complete
- **Started:** 2026-03-09
- Actions taken:
  - 在 `playground/hamilton/core/exp.py` 落地 NewtonBench 完成护栏：
    - `task_completed="true"` 需满足成功实验、`evaluate_submission` 调用、`<final_law>`+`def discovered_law` 三项约束
    - 不满足时写入 `signal.protocol.violations` 并阻断 `satisfied=true`
  - 强化 `playground/hamilton/prompts/hamilton_newtonbench_system.txt`，显式要求“先评测后完成”
  - 修复 `summarize_hamilton_run.py` 对 `evaluate_submission.py` 混合 stdout 的 JSON 解析
  - 新增汇总指标 `protocol_full_ok`（核心协议 + evaluate 调用）
  - 使用离线构造 trajectory 的方式验证 `RoundExp` 协议护栏放行/拦截行为
  - 在用户 run 目录 `runs/hamilton_20260309_154651` 上回归验证脚本输出
- Files created/modified:
  - `playground/hamilton/core/exp.py` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `scripts/newtonbench/summarize_hamilton_run.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 8.0: easy36 Pilot 首次完整运行 + 稳定性回归
- **Status:** in_progress
- **Started:** 2026-03-09
- Actions taken:
  - 停止卡住的旧 easy36 进程（卡在 `evaluate_submission.py` 子进程）。
  - 修复 NewtonBench 评测链路稳定性：
    - `evomaster/skills/newtonbench/scripts/evaluate_submission.py`
      - 新增 `--eval-timeout-sec`（默认 90s）硬超时；
      - 新增 judge 模型别名归一化（`gpt-5-mini`/`gpt-5-chat` 等）；
      - 处理 `\\n` 形式的函数文本，避免 `discovered_law` 执行报错。
    - `third_party/NewtonBench/utils/call_llm_api.py`
      - OpenAI client 支持 `OPENAI_BASE_URL` / `GPT_BASE_URL`；
      - 增加 HTTP timeout（`NEWTONBENCH_HTTP_TIMEOUT_SEC`）和 `max_retries=1`；
      - `gpt5mini` 映射改为 `gpt-5-mini`，并补 `gpt5chat` 映射。
    - NewtonBench prompts 更新评测示例，显式加入 `--eval-timeout-sec 90`。
  - 重新跑单任务 smoke：
    - run dir: `runs/hamilton_20260309_180345`
    - 结果：`protocol_full_ok=1`，`run_experiment/evaluate_submission` 均成功执行，评测不再长时间阻塞。
  - 跑完 easy36 pilot：
    - run dir: `runs/hamilton_20260309_180705`
    - 命令：`run.py --agent hamilton --config configs/hamilton/newtonbench.yaml --task-file playground/hamilton/tasks/newtonbench_easy36.json`
  - 修复汇总脚本统计偏差：
    - 原因：每个 task 日志会混入后续任务日志，导致按日志计数的脚本调用次数虚高。
    - 方案：`summarize_hamilton_run.py` 改为优先从 `trajectory.json` 精确统计 `run_script` 调用与成功次数，并改为从最后一次 trajectory 统计 token。
    - 增加汇总字段：`exact_accuracy_tasks`、`rmsle_tasks`。
    - 自动评测时若 `evaluation.error` 非空或 `rmsle` 非有限值，会写入 `evaluation_error`。
- Files created/modified:
  - `evomaster/skills/newtonbench/scripts/evaluate_submission.py` (updated)
  - `third_party/NewtonBench/utils/call_llm_api.py` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt` (updated)
  - `scripts/newtonbench/summarize_hamilton_run.py` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

## New Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 单任务稳定性回归（修复后） | `run.py --agent hamilton --config ...newtonbench.yaml --task "...m0_gravity..."` | 不再卡在 evaluate，并满足完整协议 | 完成：`protocol_full_ok=1`，`run_experiment/evaluate_submission` 成功 | ✅ |
| easy36 pilot 全量运行 | `run.py --agent hamilton --config ...newtonbench.yaml --task-file ...newtonbench_easy36.json` | 36 题可跑完且不出现长阻塞 | 36/36 完成，运行结束于 `runs/hamilton_20260309_180705` | ✅ |
| easy36 汇总（trajectory 口径） | `summarize_hamilton_run.py --run-dir ...180705 --auto-evaluate --judge-model gpt5mini` | 输出协议合规 + 指标 + 覆盖率 | 成功输出：`protocol_full_ok=34/36`、`avg_rmsle=1.5485`、`rmsle_tasks=26/36` | ✅ |

### Phase 8.1: 高风险样本护栏验证 + 汇总脚本纠偏
- **Status:** complete
- **Started:** 2026-03-09
- Actions taken:
  - 对高风险样本执行与复盘：`runs/hamilton_20260309_220308`（`m1_coulomb_force/complex_system`）。
  - 从 `task_0.log` 与 `experiment_20260309_220346.json` 双重确认：
    - `evaluate_submission` 成功执行但 `rmsle≈20.64`；
    - Agent 最终 `finish(task_completed="false")`；
    - `signal.protocol.violations` 包含 `rmsle_above_threshold`。

## Session: 2026-03-13

### Phase 11.4: P0 协议收紧（hard 合法采样 + findings 稳定化）
- **Status:** in_progress
- **Started:** 2026-03-13
- Actions taken:
  - 收紧 `run_experiment.py`：`m10_be_distribution/complex_system` 现在必须显式传 `temperature/center_frequency/bandwidth`，禁止再用 `omega` 混过 experiment API。
  - 收紧 `fit_pysr_candidates.py`：complex-system 采样缓存也改为同样的官方字段校验，避免脏样本继续进入 PySR。
  - 将 hard 单任务配置的 `newtonbench_protocol.max_rmsle` 从 `1.0` 降到 `0.01`，防止 `RMSLE≈0.04` 时 1 轮直接停机。
  - 更新 Hamilton NewtonBench prompts 与 `workspace_newtonbench/task.md`，同步 hard 任务质量门槛与官方 experiment 键要求。
  - 重写 `findings.md` 归位逻辑：`关键洞察 / 实验结果 / Worth Trying Next` 现在按 section 重建，只保留一个 `APPEND_FINDINGS/RESULTS/NEXT` marker。
  - 增加 `候选方程解析` 的系统重建逻辑：从 `plan.md` 的 `CURRENT_BEST` 读取当前最优 round / equation / metrics，避免解析块停留在旧轮次。
  - 放宽对结构族的先验偏置：`关键洞察` 与 `Worth Trying Next` 不再默认把非 `exp` 结构判错，而是按 `exp/log/代数 surrogate` 三类比较。
  - 做离线回归：
    - `py_compile` 通过；
    - 复制一份已损坏 hard findings 到临时目录后执行归位，成功收敛为单一 marker；
    - `run_experiment.py` 回归验证：错误 `omega` 输入被拒绝，合法 `center_frequency/bandwidth` 输入可执行。
    - 复制最新 hard findings 到临时目录后执行归位，`候选方程解析` 已能自动对齐到 Round 20。
- Files created/modified:
  - `evomaster/skills/newtonbench/scripts/run_experiment.py` (updated)
  - `evomaster/skills/newtonbench/scripts/fit_pysr_candidates.py` (updated)
  - `playground/hamilton/core/exp.py` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter_20rounds.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter_2rounds.yaml` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt` (updated)
  - `playground/hamilton/workspace_newtonbench/task.md` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

## Session: 2026-03-13

### Phase 11.3: P0 目标对齐（m10 complex_system）
- **Status:** in_progress
- **Started:** 2026-03-13
- Actions taken:
  - 复盘 `m10_be_distribution/complex_system` 数据链路，确认实验脚本返回的是积分量 `total_power`，而评测对比的是 `discovered_law(omega, T)` 对应的 occupation number。
  - 复盘 `fit_pysr_candidates.py`，确认其默认按出现频率自动选择目标字段，因此此前实际在用 `omega,T -> total_power` 做 PySR 拟合，目标错位。
  - 在 `fit_pysr_candidates.py` 中新增 `m10 complex_system` 专用窄带 proxy：
    - 仅当 `bandwidth / center_frequency <= 0.05` 时，生成 `derived.occupation_proxy_narrowband`；
    - proxy 公式为 `total_power / (bandwidth * center_frequency^3)`；
    - 默认优先拟合该 proxy，并在样本不足时输出明确提示。
  - 在 `generate_task_prompt.py` 追加 Hamilton 专用工作提示，要求 `m10 complex_system` 优先采窄带样本并跨数量级覆盖 `omega/T`。
  - 在 Hamilton NewtonBench system/user prompt 与 `workspace_newtonbench/task.md` 中同步加入窄带采样策略说明。
  - 复核 `third_party/NewtonBench/modules/m10_be_distribution/laws.py`：
    - `easy/v0` 真式为 `1 / (exp(C * omega / T) + 1)`；
    - 当前 smoke 采样区间会把该真式压缩到接近 `0.5` 的小指数极限，因此“常数近似”属于局部可解释现象，不应直接解读为流程失效。
  - 据此继续修 prompt 污染：
    - `generate_task_prompt.py` 改为对 `vanilla/simple/complex` 三种 system 分别追加不同的 Hamilton note；
    - 明确禁止在 `vanilla/simple` 模式下引入 `bandwidth/filter/total_power` 叙述。
  - 新增 easy smoke 回归入口：
    - `playground/hamilton/tasks/newtonbench_single_m10_easy_smoke.json`
    - `configs/hamilton/newtonbench_single_m10_easy_smoke.yaml`
  - 完成本地静态/轻量自检：
    - `py_compile` 通过；
    - `generate_task_prompt.py` 已正确输出窄带 proxy 提示；
    - proxy 逻辑自检通过，只有窄带样本会被纳入目标构造。
- Files created/modified:
  - `evomaster/skills/newtonbench/scripts/fit_pysr_candidates.py` (updated)
  - `evomaster/skills/newtonbench/scripts/generate_task_prompt.py` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt` (updated)
  - `playground/hamilton/workspace_newtonbench/task.md` (updated)
  - `playground/hamilton/tasks/newtonbench_single_m10_easy_smoke.json` (created)
  - `configs/hamilton/newtonbench_single_m10_easy_smoke.yaml` (created)
  - 修复 `scripts/newtonbench/summarize_hamilton_run.py`：
    - `final_law` 改为只从真实 finish 消息提取，避免误命中 prompt 模板；
    - 新增 `task_completed_true/task_completed_false` 汇总；
    - 新增从日志提取最后一次 evaluation（`rmsle/exact_accuracy/symbolic`）能力。
  - 对 `runs/hamilton_20260309_220308` 与 `runs/hamilton_20260309_215738` 做回归汇总，确认新口径正确。
- Files created/modified:
  - `scripts/newtonbench/summarize_hamilton_run.py` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)
  - `task_plan.md` (updated)

## Additional Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 高风险质量护栏验证（m1 complex） | `run.py --agent hamilton --config ...newtonbench.yaml --task "module: m1_coulomb_force ... system: complex_system ..."` | 低质量结果应被护栏阻断为完成态 | `finish(task_completed=false)`；记录中出现 `rmsle_above_threshold` 违规 | ✅ |
| 汇总脚本纠偏回归（高风险 run） | `summarize_hamilton_run.py --run-dir runs/hamilton_20260309_220308` | 不应误报 `final_law/protocol_full_ok`，应回填评测指标 | `with_final_law=0`、`protocol_full_ok=0`、`avg_rmsle=20.6396` | ✅ |
| 汇总脚本纠偏回归（guard smoke run） | `summarize_hamilton_run.py --run-dir runs/hamilton_20260309_215738` | 保持原先成功样本统计正确 | `task_completed_true=2`、`protocol_full_ok=2`、`avg_exact_accuracy=1.0` | ✅ |

## Session: 2026-03-11

### Phase 11 预研：回归 Hamilton 原始 PySR 流程（分析）
- **Status:** in_progress
- **Started:** 2026-03-11
- Actions taken:
  - 对比阅读 Hamilton 基线配置与 NewtonBench 配置，确认 skill 组合差异（`pysr` 在 NewtonBench 主线缺席）。
  - 复核 NewtonBench skill 三脚本职责，确认当前链路“run_experiment/evaluate_submission 不负责搜索，仅执行/评测”。
  - 复核 `RoundExp` 的 `system_backfill` 与 findings 表格写入逻辑，确认其为 L2 回填兜底机制。
  - 输出“回归 PySR 主搜索器”的改造工作包（WP1~WP7）并同步到 `task_plan.md`。
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 11.1: P0 第一批代码改造（PySR-assisted）
- **Status:** in_progress
- **Started:** 2026-03-11
- Actions taken:
  - 新增 `fit_pysr_candidates.py`（newtonbench skill）：
    - 增量采样 + 缓存 + PySR 拟合 + top-k 候选输出。
  - NewtonBench 三套配置接入 `pysr`：
    - skills 增加 `pysr`；
    - symlinks 增加 `evomaster/skills/pysr`；
    - experiment 增加 `search_mode: pysr_assisted`。
  - 更新 NewtonBench 提示词和任务模板，要求每轮至少一次 PySR 候选搜索。
  - 按“轻量化，相信 LLM”原则重写 NewtonBench prompts，删除大量硬约束。
  - 在 `HamiltonPlayground` 注入 `runtime.search_mode` 提示，支持 `pysr_assisted/llm_direct` 双模式。
  - 放宽 NewtonBench 配置护栏：移除 symbolic/exact 强门槛，保留核心协议口径。
  - `fit_pysr_candidates.py` 增加模块默认算子配置与 `--health-check`。
  - 更新 `newtonbench/SKILL.md` 与 Hamilton README 的流程文档。
  - 验证结果：
    - `fit_pysr_candidates.py --help` 正常；
    - 脚本通过 `compileall`；
    - 端到端最小样例在当前沙箱受 Julia registry 网络限制，未能完成最终拟合。
- Files created/modified:
  - `evomaster/skills/newtonbench/scripts/fit_pysr_candidates.py` (created)
  - `evomaster/skills/newtonbench/SKILL.md` (updated)
  - `configs/hamilton/newtonbench.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter_2rounds.yaml` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt` (updated)
  - `playground/hamilton/workspace_newtonbench/task.md` (updated)
  - `playground/hamilton/README.md` (updated)
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Phase 11.2: P0 减法改造（轻量回归）
- **Status:** complete
- **Started:** 2026-03-11
- Actions taken:
  - 将 NewtonBench 三套配置切回轻量默认：`search_mode=llm_direct`，移除 `pysr` 默认 skill 依赖。
  - 协议护栏收敛为核心闭环（实验/评测/final_law），关闭默认签名与质量硬门槛。
  - 关闭 NewtonBench 默认 system backfill（`auto_backfill_l2: false`），避免系统重写 findings/plan 造成“框架过重”。
  - 简化上一轮反馈注入文本，删除“硬约束”措辞，仅保留失败摘要与回滚建议。
  - 重写 NewtonBench system/user prompt 与 workspace task 模板，保留 APPEND 标记写入约束与闭环必要项。
- Files created/modified:
  - `configs/hamilton/newtonbench.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter.yaml` (updated)
  - `configs/hamilton/newtonbench_single_hard_iter_2rounds.yaml` (updated)
  - `playground/hamilton/core/exp.py` (updated)
  - `playground/hamilton/core/playground.py` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_system.txt` (updated)
  - `playground/hamilton/prompts/hamilton_newtonbench_user.txt` (updated)
  - `playground/hamilton/workspace_newtonbench/task.md` (updated)
  - `evomaster/skills/evo-protocol/references/plan_template.md` (updated)
  - `playground/hamilton/README.md` (updated)
