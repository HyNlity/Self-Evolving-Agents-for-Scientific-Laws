# Hamilton Agent 开发计划

## Goal
构建 Hamilton 符号回归 Agent：单 Agent 闭环迭代（规划→实验→验证→记录），基于 EvoMaster v0.0.2 框架，在 VIV benchmark 上发现 ODE 方程族。

## Current Phase
Phase 5

## Phases

### Phase 1: 上游合并 (EvoMaster v0.0.2) — `complete`
- [x] rsync 上游 evomaster/ 到本仓库
- [x] EnvConfig 字段加默认值（解决 pydantic ValidationError）
- [x] 删除 `create_default_registry()`，统一用 `create_registry()`
- [x] config.yaml 迁移到 v0.0.2 格式（`agents:` 复数 + per-agent tools/skills）
- [x] v0.0.2 特性验证 26/26 ✅

### Phase 2: Bug 修复 — `complete`
- [x] finish 工具描述改为 round-scoped（不再暗示"任务完全完成才能调"）
- [x] `_handle_no_tool_call` 改为诊断提示（不再说"请继续工作"导致死循环）
- [x] task_completed / satisfied 统一：去掉 signal block，直接从 finish 参数提取

### Phase 3: HCC 记忆体系 — `complete`
- [x] L2 post-check（mtime 检测 Agent 是否更新了 findings.md / plan.md）
- [x] lib/ 持久脚本复用目录
- [x] L2 写入规范（双写分离、精简记录，借鉴 ml-master-skills HCC 模式）
- [x] 删除 EUREKA_SIGNAL_BEGIN/END 及相关解析代码和常量

### Phase 4: Prompt 优化 — `complete`
- [x] system prompt 重写（139行→67行，流程文档→研究方法论）
- [x] user prompt 加 {description} 占位符，task.md 通过 --task 直接注入
- [x] PySR skill 加调参原则（搜索预算、降采样、收敛判断、小时量级时间）
- [x] 去掉 task.md 引用（Agent 不需要额外读文件）
- [x] 核心原则：假设驱动、大胆探索、从简到复杂、复用代码、充分搜索、精简记录

### Phase 5: 验证运行 — `pending`
- [ ] `python run.py --agent hamilton --task playground/hamilton/workspace/task.md`
- [ ] 验证 Agent 能正常调 finish 并结束轮次
- [ ] 验证 per-agent skill 过滤生效（只加载 pysr + evo-protocol）
- [ ] 验证 lib/ 复用
- [ ] 分析日志，迭代优化 prompt

## Key Files

| 文件 | 作用 |
|------|------|
| `configs/hamilton/config.yaml` | 配置（LLM、agent、session、max_rounds=3） |
| `playground/hamilton/core/playground.py` | 多轮编排 + workspace 初始化 |
| `playground/hamilton/core/exp.py` | 单轮执行 + task_completed 解析 |
| `playground/hamilton/prompts/hamilton_system.txt` | 系统提示词（67行，研究方法论风格） |
| `playground/hamilton/prompts/hamilton_user.txt` | 用户提示词（轮次 + 任务描述注入） |
| `playground/hamilton/workspace/task.md` | VIV benchmark 任务描述 |
| `evomaster/skills/pysr/references/api_reference.md` | PySR API + 调参原则 |
| `evomaster/agent/tools/builtin/finish.py` | finish 工具（round-scoped 描述） |
| `evomaster/agent/agent.py` | Agent 核心（含 _handle_no_tool_call 修复） |
| `evomaster/config.py` | 配置系统（EnvConfig 默认值修复） |

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| 去掉 signal block，统一用 task_completed | 消除 satisfied/task_completed 语义重叠 |
| system prompt 从 checklist 改为原则式 | Agent 应像研究员思考，不是执行流程 |
| PySR 调参写原则不写数字 | 避免过度约束，让 Agent 根据 EDA 结果判断 |
| task.md 通过 --task 注入而非 Agent 自己读 | 节省一步 tool call，任务直接在 user prompt 中 |
| L2 写入借鉴 ml-master-skills HCC | 双写分离 + 精简记录防止记忆膨胀 |
| finish 语义改为"结束本轮探索" | 鼓励大胆尝试，不要求一轮解决所有问题 |
