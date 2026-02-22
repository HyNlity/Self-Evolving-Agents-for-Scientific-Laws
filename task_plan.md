# Task Plan: EvoMaster + Hamilton 符号回归 Agent 代码审查
<!--
  目标：系统性审查当前仓库中 EvoMaster 框架实现，以及 playground/hamilton（符号回归 Hamilton agent）
  的整体设计、实现逻辑、可维护性、复用框架思想程度、提示词质量与潜在 bug。
-->

## Goal
对 `/opt/EvoMaster` 当前代码进行端到端代码审查（重点：`playground/hamilton`），输出可落地的缺陷清单、
风险点、重构/优化建议（含必要补丁），并评估 Hamilton agent 是否正确复用/对齐 EvoMaster 框架思想。

## Current Phase
Phase 2

## Phases

### Phase 1: 范围确认 & 仓库摸底
- [x] 确认 Hamilton agent 入口与运行方式
- [x] 梳理 EvoMaster 框架核心模块与扩展点（playground/skills/utils）
- [x] 记录当前工作区状态（git status/diff）及可能影响审查的改动
- [x] 将关键发现写入 findings.md
- **Status:** complete

### Phase 2: 架构与数据流审查（EvoMaster Core）
- [ ] 读核心代码（core / llm / playground 基类）
- [ ] 画出（文字版）数据流：配置 → 环境 → LLM → 工具 → 产出
- [ ] 检查抽象边界、可复用性、错误处理与日志
- **Status:** in_progress

### Phase 3: Hamilton Agent 代码审查（符号回归任务）
- [ ] 逐文件检查：核心逻辑、状态机/回合机制、工具封装、I/O
- [ ] 检查 bug：路径/文件、并发/随机性、异常处理、边界条件
- [ ] 检查工程质量：重复代码、命名、职责划分、可测试性
- **Status:** pending

### Phase 4: Prompt/Tooling 审查
- [ ] 审查 prompts：角色分工、约束、输出格式、可控性、注入面
- [ ] 审查 tools：接口一致性、失败策略、可复用性、与框架对齐
- **Status:** pending

### Phase 5: 验证与交付
- [ ] 运行最小化静态检查/自测（可行范围内）
- [ ] 汇总 findings.md：问题清单（严重度/影响/修复建议）
- [ ] 如需要，提交最小补丁以修复明确 bug 或显著不优雅点
- **Status:** pending

## Key Questions
1. Hamilton agent 的“回合/搜索/评估”循环是否正确、可复现、可诊断？
2. EvoMaster 核心抽象（Playground/LLM/Tool/Config）是否被 Hamilton agent 正确复用？
3. prompts 是否具备：明确目标、可验证输出、失败回退、工具使用边界与注入防护？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 采用文件化审查产出（findings/progress） | 方便长任务留存与复盘，避免上下文丢失 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
|       | 1       |            |

## Notes
- 审查以“当前工作区代码”为准（包含未提交改动）；将明确标注“疑似未完成/大幅改动”区域的风险。
