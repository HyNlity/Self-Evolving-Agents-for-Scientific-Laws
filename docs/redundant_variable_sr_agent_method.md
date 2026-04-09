# Redundancy-Aware Symbolic Regression Agent

工作名：`Hamilton-RVSR`

目标：在冗余变量条件下，把符号回归从“单纯误差优化”改写为“支持集识别 + 方程发现 + 假设证伪”的联合问题。

## 1. Problem Definition

给定高维观测变量集合 `X = {x_1, ..., x_d}`，其中包含：
- 真正参与机制的核心变量
- 与核心变量强相关的代理变量
- 派生变量
- 无关冗余变量

目标不是只发现一个低误差方程 `y = f(X)`，而是发现：

1. 最小且稳定的支持集 `S`
2. 支持集上的方程 `f(S)`
3. 解释为什么 `X \\ S` 是冗余、代理或伪相关

这使问题从普通 SR 变成：

`joint support-set identification + equation discovery`

## 2. Why Current Hamilton Is Not Enough

当前 `main` 上的 Hamilton 已经有三项可复用基础：
- 单 Agent 多轮迭代
- HCC 文件记忆
- 工具/技能接口

但仍有四个缺口：

1. 变量支持集没有显式状态。
2. 记忆是文本性的，缺少 machine-readable 决策状态。
3. 每轮主要围绕拟合改进，没有把证伪实验作为核心动作。
4. 当前模板与日志把 `MSE` 放在中心位置。

## 3. Method Overview

### RVSR-ANCHOR-METHOD-OVERVIEW

方法由四个核心模块构成：

1. `Variable Role Memory`
2. `Hypothesis Archive`
3. `Falsification Planner`
4. `Tool Router`

它们共同服务于一个目标：

**优先恢复稳定支持集，再恢复支持集上的方程结构。**

## 4. Core Modules

### RVSR-ANCHOR-VARIABLE-MEMORY

维护 `variable_memory.json`，按变量记录角色与证据。

建议字段：

```json
{
  "variable_name": {
    "role": "core | candidate_core | redundant | proxy | spurious | unknown",
    "evidence": {
      "leave_one_out_delta": 0.0,
      "swap_sensitivity": 0.0,
      "ood_stability": 0.0,
      "co_usage_count": 0,
      "last_updated_round": 0
    },
    "notes": []
  }
}
```

用途：
- 让 Agent 不再把变量只当作输入列，而是当作待判定角色的对象。
- 跨轮保留“哪些变量被怀疑是冗余/代理”的证据。

### RVSR-ANCHOR-HYPOTHESIS-ARCHIVE

维护 `hypothesis_archive.jsonl`，每条候选都不是单纯公式字符串，而是结构化假设。

建议字段：

```json
{
  "round": 1,
  "id": "h_0001",
  "equation": "y = a*x1 + b*x2^2",
  "support_set": ["x1", "x2"],
  "tool_path": "pysr",
  "fit_metrics": {},
  "structure_metrics": {},
  "generalization_metrics": {},
  "falsification_status": "untested | survived | rejected",
  "rejection_reason": null
}
```

用途：
- 让每轮真正积累“被尝试过的科学假设”。
- 后续可直接做 archive ablation，而不是只看文字日志。

### RVSR-ANCHOR-FALSIFICATION

维护 `falsification_log.jsonl`，每轮都要有“否掉谁”的动作。

第一版证伪动作建议只做三类：
- `leave_one_out`
- `variable_swap`
- `ood_slice_check`

核心思想：
- 不是继续平均优化全局误差
- 而是主动找最能区分两个支持集/两个候选结构的实验

输出不是“分更低了”，而是：
- 哪个变量被排除
- 哪个候选在什么情形下失效
- 哪条机制解释被否掉

### RVSR-ANCHOR-TOOL-ROUTER

维护 `routing_state.json`，明确记录为什么本轮选择某种工具。

第一版采用规则路由，不做学习型路由：

- 若变量数多且冗余明显：先做 redundancy profiling
- 若支持集较稳定但结构未知：优先 PySR
- 若怀疑存在解析变换：优先 log / ratio / separability 分析
- 若 top hypotheses 难区分：优先 falsification

这样做的目的不是复杂化，而是：
- 避免所有题都直接调用 PySR
- 避免所有轮次都追着 MSE 优化

## 5. Ranking: Reduce MSE Dominance

### RVSR-ANCHOR-SCORING

当前最优选择不再按最低 MSE 决定。

采用四层证据：

1. `feasibility`
   - 可运行
   - 数值稳定
   - 满足基本约束

2. `support_stability`
   - 去掉可疑变量后是否仍成立
   - 变量互换后是否崩溃
   - OOD 下支持集是否变化

3. `structure_quality`
   - 表达式复杂度
   - 结构可解释性
   - 是否满足先验约束

4. `fit_quality`
   - `NMSE` / `RMSLE` / relative error
   - 仅作为第四层 tie-breaker，而不是主目标

建议第一版用“分层筛选”而不是简单加权和：
- 先过滤不可行候选
- 再按支持集稳定性排序
- 再看结构质量
- 最后才看拟合指标

## 6. Round Loop

### RVSR-ANCHOR-ROUND-LOOP

每轮建议改为：

1. 读取 `plan.md`、`findings.md` 和 machine-readable state
2. 更新变量角色记忆
3. 由 Tool Router 决定本轮工具路径
4. 生成多个带支持集的候选假设
5. 运行基础拟合与结构检查
6. 选择最值得证伪的 1 到 2 个候选对
7. 运行证伪实验
8. 更新 `hypothesis_archive.jsonl`
9. 将经验证的结论提升到 `findings.md` / `plan.md`
10. 调用 `finish`

这里的关键变化是：
- 本轮输出不再只是“一个最优公式”
- 而是“支持集状态 + 候选档案 + 被否掉的假设”

## 7. Why This Fits Redundant Variables

冗余变量问题的本质不是“找到一个低误差表达式”，而是：

- 哪些变量是真正必需的
- 哪些变量只是代理
- 哪些变量在 train 上能替代、但在 OOD 下不稳定

如果方法只盯 `MSE`，就会奖励：
- 利用 proxy 变量的伪解
- 高相关冗余变量拼出来的经验式
- 对 train 有利但不稳定的支持集

因此，支持集稳定性必须成为主优化对象之一。

## 8. Method Difference vs AlphaEvolve / OpenEvolve

### RVSR-ANCHOR-COMPARISON

`AlphaEvolve / OpenEvolve`
- 搜索对象：程序
- 主记忆：program archive / population history
- 主信号：自动 evaluator 的标量分数
- 优势：广搜索、程序改写、并行演化

`Hamilton-RVSR`
- 搜索对象：`(support set, equation, mechanism explanation)`
- 主记忆：变量角色记忆 + 假设档案 + 证伪记录
- 主信号：support stability + structure + generalization + fit
- 优势：显式处理冗余变量、支持集恢复、可审计科学假设

一句话总结：

**前者优化更好的程序，后者优化更可信的科学假设。**

## 9. Mapping to Current Codebase

### RVSR-ANCHOR-CODE-MAP

建议最小改动如下：

1. [`playground/hamilton/core/playground.py`](/home/zychen/SRAgent/Self-Evolving-Agents-for-Scientific-Laws/playground/hamilton/core/playground.py)
   - 初始化新的状态文件
   - 改 `findings.md` / `plan.md` 模板，不再默认以 MSE 为中心

2. [`playground/hamilton/core/exp.py`](/home/zychen/SRAgent/Self-Evolving-Agents-for-Scientific-Laws/playground/hamilton/core/exp.py)
   - 每轮初始化 `trace.md` 时加入 support/falsification/router 字段
   - 轮后检查不只看 `plan.md` / `findings.md`，也看状态文件是否更新

3. [`playground/hamilton/prompts/hamilton_system.txt`](/home/zychen/SRAgent/Self-Evolving-Agents-for-Scientific-Laws/playground/hamilton/prompts/hamilton_system.txt)
   - 从“PySR 主力”改成“先判断变量支持集，再决定工具路径”
   - 显式要求证伪与变量角色更新

4. 新增脚本目录
   - `playground/hamilton/workspace/tools/redundancy_profile.py`
   - `playground/hamilton/workspace/tools/falsify_hypotheses.py`
   - `playground/hamilton/workspace/tools/rank_hypotheses.py`

5. 可选：新增 skill
   - `evomaster/skills/redundancy-sr/`
   - 作为 PySR 之外的冗余变量分析参考

## 10. First Implementation Slice

### RVSR-ANCHOR-FIRST-SLICE

第一刀不要同时改所有东西。建议顺序：

1. 改 workspace seed 和模板
2. 新增 4 个状态文件
3. 改 system prompt
4. 加 1 个最小 `redundancy_profile.py`
5. 先跑 1 个 toy 冗余变量任务

只要第一版能做到下面 3 点，就算成功：
- 同一轮输出多个带支持集的候选
- `findings.md` 明确记录哪个变量被判定为冗余
- 当前最优不是单纯由最低 MSE 决定

## 11. Evaluation Plan

### RVSR-ANCHOR-EVAL

至少做四类指标：

1. `support recovery`
2. `structure recovery`
3. `OOD generalization`
4. `NMSE / relative error`

至少做四组 ablation：

1. `MSE-only`
2. `+ memory`
3. `+ falsification`
4. `+ tool router`

并与以下方法对比：
- PySR baseline
- 单轮 LLM baseline
- OpenEvolve / AlphaEvolve-style program search baseline
- 当前 Hamilton baseline

## 12. Practical Positioning

这条路线的论文卖点不应写成“又一个会搜索公式的 Agent”，而应写成：

**一个面向冗余变量符号回归的、记忆增强且证伪驱动的支持集恢复方法。**

这才是与当前文献最清楚的区分点。
