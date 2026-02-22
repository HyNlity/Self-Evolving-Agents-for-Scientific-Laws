# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvoMaster is a framework for building autonomous scientific research agents. It's the engine behind the SciMaster family of agents (ML-Master, X-Master, PhysMaster).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Or use uv (faster)
uv sync

# Run a basic task
python run.py --agent minimal --task "Your task description"

# Run with custom config
python run.py --agent minimal --config configs/minimal/config.yaml --task "Your task"

# Run from task file (.txt or .md)
python run.py --agent minimal --task task.txt

# Interactive mode
python run.py --agent minimal --interactive

# Batch tasks (sequential)
python run.py --agent minimal --task-file tasks.json

# Batch tasks (parallel)
python run.py --agent minimal --task-file tasks.json --parallel

# Specify run directory
python run.py --agent minimal --task "task" --run-dir runs/my_experiment
```

Available agents: `minimal`, `minimal_kaggle`, `minimal_multi_agent`, `minimal_skill_task`, `x_master`, `mat_master`

## Architecture

EvoMaster uses a three-layer architecture:

```
Playground → Exp → Agent
```

- **Playground** (`evomaster/core/playground.py`): Workflow orchestration, configuration, multi-agent coordination, MCP connections
- **Exp** (`evomaster/core/exp.py`): Single experiment execution, creates TaskInstance, runs Agent, collects trajectory
- **Agent** (`evomaster/agent/agent.py`): LLM + Tools + Memory, executes tool calls, manages conversation history

## Key Directories

- `evomaster/agent/` - Agent components (Agent, Session, Tools, Context)
- `evomaster/core/` - BaseExp, BasePlayground, Registry
- `evomaster/env/` - Environment management (Local, Docker, K8s)
- `evomaster/skills/` - Skill system (RAG, MCP-builder, PDF, Calculation)
- `evomaster/utils/llm.py` - LLM abstraction (OpenAI, Anthropic, Google)
- `playground/` - Agent implementations (minimal, x_master, etc.)
- `configs/` - YAML configuration files
- `docs/` - API documentation

## Configuration

- Configuration files are YAML in `configs/{agent}/`
- Use `.env.template` to create `.env` with API keys
- Environment variables in config are substituted at runtime (e.g., `${OPENAI_API_KEY}`)

## Playground Registration

Playgrounds are registered via the `@register_playground` decorator in `playground/{name}/core/playground.py`. The `run.py` auto-imports all playground modules to trigger registration.

## Testing

Tests are in `evomaster/agent/test_agent_context.py`. Run directly with Python.

## Hamilton Agent (实验性)

符号回归Agent，位于 `playground/hamilton/`

### 架构
- **RoundExp**: 单轮执行单元
- **Playground**: 循环编排，多次调用RoundExp
- **AnalysisAgent**: 负责变量分析、筛选、PySR调用

### 目录结构
```
playground/hamilton/
├── core/
│   ├── playground.py
│   └── exp.py
├── prompts/
│   ├── analysis_system.txt
│   └── analysis_user.txt
├── workspace/           # Agent自行维护
│   ├── data.csv
│   ├── history.md
│   └── round{N}/
│       ├── scripts/
│       ├── results/
│       └── insight.md
└── TODO.md
```

### 开发规范

**重要**：每个playground的TODO维护在该目录下：
- Hamilton: `playground/hamilton/TODO.md`

在开发过程中，必须时刻更新对应playground的TODO.md文件，记录进度和待办事项。
