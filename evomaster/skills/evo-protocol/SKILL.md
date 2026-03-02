---
name: evo-protocol
description: "Scientific iteration protocol for autonomous research agents. Provides plan structure, attention manipulation, failure tracking, and convergence analysis. Use get_info for methodology overview, references for templates, scripts for automation."
license: null
---

# Evo Protocol

A lightweight scientific iteration methodology for research agents. Core loop:

**Hypothesize -> Experiment -> Record -> Iterate**

## Quick Start

1. Run `scripts/init_plan.py` to generate a `plan.md` from your task description
2. Before each experiment: read `plan.md` to avoid repeating failed strategies
3. After each experiment: update `plan.md` with results and new hypotheses
4. Use `scripts/check_progress.py` to get a concise progress summary
5. Use `scripts/failure_report.py` to extract failure patterns

## Core Rules

### Attention Rule: "Read Before You Act"
- Read `plan.md` before designing each experiment
- Check `Failed Approaches` to avoid repetition
- Check `Confirmed Knowledge` to build on what works

### Recording Rule: "Failures Are Knowledge"
- Every failed experiment must be recorded in `Failed Approaches`
- Include: what you tried, what happened, why it failed
- This prevents strategy loops and accelerates convergence

### Mutation Rule: "Never Repeat"
- Each new strategy must differ meaningfully from all previous attempts
- Change at least one of: variables, operators, expression structure, PySR parameters
- If stuck, try a radically different approach (new features, different target, etc.)

## Resources

| Resource | Path | Purpose |
|----------|------|---------|
| Plan template | `references/plan_template.md` | Recommended plan.md structure |
| Full rules | `references/evo_rules.md` | Complete Evo Protocol methodology |
| Convergence guide | `references/convergence_guide.md` | When to stop iterating |
| Init plan | `scripts/init_plan.py` | Generate plan.md from task |
| Check progress | `scripts/check_progress.py` | Analyze iteration progress |
| Failure report | `scripts/failure_report.py` | Extract failure patterns |

## Usage Patterns

```python
# In workspace scripts:
# Generate initial plan
python scripts/init_plan.py --task "Discover governing equation from data" --output plan.md

# Check progress mid-experiment
python scripts/check_progress.py --plan plan.md --experiment experiment.json

# After several rounds, analyze failures
python scripts/failure_report.py --experiment experiment.json --plan plan.md
```
