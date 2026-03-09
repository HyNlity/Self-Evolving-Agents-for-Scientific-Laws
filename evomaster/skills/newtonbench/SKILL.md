---
name: newtonbench
description: "NewtonBench bridge skill for Hamilton. Provides scripts to generate task prompts, run interactive experiments, and evaluate discovered laws (SA/RMSLE) against NewtonBench modules."
license: null
---

# NewtonBench Skill

This skill bridges Hamilton's single-agent workflow with NewtonBench's interactive benchmark protocol.

## What This Skill Provides

1. `generate_task_prompt.py`
- Build a domain/system-specific NewtonBench task prompt and metadata.

2. `run_experiment.py`
- Execute NewtonBench `run_experiment_for_module(...)` for one or many input sets.

3. `evaluate_submission.py`
- Evaluate a submitted `discovered_law` using module-native `evaluate_law(...)`.

## Required Environment

- Set `NEWTONBENCH_ROOT` to a NewtonBench code checkout (folder containing `modules/`).
- Install NewtonBench Python dependencies in the runtime environment.

## Typical Usage

```bash
# 1) Generate prompt package
python generate_task_prompt.py --module m0_gravity --system vanilla_equation --difficulty easy --law-version v0 --noise 0

# 2) Run interactive experiment(s)
python run_experiment.py --module m0_gravity --system vanilla_equation --difficulty easy --law-version v0 --noise 0 --inputs-json '[{"mass1": 1, "mass2": 2, "distance": 3}]' --tag

# 3) Evaluate final submitted law
python evaluate_submission.py --module m0_gravity --difficulty easy --law-version v0 --judge-model gpt41 --law-file submitted_law.py
```

