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

4. `fit_pysr_candidates.py`
- PySR-assisted path for NewtonBench:
  - optional fresh sampling via `run_experiment_for_module(...)`
  - cache samples into workspace-local jsonl
  - fit PySR and return top-k symbolic candidates + `discovered_law` templates
  - built-in module-level default operator profile (can still override via CLI)

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

# 4) PySR-assisted candidate search (collect + fit)
python fit_pysr_candidates.py \
  --module m0_gravity \
  --system vanilla_equation \
  --difficulty easy \
  --law-version v0 \
  --inputs-json '[{"mass1": 1, "mass2": 2, "distance": 3}, {"mass1": 2, "mass2": 2, "distance": 4}]' \
  --niterations 80 \
  --top-k 3

# 5) Health check (PySR + Julia)
python fit_pysr_candidates.py --health-check
```
