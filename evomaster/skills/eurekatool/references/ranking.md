# Ranking (BestEq / MSE / AltEqs)

## What gets written to insight.md

Each round, Eureka writes a minimal block:
- `BestEq`: one expression string
- `MSE`: one number (reported or recomputed)
- `AltEqs`: 2-5 candidate expressions, separated by ` ; `
- `Notes`: natural language reliability notes

## Default selection rule (tool.pick_best_equation)

Pick `BestEq` by:
1. Lowest **reported** MSE from `experiment.json`
2. Tie-break: lower complexity
3. Tie-break: lower rank

If all reported MSE are missing, fall back to rank then complexity.

## Reported vs recomputed MSE

- Reported MSE comes directly from PySR output (stored in `experiment.json`).
- Recomputed MSE comes from evaluating the expression against `data.csv` using stdlib-only safe evaluation.

If they disagree materially, keep the number you trust more and explain in `Notes`.

