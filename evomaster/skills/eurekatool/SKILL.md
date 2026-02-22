---
name: eurekatool
description: Minimal, reusable Python helpers for Eureka to analyze Hamilton PySR results (experiment.json) and write stable insight.md summaries. Skill is the layered manual; code reuse is via Python imports.
license: Proprietary
---

# EurekaTool

This is a tiny toolbox + layered manual for the Hamilton Eureka agent.

## What Tools Exist (High-Level)

- Toolbox code: `tool.py` (experiment parsing, BestEq ranking, safe expression eval, residual summary)
- Script: `scripts/round_report.py` (prints an `insight.md` block for one round)
- Full catalog (1 line per function): `usage.txt`

## How to Use This Skill (Progressive Disclosure)

1. **See what exists (catalog)**: load `usage.txt` via `use_skill(..., action="get_reference")`.
2. **Read details only when needed**: load a single topic page via `get_reference`:
   - `io.md` (files/paths + import tips)
   - `ranking.md` (BestEq/MSE/AltEqs rules)
   - `residuals.md` (residual summary + outliers)
   - `promotion.md` (when/how to promote new analysis into tools)
3. **Reuse code via Python import (no use_skill required)**:
   - Import `skills.eurekatool.tool` from scripts running in the Hamilton workspace.

## Promotion Rule (Keep It Small)

Prefer reuse first. If you write analysis that is likely to be reused in later rounds:
- Add a new function (or extend an existing one) in `tool.py`.
- Add a 1-line entry in `usage.txt` for discoverability.
- If it needs explanation, add a short note to the most relevant reference page (do not create long docs).

If it is one-off exploration, keep it in `history/roundN/scripts/` instead.
