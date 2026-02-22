#!/usr/bin/env python3
"""Print a minimal insight.md block for a given round.

This script is optional. Code reuse is via `skills.eurekatool.tool`.
"""

from __future__ import annotations

import argparse
import math

try:
    from skills.eurekatool import tool  # type: ignore[import-not-found]
except ImportError:
    import sys
    from pathlib import Path

    # Fallback: allow running from repo root (without workspace symlink).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tool  # type: ignore[no-redef]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--experiment", default="experiment.json")
    parser.add_argument("--data", default="data.csv")
    parser.add_argument("--recompute", action="store_true", help="Recompute MSE and residual summary from data.csv")
    parser.add_argument("--top_outliers", type=int, default=5)
    args = parser.parse_args()

    exp = tool.load_experiment(args.experiment)
    record = tool.get_round_record(exp, args.round)
    results = tool.get_round_results(record)

    best = tool.pick_best_equation(results)
    best_eq = (best or {}).get("equation") if isinstance(best, dict) else None
    best_eq = best_eq.strip() if isinstance(best_eq, str) else "none"

    reported_mse = None
    if isinstance(best, dict):
        try:
            reported_mse = float(best.get("mse"))
        except Exception:
            reported_mse = None
        if reported_mse is not None and not math.isfinite(reported_mse):
            reported_mse = None

    alt_eqs = tool.format_alt_eqs(results, best_eq=best_eq, max_n=5)

    notes_parts: list[str] = []
    mse_value = reported_mse

    if args.recompute and best_eq != "none":
        pysr_cfg = record.get("pysr_config", {}) if isinstance(record, dict) else {}
        y_col = pysr_cfg.get("y") if isinstance(pysr_cfg, dict) else None
        if not isinstance(y_col, str) or not y_col:
            y_col = "y"

        expr_spec = pysr_cfg.get("expression_spec", {}) if isinstance(pysr_cfg, dict) else {}
        variable_names = []
        if isinstance(expr_spec, dict):
            variable_names = expr_spec.get("variable_names", [])
        if not isinstance(variable_names, list) or not all(isinstance(x, str) for x in variable_names):
            variable_names = []

        columns = list(dict.fromkeys(variable_names + [y_col]))
        rows = tool.read_csv_rows(args.data, columns=columns)
        y_true = [r.get(y_col, float("nan")) for r in rows]
        y_pred = tool.eval_equation_on_rows(best_eq, rows, variable_names)
        computed_mse = tool.compute_mse(y_true, y_pred)
        summary = tool.residual_summary(y_true, y_pred, top_outliers=args.top_outliers)

        if math.isfinite(computed_mse):
            mse_value = computed_mse
            notes_parts.append(
                "Residual: "
                f"n_used={summary.get('n_used')} "
                f"mean={summary.get('residual_mean'):.6g} "
                f"std={summary.get('residual_std'):.6g} "
                f"p95_abs={summary.get('abs_residual_p95'):.6g}"
            )
        else:
            notes_parts.append("Residual: recompute_failed (expression eval returned NaN/inf)")

        if reported_mse is not None and math.isfinite(computed_mse):
            ratio = computed_mse / reported_mse if reported_mse != 0 else float("inf")
            notes_parts.append(f"MSE_source: recomputed (reported={reported_mse:.6g}, ratio={ratio:.3g})")

    if not notes_parts:
        notes_parts.append("Notes: fill reliability judgement (residuals/overfit/meaning/stability)")

    notes = " | ".join(notes_parts)

    print(f"## Round {args.round}")
    print("")
    print(f"BestEq: {best_eq}")
    print(f"MSE: {mse_value if mse_value is not None else 'unknown'}")
    print(f"AltEqs: {alt_eqs if alt_eqs else 'none'}")
    print(f"Notes: {notes}")
    print("")
    print("Recommendations:")
    print("- (fill next-round changes in variables/operators/constraints/template/budget)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
