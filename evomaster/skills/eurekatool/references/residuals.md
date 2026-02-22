# Residuals (stdlib-only)

## What residual_summary computes

`tool.residual_summary(y_true, y_pred)` returns a small dict:
- `n_used` / `n_skipped`
- `mse`
- `residual_mean` / `residual_std`
- `abs_residual_p50` / `abs_residual_p95` / `abs_residual_max`
- `outliers`: top-N rows by absolute residual (`row_index`, `abs_residual`)

## Limitations

- No numpy/pandas/sympy required, but it's slower on very large datasets.
- Expression evaluation is AST-whitelisted. If an expression cannot be parsed/evaluated, predictions become NaN and are skipped.

