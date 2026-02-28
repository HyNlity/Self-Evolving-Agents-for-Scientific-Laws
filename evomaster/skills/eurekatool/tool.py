"""eurekatool: stdlib-only helpers for analyzing Hamilton PySR results.

Design goals:
- No heavy dependencies (numpy/pandas/sympy) required.
- Safe evaluation of expressions (AST whitelist; no eval sandbox hacks).
- Stable, small API that can be extended iteratively.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


_ALLOWED_FUNCS: dict[str, Callable[..., float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "square": lambda x: x * x,
    "abs": abs,
    "pow": pow,
}

_ALLOWED_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def load_experiment(path: str = "experiment.json") -> dict[str, Any]:
    """Load experiment.json as a dict."""
    experiment_path = Path(path)
    if not experiment_path.exists():
        raise FileNotFoundError(f"experiment.json not found: {experiment_path}")
    try:
        return json.loads(experiment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {experiment_path}: {e}") from e


def list_rounds(experiment: dict[str, Any]) -> list[int]:
    """List available round numbers in experiment.json."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        return []
    out: list[int] = []
    for k in rounds.keys():
        try:
            out.append(int(k))
        except Exception:
            continue
    return sorted(out)


def get_round_record(experiment: dict[str, Any], round_n: int) -> dict[str, Any]:
    """Get one round record (pysr_config/results)."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        raise KeyError("experiment['rounds'] missing or not a dict")
    key = str(round_n)
    if key not in rounds:
        raise KeyError(f"Round {round_n} not found. Available: {sorted(rounds.keys())}")
    record = rounds[key]
    if not isinstance(record, dict):
        raise ValueError(f"Round {round_n} record is not an object")
    return record


def get_round_results(round_record: dict[str, Any]) -> list[dict[str, Any]]:
    """Get the results list for a round."""
    results = round_record.get("results", [])
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    return []


def pick_best_equation(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick BestEq (lowest reported MSE; tie-break by complexity/rank)."""
    if not results:
        return None

    def _as_float(v: Any) -> float | None:
        try:
            x = float(v)
        except Exception:
            return None
        if not math.isfinite(x):
            return None
        return x

    def _as_int(v: Any) -> int | None:
        try:
            return int(v)
        except Exception:
            return None

    scored: list[tuple[tuple[float, int, int], dict[str, Any]]] = []
    fallback: list[tuple[tuple[int, int], dict[str, Any]]] = []

    for r in results:
        eq = r.get("equation")
        if not isinstance(eq, str) or not eq.strip():
            continue

        mse = _as_float(r.get("mse"))
        complexity = _as_int(r.get("complexity"))
        rank = _as_int(r.get("rank"))

        if mse is not None:
            scored.append(((mse, complexity if complexity is not None else 10**9, rank if rank is not None else 10**9), r))
        else:
            fallback.append(((rank if rank is not None else 10**9, complexity if complexity is not None else 10**9), r))

    if scored:
        scored.sort(key=lambda x: x[0])
        return scored[0][1]
    if fallback:
        fallback.sort(key=lambda x: x[0])
        return fallback[0][1]
    return None


def format_alt_eqs(
    results: list[dict[str, Any]],
    best_eq: str | None = None,
    max_n: int = 5,
) -> str:
    """Format 2-5 alternative equations into one line."""
    alts: list[str] = []
    for r in results:
        eq = r.get("equation")
        if not isinstance(eq, str):
            continue
        eq = eq.strip()
        if not eq:
            continue
        if best_eq is not None and eq == best_eq:
            continue
        alts.append(eq)
        if len(alts) >= max_n:
            break
    return " ; ".join(alts)


def read_csv_rows(path: str = "data.csv", columns: list[str] | None = None) -> list[dict[str, float]]:
    """Read numeric CSV rows as dicts of floats; non-numeric values become NaN."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return rows

        for raw_row in reader:
            row: dict[str, float] = {}
            if columns is None:
                items = raw_row.items()
            else:
                items = ((c, raw_row.get(c, "")) for c in columns)

            for k, v in items:
                if k is None:
                    continue
                s = (v or "").strip()
                if s == "":
                    row[k] = float("nan")
                    continue
                try:
                    row[k] = float(s)
                except Exception:
                    row[k] = float("nan")
            rows.append(row)
    return rows


def build_row_env(row: dict[str, float], variable_names: list[str]) -> dict[str, float]:
    """Build a safe eval environment for one row (x0/x1... + column names)."""
    env: dict[str, float] = {}

    # Prefer explicit variable order from expression_spec.variable_names.
    for i, col in enumerate(variable_names):
        v = row.get(col, float("nan"))
        env[f"x{i}"] = v
        if _is_identifier(col):
            env[col] = v

    # Also allow directly using columns named like x0/x1 if present in CSV.
    for k, v in row.items():
        if k and _is_identifier(k) and k.startswith("x") and k[1:].isdigit():
            env[k] = v

    return env


def safe_eval_expr(expr: str, env: dict[str, float]) -> float:
    """Safely evaluate an expression on one row (AST whitelist)."""
    expr = _normalize_expr(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return float("nan")
    try:
        return float(_eval_node(tree.body, env))
    except Exception:
        return float("nan")


def eval_equation_on_rows(expr: str, rows: list[dict[str, float]], variable_names: list[str]) -> list[float]:
    """Evaluate one expression on many rows."""
    out: list[float] = []
    for row in rows:
        env = build_row_env(row, variable_names)
        out.append(safe_eval_expr(expr, env))
    return out


def compute_mse(y_true: list[float], y_pred: list[float]) -> float:
    """Mean squared error (skips NaN/inf pairs)."""
    n = 0
    acc = 0.0
    for yt, yp in zip(y_true, y_pred):
        if not (math.isfinite(yt) and math.isfinite(yp)):
            continue
        diff = yt - yp
        acc += diff * diff
        n += 1
    return (acc / n) if n > 0 else float("nan")


def residual_summary(
    y_true: list[float],
    y_pred: list[float],
    top_outliers: int = 5,
) -> dict[str, Any]:
    """Residual stats + outlier indices."""
    residuals: list[float] = []
    abs_residuals: list[float] = []
    outlier_candidates: list[tuple[float, int]] = []

    used = 0
    skipped = 0
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if not (math.isfinite(yt) and math.isfinite(yp)):
            skipped += 1
            continue
        r = yt - yp
        used += 1
        residuals.append(r)
        ar = abs(r)
        abs_residuals.append(ar)
        outlier_candidates.append((ar, i))

    if used == 0:
        return {
            "n_used": 0,
            "n_skipped": skipped,
            "mse": float("nan"),
        }

    mse = compute_mse(y_true, y_pred)
    mean_r = statistics.fmean(residuals)
    std_r = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0

    outlier_candidates.sort(reverse=True)
    outliers = [{"row_index": idx, "abs_residual": ar} for ar, idx in outlier_candidates[: max(0, top_outliers)]]

    return {
        "n_used": used,
        "n_skipped": skipped,
        "mse": mse,
        "residual_mean": mean_r,
        "residual_std": std_r,
        "abs_residual_p50": _percentile(abs_residuals, 50),
        "abs_residual_p95": _percentile(abs_residuals, 95),
        "abs_residual_max": max(abs_residuals) if abs_residuals else float("nan"),
        "outliers": outliers,
    }


def _normalize_expr(expr: str) -> str:
    s = expr.strip()
    # Common prefix patterns.
    for prefix in ("y =", "y="):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    # PySR sometimes prints '^' for power; Python needs '**'.
    s = s.replace("^", "**")
    return s


def _is_identifier(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(ch.isalnum() or ch == "_" for ch in name)


def _eval_node(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant):  # py3.8+
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Non-numeric constant")

    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise KeyError(f"Unknown name: {node.id}")

    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +v
        if isinstance(node.op, ast.USub):
            return -v
        raise ValueError("Unsupported unary op")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary op")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed")
        fn = _ALLOWED_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        if node.keywords:
            raise ValueError("Keywords not allowed")
        args = [_eval_node(a, env) for a in node.args]
        return float(fn(*args))

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))

    xs = sorted(values)
    # Linear interpolation between closest ranks.
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[int(f)] * (c - k)
    d1 = xs[int(c)] * (k - f)
    return float(d0 + d1)
