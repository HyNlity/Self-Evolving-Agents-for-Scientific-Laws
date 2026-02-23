#!/usr/bin/env python3
"""直接 PySR 基线脚本（x1~x10 全量输入）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a direct PySR baseline on x1-x10.")
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to train csv (must contain x1..x10 and y).",
    )
    parser.add_argument(
        "--ood-csv",
        required=True,
        help="Path to OOD csv (must contain x1..x10 and y).",
    )
    parser.add_argument(
        "--y-col",
        default="y",
        help="Target column name (default: y).",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=10,
        help="Number of top equations to print/report.",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        default=200,
        help="PySR niterations.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=200000,
        help="PySR max_evals.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="PySR timeout in seconds.",
    )
    parser.add_argument(
        "--out-json",
        default="playground/hamilton/history/round1/scripts/pysr_direct_baseline_result.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random_state for reproducibility.",
    )
    return parser.parse_args()


def load_xy(csv_path: str, y_col: str, var_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [c for c in var_names + [y_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    x = df[var_names].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    return x, y


def summarize_top_equations(equations_df: pd.DataFrame, select_k: int) -> list[dict]:
    if equations_df is None or equations_df.empty:
        return []
    if "loss" in equations_df.columns:
        equations_df = equations_df.sort_values("loss").reset_index(drop=True)
    top = equations_df.head(select_k)
    out = []
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        eq = str(getattr(row, "equation", ""))
        loss = getattr(row, "loss", np.nan)
        complexity = getattr(row, "complexity", np.nan)
        out.append(
            {
                "rank": rank,
                "equation": eq,
                "loss": float(loss) if pd.notna(loss) else None,
                "complexity": int(complexity) if pd.notna(complexity) else None,
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    var_names = [f"x{i}" for i in range(1, 11)]
    X_train, y_train = load_xy(args.train_csv, args.y_col, var_names)
    X_ood, y_ood = load_xy(args.ood_csv, args.y_col, var_names)

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=args.niterations,
        max_evals=args.max_evals,
        timeout_in_seconds=args.timeout,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        model_selection="best",
        random_state=args.seed,
        procs=0,
        verbosity=0,
    )
    model.fit(X_train, y_train, variable_names=var_names)

    equations = model.equations_
    if isinstance(equations, list):
        equations_df = equations[0]
    else:
        equations_df = equations

    top = summarize_top_equations(equations_df, args.select_k)
    print(f"Top-{len(top)} equations (by loss):")
    for item in top:
        print(
            f"#{item['rank']}: {item['equation']}"
            f" | loss={item['loss']} | complexity={item['complexity']}"
        )

    train_pred = model.predict(X_train)
    ood_pred = model.predict(X_ood)
    train_mse = float(mean_squared_error(y_train, train_pred))
    ood_mse = float(mean_squared_error(y_ood, ood_pred))

    print(f"Best equation: {model.sympy()}")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"OOD MSE: {ood_mse:.6f}")
    print(f"OOD gap: {ood_mse - train_mse:.6f}")

    result = {
        "settings": {
            "niterations": args.niterations,
            "max_evals": args.max_evals,
            "timeout_in_seconds": args.timeout,
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["sin", "cos", "exp", "log"],
            "select_k": args.select_k,
            "seed": args.seed,
            "train_csv": str(Path(args.train_csv).resolve()),
            "ood_csv": str(Path(args.ood_csv).resolve()),
        },
        "top_equations": top,
        "best_equation": str(model.sympy()),
        "train_mse": train_mse,
        "ood_mse": ood_mse,
        "ood_gap": ood_mse - train_mse,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Result JSON: {out_path}")


if __name__ == "__main__":
    run(parse_args())
