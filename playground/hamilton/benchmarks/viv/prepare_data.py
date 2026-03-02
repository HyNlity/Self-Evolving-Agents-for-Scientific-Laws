#!/usr/bin/env python3
"""Prepare VIV benchmark data for Hamilton agent.

Takes the raw VIV wind tunnel time-series (from EvLOWN paper) and produces:
  - data.csv      : training set (U254 Z2S — zero-to-stable, 2.54 m/s)
  - data_ood.csv  : test set    (U254 B2S — big-to-stable,  same wind speed, different IC)

The key idea:
  - Same physical system, same wind speed
  - DIFFERENT initial conditions → no data leakage
  - Training: amplitude grows from ~0 → stable VIV amplitude
  - Testing:  amplitude decays from 15mm → same stable VIV amplitude
  - A correct governing equation should predict BOTH trajectories

Data columns:
  t   — time (s)
  x   — displacement (mm)
  v   — velocity (mm/s), computed via central finite difference
  a   — acceleration (mm/s^2), computed via central finite difference

The agent's goal is to discover the ODE:  x'' = f(x, x')
i.e., find f such that a ≈ f(x, v).

Subsampling: raw data is 300 Hz (~90k points).
We subsample to 100 Hz to reduce size while preserving dynamics.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def load_raw(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw CSV (columns: index, t, x) → (t, x)."""
    df = pd.read_csv(path)
    return np.array(df["t"]), np.array(df["x"])


def compute_derivatives(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central finite difference for velocity and acceleration."""
    dt = np.gradient(t)
    v = np.gradient(x, t, edge_order=2)
    a = np.gradient(v, t, edge_order=2)
    return v, a


def subsample(t, x, v, a, factor=3):
    """Take every `factor`-th point (300Hz → 100Hz when factor=3)."""
    idx = np.arange(0, len(t), factor)
    return t[idx], x[idx], v[idx], a[idx]


def make_dataframe(t, x, v, a) -> pd.DataFrame:
    """Build DataFrame, trimming edges to avoid derivative boundary artifacts."""
    # Trim 50 points from each end (derivative edge effects)
    trim = 50
    sl = slice(trim, -trim)
    return pd.DataFrame({
        "t": t[sl],
        "x": x[sl],
        "v": v[sl],
        "a": a[sl],
    })


def main():
    parser = argparse.ArgumentParser(description="Prepare VIV benchmark data")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "paper", "EvLOWN-main", "VIV", "Data"),
        help="Path to raw VIV Data/ directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(__file__),
        help="Output directory for data.csv and data_ood.csv",
    )
    parser.add_argument(
        "--train-file", default="U254_Z2S.csv",
        help="Training data filename (default: U254_Z2S.csv)",
    )
    parser.add_argument(
        "--test-file", default="U254_B2S.csv",
        help="Test data filename (default: U254_B2S.csv)",
    )
    parser.add_argument(
        "--subsample", type=int, default=3,
        help="Subsample factor (default: 3, i.e. 300Hz→100Hz)",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Data dir:   {data_dir}")
    print(f"Output dir: {output_dir}")

    # --- Training data ---
    train_path = os.path.join(data_dir, args.train_file)
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found")
        sys.exit(1)

    t, x = load_raw(train_path)
    v, a = compute_derivatives(t, x)
    t, x, v, a = subsample(t, x, v, a, factor=args.subsample)
    df_train = make_dataframe(t, x, v, a)

    train_out = os.path.join(output_dir, "data.csv")
    df_train.to_csv(train_out, index=False)
    print(f"Training data: {train_out}  ({len(df_train)} rows)")
    print(f"  t range:  [{df_train['t'].min():.3f}, {df_train['t'].max():.3f}] s")
    print(f"  x range:  [{df_train['x'].min():.3f}, {df_train['x'].max():.3f}] mm")
    print(f"  v range:  [{df_train['v'].min():.3f}, {df_train['v'].max():.3f}] mm/s")
    print(f"  a range:  [{df_train['a'].min():.3f}, {df_train['a'].max():.3f}] mm/s^2")

    # --- Test data (OOD: different initial condition) ---
    test_path = os.path.join(data_dir, args.test_file)
    if not os.path.exists(test_path):
        print(f"WARNING: {test_path} not found, skipping OOD data")
    else:
        t2, x2 = load_raw(test_path)
        v2, a2 = compute_derivatives(t2, x2)
        t2, x2, v2, a2 = subsample(t2, x2, v2, a2, factor=args.subsample)
        df_test = make_dataframe(t2, x2, v2, a2)

        test_out = os.path.join(output_dir, "data_ood.csv")
        df_test.to_csv(test_out, index=False)
        print(f"\nTest data (OOD): {test_out}  ({len(df_test)} rows)")
        print(f"  t range:  [{df_test['t'].min():.3f}, {df_test['t'].max():.3f}] s")
        print(f"  x range:  [{df_test['x'].min():.3f}, {df_test['x'].max():.3f}] mm")
        print(f"  v range:  [{df_test['v'].min():.3f}, {df_test['v'].max():.3f}] mm/s")
        print(f"  a range:  [{df_test['a'].min():.3f}, {df_test['a'].max():.3f}] mm/s^2")

    print("\nDone.")


if __name__ == "__main__":
    main()
