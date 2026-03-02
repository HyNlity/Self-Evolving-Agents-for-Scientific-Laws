#!/usr/bin/env python3
"""Check iteration progress by analyzing plan.md and experiment.json.

Usage:
    python check_progress.py --plan plan.md --experiment experiment.json

Output: concise summary of iteration progress including:
  - Number of strategies tried
  - MSE trend
  - Duplicate strategy detection
  - Suggested next direction
"""

import argparse
import json
import os
import re
import sys


def _load_json(path):
    """Load a JSON file, return empty dict on failure."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_text(path):
    """Load a text file, return empty string on failure."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _extract_failed_approaches(plan_text):
    """Extract failed approaches from the plan.md table."""
    approaches = []
    in_table = False
    for line in plan_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("| Round"):
            in_table = True
            continue
        if in_table and stripped.startswith("|---"):
            continue
        if in_table and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) >= 5:
                approaches.append({
                    "round": cells[0],
                    "strategy": cells[1],
                    "variables": cells[2],
                    "template": cells[3],
                    "mse": cells[4],
                })
        elif in_table and not stripped.startswith("|"):
            in_table = False
    return approaches


def _get_mse_trend(experiment):
    """Extract MSE values across rounds from experiment.json."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        return []

    trend = []
    for rkey in sorted(rounds.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        rdata = rounds[rkey]
        results = rdata.get("results", [])
        if not isinstance(results, list) or not results:
            trend.append({"round": rkey, "best_mse": None})
            continue
        best_mse = None
        for r in results:
            mse = r.get("mse")
            if mse is not None:
                try:
                    mse_v = float(mse)
                    if best_mse is None or mse_v < best_mse:
                        best_mse = mse_v
                except (ValueError, TypeError):
                    pass
        trend.append({"round": rkey, "best_mse": best_mse})
    return trend


def _detect_duplicate_strategies(failed):
    """Detect potentially duplicate strategies."""
    seen = {}
    duplicates = []
    for entry in failed:
        key = (entry.get("variables", "").strip().lower(), entry.get("template", "").strip().lower())
        if key in seen and key != ("", ""):
            duplicates.append((seen[key], entry))
        else:
            seen[key] = entry
    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Check iteration progress")
    parser.add_argument("--plan", default="plan.md", help="Path to plan.md")
    parser.add_argument("--experiment", default="experiment.json", help="Path to experiment.json")
    args = parser.parse_args()

    plan_text = _load_text(args.plan)
    experiment = _load_json(args.experiment)

    print("=" * 50)
    print("  Iteration Progress Report")
    print("=" * 50)

    # 1. Rounds completed
    rounds = experiment.get("rounds", {})
    n_rounds = len(rounds) if isinstance(rounds, dict) else 0
    print(f"\nRounds completed: {n_rounds}")

    # 2. Failed approaches
    failed = _extract_failed_approaches(plan_text)
    print(f"Failed approaches recorded: {len(failed)}")

    # 3. MSE trend
    trend = _get_mse_trend(experiment)
    if trend:
        print("\nMSE Trend:")
        for t in trend:
            mse_str = f"{t['best_mse']:.6f}" if t["best_mse"] is not None else "N/A"
            print(f"  Round {t['round']}: {mse_str}")

        # Check for plateau
        valid_mses = [t["best_mse"] for t in trend if t["best_mse"] is not None]
        if len(valid_mses) >= 3:
            recent = valid_mses[-3:]
            if max(recent) > 0:
                variation = (max(recent) - min(recent)) / max(recent)
                if variation < 0.01:
                    print("  WARNING: MSE plateau detected (< 1% change in last 3 rounds)")
    else:
        print("\nMSE Trend: no data")

    # 4. Duplicate detection
    duplicates = _detect_duplicate_strategies(failed)
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} potentially duplicate strategies detected:")
        for orig, dup in duplicates:
            print(f"  - Round {orig['round']} and Round {dup['round']}: {dup['strategy']}")

    # 5. Current best from plan
    best_match = re.search(r"Best equation:\s*(.+)", plan_text)
    best_mse_match = re.search(r"Best MSE:\s*(.+)", plan_text)
    if best_match:
        print(f"\nCurrent best equation: {best_match.group(1).strip()}")
    if best_mse_match:
        print(f"Current best MSE: {best_mse_match.group(1).strip()}")

    # 6. Strategy queue
    strategy_section = ""
    if "## Strategy Queue" in plan_text:
        start = plan_text.index("## Strategy Queue")
        end = plan_text.find("\n## ", start + 20)
        strategy_section = plan_text[start:end] if end != -1 else plan_text[start:]

    pending = [l.strip() for l in strategy_section.split("\n") if l.strip().startswith("- ") or l.strip().startswith("1.")]
    if pending:
        print(f"\nPending strategies: {len(pending)}")
        for p in pending[:3]:
            print(f"  {p}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
