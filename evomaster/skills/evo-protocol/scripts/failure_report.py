#!/usr/bin/env python3
"""Extract failure patterns from experiment.json and plan.md.

Usage:
    python failure_report.py --experiment experiment.json --plan plan.md

Output: analysis of failure patterns including:
  - Variable combinations that performed poorly
  - Expression templates that didn't help
  - Suggested exclusions for future rounds
"""

import argparse
import json
import os
import sys


def _load_json(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_text(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _analyze_variable_performance(experiment):
    """Analyze which variable combinations performed well/poorly."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        return [], []

    var_mse = {}  # variable_set -> list of MSEs
    for rkey, rdata in rounds.items():
        config = rdata.get("pysr_config", {})
        vars_used = config.get("variable_names", [])
        results = rdata.get("results", [])

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

        if vars_used and best_mse is not None:
            key = tuple(sorted(vars_used))
            if key not in var_mse:
                var_mse[key] = []
            var_mse[key].append(best_mse)

    # Sort by average MSE
    ranked = []
    for vars_key, mses in var_mse.items():
        avg_mse = sum(mses) / len(mses)
        ranked.append({
            "variables": list(vars_key),
            "avg_mse": avg_mse,
            "n_trials": len(mses),
            "min_mse": min(mses),
        })
    ranked.sort(key=lambda x: x["avg_mse"])

    good = [r for r in ranked if r == ranked[0]] if ranked else []
    bad = [r for r in ranked if r["avg_mse"] > ranked[0]["avg_mse"] * 2] if ranked and ranked[0]["avg_mse"] > 0 else []

    return good, bad


def _analyze_template_performance(experiment):
    """Analyze which expression templates performed well/poorly."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        return []

    template_mse = {}
    for rkey, rdata in rounds.items():
        config = rdata.get("pysr_config", {})
        spec = config.get("expression_spec", {})
        combine = spec.get("combine", "free search")
        if not combine:
            combine = "free search"

        results = rdata.get("results", [])
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

        if best_mse is not None:
            if combine not in template_mse:
                template_mse[combine] = []
            template_mse[combine].append(best_mse)

    ranked = []
    for template, mses in template_mse.items():
        ranked.append({
            "template": template,
            "avg_mse": sum(mses) / len(mses),
            "n_trials": len(mses),
            "min_mse": min(mses),
        })
    ranked.sort(key=lambda x: x["avg_mse"])
    return ranked


def _find_individual_variable_signal(experiment):
    """Check which individual variables appear in the best equations."""
    rounds = experiment.get("rounds", {})
    if not isinstance(rounds, dict):
        return {}

    var_count_good = {}
    var_count_total = {}

    all_rounds = sorted(rounds.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    if not all_rounds:
        return {}

    # Find global best MSE threshold (top 50%)
    all_mses = []
    for rdata in rounds.values():
        for r in rdata.get("results", []):
            mse = r.get("mse")
            if mse is not None:
                try:
                    all_mses.append(float(mse))
                except (ValueError, TypeError):
                    pass
    if not all_mses:
        return {}

    median_mse = sorted(all_mses)[len(all_mses) // 2]

    for rdata in rounds.values():
        config = rdata.get("pysr_config", {})
        vars_used = config.get("variable_names", [])
        results = rdata.get("results", [])

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

        for v in vars_used:
            var_count_total[v] = var_count_total.get(v, 0) + 1
            if best_mse is not None and best_mse <= median_mse:
                var_count_good[v] = var_count_good.get(v, 0) + 1

    return {
        v: {
            "total_appearances": var_count_total.get(v, 0),
            "good_results": var_count_good.get(v, 0),
        }
        for v in var_count_total
    }


def main():
    parser = argparse.ArgumentParser(description="Extract failure patterns")
    parser.add_argument("--experiment", default="experiment.json", help="Path to experiment.json")
    parser.add_argument("--plan", default="plan.md", help="Path to plan.md")
    args = parser.parse_args()

    experiment = _load_json(args.experiment)
    plan_text = _load_text(args.plan)

    print("=" * 50)
    print("  Failure Pattern Report")
    print("=" * 50)

    # 1. Variable performance
    good, bad = _analyze_variable_performance(experiment)
    if good:
        print("\nBest variable combinations:")
        for g in good:
            print(f"  {', '.join(g['variables'])} -> avg MSE: {g['avg_mse']:.6f} ({g['n_trials']} trials)")
    if bad:
        print("\nPoor variable combinations (>2x worse than best):")
        for b in bad:
            print(f"  {', '.join(b['variables'])} -> avg MSE: {b['avg_mse']:.6f} ({b['n_trials']} trials)")
        print("  SUGGESTION: avoid these combinations in future rounds")

    # 2. Template performance
    templates = _analyze_template_performance(experiment)
    if templates:
        print("\nExpression template performance:")
        for t in templates:
            print(f"  '{t['template']}' -> avg MSE: {t['avg_mse']:.6f} ({t['n_trials']} trials)")
        if len(templates) > 1:
            worst = templates[-1]
            print(f"  SUGGESTION: avoid template '{worst['template']}' (worst performer)")

    # 3. Individual variable signal
    var_signal = _find_individual_variable_signal(experiment)
    if var_signal:
        print("\nIndividual variable effectiveness:")
        for v, stats in sorted(var_signal.items(), key=lambda x: x[1]["good_results"], reverse=True):
            ratio = stats["good_results"] / stats["total_appearances"] if stats["total_appearances"] > 0 else 0
            marker = "***" if ratio >= 0.5 else "   "
            print(f"  {marker} {v}: in {stats['good_results']}/{stats['total_appearances']} good results ({ratio:.0%})")

        # Variables that never appear in good results
        never_good = [v for v, s in var_signal.items() if s["good_results"] == 0 and s["total_appearances"] >= 2]
        if never_good:
            print(f"\n  SUGGESTION: consider eliminating: {', '.join(never_good)}")
            print("  (appeared 2+ times but never in a good result)")

    if not good and not bad and not templates and not var_signal:
        print("\nInsufficient data for failure analysis. Run more rounds first.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
