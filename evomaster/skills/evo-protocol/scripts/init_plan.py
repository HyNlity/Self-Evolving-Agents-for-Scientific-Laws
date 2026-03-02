#!/usr/bin/env python3
"""Generate a plan.md from a task description using the Evo Protocol template.

Usage:
    python init_plan.py --task "Discover the governing equation" --output plan.md
    python init_plan.py --task "Find symbolic expression for y" --output plan.md --data data.csv
"""

import argparse
import csv
import os
import sys


def _read_template():
    """Read the plan template from references/plan_template.md."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "..", "references", "plan_template.md")
    if not os.path.exists(template_path):
        # Fallback minimal template
        return (
            "# Research Plan\n\n"
            "## Task\n{task_description}\n\n"
            "## Data Overview\n(TBD)\n\n"
            "## Current Hypotheses\n1. TBD\n\n"
            "## Confirmed Knowledge\n- Best equation: none\n- Best MSE: unknown\n\n"
            "## Strategy Queue\n1. EDA + baseline PySR\n\n"
            "## Failed Approaches\n"
            "| Round | Strategy | Variables | Template/Params | MSE | Why Failed |\n"
            "|-------|----------|-----------|-----------------|-----|------------|\n"
        )
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def _quick_data_summary(data_path):
    """Generate a quick data summary from a CSV file (stdlib-only)."""
    if not data_path or not os.path.exists(data_path):
        return None

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            rows = list(reader)

        n_rows = len(rows)
        n_cols = len(header)
        summary = f"- Columns: {', '.join(header)}\n"
        summary += f"- Data size: {n_rows} rows x {n_cols} columns\n"

        # Check for missing values
        missing_count = 0
        for row in rows:
            for val in row:
                if val.strip() == "":
                    missing_count += 1
        if missing_count > 0:
            summary += f"- Missing values: {missing_count} cells\n"
        else:
            summary += "- Missing values: none\n"

        # Check for time-like column
        time_cols = [c for c in header if c.lower() in ("t", "time", "timestamp", "dt")]
        if time_cols:
            summary += f"- Time column: {', '.join(time_cols)} (consider dynamics)\n"
        else:
            summary += "- Time column: none detected\n"

        return summary

    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate plan.md from task description")
    parser.add_argument("--task", required=True, help="Task description")
    parser.add_argument("--output", default="plan.md", help="Output file path (default: plan.md)")
    parser.add_argument("--data", default=None, help="Optional: path to data.csv for auto-summary")
    args = parser.parse_args()

    template = _read_template()
    plan = template.replace("{task_description}", args.task)

    # If data file provided, fill in data overview
    if args.data:
        summary = _quick_data_summary(args.data)
        if summary:
            # Replace the TBD data overview section
            old_section = (
                "(Fill after first-round EDA: variable list, basic statistics, initial observations)\n\n"
                "- Columns: TBD\n"
                "- Target variable(s): TBD\n"
                "- Time column: TBD (if present, consider dynamics)\n"
                "- Data size: TBD rows x TBD columns\n"
                "- Missing values: TBD\n"
                "- Notable patterns: TBD"
            )
            new_section = "(Auto-generated from data file)\n\n" + summary + "- Notable patterns: TBD"
            plan = plan.replace(old_section, new_section)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(plan)

    print(f"Plan generated: {args.output}")
    if args.data:
        print(f"Data summary included from: {args.data}")


if __name__ == "__main__":
    main()
