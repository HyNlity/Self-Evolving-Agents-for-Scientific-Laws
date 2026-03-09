#!/usr/bin/env python3
"""Generate NewtonBench task-file JSON for Hamilton batch runs."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def parse_csv_arg(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def parse_float_csv_arg(raw: str) -> list[float]:
    out: list[float] = []
    for p in parse_csv_arg(raw):
        out.append(float(p))
    return out


def resolve_newtonbench_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "third_party/NewtonBench")
    candidates.append(Path("third_party/NewtonBench"))

    for c in candidates:
        if (c / "modules").exists():
            return c.resolve()
    raise FileNotFoundError(
        "NewtonBench root not found. Set --newtonbench-root or clone to third_party/NewtonBench."
    )


def discover_modules(nb_root: Path) -> list[str]:
    modules_dir = nb_root / "modules"
    out: list[str] = []
    for p in modules_dir.iterdir():
        if p.is_dir() and p.name.startswith("m"):
            out.append(p.name)
    return sorted(out)


def available_law_versions(nb_root: Path, module_name: str, difficulty: str) -> list[str]:
    root_str = str(nb_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        laws_mod = importlib.import_module(f"modules.{module_name}.laws")
    except Exception:
        return ["v0"]

    fn = getattr(laws_mod, "get_available_law_versions", None)
    if not callable(fn):
        return ["v0"]

    try:
        versions = fn(difficulty)
    except Exception:
        return ["v0"]

    if not isinstance(versions, list):
        return ["v0"]

    out: list[str] = []
    for v in versions:
        if v is None:
            out.append("none")
        else:
            out.append(str(v))
    return out or ["v0"]


def normalize_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_task_description(
    module: str,
    system: str,
    difficulty: str,
    law_version: str,
    noise: float,
    code_assisted: bool,
) -> str:
    return (
        "profile: newtonbench\n"
        f"module: {module}\n"
        f"system: {system}\n"
        f"difficulty: {difficulty}\n"
        f"law_version: {law_version}\n"
        f"noise: {noise}\n"
        f"code_assisted: {str(code_assisted).lower()}"
    )


def make_task_id(
    index: int,
    module: str,
    system: str,
    difficulty: str,
    law_version: str,
    noise: float,
) -> str:
    noise_tag = str(noise).replace(".", "p").replace("-", "m")
    return f"nb_{index:04d}_{module}_{difficulty}_{system}_{law_version}_n{noise_tag}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate NewtonBench tasks for Hamilton --task-file")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--newtonbench-root", default=None, help="Path to NewtonBench root")
    parser.add_argument("--modules", default="all", help="Comma list or 'all'")
    parser.add_argument(
        "--systems",
        default="vanilla_equation,simple_system,complex_system",
        help="Comma list of systems",
    )
    parser.add_argument("--difficulties", default="easy", help="Comma list of difficulties")
    parser.add_argument(
        "--law-versions",
        default="v0",
        help="Comma list (e.g. v0,v1) or 'all' to query each module/difficulty",
    )
    parser.add_argument("--noise-levels", default="0.0", help="Comma list of noise levels")
    parser.add_argument("--code-assisted", default="false", help="true/false")
    args = parser.parse_args()

    nb_root = resolve_newtonbench_root(args.newtonbench_root)

    if args.modules.strip().lower() == "all":
        modules = discover_modules(nb_root)
    else:
        modules = parse_csv_arg(args.modules)

    systems = parse_csv_arg(args.systems)
    difficulties = parse_csv_arg(args.difficulties)
    noise_levels = parse_float_csv_arg(args.noise_levels)
    code_assisted = normalize_bool(args.code_assisted)

    if not modules:
        raise ValueError("No modules selected.")
    if not systems:
        raise ValueError("No systems selected.")
    if not difficulties:
        raise ValueError("No difficulties selected.")
    if not noise_levels:
        raise ValueError("No noise levels selected.")

    use_all_versions = args.law_versions.strip().lower() == "all"
    explicit_versions = parse_csv_arg(args.law_versions) if not use_all_versions else []

    tasks: list[dict[str, Any]] = []
    idx = 0
    for module in modules:
        for difficulty in difficulties:
            if use_all_versions:
                versions = available_law_versions(nb_root, module, difficulty)
            else:
                versions = explicit_versions
            for law_version in versions:
                for system in systems:
                    for noise in noise_levels:
                        idx += 1
                        task_id = make_task_id(
                            idx,
                            module=module,
                            system=system,
                            difficulty=difficulty,
                            law_version=law_version,
                            noise=noise,
                        )
                        description = build_task_description(
                            module=module,
                            system=system,
                            difficulty=difficulty,
                            law_version=law_version,
                            noise=noise,
                            code_assisted=code_assisted,
                        )
                        tasks.append(
                            {
                                "id": task_id,
                                "description": description,
                                "meta": {
                                    "profile": "newtonbench",
                                    "module": module,
                                    "system": system,
                                    "difficulty": difficulty,
                                    "law_version": law_version,
                                    "noise": noise,
                                    "code_assisted": code_assisted,
                                },
                            }
                        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "output": str(output_path),
        "total_tasks": len(tasks),
        "modules": len(modules),
        "systems": systems,
        "difficulties": difficulties,
        "law_versions_mode": "all" if use_all_versions else explicit_versions,
        "noise_levels": noise_levels,
        "code_assisted": code_assisted,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
