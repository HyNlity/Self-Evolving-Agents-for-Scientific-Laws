#!/usr/bin/env python3
"""Run NewtonBench interactive experiment(s) for one task setting."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def resolve_newtonbench_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_root = os.environ.get("NEWTONBENCH_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    # repo-root fallback (scripts/ -> newtonbench/ -> skills/ -> evomaster/ -> <repo-root>)
    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root / "third_party/NewtonBench")
    candidates.append(Path("third_party/NewtonBench"))

    for c in candidates:
        if (c / "modules").exists():
            return c.resolve()
    raise FileNotFoundError(
        "NewtonBench root not found. Set --newtonbench-root or NEWTONBENCH_ROOT."
    )


def load_module(nb_root: Path, module_name: str):
    root_str = str(nb_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module(f"modules.{module_name}")


def parse_inputs(inputs_json: str | None, inputs_file: str | None) -> list[dict[str, Any]]:
    if inputs_json:
        raw = json.loads(inputs_json)
    elif inputs_file:
        raw = json.loads(Path(inputs_file).read_text(encoding="utf-8"))
    else:
        raise ValueError("Provide --inputs-json or --inputs-file.")

    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        if len(raw) > 20:
            raise ValueError("NewtonBench protocol allows at most 20 input sets per call.")
        return raw
    raise ValueError("Inputs must be a JSON object or JSON list.")


def normalize_law_version(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"", "none", "null", "random"}:
        return None
    return value


def normalize_payload(module_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort key normalization for common model-generated aliases."""
    data = dict(payload)
    # m0_gravity frequently receives m1/m2/r from models; map to official names.
    if module_name == "m0_gravity":
        alias_map = {"m1": "mass1", "m2": "mass2", "r": "distance"}
        for src, dst in alias_map.items():
            if src in data and dst not in data:
                data[dst] = data[src]
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NewtonBench experiments")
    parser.add_argument("--module", required=True, help="NewtonBench module, e.g. m0_gravity")
    parser.add_argument(
        "--system",
        default="vanilla_equation",
        choices=["vanilla_equation", "simple_system", "complex_system"],
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--law-version", default="v0", help="Law version, e.g. v0 or none")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level")
    parser.add_argument("--inputs-json", default=None, help="Input parameter sets as JSON string")
    parser.add_argument("--inputs-file", default=None, help="Input parameter sets JSON file")
    parser.add_argument("--newtonbench-root", default=None, help="Path to NewtonBench repo root")
    parser.add_argument(
        "--tag",
        nargs="?",
        const="true",
        default=None,
        help="Emit output wrapped in <experiment_output> tags. Accepts '--tag' or '--tag anything'.",
    )
    args = parser.parse_args()

    try:
        nb_root = resolve_newtonbench_root(args.newtonbench_root)
        module = load_module(nb_root, args.module)
        inputs = parse_inputs(args.inputs_json, args.inputs_file)
        law_version = normalize_law_version(args.law_version)
    except Exception as e:
        print(
            "Failed to prepare NewtonBench experiment call. "
            "Please check NEWTONBENCH_ROOT path, dependencies, and input JSON.\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        return 2

    results: list[Any] = []
    for payload in inputs:
        normalized_payload = normalize_payload(args.module, payload)
        result = module.run_experiment_for_module(
            **normalized_payload,
            noise_level=args.noise,
            difficulty=args.difficulty,
            system=args.system,
            law_version=law_version,
        )
        results.append(result)

    emit_tag = args.tag is not None and str(args.tag).strip().lower() not in {"", "0", "false", "no"}
    if emit_tag:
        print("<experiment_output>")
        print(json.dumps(results, ensure_ascii=False))
        print("</experiment_output>")
    else:
        print(
            json.dumps(
                {
                    "module": args.module,
                    "system": args.system,
                    "difficulty": args.difficulty,
                    "law_version": law_version,
                    "noise": args.noise,
                    "num_inputs": len(inputs),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
