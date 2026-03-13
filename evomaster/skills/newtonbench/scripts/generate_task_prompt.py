#!/usr/bin/env python3
"""Generate NewtonBench task prompt package for a specific module setting.

Usage example:
  python generate_task_prompt.py \
    --module m0_gravity \
    --system vanilla_equation \
    --difficulty easy \
    --law-version v0 \
    --noise 0 \
    --code-assisted
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def append_hamilton_task_hint(task_prompt: str, *, module_name: str, system: str) -> str:
    if module_name != "m10_be_distribution":
        return task_prompt

    if system == "vanilla_equation":
        hint = """

**Hamilton Working Note (direct occupation mode):**
- This system directly measures `n(omega, T)`. There is no filter, no bandwidth, and no `total_power` proxy in this mode.
- Only reason with the controls actually listed in this task: `omega` and `temperature`.
- If your measured outputs stay close to `0.5`, you are likely trapped in a small-exponent asymptotic regime. Expand the `omega / T` ratio aggressively across orders of magnitude before fitting candidates.
"""
        return task_prompt + hint

    if system == "simple_system":
        hint = """

**Hamilton Working Note (radiance proxy):**
- This system measures `spectral_radiance` at a single probe frequency. There is no bandwidth parameter in this mode.
- Use only the controls listed in this task: `temperature` and `probe_frequency`.
- Because `R(omega) ∝ n(omega, T) * omega^3`, you may use `spectral_radiance / probe_frequency^3` as a working proxy for `n(omega, T)` when comparing candidate structures.
"""
        return task_prompt + hint

    if system == "complex_system":
        hint = """

**Hamilton Working Note (narrow-band proxy):**
- This system uses a filter + calorimeter. Only in this mode should you reason about `center_frequency`, `bandwidth`, and `total_power`.
- Prioritize experiments with a narrow filter: `bandwidth / center_frequency <= 0.05`.
- In that regime, you can treat `total_power / (bandwidth * center_frequency^3)` as a practical proxy for `n(omega, T)` at `omega = center_frequency`.
- Collect multiple narrow-band samples across orders of magnitude in both `center_frequency` and `temperature` before fitting symbolic candidates.
"""
        return task_prompt + hint

    return task_prompt


def resolve_newtonbench_root(explicit: str | None) -> Path:
    """Resolve NewtonBench root path from arg/env/defaults."""
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
        "NewtonBench root not found. Set --newtonbench-root or NEWTONBENCH_ROOT. "
        "Expected a directory containing 'modules/'."
    )


def load_module(nb_root: Path, module_name: str):
    """Import modules.<module_name> from NewtonBench root."""
    root_str = str(nb_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module(f"modules.{module_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate NewtonBench task prompt package")
    parser.add_argument("--module", required=True, help="NewtonBench module name, e.g. m0_gravity")
    parser.add_argument(
        "--system",
        default="vanilla_equation",
        choices=["vanilla_equation", "simple_system", "complex_system"],
        help="Model system setting",
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Equation difficulty",
    )
    parser.add_argument("--law-version", default="v0", help="Law version, e.g. v0")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level")
    parser.add_argument("--code-assisted", action="store_true", help="Generate code-assisted prompt mode")
    parser.add_argument("--newtonbench-root", default=None, help="Path to NewtonBench repo root")
    parser.add_argument("--output", default="-", help="Output file path, '-' for stdout")
    args = parser.parse_args()

    try:
        nb_root = resolve_newtonbench_root(args.newtonbench_root)
        module = load_module(nb_root, args.module)
    except Exception as e:
        print(
            "Failed to load NewtonBench module. "
            "Please check NEWTONBENCH_ROOT path and dependencies (e.g. numpy/scipy).\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        return 2

    task_prompt = module.get_task_prompt(
        args.system,
        is_code_assisted=args.code_assisted,
        noise_level=args.noise,
    )
    task_prompt = append_hamilton_task_hint(
        task_prompt,
        module_name=args.module,
        system=args.system,
    )

    payload = {
        "module": args.module,
        "system": args.system,
        "difficulty": args.difficulty,
        "law_version": args.law_version,
        "noise": args.noise,
        "code_assisted": bool(args.code_assisted),
        "newtonbench_root": str(nb_root),
        "function_signature": getattr(module, "FUNCTION_SIGNATURE", ""),
        "param_description": getattr(module, "PARAM_DESCRIPTION", ""),
        "task_prompt": task_prompt,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote task package to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
