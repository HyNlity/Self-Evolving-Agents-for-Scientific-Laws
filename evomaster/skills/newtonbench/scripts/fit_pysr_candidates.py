#!/usr/bin/env python3
"""Collect NewtonBench samples and fit PySR candidate equations.

This script is designed for Hamilton's `pysr_assisted` mode:
1) Optionally query new experiment points from NewtonBench.
2) Persist samples in a workspace-local cache file (jsonl).
3) Fit PySR on accumulated samples.
4) Return top-k symbolic candidates and generated discovered_law templates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SampleRecord:
    input_payload: dict[str, Any]
    result_payload: Any
    targets: dict[str, float]
    input_signature: str


def resolve_newtonbench_root(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_root = os.environ.get("NEWTONBENCH_ROOT")
    if env_root:
        candidates.append(Path(env_root))
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


def normalize_law_version(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"", "none", "null", "random"}:
        return None
    return value


def parse_inputs(inputs_json: str | None, inputs_file: str | None) -> list[dict[str, Any]]:
    if inputs_json:
        raw = json.loads(inputs_json)
    elif inputs_file:
        raw = json.loads(Path(inputs_file).read_text(encoding="utf-8"))
    else:
        return []

    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        if len(raw) > 20:
            raise ValueError("At most 20 input sets are allowed per call.")
        out: list[dict[str, Any]] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"inputs[{idx}] must be JSON object")
            out.append(item)
        return out
    raise ValueError("Inputs must be a JSON object or list of JSON objects.")


def normalize_payload(module_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if module_name == "m0_gravity":
        alias_map = {"m1": "mass1", "m2": "mass2", "r": "distance"}
        for src, dst in alias_map.items():
            if src in data and dst not in data:
                data[dst] = data[src]
    if module_name == "m10_be_distribution":
        # NewtonBench m10 uses temperature/center_frequency in experiment API
        # but discovered_law signature is discovered_law(omega, T).
        if "omega" in data and "center_frequency" not in data:
            data["center_frequency"] = data["omega"]
        if "T" in data and "temperature" not in data:
            data["temperature"] = data["T"]
        if "center_frequency" in data and "omega" not in data:
            data["omega"] = data["center_frequency"]
        if "temperature" in data and "T" not in data:
            data["T"] = data["temperature"]
    return data


def parse_signature_params(signature: str) -> list[str]:
    if not isinstance(signature, str):
        return []
    m = re.search(r"def\s+discovered_law\s*\((.*?)\)\s*:", signature, flags=re.S)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    names: list[str] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if "=" in part:
            part = part.split("=", 1)[0].strip()
        if ":" in part:
            part = part.split(":", 1)[0].strip()
        if part:
            names.append(part)
    return names


def parse_feature_order(raw: str | None, default_order: list[str]) -> list[str]:
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            return parts
    return default_order


def _coerce_finite_float(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
    try:
        value = float(x)
    except Exception:
        return None
    if math.isfinite(value):
        return value
    return None


def _is_number(x: Any) -> bool:
    return _coerce_finite_float(x) is not None


def flatten_numeric_targets(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    numeric = _coerce_finite_float(obj)
    if numeric is not None:
        key = prefix if prefix else "value"
        out[key] = numeric
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric_targets(v, prefix=child_prefix))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(flatten_numeric_targets(v, prefix=child_prefix))
        return out
    return out


def build_input_signature(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def default_dataset_file(
    workspace: Path,
    module: str,
    system: str,
    difficulty: str,
    law_version: str | None,
    noise: float,
) -> Path:
    law = law_version if law_version is not None else "none"
    safe_noise = str(noise).replace(".", "p")
    name = f"{module}__{system}__{difficulty}__{law}__n{safe_noise}.jsonl"
    return workspace / ".cache" / "newtonbench" / name


def load_records(path: Path) -> list[SampleRecord]:
    if not path.exists():
        return []
    records: list[SampleRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        in_payload = payload.get("input")
        targets = payload.get("targets")
        signature = payload.get("input_signature")
        if not isinstance(in_payload, dict):
            continue
        if not isinstance(targets, dict):
            continue
        parsed_targets: dict[str, float] = {}
        for k, v in targets.items():
            numeric = _coerce_finite_float(v)
            if numeric is not None:
                parsed_targets[str(k)] = numeric
        if not parsed_targets:
            continue
        if not isinstance(signature, str) or not signature:
            signature = build_input_signature(in_payload)
        records.append(
            SampleRecord(
                input_payload=in_payload,
                result_payload=payload.get("result"),
                targets=parsed_targets,
                input_signature=signature,
            )
        )
    return records


def append_records(path: Path, records: list[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            payload = {
                "input": rec.input_payload,
                "result": rec.result_payload,
                "targets": rec.targets,
                "input_signature": rec.input_signature,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def collect_new_samples(
    module,
    module_name: str,
    inputs: list[dict[str, Any]],
    *,
    system: str,
    difficulty: str,
    law_version: str | None,
    noise: float,
    dedupe_signatures: set[str],
) -> list[SampleRecord]:
    new_records: list[SampleRecord] = []
    for payload in inputs:
        normalized_payload = normalize_payload(module_name, payload)
        signature = build_input_signature(normalized_payload)
        if signature in dedupe_signatures:
            continue
        result = module.run_experiment_for_module(
            **normalized_payload,
            noise_level=noise,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
        )
        targets = flatten_numeric_targets(result)
        if not targets:
            continue
        new_records.append(
            SampleRecord(
                input_payload=normalized_payload,
                result_payload=result,
                targets=targets,
                input_signature=signature,
            )
        )
        dedupe_signatures.add(signature)
    return new_records


def choose_target_key(records: list[SampleRecord], forced: str | None) -> str:
    if forced:
        return forced
    freq: dict[str, int] = {}
    for rec in records:
        for key in rec.targets:
            freq[key] = freq.get(key, 0) + 1
    if not freq:
        raise ValueError("No numeric targets found in cached experiment records.")
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_dataset_matrix(
    records: list[SampleRecord],
    feature_order: list[str],
    target_key: str,
) -> tuple[list[list[float]], list[float]]:
    X: list[list[float]] = []
    y: list[float] = []
    for rec in records:
        if target_key not in rec.targets:
            continue
        target_value = _coerce_finite_float(rec.targets[target_key])
        if target_value is None:
            continue
        row: list[float] = []
        ok = True
        for name in feature_order:
            value = rec.input_payload.get(name)
            numeric = _coerce_finite_float(value)
            if numeric is None:
                ok = False
                break
            row.append(numeric)
        if not ok:
            continue
        X.append(row)
        y.append(target_value)
    return X, y


def take_tail_points(X: list[list[float]], y: list[float], max_points: int | None) -> tuple[list[list[float]], list[float]]:
    if max_points is None or max_points <= 0:
        return X, y
    if len(X) <= max_points:
        return X, y
    return X[-max_points:], y[-max_points:]


def normalize_ops(raw: str | None, default_ops: list[str]) -> list[str]:
    if raw is None:
        return list(default_ops)
    values = [v.strip() for v in raw.split(",")]
    return [v for v in values if v]


def default_operator_profile(module_name: str) -> tuple[list[str], list[str]]:
    m = str(module_name).strip().lower()
    common_binary = ["+", "-", "*", "/"]
    common_unary = ["log", "sqrt"]
    if m in {"m10_be_distribution"}:
        return (common_binary, ["log", "sqrt", "exp"])
    if m in {"m7_malus_law", "m4_snell_law"}:
        return (common_binary, ["sin", "cos", "sqrt"])
    if m in {"m0_gravity", "m1_coulomb_force", "m2_magnetic_force", "m3_fourier_law"}:
        return (common_binary, ["log", "sqrt"])
    return (common_binary, common_unary)


def cast_optional_float(value: Any) -> float | None:
    try:
        x = float(value)
        if x == x and x not in (float("inf"), float("-inf")):
            return x
        return None
    except Exception:
        return None


def sympy_to_python_expr(expr: str) -> str:
    out = str(expr)
    out = re.sub(r"\bAbs\s*\(", "abs(", out)
    for fn in ("log", "sqrt", "sin", "cos", "tan", "exp"):
        out = re.sub(rf"\b{fn}\s*\(", f"math.{fn}(", out)
    out = re.sub(r"\bpi\b", "math.pi", out)
    return out


def render_discovered_law(feature_order: list[str], expr: str) -> str:
    args = ", ".join(feature_order)
    body = sympy_to_python_expr(expr)
    return "\n".join(
        [
            f"def discovered_law({args}):",
            "    import math",
            f"    return {body}",
        ]
    )


def normalize_expression(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        expr = raw.strip()
        return expr or None
    numeric = _coerce_finite_float(raw)
    if numeric is not None:
        return repr(numeric)
    try:
        expr = str(raw).strip()
    except Exception:
        return None
    if not expr:
        return None
    lowered = expr.lower()
    if lowered in {"nan", "none"}:
        return None
    return expr


def iter_equation_rows(equations: Any):
    if equations is None:
        return
    if hasattr(equations, "iterrows"):
        for _, row in equations.iterrows():
            yield row
        return
    if isinstance(equations, list):
        for part in equations:
            if hasattr(part, "iterrows"):
                for _, row in part.iterrows():
                    yield row


def collect_candidates_from_model(
    model: Any,
    feature_order: list[str],
    top_k: int,
    output_scale: float = 1.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    equations = getattr(model, "equations_", None)
    for row in iter_equation_rows(equations):
        expr = normalize_expression(row.get("sympy_format"))
        if expr is None:
            expr = normalize_expression(row.get("equation"))
        if expr is None:
            continue
        expr_for_output = expr
        if abs(output_scale - 1.0) > 1e-12:
            expr_for_output = f"({repr(float(output_scale))})*({expr})"

        candidate = {
            "expression": expr_for_output,
            "loss": cast_optional_float(row.get("loss")),
            "complexity": cast_optional_float(row.get("complexity")),
            "score": cast_optional_float(row.get("score")),
        }
        rows.append(candidate)

    # Fallback: some PySR versions may have sparse/odd equations_ rows;
    # get_best() still often returns a usable selected equation row.
    if not rows:
        try:
            best = model.get_best()
            best_rows = best if isinstance(best, list) else [best]
            for row in best_rows:
                expr = normalize_expression(row.get("sympy_format"))
                if expr is None:
                    expr = normalize_expression(row.get("equation"))
                if expr is None:
                    continue
                expr_for_output = expr
                if abs(output_scale - 1.0) > 1e-12:
                    expr_for_output = f"({repr(float(output_scale))})*({expr})"

                rows.append(
                    {
                        "expression": expr_for_output,
                        "loss": cast_optional_float(row.get("loss")),
                        "complexity": cast_optional_float(row.get("complexity")),
                        "score": cast_optional_float(row.get("score")),
                    }
                )
        except Exception:
            pass

    rows = sorted(
        rows,
        key=lambda item: (
            item["loss"] if isinstance(item["loss"], float) else float("inf"),
            item["complexity"] if isinstance(item["complexity"], float) else float("inf"),
        ),
    )

    out: list[dict[str, Any]] = []
    seen_expr: set[str] = set()
    for item in rows:
        expr = item["expression"]
        if expr in seen_expr:
            continue
        seen_expr.add(expr)
        enriched = dict(item)
        enriched["rank"] = len(out) + 1
        enriched["discovered_law"] = render_discovered_law(feature_order, expr)
        out.append(enriched)
        if len(out) >= max(1, top_k):
            break
    return out


def fit_pysr_candidates(
    X: list[list[float]],
    y: list[float],
    *,
    feature_order: list[str],
    niterations: int,
    maxsize: int,
    populations: int,
    parsimony: float,
    binary_operators: list[str],
    unary_operators: list[str],
    top_k: int,
    random_state: int,
) -> list[dict[str, Any]]:
    try:
        from pysr import PySRRegressor
    except Exception as e:
        raise RuntimeError(
            "PySR import failed. Ensure `pysr` is installed and Julia runtime is ready. "
            "If needed, run once with a writable JULIA_DEPOT_PATH."
        ) from e

    import numpy as np

    model = PySRRegressor(
        niterations=niterations,
        maxsize=maxsize,
        populations=populations,
        parsimony=parsimony,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        random_state=random_state,
        precision=64,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
        progress=False,
        model_selection="best",
    )
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y_np)):
        raise ValueError("Non-finite targets detected after dataset build.")

    # Large NewtonBench targets (e.g., total_power) can exceed float32 range and
    # destabilize symbolic search. Fit on scaled targets, then scale back expression.
    y_abs_max = float(np.max(np.abs(y_np))) if y_np.size else 1.0
    y_scale = y_abs_max if y_abs_max > 0 else 1.0
    y_fit = y_np / y_scale

    model.fit(X_np, y_fit, variable_names=feature_order)

    return collect_candidates_from_model(
        model=model,
        feature_order=feature_order,
        top_k=top_k,
        output_scale=y_scale,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect NewtonBench samples and fit PySR top-k candidates"
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only check PySR import and exit without fitting",
    )
    parser.add_argument("--module", required=True, help="NewtonBench module, e.g. m10_be_distribution")
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
    parser.add_argument("--noise", type=float, default=0.0, help="Noise level for experiment sampling")
    parser.add_argument("--inputs-json", default=None, help="Optional new sample inputs as JSON")
    parser.add_argument("--inputs-file", default=None, help="Optional new sample inputs file")
    parser.add_argument("--target-key", default=None, help="Target path from experiment result payload")
    parser.add_argument("--feature-order", default=None, help="Comma-separated feature order")
    parser.add_argument("--dataset-file", default=None, help="Cache jsonl path")
    parser.add_argument("--newtonbench-root", default=None, help="Path to NewtonBench root")
    parser.add_argument("--max-points", type=int, default=2000, help="Max recent points used for fitting")
    parser.add_argument("--top-k", type=int, default=5, help="Top candidate count in output")
    parser.add_argument("--niterations", type=int, default=120, help="PySR niterations")
    parser.add_argument("--maxsize", type=int, default=20, help="PySR max expression size")
    parser.add_argument("--populations", type=int, default=8, help="PySR population count")
    parser.add_argument("--parsimony", type=float, default=1e-3, help="PySR parsimony coefficient")
    parser.add_argument(
        "--binary-operators",
        default=None,
        help="Comma-separated PySR binary operators",
    )
    parser.add_argument(
        "--unary-operators",
        default=None,
        help="Comma-separated PySR unary operators",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for PySR")
    parser.add_argument("--output", default="-", help="Output JSON path, '-' for stdout")
    args = parser.parse_args()

    if args.health_check:
        try:
            import pysr  # noqa: F401
            print(
                json.dumps(
                    {
                        "ok": True,
                        "pysr_version": getattr(sys.modules.get("pysr"), "__version__", None),
                        "julia_depot_path": os.environ.get("JULIA_DEPOT_PATH"),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        except Exception as e:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(e),
                        "hint": "Set writable JULIA_DEPOT_PATH and ensure Julia registry/packages are available.",
                    },
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 4

    try:
        nb_root = resolve_newtonbench_root(args.newtonbench_root)
        module = load_module(nb_root, args.module)
        law_version = normalize_law_version(args.law_version)
        inputs = parse_inputs(args.inputs_json, args.inputs_file)
    except Exception as e:
        print(f"Failed to prepare task context: {e}", file=sys.stderr)
        return 2

    signature = getattr(module, "FUNCTION_SIGNATURE", "")
    default_features = parse_signature_params(signature)
    if not default_features:
        print(
            "Cannot infer feature order from FUNCTION_SIGNATURE. "
            "Please pass --feature-order explicitly.",
            file=sys.stderr,
        )
        return 2
    feature_order = parse_feature_order(args.feature_order, default_features)

    workspace = Path.cwd()
    dataset_file = (
        Path(args.dataset_file).resolve()
        if args.dataset_file
        else default_dataset_file(
            workspace=workspace,
            module=args.module,
            system=args.system,
            difficulty=args.difficulty,
            law_version=law_version,
            noise=args.noise,
        )
    )

    existing_records = load_records(dataset_file)
    dedupe_signatures = {rec.input_signature for rec in existing_records}

    try:
        new_records = collect_new_samples(
            module=module,
            module_name=args.module,
            inputs=inputs,
            system=args.system,
            difficulty=args.difficulty,
            law_version=law_version,
            noise=args.noise,
            dedupe_signatures=dedupe_signatures,
        )
    except Exception as e:
        print(f"Failed to run experiments for sampling: {e}", file=sys.stderr)
        return 3

    if new_records:
        append_records(dataset_file, new_records)
        existing_records.extend(new_records)

    if not existing_records:
        print(
            "No samples available. Provide --inputs-json/--inputs-file to collect data first.",
            file=sys.stderr,
        )
        return 3

    try:
        target_key = choose_target_key(existing_records, args.target_key)
        X, y = build_dataset_matrix(existing_records, feature_order, target_key)
        X, y = take_tail_points(X, y, args.max_points)
    except Exception as e:
        print(f"Failed to build fitting dataset: {e}", file=sys.stderr)
        return 3

    if len(X) < 6:
        available_input_keys = sorted(
            {k for rec in existing_records for k in rec.input_payload.keys()}
        )
        print(
            "Not enough valid samples for PySR fitting "
            f"(got {len(X)} rows, need >= 6). "
            f"feature_order={feature_order}, available_input_keys={available_input_keys}, "
            f"target_key={target_key}, records_total={len(existing_records)}",
            file=sys.stderr,
        )
        return 3

    profile_binary, profile_unary = default_operator_profile(args.module)
    binary_ops = normalize_ops(args.binary_operators, profile_binary)
    unary_ops = normalize_ops(args.unary_operators, profile_unary)

    fit_attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    attempt_configs: list[dict[str, Any]] = [
        {
            "label": "default",
            "niterations": args.niterations,
            "maxsize": args.maxsize,
            "populations": args.populations,
            "parsimony": args.parsimony,
            "binary_operators": binary_ops,
            "unary_operators": unary_ops,
            "random_state": args.random_state,
        },
        {
            "label": "fallback_relaxed",
            "niterations": max(args.niterations * 2, 200),
            "maxsize": max(args.maxsize + 8, 28),
            "populations": max(args.populations + 4, 12),
            "parsimony": max(args.parsimony * 0.3, 1e-6),
            "binary_operators": binary_ops,
            "unary_operators": sorted(set(unary_ops + ["exp", "log", "sqrt"])),
            "random_state": args.random_state + 1,
        },
    ]

    for cfg in attempt_configs:
        try:
            attempt_candidates = fit_pysr_candidates(
                X=X,
                y=y,
                feature_order=feature_order,
                niterations=int(cfg["niterations"]),
                maxsize=int(cfg["maxsize"]),
                populations=int(cfg["populations"]),
                parsimony=float(cfg["parsimony"]),
                binary_operators=list(cfg["binary_operators"]),
                unary_operators=list(cfg["unary_operators"]),
                top_k=args.top_k,
                random_state=int(cfg["random_state"]),
            )
        except Exception as e:
            fit_attempts.append(
                {
                    "label": cfg["label"],
                    "status": "error",
                    "error": str(e),
                    "niterations": cfg["niterations"],
                    "maxsize": cfg["maxsize"],
                    "populations": cfg["populations"],
                    "parsimony": cfg["parsimony"],
                }
            )
            continue

        fit_attempts.append(
            {
                "label": cfg["label"],
                "status": "ok",
                "candidate_count": len(attempt_candidates),
                "niterations": cfg["niterations"],
                "maxsize": cfg["maxsize"],
                "populations": cfg["populations"],
                "parsimony": cfg["parsimony"],
            }
        )
        if attempt_candidates:
            candidates = attempt_candidates
            break

    if not candidates:
        print(
            "PySR fitting produced no candidate equations after retry. "
            "Try widening sample diversity/scales or further increasing niterations/populations.",
            file=sys.stderr,
        )
        print(json.dumps({"fit_attempts": fit_attempts}, ensure_ascii=False), file=sys.stderr)
        return 4

    payload = {
        "module": args.module,
        "system": args.system,
        "difficulty": args.difficulty,
        "law_version": law_version,
        "noise": args.noise,
        "function_signature": signature,
        "feature_order": feature_order,
        "target_key": target_key,
        "dataset_file": str(dataset_file),
        "records_total": len(existing_records),
        "records_added": len(new_records),
        "fit_rows": len(X),
        "top_k": max(1, args.top_k),
        "fit_attempts": fit_attempts,
        "candidates": candidates,
        "best_candidate": candidates[0] if candidates else None,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote PySR candidate package to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
