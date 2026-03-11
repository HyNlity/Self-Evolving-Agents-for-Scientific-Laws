#!/usr/bin/env python3
"""Evaluate a discovered law against NewtonBench module evaluator."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import signal
import sys
from pathlib import Path


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


def extract_law_code(text: str) -> str:
    """Extract discovered_law function from raw text or <final_law> block."""
    # prefer content inside <final_law> ... </final_law>
    start = text.rfind("<final_law>")
    if start >= 0:
        end = text.find("</final_law>", start)
        if end > start:
            text = text[start + len("<final_law>") : end].strip()

    # extract last discovered_law function definition
    matches = re.findall(r"(def\s+discovered_law\s*\(.*?(?=\ndef\s+|\Z))", text, flags=re.S)
    if matches:
        code = matches[-1].strip()
    else:
        code = text.strip()
    # Some shells/loggers pass escaped newlines in one line. Normalize for exec().
    if "\\n" in code and "\n" not in code:
        code = code.replace("\\n", "\n")
    return code


def normalize_law_version(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"", "none", "null", "random"}:
        return None
    return value


def normalize_judge_model(value: str) -> str:
    """Normalize common aliases to NewtonBench's internal model keys."""
    v = (value or "").strip().lower().replace("_", "-")
    alias_map = {
        "gpt5": "gpt5",
        "gpt-5": "gpt5",
        "gpt5mini": "gpt5mini",
        "gpt5-mini": "gpt5mini",
        "gpt-5-mini": "gpt5mini",
        "gpt5chat": "gpt5chat",
        "gpt-5-chat": "gpt5chat",
        "gpt41": "gpt41",
        "gpt-4": "gpt41",
        "gpt-4.1": "gpt41",
        "gpt41mini": "gpt41mini",
        "gpt-4.1-mini": "gpt41mini",
        "o4mini": "o4mini",
        "o4-mini": "o4mini",
    }
    return alias_map.get(v, value)


class EvaluationTimeoutError(TimeoutError):
    """Raised when evaluate_law exceeds wall-clock timeout."""


def _raise_eval_timeout(signum, frame):  # noqa: ARG001
    raise EvaluationTimeoutError("evaluate_law timed out")


def build_evaluate_kwargs(module, law_code: str, difficulty: str, law_version: str | None, judge_model: str):
    """Build kwargs compatible with module.evaluate_law signature."""
    sig = inspect.signature(module.evaluate_law)
    kwargs = {"llm_function_str": law_code}
    for name in sig.parameters:
        if name == "llm_function_str":
            continue
        if name == "param_description":
            kwargs[name] = getattr(module, "PARAM_DESCRIPTION", "")
        elif name == "difficulty":
            kwargs[name] = difficulty
        elif name == "law_version":
            kwargs[name] = law_version
        elif name == "judge_model_name":
            kwargs[name] = judge_model
    return kwargs


def sanitize_evaluation(metrics: Any, reveal_ground_truth: bool) -> dict[str, Any]:
    """Normalize evaluation payload and optionally hide ground-truth law."""
    if not isinstance(metrics, dict):
        return {}
    out = dict(metrics)
    if not reveal_ground_truth:
        out.pop("ground_truth_law", None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate discovered law with NewtonBench module")
    parser.add_argument("--module", required=True, help="NewtonBench module, e.g. m0_gravity")
    parser.add_argument(
        "--difficulty",
        default="easy",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument("--law-version", default="v0", help="Law version, e.g. v0 or none")
    parser.add_argument("--judge-model", default="gpt5mini", help="Judge model name")
    parser.add_argument(
        "--eval-timeout-sec",
        type=int,
        default=int(os.environ.get("NEWTONBENCH_EVAL_TIMEOUT_SEC", "90")),
        help="Hard timeout (seconds) for module.evaluate_law",
    )
    parser.add_argument("--law-file", default=None, help="Path to file containing discovered law")
    parser.add_argument("--law-text", default=None, help="Raw discovered law text")
    parser.add_argument("--newtonbench-root", default=None, help="Path to NewtonBench repo root")
    parser.add_argument(
        "--reveal-ground-truth",
        action="store_true",
        help="Include ground_truth_law in output JSON (default: hidden to avoid leakage).",
    )
    args = parser.parse_args()

    if not args.law_file and not args.law_text:
        raise ValueError("Provide --law-file or --law-text.")

    raw_text = args.law_text or Path(args.law_file).read_text(encoding="utf-8")
    law_code = extract_law_code(raw_text)
    law_version = normalize_law_version(args.law_version)
    judge_model = normalize_judge_model(args.judge_model)

    try:
        nb_root = resolve_newtonbench_root(args.newtonbench_root)
        module = load_module(nb_root, args.module)
    except Exception as e:
        print(
            "Failed to load NewtonBench module for evaluation. "
            "Please check NEWTONBENCH_ROOT path and dependencies.\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        return 2

    kwargs = build_evaluate_kwargs(
        module=module,
        law_code=law_code,
        difficulty=args.difficulty,
        law_version=law_version,
        judge_model=judge_model,
    )
    try:
        if args.eval_timeout_sec and args.eval_timeout_sec > 0:
            prev_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_eval_timeout)
            signal.alarm(int(args.eval_timeout_sec))
            try:
                metrics = module.evaluate_law(**kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)
        else:
            metrics = module.evaluate_law(**kwargs)
    except EvaluationTimeoutError:
        print(
            f"evaluate_law timed out after {args.eval_timeout_sec}s",
            file=sys.stderr,
        )
        return 124
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        return 3

    sanitized_metrics = sanitize_evaluation(
        metrics=metrics,
        reveal_ground_truth=bool(args.reveal_ground_truth),
    )

    print(
        json.dumps(
            {
                "module": args.module,
                "difficulty": args.difficulty,
                "law_version": law_version,
                "judge_model": judge_model,
                "submitted_law": law_code,
                "evaluation": sanitized_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
