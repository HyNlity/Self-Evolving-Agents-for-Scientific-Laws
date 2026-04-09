"""Evaluation rubric materialization for Hamilton tasks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .constants import EVALUATION_CONTEXT_FILE, EVALUATION_CONTEXT_MD


METRIC_LIBRARY: dict[str, dict[str, str]] = {
    "mse": {
        "name": "mse",
        "category": "fit",
        "preferred_direction": "lower",
        "description": "Mean squared error on in-domain observations.",
    },
    "mae": {
        "name": "mae",
        "category": "fit",
        "preferred_direction": "lower",
        "description": "Mean absolute error on in-domain observations.",
    },
    "r2": {
        "name": "r2",
        "category": "fit",
        "preferred_direction": "higher",
        "description": "Explained variance or R^2 goodness-of-fit.",
    },
    "nrmse": {
        "name": "nrmse",
        "category": "fit",
        "preferred_direction": "lower",
        "description": "Normalized RMSE for scale-robust fit comparison.",
    },
    "support_set_stability": {
        "name": "support_set_stability",
        "category": "structure",
        "preferred_direction": "higher",
        "description": "Support-set consistency under resampling, ablation, or condition changes.",
    },
    "structure_consistency": {
        "name": "structure_consistency",
        "category": "structure",
        "preferred_direction": "higher",
        "description": "Whether the discovered symbolic structure remains stable across runs or regimes.",
    },
    "symbolic_complexity": {
        "name": "symbolic_complexity",
        "category": "structure",
        "preferred_direction": "lower",
        "description": "Expression complexity or description length of the candidate law.",
    },
    "sparsity": {
        "name": "sparsity",
        "category": "structure",
        "preferred_direction": "higher",
        "description": "Parsimony of the support set or operator inventory.",
    },
    "ood_slice_consistency": {
        "name": "ood_slice_consistency",
        "category": "generalization",
        "preferred_direction": "higher",
        "description": "Consistency on out-of-distribution slices or regimes.",
    },
    "cross_condition_transfer": {
        "name": "cross_condition_transfer",
        "category": "generalization",
        "preferred_direction": "higher",
        "description": "Transfer quality across operating conditions, speeds, or environments.",
    },
    "rollout_stability": {
        "name": "rollout_stability",
        "category": "dynamics",
        "preferred_direction": "higher",
        "description": "Numerical stability of long-horizon simulation or integration.",
    },
    "limit_cycle_fidelity": {
        "name": "limit_cycle_fidelity",
        "category": "dynamics",
        "preferred_direction": "higher",
        "description": "Ability to reproduce attractors or limit-cycle geometry.",
    },
    "amplitude_error": {
        "name": "amplitude_error",
        "category": "dynamics",
        "preferred_direction": "lower",
        "description": "Error of steady-state or transient amplitude response.",
    },
    "phase_error": {
        "name": "phase_error",
        "category": "dynamics",
        "preferred_direction": "lower",
        "description": "Phase mismatch against observed trajectories.",
    },
    "frequency_error": {
        "name": "frequency_error",
        "category": "dynamics",
        "preferred_direction": "lower",
        "description": "Frequency mismatch against observed trajectories.",
    },
    "monotonicity_consistency": {
        "name": "monotonicity_consistency",
        "category": "physics",
        "preferred_direction": "higher",
        "description": "Whether physically expected monotonic relationships are respected.",
    },
    "symmetry_consistency": {
        "name": "symmetry_consistency",
        "category": "physics",
        "preferred_direction": "higher",
        "description": "Whether expected invariances or symmetries are preserved.",
    },
    "conservation_residual": {
        "name": "conservation_residual",
        "category": "physics",
        "preferred_direction": "lower",
        "description": "Residual against conservation-law or balance constraints.",
    },
    "dimension_consistency": {
        "name": "dimension_consistency",
        "category": "physics",
        "preferred_direction": "higher",
        "description": "Whether the expression is dimensionally or unit consistent.",
    },
}

PROFILE_LIBRARY: dict[str, list[str]] = {
    "sr_regression_basic": ["mse", "mae", "r2", "nrmse", "symbolic_complexity"],
    "sr_structure_discovery": [
        "r2",
        "support_set_stability",
        "structure_consistency",
        "symbolic_complexity",
        "sparsity",
    ],
    "dynamics_identification": [
        "r2",
        "rollout_stability",
        "limit_cycle_fidelity",
        "amplitude_error",
        "phase_error",
        "frequency_error",
        "ood_slice_consistency",
    ],
    "physics_law_discovery": [
        "r2",
        "structure_consistency",
        "monotonicity_consistency",
        "symmetry_consistency",
        "conservation_residual",
        "dimension_consistency",
    ],
}

PAPER_METRIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "support_set_stability": ("support set", "支持集", "ablation", "消融", "redund", "proxy"),
    "structure_consistency": ("structure", "结构", "operator", "symbolic", "equation family"),
    "ood_slice_consistency": ("ood", "out-of-distribution", "泛化", "cross-condition", "cross speed"),
    "cross_condition_transfer": ("cross-speed", "wind speed", "工况", "transfer", "迁移"),
    "rollout_stability": ("rollout", "integration", "trajectory", "stable", "积分", "仿真"),
    "limit_cycle_fidelity": ("limit cycle", "attractor", "phase portrait", "极限环"),
    "amplitude_error": ("amplitude", "振幅"),
    "phase_error": ("phase", "相位"),
    "frequency_error": ("frequency", "频率"),
    "monotonicity_consistency": ("monotonic", "单调"),
    "symmetry_consistency": ("symmetry", "对称"),
    "conservation_residual": ("conservation", "守恒", "residual", "残差"),
    "dimension_consistency": ("dimension", "dimensional", "unit", "量纲", "单位"),
}


def infer_evaluation_profile(task_description: str, task_contract: dict[str, Any]) -> str:
    protocol = task_contract.get("protocol", {}) if isinstance(task_contract, dict) else {}
    explicit = str(protocol.get("evaluation_profile", "auto")).strip().lower()
    if explicit and explicit != "auto":
        return explicit

    text = (task_description or "").lower()
    if any(keyword in text for keyword in ("viv", "limit cycle", "trajectory", "rollout", "dynamics", "ode", "极限环", "动力学")):
        return "dynamics_identification"
    if any(keyword in text for keyword in ("physics", "law discovery", "newtonbench", "守恒", "物理")):
        return "physics_law_discovery"
    if any(keyword in text for keyword in ("support set", "symbolic regression", "support-set", "支持集", "方程结构")):
        return "sr_structure_discovery"
    return "sr_regression_basic"


def build_evaluation_context(
    task_description: str,
    task_contract: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    protocol = task_contract.get("protocol", {}) if isinstance(task_contract, dict) else {}
    profile = infer_evaluation_profile(task_description, task_contract)
    metric_names: list[str] = list(PROFILE_LIBRARY.get(profile, PROFILE_LIBRARY["sr_regression_basic"]))

    explicit_metrics = protocol.get("evaluation_metrics", [])
    if isinstance(explicit_metrics, str):
        explicit_metrics = [explicit_metrics]
    if isinstance(explicit_metrics, list):
        for metric_name in explicit_metrics:
            normalized_name = str(metric_name).strip().lower()
            if normalized_name and normalized_name not in metric_names:
                metric_names.append(normalized_name)

    paper_sources = protocol.get("paper_rubric_sources", [])
    if isinstance(paper_sources, str):
        paper_sources = [paper_sources]

    source_notes: list[dict[str, Any]] = []
    for source in paper_sources if isinstance(paper_sources, list) else []:
        parsed = _parse_rubric_source(str(source), project_root)
        source_notes.append(parsed["note"])
        for metric_name in parsed["metrics"]:
            if metric_name not in metric_names:
                metric_names.append(metric_name)

    metrics = [_metric_spec(name) for name in metric_names]
    categories = list(dict.fromkeys(metric["category"] for metric in metrics))

    return {
        "profile": profile,
        "metrics": metrics,
        "focus_dimensions": categories,
        "paper_sources": source_notes,
        "review_focus_hint": protocol.get("review_focus", []),
    }


def materialize_evaluation_context(workspace: Path, context: dict[str, Any]) -> tuple[Path, Path]:
    workspace.mkdir(parents=True, exist_ok=True)
    json_path = workspace / EVALUATION_CONTEXT_FILE
    md_path = workspace / EVALUATION_CONTEXT_MD
    json_path.write_text(json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_evaluation_markdown(context), encoding="utf-8")
    return json_path, md_path


def _metric_spec(metric_name: str) -> dict[str, str]:
    normalized_name = metric_name.strip().lower()
    default = {
        "name": normalized_name,
        "category": "custom",
        "preferred_direction": "task_defined",
        "description": "Task-specific metric supplied by the task contract or paper rubric.",
    }
    return dict(METRIC_LIBRARY.get(normalized_name, default))


def _render_evaluation_markdown(context: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Rubric",
        "",
        f"- profile: {context.get('profile', 'unknown')}",
        f"- focus_dimensions: {', '.join(context.get('focus_dimensions', [])) or 'none'}",
        "",
        "## Metrics",
    ]
    for metric in context.get("metrics", []):
        lines.extend(
            [
                f"### {metric.get('name', 'unknown')}",
                f"- category: {metric.get('category', 'custom')}",
                f"- preferred_direction: {metric.get('preferred_direction', 'task_defined')}",
                f"- description: {metric.get('description', '')}",
                "",
            ]
        )
    if context.get("paper_sources"):
        lines.append("## Paper Rubric Sources")
        for source in context["paper_sources"]:
            lines.extend(
                [
                    f"- source: {source.get('source', '')}",
                    f"  status: {source.get('status', '')}",
                    f"  note: {source.get('note', '')}",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def _parse_rubric_source(source: str, project_root: Path) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()

    if not source_path.exists():
        return {
            "metrics": [],
            "note": {"source": source, "status": "missing", "note": "Rubric source path does not exist."},
        }

    text = _read_source_text(source_path)
    if not text:
        return {
            "metrics": [],
            "note": {
                "source": str(source_path),
                "status": "unparsed",
                "note": "Rubric source could not be parsed; fallback to builtin metrics only.",
            },
        }

    normalized = text.lower()
    metrics: list[str] = []
    for metric_name, keywords in PAPER_METRIC_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            metrics.append(metric_name)

    return {
        "metrics": list(dict.fromkeys(metrics)),
        "note": {
            "source": str(source_path),
            "status": "parsed",
            "note": f"Matched {len(metrics)} rubric metric hints from source text.",
        },
    }


def _read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".py", ".json", ".yaml", ".yml"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _read_pdf_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_text(path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return ""
    try:
        completed = subprocess.run(
            [pdftotext, str(path), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    text = completed.stdout.strip()
    text = re.sub(r"\s+", " ", text)
    return text[:200000]
