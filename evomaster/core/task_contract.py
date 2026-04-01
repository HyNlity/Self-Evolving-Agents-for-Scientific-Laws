"""Framework-level task contract parsing and materialization helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TASK_CONTRACT_FILE = "task_contract.json"
DEFAULT_EVIDENCE_POLICY = "advisory"
ALLOWED_EVIDENCE_POLICIES = {"advisory", "blocking"}
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)


@dataclass(frozen=True)
class TaskContractBundle:
    """Normalized task body plus optional protocol contract."""

    task_body: str
    contract: dict[str, Any]


def _normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    for item in value:
        if not item:
            continue
        normalized_item = str(item).strip()
        if normalized_item and normalized_item not in normalized:
            normalized.append(normalized_item)
    return normalized


def default_task_contract() -> dict[str, Any]:
    return {
        "protocol": {
            "evidence_policy": DEFAULT_EVIDENCE_POLICY,
            "review_focus": [],
            "required_evidence": {},
        },
        "meta": {
            "has_front_matter": False,
            "parse_errors": [],
        },
    }


def normalize_task_contract(frontmatter: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize task protocol config with safe defaults."""
    contract = default_task_contract()
    parse_errors: list[str] = []

    if frontmatter is None:
        return contract

    if not isinstance(frontmatter, dict):
        parse_errors.append("front_matter_must_be_a_mapping")
        contract["meta"]["parse_errors"] = parse_errors
        contract["meta"]["has_front_matter"] = True
        return contract

    protocol = frontmatter.get("protocol", {})
    if protocol is None:
        protocol = {}
    if not isinstance(protocol, dict):
        parse_errors.append("protocol_must_be_a_mapping")
        protocol = {}

    evidence_policy = str(protocol.get("evidence_policy", DEFAULT_EVIDENCE_POLICY)).strip().lower()
    if evidence_policy not in ALLOWED_EVIDENCE_POLICIES:
        parse_errors.append(f"invalid_evidence_policy:{evidence_policy}")
        evidence_policy = DEFAULT_EVIDENCE_POLICY

    review_focus = _normalize_str_list(protocol.get("review_focus", []))

    raw_required_evidence = protocol.get("required_evidence", {})
    normalized_required_evidence: dict[str, list[str]] = {}
    if raw_required_evidence is None:
        raw_required_evidence = {}
    if not isinstance(raw_required_evidence, dict):
        parse_errors.append("required_evidence_must_be_a_mapping")
        raw_required_evidence = {}

    for capability, evidence_names in raw_required_evidence.items():
        capability_key = str(capability).strip()
        if not capability_key:
            continue
        normalized_names = _normalize_str_list(evidence_names)
        if normalized_names:
            normalized_required_evidence[capability_key] = normalized_names

    contract["protocol"] = {
        "evidence_policy": evidence_policy,
        "review_focus": review_focus,
        "required_evidence": normalized_required_evidence,
    }
    contract["meta"] = {
        "has_front_matter": True,
        "parse_errors": parse_errors,
    }
    return contract


def parse_task_description_with_contract(task_description: str) -> TaskContractBundle:
    """Split optional YAML front matter from the task body and normalize it."""
    raw_text = task_description or ""
    match = FRONT_MATTER_RE.match(raw_text)
    if not match:
        return TaskContractBundle(task_body=raw_text.strip(), contract=default_task_contract())

    frontmatter_text = match.group(1)
    task_body = raw_text[match.end():].lstrip("\n").rstrip()
    try:
        parsed_frontmatter = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
    except yaml.YAMLError as exc:
        contract = default_task_contract()
        contract["meta"]["has_front_matter"] = True
        contract["meta"]["parse_errors"] = [f"invalid_front_matter_yaml:{exc.__class__.__name__}"]
        return TaskContractBundle(task_body=task_body, contract=contract)

    contract = normalize_task_contract(parsed_frontmatter)
    contract.setdefault("meta", {})
    contract["meta"]["has_front_matter"] = True
    return TaskContractBundle(task_body=task_body, contract=contract)


def write_task_contract(workspace_path: str | Path, task_contract: dict[str, Any]) -> Path:
    """Persist the normalized task contract for runtime consumption."""
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    contract_path = workspace / TASK_CONTRACT_FILE
    contract_path.write_text(json.dumps(task_contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract_path
