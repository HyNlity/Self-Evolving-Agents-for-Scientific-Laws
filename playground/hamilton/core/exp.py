"""Hamilton Round Exp - proposer/critic round orchestration."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from evomaster.core.task_contract import default_task_contract, normalize_task_contract
from evomaster.utils.types import TaskInstance

from .constants import (
    CRITIC_CHALLENGE_TYPES,
    CRITIC_REPORT_JSON,
    CRITIC_REPORT_MD,
    DEBATE_STATE_FILE,
    ENV_CAPABILITIES_FILE,
)

DEFAULT_REQUIRED_L2_FILES = (
    "plan.md",
    "findings.md",
)

DEFAULT_MACHINE_STATE_FILES = (
    "variable_memory.json",
    "routing_state.json",
    "hypothesis_archive.jsonl",
    "falsification_log.jsonl",
)

TRACE_METRICS_HEADING = "### 指标记录"
TRACE_FALSIFICATION_HEADING = "### 证伪实验"

COMPLETED_FIT_PATTERNS = (
    r"(已完成|完成了|进行了|执行了|得到|产出|生成).{0,24}(拟合|回归|fit|calibration|estimate)",
    r"(拟合|回归|fit|calibration|estimate).{0,24}(已完成|完成|得到|产出|确认|表明)",
)

COMPLETED_INTEGRATION_PATTERNS = (
    r"(已完成|完成了|进行了|执行了|得到|产出|生成).{0,24}(积分|仿真|模拟|rollout|simulation|integration|evaluation)",
    r"(积分|仿真|模拟|rollout|simulation|integration|evaluation).{0,24}(已完成|完成|得到|产出|确认|表明)",
)

COMPLETED_FALSIFICATION_PATTERNS = (
    r"(已完成|完成了|进行了|执行了|得到|产出|生成).{0,24}(证伪|消融|去除|互换)",
    r"(证伪结果|消融实验|去除.+项|互换.+系数).{0,24}(已完成|完成|表明|确认|导致|失效|发散)",
)

SUPPORTED_CONTRACT_EVIDENCE = {
    "plan",
    "findings",
    "trace",
    "machine_state",
    "script",
    "result",
    "trace_metrics",
    "trace_falsification",
}

CLAIMED_ROUND_ARTIFACT_RE = re.compile(
    r"(?P<path>(?:history/round\d+/(?:scripts|results)/[^\s`'\"，,；;]+|(?:scripts|results)/[^\s`'\"，,；;]+))"
)

QUANTIFIED_RESULT_PATTERNS = (
    r"(R²|R\^2|fit_R2|amplitude_error|RMSE|MAE|MSE|loss|accuracy|precision|recall|F1|AUC)\s*(?:[:=≈<>]|约)?\s*[-+]?\d",
    r"(稳态振幅|振幅误差|相对误差|系数表|系数趋势|趋势图|coefficient table).{0,16}(≈|=|约)\s*[-+]?\d",
    r"(ω²|omega\^2|系数|coefficient|coefficients).{0,16}(≈|=|约)\s*[-+]?\d",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _truncate(text: str, limit: int = 1200) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 32] + "\n... [truncated] ..."


def _file_fingerprint(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _round_file_inventory(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: str(path))


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"line {line_no} is not a JSON object")
        objects.append(payload)
    return objects


class RoundExp(BaseExp):
    """Single Hamilton round with proposer/critic orchestration."""

    def __init__(
        self,
        hamilton_agent,
        critic_agent,
        config,
        round_num,
        critic_policy: dict[str, Any] | None = None,
        completion_policy: dict[str, Any] | None = None,
        task_contract: dict[str, Any] | None = None,
    ):
        super().__init__(hamilton_agent, config)
        self.hamilton_agent = hamilton_agent
        self.critic_agent = critic_agent
        self.round_num = round_num
        self.critic_policy = critic_policy or {}
        self.completion_policy = completion_policy or {}
        if task_contract and isinstance(task_contract, dict) and isinstance(task_contract.get("protocol"), dict):
            normalized_contract = dict(task_contract)
            normalized_contract.setdefault("meta", {"has_front_matter": False, "parse_errors": []})
            self.task_contract = normalized_contract
        elif task_contract:
            self.task_contract = normalize_task_contract(task_contract)
        else:
            self.task_contract = default_task_contract()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return f"Round_{self.round_num}"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """Execute one Hamilton round followed by one critic review."""
        self.logger.info("Starting Round %s", self.round_num)

        self._ensure_round_dirs()
        self._init_trace()
        self._init_critic_artifacts()

        artifact_snapshot = self._snapshot_completion_artifacts()

        hamilton_result, hamilton_signal, hamilton_trajectory = self._run_hamilton(
            task_description,
            task_id,
            artifact_snapshot,
        )

        # L2 promotion is still Hamilton's responsibility; critic only audits.
        self._check_l2_promotion(artifact_snapshot)

        critic_result, critic_signal, critic_trajectory = self._run_critic(
            task_description=task_description,
            task_id=task_id,
            hamilton_result=hamilton_result,
            hamilton_signal=hamilton_signal,
        )

        critic_report = self._load_critic_report()
        aggregated_signal = self._aggregate_signals(hamilton_signal, critic_signal, critic_report)
        self._update_debate_state(hamilton_signal, critic_signal, critic_report, aggregated_signal)

        findings_content = self._read_findings()
        self.logger.info("Round %s completed", self.round_num)

        return {
            "round": self.round_num,
            "hamilton_result": hamilton_result,
            "critic_result": critic_result,
            "agent_result": hamilton_result,
            "signal": aggregated_signal,
            "hamilton_signal": hamilton_signal,
            "critic_signal": critic_signal,
            "critic_report": critic_report,
            "findings": findings_content,
            "trajectory": {"hamilton": hamilton_trajectory, "critic": critic_trajectory},
        }

    def _run_hamilton(
        self,
        task_description: str,
        task_id: str,
        artifact_snapshot: dict[str, tuple[int, int]],
    ) -> tuple[str, dict[str, Any], Any]:
        max_repair_attempts = max(int(self.completion_policy.get("max_hamilton_repair_attempts", 2) or 0), 0)
        attempts: list[dict[str, Any]] = []
        repair_instruction = ""

        for attempt_index in range(max_repair_attempts + 1):
            BaseAgent.set_exp_info(
                exp_name=f"{self.exp_name}_Hamilton_Attempt{attempt_index + 1}",
                exp_index=self.round_num,
            )
            self.logger.info(
                "[Round %s] Running Hamilton proposer (attempt %s/%s)...",
                self.round_num,
                attempt_index + 1,
                max_repair_attempts + 1,
            )
            task = TaskInstance(
                task_id=f"{task_id}_round{self.round_num}_hamilton_attempt{attempt_index + 1}",
                task_type="hamilton",
                description=self._build_hamilton_task_description(task_description, repair_instruction),
                input_data={
                    "round": self.round_num,
                    "repair_attempt": attempt_index,
                    "task_contract_policy": self._task_evidence_policy(),
                    "task_review_focus": ", ".join(self._task_protocol().get("review_focus", [])),
                },
            )
            trajectory = self.hamilton_agent.run(task)
            result = self._extract_agent_response(trajectory)
            signal = self._parse_signal(result, trajectory, actor="hamilton")
            completion_gate = self._evaluate_hamilton_completion(artifact_snapshot, signal)
            attempts.append(completion_gate)
            signal["completion_gate"] = completion_gate

            if completion_gate.get("accepted", True):
                self.logger.info(
                    "[Round %s] Hamilton proposer completed after %s attempt(s).",
                    self.round_num,
                    attempt_index + 1,
                )
                return result, signal, trajectory

            missing_summary = ", ".join(completion_gate.get("missing_artifacts", []))
            self.logger.warning(
                "[Round %s] Hamilton finish rejected by completion gate (attempt %s/%s). Missing updates: %s",
                self.round_num,
                attempt_index + 1,
                max_repair_attempts + 1,
                missing_summary or "unknown",
            )
            repair_instruction = self._build_hamilton_repair_instruction(completion_gate, result)

        final_gate = attempts[-1] if attempts else {"accepted": False, "missing_artifacts": []}
        signal["completion_gate"] = final_gate
        signal["satisfied"] = False
        signal["task_completed"] = "false"
        signal["notes"] = _truncate(
            f"{signal.get('notes', '')}\nCompletion gate rejected finish because required artifacts "
            f"were not updated: {', '.join(final_gate.get('missing_artifacts', []))}".strip(),
            500,
        )
        self.logger.info("[Round %s] Hamilton proposer completed with unresolved completion gate blockers.", self.round_num)
        return result, signal, trajectory

    def _run_critic(
        self,
        *,
        task_description: str,
        task_id: str,
        hamilton_result: str,
        hamilton_signal: dict[str, Any],
    ) -> tuple[str, dict[str, Any], Any]:
        BaseAgent.set_exp_info(exp_name=f"{self.exp_name}_Critic", exp_index=self.round_num)
        self.logger.info("[Round %s] Running Critic challenger...", self.round_num)
        report_md, report_json = self._critic_report_paths()
        task = TaskInstance(
            task_id=f"{task_id}_round{self.round_num}_critic",
            task_type="hamilton_critic",
            description=task_description,
            input_data={
                "round": self.round_num,
                "hamilton_summary": _truncate(hamilton_result, 1800),
                "hamilton_task_completed": hamilton_signal.get("task_completed", "false"),
                "challenge_taxonomy": ", ".join(CRITIC_CHALLENGE_TYPES),
                "task_contract_policy": self._task_evidence_policy(),
                "task_review_focus": ", ".join(self._task_protocol().get("review_focus", [])),
                "trace_file": str((self.run_dir / "history" / f"round{self.round_num}" / "trace.md").relative_to(self.run_dir)),
                "critic_report_md": str(report_md.relative_to(self.run_dir)),
                "critic_report_json": str(report_json.relative_to(self.run_dir)),
            },
        )
        trajectory = self.critic_agent.run(task)
        result = self._extract_agent_response(trajectory)
        signal = self._parse_signal(result, trajectory, actor="critic")
        self.logger.info("[Round %s] Critic challenger completed", self.round_num)
        return result, signal, trajectory

    def _ensure_round_dirs(self) -> None:
        if not self.run_dir:
            return
        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        (round_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (round_dir / "results").mkdir(parents=True, exist_ok=True)

    def _init_trace(self) -> None:
        if not self.run_dir:
            return

        trace_file = self.run_dir / "history" / f"round{self.round_num}" / "trace.md"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        l1_template = f"""# 执行日志 — 第 {self.round_num} 轮

### 本轮策略
- 路由决策：
- 目标支持集：
- 主要证伪对象：

### 操作记录
（记录执行的脚本、使用的参数、观察到的现象）

### 指标记录
| 实验 | 方法 | 支持集 | 关键参数/模板 | Fit | Support Stability | Structure | 备注 |
|------|------|--------|----------------|-----|-------------------|-----------|------|

### 证伪实验
| 实验 | 比较候选 | 证伪方式 | 结果 | 结论 |
|------|----------|----------|------|------|

### 工作笔记
（变量角色变化、中间观察、思考过程）
"""
        trace_file.write_text(l1_template, encoding="utf-8")
        self.logger.info("创建 history/round%s/trace.md（L1）", self.round_num)

    def _init_critic_artifacts(self) -> None:
        report_md, report_json = self._critic_report_paths()
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_json.parent.mkdir(parents=True, exist_ok=True)

    def _critic_report_paths(self) -> tuple[Path, Path]:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        report_md = round_dir / self.critic_policy.get("report_md_name", CRITIC_REPORT_MD)
        report_json = round_dir / self.critic_policy.get("report_json_name", CRITIC_REPORT_JSON)
        return report_md, report_json

    def _load_critic_report(self) -> dict[str, Any]:
        report_md, report_json = self._critic_report_paths()
        payload = _read_json(report_json, {})
        if not payload:
            payload = {
                "round": self.round_num,
                "blocking": True,
                "approved": False,
                "summary": _truncate(report_md.read_text(encoding="utf-8") if report_md.exists() else "", 800),
                "challenges": [],
                "required_evidence": [],
                "updated_at": _utc_now(),
            }
        payload.setdefault("round", self.round_num)
        payload.setdefault("blocking", not payload.get("approved", False))
        payload.setdefault("approved", not payload.get("blocking", True))
        payload.setdefault("summary", "")
        payload.setdefault("challenges", [])
        payload.setdefault("required_evidence", [])
        payload["challenges"] = self._normalize_challenges(payload.get("challenges"))
        return payload

    def _normalize_challenges(self, challenges: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if not isinstance(challenges, list):
            return normalized
        for item in challenges:
            if isinstance(item, str):
                normalized.append(
                    {
                        "challenge_type": "evidence_gap_attack",
                        "title": item[:120],
                        "summary": item[:500],
                        "blocking": True,
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            challenge_type = item.get("challenge_type", "evidence_gap_attack")
            if challenge_type not in CRITIC_CHALLENGE_TYPES:
                challenge_type = "evidence_gap_attack"
            normalized.append(
                {
                    "challenge_type": challenge_type,
                    "title": item.get("title", challenge_type),
                    "summary": item.get("summary", ""),
                    "blocking": bool(item.get("blocking", True)),
                    "required_evidence": item.get("required_evidence", ""),
                    "blocking_reason": item.get("blocking_reason", ""),
                }
            )
        return normalized

    def _read_findings(self) -> str:
        if not self.run_dir:
            return ""
        findings_file = self.run_dir / "findings.md"
        if findings_file.exists():
            return findings_file.read_text(encoding="utf-8")
        return ""

    def _tracked_l2_files(self) -> list[str]:
        return [
            "findings.md",
            "plan.md",
            "variable_memory.json",
            "routing_state.json",
            "hypothesis_archive.jsonl",
            "falsification_log.jsonl",
        ]

    def _round_trace_relpath(self) -> str:
        return str(Path("history") / f"round{self.round_num}" / "trace.md")

    def _completion_policy_bool(self, key: str, default: bool) -> bool:
        value = self.completion_policy.get(key, default)
        if isinstance(value, bool):
            return value
        return bool(value)

    def _required_l2_files(self) -> list[str]:
        value = self.completion_policy.get("required_l2_files")
        if isinstance(value, list):
            files = [str(item) for item in value if item]
            if files:
                return files
        return list(DEFAULT_REQUIRED_L2_FILES)

    def _machine_state_files(self) -> list[str]:
        value = self.completion_policy.get("machine_state_any_of")
        if isinstance(value, list):
            files = [str(item) for item in value if item]
            if files:
                return files
        return list(DEFAULT_MACHINE_STATE_FILES)

    def _task_protocol(self) -> dict[str, Any]:
        protocol = self.task_contract.get("protocol", {})
        return protocol if isinstance(protocol, dict) else {}

    def _task_evidence_policy(self) -> str:
        evidence_policy = str(self._task_protocol().get("evidence_policy", "advisory")).strip().lower()
        return evidence_policy if evidence_policy in {"advisory", "blocking"} else "advisory"

    def _task_required_evidence(self) -> dict[str, list[str]]:
        required_evidence = self._task_protocol().get("required_evidence", {})
        if not isinstance(required_evidence, dict):
            return {}

        normalized: dict[str, list[str]] = {}
        for capability, evidence_names in required_evidence.items():
            capability_key = str(capability).strip()
            if not capability_key:
                continue
            if isinstance(evidence_names, str):
                evidence_names = [evidence_names]
            if not isinstance(evidence_names, list):
                continue
            normalized_names = [str(name).strip() for name in evidence_names if str(name).strip()]
            if normalized_names:
                normalized[capability_key] = normalized_names
        return normalized

    def _tracked_completion_artifacts(self) -> list[str]:
        tracked = self._tracked_l2_files()
        if self._completion_policy_bool("require_trace_update", True):
            tracked.append(self._round_trace_relpath())
        return list(dict.fromkeys(tracked))

    def _round_dir(self) -> Path:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        return self.run_dir / "history" / f"round{self.round_num}"

    def _round_scripts_relpath(self) -> str:
        return str(Path("history") / f"round{self.round_num}" / "scripts" / "*")

    def _round_results_relpath(self) -> str:
        return str(Path("history") / f"round{self.round_num}" / "results" / "*")

    def _round_file_inventory(self, dirname: str) -> list[str]:
        if not self.run_dir:
            return []
        root = self._round_dir() / dirname
        return [str(path.relative_to(self.run_dir)) for path in _round_file_inventory(root)]

    def _read_round_trace_text(self) -> str:
        if not self.run_dir:
            return ""
        trace_path = self.run_dir / self._round_trace_relpath()
        if not trace_path.exists():
            return ""
        return trace_path.read_text(encoding="utf-8")

    def _trace_section_lines(self, trace_text: str, heading: str) -> list[str]:
        lines = trace_text.splitlines()
        capture = False
        collected: list[str] = []
        for line in lines:
            if line.strip() == heading:
                capture = True
                continue
            if capture and line.startswith("### "):
                break
            if capture:
                collected.append(line)
        return collected

    def _trace_section_has_table_row(self, trace_text: str, heading: str) -> bool:
        lines = self._trace_section_lines(trace_text, heading)
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("（") and stripped.endswith("）"):
                continue
            if stripped.startswith("|------"):
                continue
            if heading == TRACE_METRICS_HEADING and stripped.startswith("| 实验 | 方法 |"):
                continue
            if heading == TRACE_FALSIFICATION_HEADING and stripped.startswith("| 实验 | 比较候选 |"):
                continue
            if stripped.startswith("|") and stripped.endswith("|"):
                return True
        return False

    def _contract_evidence_artifact_label(self, evidence_name: str) -> str:
        trace_relpath = self._round_trace_relpath()
        mapping = {
            "plan": "plan.md",
            "findings": "findings.md",
            "trace": trace_relpath,
            "machine_state": "machine_state_any_of",
            "script": self._round_scripts_relpath(),
            "result": self._round_results_relpath(),
            "trace_metrics": f"{trace_relpath}#指标记录",
            "trace_falsification": f"{trace_relpath}#证伪实验",
        }
        return mapping.get(evidence_name, evidence_name)

    def _machine_state_validity_artifact_label(self, name: str) -> str:
        if name.endswith(".jsonl"):
            return f"{name}#valid_jsonl"
        return f"{name}#valid_json"

    def _validate_machine_state_files(self, updated_machine_states: list[str]) -> dict[str, str]:
        if not self.run_dir:
            return {}

        errors: dict[str, str] = {}
        for name in updated_machine_states:
            path = self.run_dir / name
            if not path.exists():
                errors[name] = "file missing after update"
                continue

            try:
                if name.endswith(".jsonl"):
                    _read_jsonl_objects(path)
                else:
                    json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors[name] = str(exc)
        return errors

    def _evaluate_task_contract_evidence(
        self,
        *,
        updated: list[str],
        trace_text: str,
        trace_metrics_logged: bool,
        trace_falsification_logged: bool,
        machine_state_updated: list[str],
        round_scripts: list[str],
        round_results: list[str],
    ) -> dict[str, Any]:
        required_evidence = self._task_required_evidence()
        if not required_evidence:
            return {
                "missing_by_capability": {},
                "missing_artifacts": [],
                "warnings": [],
                "evidence_status": {},
            }

        evidence_status = {
            "plan": "plan.md" in updated,
            "findings": "findings.md" in updated,
            "trace": self._round_trace_relpath() in updated,
            "machine_state": bool(machine_state_updated),
            "script": bool(round_scripts),
            "result": bool(round_results),
            "trace_metrics": trace_metrics_logged,
            "trace_falsification": trace_falsification_logged,
        }
        missing_by_capability: dict[str, list[str]] = {}
        warnings: list[str] = []

        for capability, evidence_names in required_evidence.items():
            missing_for_capability: list[str] = []
            for evidence_name in evidence_names:
                if evidence_name not in SUPPORTED_CONTRACT_EVIDENCE:
                    warnings.append(
                        f"Unsupported task contract evidence `{evidence_name}` for capability `{capability}` ignored."
                    )
                    continue
                if not evidence_status.get(evidence_name, False):
                    missing_for_capability.append(self._contract_evidence_artifact_label(evidence_name))
            if missing_for_capability:
                missing_by_capability[capability] = list(dict.fromkeys(missing_for_capability))

        for capability, missing in missing_by_capability.items():
            warnings.append(
                f"Task contract evidence missing for `{capability}`: {', '.join(missing)}"
            )

        missing_artifacts = [
            artifact
            for missing in missing_by_capability.values()
            for artifact in missing
        ]
        return {
            "missing_by_capability": missing_by_capability,
            "missing_artifacts": list(dict.fromkeys(missing_artifacts)),
            "warnings": warnings,
            "evidence_status": evidence_status,
        }

    def _classify_execution_claims(self, text: str) -> dict[str, bool]:
        if not text:
            return {"fit": False, "integration": False, "falsification": False, "any": False}

        fit_claim = any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in COMPLETED_FIT_PATTERNS)
        integration_claim = any(
            re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in COMPLETED_INTEGRATION_PATTERNS
        )
        falsification_claim = any(
            re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in COMPLETED_FALSIFICATION_PATTERNS
        )

        any_claim = fit_claim or integration_claim or falsification_claim
        if not any_claim:
            return {"fit": False, "integration": False, "falsification": False, "any": False}

        return {
            "fit": fit_claim,
            "integration": integration_claim,
            "falsification": falsification_claim,
            "any": any_claim,
        }

    def _snapshot_completion_artifacts(self) -> dict[str, tuple[int, int]]:
        if not self.run_dir:
            return {}
        snapshot: dict[str, tuple[int, int]] = {}
        for name in self._tracked_completion_artifacts():
            path = self.run_dir / name
            snapshot[name] = _file_fingerprint(path)
        return snapshot

    def _diff_artifacts(self, before: dict[str, tuple[int, int]]) -> dict[str, list[str]]:
        if not self.run_dir:
            return {"updated": [], "unchanged": []}
        updated: list[str] = []
        unchanged: list[str] = []
        for name in before:
            path = self.run_dir / name
            if _file_fingerprint(path) != before.get(name, (0, 0)):
                updated.append(name)
            else:
                unchanged.append(name)
        return {"updated": updated, "unchanged": unchanged}

    def _check_l2_promotion(self, before: dict[str, tuple[int, int]]) -> None:
        if not self.run_dir or not before:
            return
        diff = self._diff_artifacts(before)
        unchanged = [name for name in self._tracked_l2_files() if name in diff["unchanged"]]
        if unchanged:
            self.logger.warning(
                "[Round %s] L2 files not updated: %s. Hamilton may have skipped promotion.",
                self.round_num,
                unchanged,
            )

    def _rewrite_text_if_changed(self, path: Path, text: str) -> None:
        if not path.exists():
            return
        current = path.read_text(encoding="utf-8")
        if current != text:
            path.write_text(text, encoding="utf-8")

    def _collapse_blank_lines(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\\n", "\n")
        return re.sub(r"\n{3,}", "\n\n", text).rstrip() + "\n"

    def _sanitize_plan_text(self, text: str) -> str:
        text = self._collapse_blank_lines(text)
        heading = "## 当前路由策略"
        next_heading = "## 当前假设"
        if heading not in text or next_heading not in text:
            return text

        before, remainder = text.split(heading, 1)
        section, after = remainder.split(next_heading, 1)
        bullet_lines: list[str] = []
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("- "):
                bullet_lines.append(line)

        if not bullet_lines:
            return text

        latest_by_key: dict[str, str] = {}
        observed_order: list[str] = []
        for line in bullet_lines:
            content = line[2:].strip()
            separator = "：" if "：" in content else ":" if ":" in content else None
            key = content.split(separator, 1)[0].strip() if separator else content
            if key not in observed_order:
                observed_order.append(key)
            latest_by_key[key] = f"- {content}"

        preferred_order = [
            "当前阶段",
            "选择理由",
            "下一轮优先工具",
            "本轮编号",
            "本轮重点",
            "路由理由",
            "拟合理由",
            "变量分析理由",
            "下一轮任务",
            "次优优先级",
        ]
        rendered_lines = [
            latest_by_key[key]
            for key in [*preferred_order, *observed_order]
            if key in latest_by_key
        ]
        rendered_lines = list(dict.fromkeys(rendered_lines))
        rebuilt = before + heading + "\n" + "\n".join(rendered_lines) + "\n\n" + next_heading + after
        return self._collapse_blank_lines(rebuilt)

    def _sanitize_markdown_table_section(self, text: str, start_heading: str, end_heading: str) -> str:
        if start_heading not in text or end_heading not in text:
            return text

        before, remainder = text.split(start_heading, 1)
        section, after = remainder.split(end_heading, 1)
        prefix_lines: list[str] = []
        header_lines: list[str] = []
        row_map: dict[str, str] = {}
        row_order: list[str] = []

        for raw_line in section.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("|"):
                if len(header_lines) < 2:
                    header_lines.append(line)
                    continue
                cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
                key = cells[0] if cells else stripped
                if key not in row_order:
                    row_order.append(key)
                row_map[key] = line
            else:
                prefix_lines.append(line)

        rebuilt_lines = [*prefix_lines, *header_lines, *(row_map[key] for key in row_order)]
        rebuilt_section = ("\n".join(rebuilt_lines).strip() + "\n") if rebuilt_lines else ""
        rebuilt = before + start_heading + "\n" + rebuilt_section + "\n" + end_heading + after
        return self._collapse_blank_lines(rebuilt)

    def _sanitize_findings_text(self, text: str) -> str:
        text = self._collapse_blank_lines(text)
        text = self._sanitize_markdown_table_section(text, "## 实验结果", "## 证伪记录")
        text = self._sanitize_markdown_table_section(text, "## 证伪记录", "## 最优方程演化")
        return self._collapse_blank_lines(text)

    def _sanitize_jsonl_text(self, text: str) -> str:
        seen: set[str] = set()
        lines: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            lines.append(stripped)
        return ("\n".join(lines) + ("\n" if lines else ""))

    def _normalize_updated_workspace_state(self, updated_candidates: list[str]) -> None:
        if not self.run_dir or not updated_candidates:
            return

        for relpath in updated_candidates:
            path = self.run_dir / relpath
            if not path.exists() or not path.is_file():
                continue
            if relpath == "plan.md":
                self._rewrite_text_if_changed(path, self._sanitize_plan_text(path.read_text(encoding="utf-8")))
            elif relpath == "findings.md":
                self._rewrite_text_if_changed(path, self._sanitize_findings_text(path.read_text(encoding="utf-8")))
            elif relpath.endswith(".jsonl"):
                self._rewrite_text_if_changed(path, self._sanitize_jsonl_text(path.read_text(encoding="utf-8")))

    def _read_artifact_text(self, relpath: str) -> str:
        if not self.run_dir:
            return ""
        path = self.run_dir / relpath
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _normalize_claimed_artifact_ref(self, raw_path: str) -> str | None:
        raw = raw_path.strip().strip("`'\"),.;:，。；）]")
        if not raw:
            return None
        normalized = raw.lstrip("./")
        if normalized.startswith("history/round"):
            return normalized if Path(normalized).suffix else None
        if normalized.startswith("scripts/") or normalized.startswith("results/"):
            candidate = Path("history") / f"round{self.round_num}" / normalized
            return str(candidate) if candidate.suffix else None
        return normalized if Path(normalized).suffix else None

    def _extract_claimed_artifact_refs(self, *texts: str) -> list[str]:
        refs: list[str] = []
        for text in texts:
            if not text:
                continue
            for match in CLAIMED_ROUND_ARTIFACT_RE.finditer(text):
                normalized = self._normalize_claimed_artifact_ref(match.group("path"))
                if normalized:
                    refs.append(normalized)
        return list(dict.fromkeys(refs))

    def _extract_quantified_result_claims(self, *texts: str) -> list[str]:
        claims: list[str] = []
        for text in texts:
            if not text:
                continue
            for pattern in QUANTIFIED_RESULT_PATTERNS:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 60)
                    snippet = " ".join(text[start:end].split())
                    if snippet:
                        claims.append(snippet[:220])
        return list(dict.fromkeys(claims))

    def _evaluate_claim_evidence_binding(
        self,
        *,
        updated: list[str],
        signal: dict[str, Any],
        round_scripts: list[str],
        round_results: list[str],
    ) -> dict[str, Any]:
        artifact_texts = {name: self._read_artifact_text(name) for name in updated}
        finish_message = signal.get("finish_message", "")
        all_texts = [finish_message, *artifact_texts.values()]
        claimed_artifact_refs = self._extract_claimed_artifact_refs(*all_texts)
        missing_claimed_artifacts = [
            ref for ref in claimed_artifact_refs if self.run_dir and not (self.run_dir / ref).exists()
        ]
        quantified_claims = self._extract_quantified_result_claims(*all_texts)

        missing_artifacts: list[str] = []
        warnings: list[str] = []
        if missing_claimed_artifacts:
            missing_artifacts.extend(missing_claimed_artifacts)
            warnings.append(
                "Claimed script/result artifacts are missing: "
                + ", ".join(f"`{item}`" for item in missing_claimed_artifacts)
            )
        if quantified_claims and not round_results:
            missing_artifacts.append(self._round_results_relpath())
            warnings.append(
                "Quantified claims were recorded without any round result artifacts: "
                + " | ".join(quantified_claims[:3])
            )

        return {
            "claimed_artifact_refs": claimed_artifact_refs,
            "missing_claimed_artifacts": list(dict.fromkeys(missing_claimed_artifacts)),
            "quantified_claims": quantified_claims,
            "warnings": warnings,
            "missing_artifacts": list(dict.fromkeys(missing_artifacts)),
            "round_scripts": round_scripts,
            "round_results": round_results,
            "artifact_texts": artifact_texts,
        }

    def _build_hamilton_task_description(self, task_description: str, repair_instruction: str) -> str:
        if not repair_instruction:
            return task_description
        return (
            f"{task_description}\n\n"
            "## System Completion Gate Feedback\n"
            f"{repair_instruction}\n"
        )

    def _evaluate_hamilton_completion(
        self,
        artifact_snapshot: dict[str, tuple[int, int]],
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        pre_sanitize_diff = self._diff_artifacts(artifact_snapshot)
        self._normalize_updated_workspace_state(pre_sanitize_diff["updated"])
        diff = self._diff_artifacts(artifact_snapshot)
        updated = diff["updated"]
        require_artifact_updates = self._completion_policy_bool("enforce_artifact_updates", True)
        require_trace_update = self._completion_policy_bool("require_trace_update", True)
        require_machine_state_update = self._completion_policy_bool("require_machine_state_update", True)
        enforce_on_any_finish = self._completion_policy_bool("enforce_artifact_updates_on_any_finish", True)
        finish_called = signal.get("finish_called", False)
        task_completed = signal.get("task_completed") == "true"

        required_l2 = self._required_l2_files()
        missing_required = [name for name in required_l2 if name not in updated]
        trace_relpath = self._round_trace_relpath()
        trace_updated = trace_relpath in updated
        machine_state_updated = [name for name in self._machine_state_files() if name in updated]
        invalid_machine_state_files = self._validate_machine_state_files(machine_state_updated)
        valid_machine_state_updates = [
            name for name in machine_state_updated if name not in invalid_machine_state_files
        ]
        trace_text = self._read_round_trace_text()
        trace_metrics_logged = self._trace_section_has_table_row(trace_text, TRACE_METRICS_HEADING)
        trace_falsification_logged = self._trace_section_has_table_row(trace_text, TRACE_FALSIFICATION_HEADING)
        round_scripts = self._round_file_inventory("scripts")
        round_results = self._round_file_inventory("results")
        execution_claims_debug = self._classify_execution_claims(signal.get("finish_message", ""))
        evidence_policy = self._task_evidence_policy()
        required_evidence = self._task_required_evidence()
        contract_evidence = self._evaluate_task_contract_evidence(
            updated=updated,
            trace_text=trace_text,
            trace_metrics_logged=trace_metrics_logged,
            trace_falsification_logged=trace_falsification_logged,
            machine_state_updated=valid_machine_state_updates,
            round_scripts=round_scripts,
            round_results=round_results,
        )

        missing_minimal_artifacts: list[str] = []
        should_enforce = require_artifact_updates and finish_called and (enforce_on_any_finish or task_completed)
        if should_enforce:
            missing_minimal_artifacts.extend(missing_required)
            if require_trace_update and not trace_updated:
                missing_minimal_artifacts.append(trace_relpath)
            if require_machine_state_update and not valid_machine_state_updates:
                missing_minimal_artifacts.append("machine_state_any_of")
            if invalid_machine_state_files:
                missing_minimal_artifacts.extend(
                    self._machine_state_validity_artifact_label(name)
                    for name in invalid_machine_state_files
                )

        enforce_task_contract_evidence = self._completion_policy_bool("enforce_task_contract_evidence", True)
        missing_contract_evidence: list[str] = []
        advisory_warnings: list[str] = []
        if should_enforce and enforce_task_contract_evidence and required_evidence:
            if evidence_policy == "blocking":
                missing_contract_evidence = contract_evidence["missing_artifacts"]
            else:
                advisory_warnings = contract_evidence["warnings"]
        elif required_evidence:
            advisory_warnings = contract_evidence["warnings"]

        evidence_binding = self._evaluate_claim_evidence_binding(
            updated=updated,
            signal=signal,
            round_scripts=round_scripts,
            round_results=round_results,
        )

        missing_artifacts = list(
            dict.fromkeys(
                [
                    *missing_minimal_artifacts,
                    *missing_contract_evidence,
                    *evidence_binding["missing_artifacts"],
                ]
            )
        )
        accepted = not should_enforce or not missing_artifacts
        reason = "completion_gate_passed"
        if not accepted:
            if evidence_binding["missing_artifacts"]:
                reason = "finish_rejected_until_claimed_results_are_backed_by_real_artifacts"
            elif missing_contract_evidence:
                reason = "finish_rejected_until_task_contract_evidence_is_recorded"
            else:
                reason = "finish_rejected_until_minimal_round_artifacts_are_updated"
        return {
            "accepted": accepted,
            "finish_called": finish_called,
            "task_completed_claimed": task_completed,
            "updated": updated,
            "unchanged": diff["unchanged"],
            "missing_required_l2": missing_required,
            "trace_updated": trace_updated,
            "trace_metrics_logged": trace_metrics_logged,
            "trace_falsification_logged": trace_falsification_logged,
            "machine_state_candidates": self._machine_state_files(),
            "machine_state_updated": valid_machine_state_updates,
            "machine_state_updated_raw": machine_state_updated,
            "invalid_machine_state_files": invalid_machine_state_files,
            "round_scripts": round_scripts,
            "round_results": round_results,
            "task_contract_evidence_policy": evidence_policy,
            "required_evidence": required_evidence,
            "missing_contract_evidence": missing_contract_evidence,
            "missing_contract_evidence_by_capability": contract_evidence["missing_by_capability"],
            "contract_evidence_status": contract_evidence["evidence_status"],
            "execution_claims_debug": execution_claims_debug,
            "claim_evidence_binding": {
                "claimed_artifact_refs": evidence_binding["claimed_artifact_refs"],
                "missing_claimed_artifacts": evidence_binding["missing_claimed_artifacts"],
                "quantified_claims": evidence_binding["quantified_claims"],
            },
            "missing_minimal_artifacts": list(dict.fromkeys(missing_minimal_artifacts)),
            "missing_artifacts": missing_artifacts,
            "reason": reason,
            "advisory_warnings": list(dict.fromkeys([*advisory_warnings, *evidence_binding["warnings"]])),
        }

    def _build_hamilton_repair_instruction(self, completion_gate: dict[str, Any], result: str) -> str:
        missing = completion_gate.get("missing_artifacts", [])
        missing_lines = "\n".join(f"- `{name}`" for name in missing) if missing else "- `(unknown)`"
        updated = completion_gate.get("updated", [])
        updated_lines = "\n".join(f"- `{name}`" for name in updated) if updated else "- `(none)`"
        invalid_machine_states = completion_gate.get("invalid_machine_state_files", {})
        invalid_machine_state_lines = (
            "\n".join(f"- `{name}`: {reason}" for name, reason in invalid_machine_states.items())
            if invalid_machine_states
            else "- `(none)`"
        )
        evidence_policy = completion_gate.get("task_contract_evidence_policy", "advisory")
        required_evidence = completion_gate.get("required_evidence", {})
        required_evidence_lines = (
            "\n".join(
                f"- `{capability}` -> {', '.join(f'`{name}`' for name in evidence_names)}"
                for capability, evidence_names in required_evidence.items()
            )
            if required_evidence
            else "- `(none)`"
        )
        machine_state_candidates = completion_gate.get("machine_state_candidates", [])
        machine_state_candidate_lines = (
            "\n".join(f"- `{name}`" for name in machine_state_candidates)
            if machine_state_candidates
            else "- `(none)`"
        )
        missing_contract_evidence = completion_gate.get("missing_contract_evidence", [])
        missing_contract_lines = (
            "\n".join(f"- `{name}`" for name in missing_contract_evidence)
            if missing_contract_evidence
            else "- `(none)`"
        )
        claim_binding = completion_gate.get("claim_evidence_binding", {})
        missing_claimed_artifacts = claim_binding.get("missing_claimed_artifacts", [])
        missing_claimed_artifact_lines = (
            "\n".join(f"- `{name}`" for name in missing_claimed_artifacts)
            if missing_claimed_artifacts
            else "- `(none)`"
        )
        quantified_claims = claim_binding.get("quantified_claims", [])
        quantified_claim_lines = (
            "\n".join(f"- `{snippet}`" for snippet in quantified_claims[:5])
            if quantified_claims
            else "- `(none)`"
        )
        advisory_warnings = completion_gate.get("advisory_warnings", [])
        advisory_warning_lines = (
            "\n".join(f"- {warning}" for warning in advisory_warnings)
            if advisory_warnings
            else "- `(none)`"
        )
        return (
            "Your previous `finish(...)` was rejected because the required round artifacts were not actually "
            "updated with tool calls, or because the blocking task contract evidence was still missing.\n\n"
            "Files that are still missing required updates:\n"
            f"{missing_lines}\n\n"
            "Files that have been updated so far in this round:\n"
            f"{updated_lines}\n\n"
            "Machine-readable state files that are still malformed and must be repaired:\n"
            f"{invalid_machine_state_lines}\n\n"
            "Current task contract evidence policy:\n"
            f"- `{evidence_policy}`\n\n"
            "Current task contract required evidence:\n"
            f"{required_evidence_lines}\n\n"
            "Any one of these machine-readable state files can satisfy the machine-state requirement:\n"
            f"{machine_state_candidate_lines}\n\n"
            "Blocking task contract evidence still missing:\n"
            f"{missing_contract_lines}\n\n"
            "Claimed script/result artifacts that still do not exist:\n"
            f"{missing_claimed_artifact_lines}\n\n"
            "Quantified result claims that currently lack backing result artifacts:\n"
            f"{quantified_claim_lines}\n\n"
            "Current advisory warnings:\n"
            f"{advisory_warning_lines}\n\n"
            "These files already exist because the system created the workspace scaffold. Use "
            "`str_replace_editor` with `view`, `str_replace`, or `insert` to modify the existing files, and use "
            "`execute_bash` to run scripts when you need real computation. If `task_contract.json` declares "
            "blocking evidence, you must leave the corresponding scripts/results/trace rows before finishing.\n"
            "Before choosing Python packages or claiming that a script ran successfully, read "
            f"`{ENV_CAPABILITIES_FILE}`. Only assume packages listed there as available. If a dependency is "
            "missing, either switch to a fallback that the environment supports or record the blocker explicitly.\n"
            "Do not write numeric metrics, coefficient values, or result-file paths into `findings.md`, "
            "`plan.md`, machine-readable state, or `finish.message` unless the corresponding result artifact "
            "already exists in this round.\n"
            "When editing machine-readable state files, keep them valid JSON or JSONL. If you updated a state file "
            "incorrectly, repair that exact file before calling `finish` again.\n"
            "Prefer replacing or tightening the current round summary instead of appending repeated sections over "
            "and over. Keep `plan.md` and `findings.md` compact and auditable.\n"
            "Do not invent `/workspace/...` paths if the current workspace root is different; prefer the actual "
            "working directory shown by the system, or confirm it with `execute_bash pwd` before editing.\n"
            "Do not use `create` on an existing file, and do not merely describe the edits in prose. Re-read "
            "the current workspace if needed, make the actual edits, then call "
            "`finish` again.\n\n"
            "Previous finish summary:\n"
            f"{_truncate(result, 900)}"
        )

    def _extract_agent_response(self, trajectory) -> str:
        return super()._extract_agent_response(trajectory)

    def _parse_signal(self, agent_message: str, trajectory, actor: str) -> dict[str, Any]:
        task_completed = self._extract_task_completed(trajectory)
        finish_called = self._has_finish_tool_call(trajectory)
        if task_completed is None:
            self.logger.warning(
                "[Round %s] Could not extract task_completed from %s trajectory; defaulting to false.",
                self.round_num,
                actor,
            )
            task_completed = "false"

        finish_message = self._extract_finish_message_from_trajectory(trajectory)
        satisfied = task_completed == "true"
        return {
            "round": self.round_num,
            "actor": actor,
            "satisfied": satisfied,
            "task_completed": task_completed,
            "finish_called": finish_called,
            "finish_message": finish_message or agent_message,
            "notes": _truncate(finish_message or agent_message, 500),
            "approved": satisfied if actor == "critic" else None,
        }

    def _aggregate_signals(
        self,
        hamilton_signal: dict[str, Any],
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
    ) -> dict[str, Any]:
        critic_blocking = bool(critic_report.get("blocking", not critic_signal.get("approved", False)))
        critic_approved = bool(critic_signal.get("approved", False)) and not critic_blocking
        unresolved = [
            challenge["challenge_type"]
            for challenge in critic_report.get("challenges", [])
            if challenge.get("blocking", True)
        ]
        completion_gate = hamilton_signal.get("completion_gate") or {}
        if not completion_gate.get("accepted", True):
            unresolved = ["hamilton_artifact_gate", *unresolved]
        satisfied = bool(hamilton_signal.get("satisfied", False)) and critic_approved and not unresolved
        return {
            "round": self.round_num,
            "satisfied": satisfied,
            "task_completed": "true" if satisfied else "false",
            "hamilton_satisfied": bool(hamilton_signal.get("satisfied", False)),
            "critic_approved": critic_approved,
            "blocked_by": unresolved,
            "missing_artifacts": completion_gate.get("missing_artifacts", []),
            "advisory_warnings": completion_gate.get("advisory_warnings", []),
            "task_contract_evidence_policy": completion_gate.get("task_contract_evidence_policy", "advisory"),
            "notes": _truncate(
                " | ".join(
                    part
                    for part in (
                        hamilton_signal.get("notes", ""),
                        critic_report.get("summary", ""),
                    )
                    if part
                ),
                700,
            ),
        }

    def _update_debate_state(
        self,
        hamilton_signal: dict[str, Any],
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
        aggregated_signal: dict[str, Any],
    ) -> None:
        if not self.run_dir:
            return

        state_path = self.run_dir / DEBATE_STATE_FILE
        state = _read_json(
            state_path,
            {
                "task_id": "",
                "task_hash": "",
                "unresolved_challenges": [],
                "resolved_challenges": [],
                "rounds": [],
                "updated_at": _utc_now(),
            },
        )

        challenges = critic_report.get("challenges", [])
        unresolved = []
        resolved = []
        for challenge in challenges:
            challenge_copy = dict(challenge)
            challenge_copy["round"] = self.round_num
            if challenge_copy.get("blocking", True):
                unresolved.append(challenge_copy)
            else:
                resolved.append(challenge_copy)

        if aggregated_signal.get("satisfied", False):
            state["resolved_challenges"] = state.get("resolved_challenges", []) + unresolved + resolved
            state["unresolved_challenges"] = []
        else:
            state["resolved_challenges"] = state.get("resolved_challenges", []) + resolved
            state["unresolved_challenges"] = unresolved

        state.setdefault("rounds", [])
        state["rounds"].append(
            {
                "round": self.round_num,
                "hamilton_signal": hamilton_signal,
                "critic_signal": critic_signal,
                "aggregated_signal": aggregated_signal,
                "critic_report": {
                    "summary": critic_report.get("summary", ""),
                    "blocking": critic_report.get("blocking", True),
                    "challenge_types": [item.get("challenge_type") for item in challenges],
                },
                "updated_at": _utc_now(),
            }
        )
        state["updated_at"] = _utc_now()
        _write_json(state_path, state)

    def _extract_task_completed(self, trajectory) -> str | None:
        try:
            steps = getattr(trajectory, "steps", None)
            if not isinstance(steps, list):
                return None
            for step in reversed(steps):
                assistant_message = getattr(step, "assistant_message", None)
                tool_calls = getattr(assistant_message, "tool_calls", None)
                if not tool_calls:
                    continue
                for tc in reversed(tool_calls):
                    fn = getattr(tc, "function", None)
                    if not fn or getattr(fn, "name", None) != "finish":
                        continue
                    args = getattr(fn, "arguments", "") or ""
                    try:
                        parsed = json.loads(args) if isinstance(args, str) and args.strip() else {}
                    except Exception:
                        return None
                    if isinstance(parsed, dict):
                        return parsed.get("task_completed")
        except Exception:
            return None
        return None

    def _has_finish_tool_call(self, trajectory) -> bool:
        try:
            steps = getattr(trajectory, "steps", None)
            if not isinstance(steps, list):
                return False
            for step in reversed(steps):
                assistant_message = getattr(step, "assistant_message", None)
                tool_calls = getattr(assistant_message, "tool_calls", None)
                if not tool_calls:
                    continue
                for tc in reversed(tool_calls):
                    fn = getattr(tc, "function", None)
                    if fn and getattr(fn, "name", None) == "finish":
                        return True
        except Exception:
            return False
        return False

    def _extract_finish_message_from_trajectory(self, trajectory) -> str:
        try:
            steps = getattr(trajectory, "steps", None)
            if not isinstance(steps, list):
                return ""
            for step in reversed(steps):
                assistant_message = getattr(step, "assistant_message", None)
                tool_calls = getattr(assistant_message, "tool_calls", None)
                if not tool_calls:
                    continue
                for tc in reversed(tool_calls):
                    fn = getattr(tc, "function", None)
                    if not fn or getattr(fn, "name", None) != "finish":
                        continue
                    args = getattr(fn, "arguments", "") or ""
                    try:
                        parsed = json.loads(args) if isinstance(args, str) and args.strip() else {}
                    except Exception:
                        return args
                    if isinstance(parsed, dict):
                        msg = parsed.get("message")
                        if isinstance(msg, str):
                            return msg
                        return json.dumps(parsed, ensure_ascii=False)
                    return str(parsed)
        except Exception:
            return ""
        return ""
