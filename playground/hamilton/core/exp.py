"""Hamilton Round Exp - proposer/critic round orchestration."""

from __future__ import annotations

import csv
import hashlib
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
    CRITIC_ATTACK_LOG_JSONL,
    CRITIC_ATTACK_PLAN_JSON,
    CRITIC_ATTACKS_DIR,
    CRITIC_CHALLENGE_TYPES,
    CRITIC_INTERVENTION_TYPES,
    CRITIC_REPORT_JSON,
    CRITIC_REPORT_MD,
    CRITIC_SCHEDULER_STATE_FILE,
    DEBATE_STATE_FILE,
    ENV_CAPABILITIES_FILE,
    EVALUATION_CONTEXT_FILE,
    HCC_LEDGER_FILE,
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

STRONG_CONCLUSION_PATTERNS = (
    r"(支持集|support[\s_-]*set).{0,24}(稳定|收敛|confirmed|stable|确定)",
    r"(方程|equation|structure).{0,32}(为|=|≈|confirmed|identified|candidate)",
    r"(系数|coefficient|coefficients|ω²|omega\^2).{0,24}(=|≈|约)\s*[-+]?\d",
)

CHALLENGE_KEYWORDS = {
    "support_set_attack": ("support", "支持集", "ablation", "消融", "redund", "proxy"),
    "structure_attack": ("structure", "结构", "equation", "方程", "operator"),
    "ood_generalization_attack": ("ood", "泛化", "cross-condition", "工况", "transfer"),
    "physics_consistency_attack": ("physics", "物理", "monotonic", "symmetry", "守恒", "dimension"),
    "numerical_stability_attack": ("stable", "stability", "rollout", "trajectory", "积分", "振幅", "phase"),
    "evidence_gap_attack": ("evidence", "validation", "结果", "证据", "metrics"),
}

CLAIMED_ROUND_ARTIFACT_RE = re.compile(
    r"(?P<path>(?:history/round\d+/(?:scripts|results)/[^\s`'\"，,；;。:：()（）【】]+|(?:scripts|results)/[^\s`'\"，,；;。:：()（）【】]+))"
)

QUANTIFIED_RESULT_PATTERNS = (
    r"(R²|R\^2|fit_R2|amplitude_error|RMSE|MAE|MSE|loss|accuracy|precision|recall|F1|AUC)\s*(?:[:=≈<>]|约)?\s*[-+]?\d",
    r"(稳态振幅|振幅误差|相对误差|系数表|系数趋势|趋势图|coefficient table).{0,16}(≈|=|约)\s*[-+]?\d",
    r"(ω²|omega\^2|系数|coefficient|coefficients).{0,16}(≈|=|约)\s*[-+]?\d",
)

SAFE_VARIABLE_MEMORY_TOP_LEVEL_KEYS = {
    "variables",
    "last_updated_round",
    "notes",
}

SAFE_VARIABLE_MEMORY_VARIABLE_KEYS = {
    "role",
    "unit",
    "notes",
    "evidence",
    "is_redundant",
    "is_proxy",
    "is_target",
    "redundant",
    "proxy",
    "target",
}

UNSAFE_VARIABLE_MEMORY_NOTE_PATTERNS = (
    r"(拟合|回归|fit|equation|方程|support[\s_-]*set|支持集|结构一致|结构稳定|coefficient|coefficients|系数)",
    r"(极限环|limit[\s_-]*cycle|amplitude|振幅|phase|frequency|rollout|积分|仿真|simulation)",
    r"(物理预期|physics|单调变化|跨风速一致|风速变化|critic|攻击|证伪|消融)",
    r"(restoring[\s_-]*force|damping|self[\s_-]*excitation|negative[\s_-]*damping|positive[\s_-]*damping)",
)

UNSAFE_VARIABLE_MEMORY_EVIDENCE_PATTERNS = (
    r"(跨风速|风速变化|结构一致|结构稳定|support[\s_-]*set|支持集)",
    r"(系数|coefficient|coefficients|恢复力项|阻尼项|自激|极限环|amplitude|phase|frequency)",
    r"(critic|攻击|证伪|消融|物理解释|物理预期|单调变化)",
    r"(restoring[\s_-]*force|damping|self[\s_-]*excitation|negative[\s_-]*damping|positive[\s_-]*damping)",
)

PLANNED_ARTIFACT_CONTEXT_PATTERNS = (
    r"(下一轮|下一步|后续|计划|待验证|待生成|建议|优先|将要|将会|准备)",
    r"(need to|next round|planned|plan to|will run|to be generated|follow-up)",
    r"(next_tools|next_action|next_stage|建议先执行|下一轮优先工具|当前仍缺|缺少的证据文件)",
)

DEFAULT_CRITIC_IMMEDIATE_TRIGGERS = (
    "hamilton_finish_true",
    "new_results_artifact",
    "quantified_result_claim",
    "strong_conclusion_text",
    "challenge_response",
)

COEF_TABLE_FILENAMES = {
    "coef_table.csv",
    "coefficients.csv",
    "coefficients_table.csv",
}

ROLLOUT_RESULT_FILENAMES = {
    "amplitude_error.json",
    "rollout_metrics.csv",
}

ABLATION_RESULT_FILENAMES = {
    "support_ablation.json",
    "support_ablation_no_v3v5.json",
}

ABLATION_ROLLOUT_RESULT_FILENAMES = {
    "ablation_amplitude_error.json",
    "ablation_rollout.json",
    "ablation_rollout_metrics.csv",
}

SUPPORT_SET_TOKEN_ALIASES = {
    "x": "x",
    "x3": "x^3",
    "x^3": "x^3",
    "v": "v",
    "v3": "v^3",
    "v^3": "v^3",
    "v5": "v^5",
    "v^5": "v^5",
}


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
        """Execute one Hamilton round, then trigger Critic conditionally."""
        self.logger.info("Starting Round %s", self.round_num)

        self._ensure_round_dirs()
        self._init_trace()

        artifact_snapshot = self._snapshot_completion_artifacts()

        hamilton_result, hamilton_signal, hamilton_trajectory = self._run_hamilton(
            task_description,
            task_id,
            artifact_snapshot,
        )

        # L2 promotion is still Hamilton's responsibility; critic only audits.
        self._check_l2_promotion(artifact_snapshot)
        critic_schedule = self._evaluate_critic_schedule(hamilton_signal)

        critic_result = ""
        critic_signal = self._build_skipped_critic_signal(critic_schedule)
        critic_trajectory = None
        critic_report = self._build_skipped_critic_report(critic_schedule)

        if critic_schedule.get("run_critic", False):
            self._init_critic_artifacts()
            critic_result, critic_signal, critic_trajectory = self._run_critic(
                task_description=task_description,
                task_id=task_id,
                hamilton_result=hamilton_result,
                hamilton_signal=hamilton_signal,
                critic_schedule=critic_schedule,
            )
            critic_report = self._load_critic_report(critic_signal=critic_signal)
            critic_signal = self._align_critic_signal_with_report(critic_signal, critic_report)

        aggregated_signal = self._aggregate_signals(
            hamilton_signal,
            critic_signal,
            critic_report,
            critic_schedule=critic_schedule,
        )
        self._update_debate_state(
            hamilton_signal,
            critic_signal,
            critic_report,
            aggregated_signal,
            critic_schedule=critic_schedule,
        )
        self._update_critic_scheduler_state(critic_schedule, critic_report, aggregated_signal)
        self._append_hcc_ledger_entries(
            hamilton_signal=hamilton_signal,
            critic_signal=critic_signal,
            critic_report=critic_report,
            aggregated_signal=aggregated_signal,
            critic_schedule=critic_schedule,
        )
        self._sync_round_fact_ledgers(
            hamilton_signal=hamilton_signal,
            critic_signal=critic_signal,
            critic_report=critic_report,
            aggregated_signal=aggregated_signal,
            critic_schedule=critic_schedule,
        )

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
            "critic_schedule": critic_schedule,
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
            self._canonicalize_routing_state()
            routing_state = _read_json(self.run_dir / "routing_state.json", {}) if self.run_dir else {}
            current_strategy = str(routing_state.get("current_strategy", "analytic_fit") or "analytic_fit").strip()
            current_focus = str(routing_state.get("current_focus", "") or "").strip()
            next_tools = routing_state.get("next_tools", []) if isinstance(routing_state, dict) else []
            recommended_command = self._recommended_command_for_strategy(current_strategy, next_tools)
            primary_artifact = self._primary_artifact_for_strategy(current_strategy)
            stage_input_artifact = self._stage_input_artifact_for_strategy(current_strategy)
            stage_artifact_missing = not (self.run_dir / primary_artifact).exists() if self.run_dir else True

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
                    "current_strategy": current_strategy,
                    "current_focus": current_focus,
                    "recommended_command": recommended_command,
                    "primary_artifact": primary_artifact,
                    "stage_input_artifact": stage_input_artifact or "无（需先生成）",
                    "stage_artifact_missing": "true" if stage_artifact_missing else "false",
                },
            )
            trajectory = self.hamilton_agent.run(task)
            result = self._extract_agent_response(trajectory)
            signal = self._parse_signal(result, trajectory, actor="hamilton")
            signal["executed_strategy"] = current_strategy
            signal["current_focus"] = current_focus
            signal["recommended_command"] = recommended_command
            signal["stage_input_artifact"] = stage_input_artifact or ""
            signal["primary_artifact"] = primary_artifact
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
        critic_schedule: dict[str, Any],
    ) -> tuple[str, dict[str, Any], Any]:
        BaseAgent.set_exp_info(exp_name=f"{self.exp_name}_Critic", exp_index=self.round_num)
        self.logger.info("[Round %s] Running Critic challenger...", self.round_num)
        report_md, report_json = self._critic_report_paths()
        attack_plan_path, attack_log_path, attack_scripts_dir, attack_results_dir = self._critic_attack_paths()
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
                "evaluation_context_file": EVALUATION_CONTEXT_FILE,
                "hcc_ledger_file": HCC_LEDGER_FILE,
                "critic_scheduler_state_file": CRITIC_SCHEDULER_STATE_FILE,
                "critic_attack_plan_json": str(attack_plan_path.relative_to(self.run_dir)),
                "critic_attack_log_jsonl": str(attack_log_path.relative_to(self.run_dir)),
                "critic_attack_scripts_dir": str(attack_scripts_dir.relative_to(self.run_dir)),
                "critic_attack_results_dir": str(attack_results_dir.relative_to(self.run_dir)),
                "critic_schedule_reasons": ", ".join(critic_schedule.get("trigger_reasons", [])),
                "critic_execution_mode": self._critic_execution_mode(),
                "critic_max_interventions": str(self._critic_max_interventions()),
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
        l1_template = f"""<!-- EVO_SCAFFOLD_OVERWRITABLE -->
# 执行日志 — 第 {self.round_num} 轮

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
        attack_plan_path, attack_log_path, attack_scripts_dir, attack_results_dir = self._critic_attack_paths()
        attack_plan_path.parent.mkdir(parents=True, exist_ok=True)
        attack_log_path.parent.mkdir(parents=True, exist_ok=True)
        attack_scripts_dir.mkdir(parents=True, exist_ok=True)
        attack_results_dir.mkdir(parents=True, exist_ok=True)
        if not report_md.exists():
            report_md.write_text(
                "<!-- EVO_SCAFFOLD_OVERWRITABLE -->\n# Critic 审核摘要\n\n（本轮待写）\n",
                encoding="utf-8",
            )
        if not report_json.exists():
            _write_json(
                report_json,
                {
                    "__placeholder__": True,
                    "round": self.round_num,
                    "approved": False,
                    "blocking": True,
                    "summary": "",
                    "required_evidence": [],
                    "challenges": [],
                    "attack_execution": {
                        "execution_mode": self._critic_execution_mode(),
                        "executed_interventions": 0,
                        "status": "not_executed",
                    },
                },
            )
        if not attack_plan_path.exists():
            _write_json(
                attack_plan_path,
                {
                    "__placeholder__": True,
                    "round": self.round_num,
                    "execution_mode": self._critic_execution_mode(),
                    "max_interventions": self._critic_max_interventions(),
                    "interventions": [],
                },
            )
        if not attack_log_path.exists():
            attack_log_path.write_text("", encoding="utf-8")

    def _critic_report_paths(self) -> tuple[Path, Path]:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        report_md = round_dir / self.critic_policy.get("report_md_name", CRITIC_REPORT_MD)
        report_json = round_dir / self.critic_policy.get("report_json_name", CRITIC_REPORT_JSON)
        return report_md, report_json

    def _critic_attack_paths(self) -> tuple[Path, Path, Path, Path]:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        attacks_root = round_dir / CRITIC_ATTACKS_DIR
        return (
            round_dir / CRITIC_ATTACK_PLAN_JSON,
            round_dir / CRITIC_ATTACK_LOG_JSONL,
            attacks_root / "scripts",
            attacks_root / "results",
        )

    def _critic_scheduler_state_path(self) -> Path:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        return self.run_dir / CRITIC_SCHEDULER_STATE_FILE

    def _critic_periodic_every_n_rounds(self) -> int:
        raw_value = self.critic_policy.get("periodic_every_n_rounds", 3)
        try:
            value = int(raw_value)
        except Exception:
            value = 3
        return max(1, value)

    def _critic_schedule_mode(self) -> str:
        mode = str(self.critic_policy.get("schedule_mode", "hybrid")).strip().lower()
        return mode if mode in {"hybrid", "event_only", "periodic"} else "hybrid"

    def _critic_execution_mode(self) -> str:
        mode = str(self.critic_policy.get("execution_mode", "light_self_execute")).strip().lower()
        return mode or "light_self_execute"

    def _critic_immediate_triggers(self) -> set[str]:
        raw_value = self.critic_policy.get("immediate_triggers", list(DEFAULT_CRITIC_IMMEDIATE_TRIGGERS))
        if isinstance(raw_value, str):
            raw_value = [raw_value]
        if not isinstance(raw_value, list):
            raw_value = list(DEFAULT_CRITIC_IMMEDIATE_TRIGGERS)
        triggers = {str(item).strip() for item in raw_value if str(item).strip()}
        return triggers or set(DEFAULT_CRITIC_IMMEDIATE_TRIGGERS)

    def _critic_max_interventions(self) -> int:
        raw_value = self.critic_policy.get("max_interventions_per_round", 2)
        try:
            value = int(raw_value)
        except Exception:
            value = 2
        return max(0, value)

    def _load_critic_scheduler_state(self) -> dict[str, Any]:
        path = self._critic_scheduler_state_path()
        state = _read_json(
            path,
            {
                "last_critic_round": 0,
                "next_periodic_round": self._critic_periodic_every_n_rounds(),
                "pending_trigger_reasons": [],
                "pending_attack_backlog": [],
                "last_critic_outcome": {},
                "updated_at": _utc_now(),
            },
        )
        if not isinstance(state, dict):
            state = {}
        state.setdefault("last_critic_round", 0)
        state.setdefault("next_periodic_round", self._critic_periodic_every_n_rounds())
        state.setdefault("pending_trigger_reasons", [])
        state.setdefault("pending_attack_backlog", [])
        state.setdefault("last_critic_outcome", {})
        state.setdefault("updated_at", _utc_now())
        return state

    def _write_critic_scheduler_state(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utc_now()
        _write_json(self._critic_scheduler_state_path(), payload)

    def _fallback_critic_report(self, *, summary: str, reason: str) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "approved": False,
            "blocking": True,
            "summary": _truncate(summary.strip() or f"Critic review is incomplete: {reason}.", 800),
            "required_evidence": ["Provide a valid `critic_report.json` with structured challenges or approve explicitly."],
            "attack_execution": {
                "execution_mode": self._critic_execution_mode(),
                "executed_interventions": 0,
                "status": "not_executed",
            },
            "challenges": [
                {
                    "challenge_type": "evidence_gap_attack",
                    "title": "Critic structured report missing",
                    "summary": f"Critic review is incomplete because {reason}.",
                    "blocking": True,
                    "required_evidence": "Write `critic_report.json` and, for any blocking claim, attach an intervention or a non-execution reason.",
                    "blocking_reason": "Structured critic output is the only supported review source for gate and HCC updates.",
                    "intervention_type": "",
                    "target_claim": "critic review completed",
                    "execution_status": "not_executed",
                    "expected_artifacts": [],
                }
            ],
            "updated_at": _utc_now(),
        }

    def _enforce_critic_report_contract(
        self,
        payload: dict[str, Any],
        *,
        attack_plan: dict[str, Any],
        attack_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = dict(payload)
        challenges = normalized.get("challenges", [])
        if normalized.get("blocking", True) and not challenges:
            fallback = self._fallback_critic_report(
                summary=normalized.get("summary", ""),
                reason="blocking review lacked structured challenges",
            )
            normalized["approved"] = False
            normalized["blocking"] = True
            normalized["required_evidence"] = fallback["required_evidence"]
            normalized["challenges"] = fallback["challenges"]
            challenges = normalized["challenges"]

        if not normalized.get("summary", "").strip():
            if normalized.get("approved", False):
                normalized["summary"] = "Critic completed a structured review and approved the current round."
            elif challenges:
                normalized["summary"] = challenges[0].get("summary", "") or "Critic found unresolved blocking issues."
            else:
                normalized["summary"] = "Critic review remains incomplete."

        interventions = attack_plan.get("interventions", []) if isinstance(attack_plan, dict) else []
        has_attack_log = any(isinstance(item, dict) for item in attack_log)
        for challenge in challenges:
            if not isinstance(challenge, dict) or not challenge.get("blocking", True):
                continue
            execution_status = challenge.get("execution_status", "not_executed")
            if execution_status == "executed":
                continue
            has_intervention = bool(challenge.get("intervention_type")) or bool(interventions)
            has_non_exec_reason = bool(challenge.get("required_evidence")) or bool(challenge.get("blocking_reason"))
            if not has_intervention:
                challenge["intervention_type"] = ""
            if not has_non_exec_reason:
                challenge["required_evidence"] = (
                    "No lightweight attack was executed in this round; record the minimum next validation path before approval."
                )
                challenge["blocking_reason"] = "Blocking challenge lacked both an executed intervention and an explicit non-execution reason."
            if execution_status == "not_executed" and has_attack_log:
                challenge["execution_status"] = "planned"

        normalized["updated_at"] = _utc_now()
        return normalized

    def _load_critic_report(self, critic_signal: dict[str, Any] | None = None) -> dict[str, Any]:
        report_md, report_json = self._critic_report_paths()
        payload = _read_json(report_json, {})
        report_md_text = self._normalize_critic_report_md_text(
            report_md.read_text(encoding="utf-8") if report_md.exists() else ""
        )
        attack_plan = self._load_critic_attack_plan()
        attack_log = self._load_critic_attack_log()
        attack_results = self._round_file_inventory(f"{CRITIC_ATTACKS_DIR}/results")
        if not payload or payload.get("__placeholder__", False):
            payload = self._synthesize_critic_report_from_artifacts(
                critic_signal=critic_signal or {},
                report_md_text=report_md_text,
                attack_plan=attack_plan,
                attack_log=attack_log,
                attack_results=attack_results,
            )
            if not payload:
                payload = self._fallback_critic_report(
                    summary=report_md_text,
                    reason="critic did not materialize a structured report",
                )
        payload.setdefault("round", self.round_num)
        payload.setdefault("blocking", not payload.get("approved", False))
        payload.setdefault("approved", not payload.get("blocking", True))
        payload.setdefault("summary", _truncate(report_md_text, 800))
        payload.setdefault("challenges", [])
        payload.setdefault("required_evidence", [])
        payload.setdefault(
            "attack_execution",
            {
                "execution_mode": self._critic_execution_mode(),
                "executed_interventions": 0,
                "status": "not_executed",
            },
        )
        payload["challenges"] = self._normalize_challenges(payload.get("challenges"))
        executed_count = self._count_executed_interventions(attack_log, attack_results)
        payload["attack_execution"] = {
            "execution_mode": payload.get("attack_execution", {}).get("execution_mode", self._critic_execution_mode()),
            "executed_interventions": executed_count,
            "status": "executed" if executed_count else ("planned" if attack_plan.get("interventions") else "not_executed"),
            "attack_result_artifacts": attack_results,
        }
        payload = self._enforce_critic_report_contract(payload, attack_plan=attack_plan, attack_log=attack_log)
        self._materialize_critic_artifacts_from_report(payload, attack_plan=attack_plan, attack_log=attack_log)
        _write_json(report_json, payload)
        if not report_md.exists() or not report_md_text.strip():
            report_md.write_text(f"# Critic 审核摘要\n\n{payload.get('summary', '').strip()}\n", encoding="utf-8")
        return payload

    def _normalize_critic_report_md_text(self, text: str) -> str:
        cleaned = text.replace("<!-- EVO_SCAFFOLD_OVERWRITABLE -->", "").strip()
        if cleaned in {"", "# Critic 审核摘要", "（本轮待写）", "# Critic 审核摘要\n\n（本轮待写）"}:
            return ""
        return text

    def _load_critic_attack_plan(self) -> dict[str, Any]:
        attack_plan_path, _, _, _ = self._critic_attack_paths()
        payload = _read_json(attack_plan_path, {})
        if not payload:
            payload = {
                "round": self.round_num,
                "execution_mode": self._critic_execution_mode(),
                "max_interventions": self._critic_max_interventions(),
                "interventions": [],
            }
        if not isinstance(payload, dict):
            payload = {}
        interventions = payload.get("interventions")
        if not isinstance(interventions, list):
            attacks_alias = payload.get("attacks", [])
            interventions = attacks_alias if isinstance(attacks_alias, list) else []
        payload["interventions"] = [item for item in interventions if isinstance(item, dict)]
        payload.setdefault("round", self.round_num)
        payload.setdefault("execution_mode", self._critic_execution_mode())
        payload.setdefault("max_interventions", self._critic_max_interventions())
        return payload

    def _compact_text(self, text: str, *, limit: int = 260) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _extract_recovered_critic_challenges(self, summary: str, *, approved: bool) -> list[dict[str, Any]]:
        compact_summary = self._compact_text(summary, limit=400)
        expected_artifacts = self._normalize_expected_artifacts(self._extract_claimed_artifact_refs(summary))
        challenges: list[dict[str, Any]] = []
        lower_summary = summary.lower()

        if any(keyword in summary for keyword in ("积分", "solve_ivp", "极限环", "振幅误差", "amplitude_error")):
            challenges.append(
                {
                    "challenge_type": "evidence_gap_attack",
                    "title": "缺乏积分验证",
                    "summary": "尚未留下长期积分或稳态振幅误差结果，无法确认极限环复现。",
                    "blocking": not approved,
                    "required_evidence": "从两种初始条件积分到稳态，计算 amplitude_error 并保存结果文件。",
                    "blocking_reason": "缺少长期积分与振幅误差结果，当前结论无法支撑动力学复现。",
                    "intervention_type": "",
                    "target_claim": "当前候选已能复现极限环动力学",
                    "execution_status": "not_executed",
                    "expected_artifacts": expected_artifacts,
                }
            )

        if any(keyword in summary for keyword in ("支持集", "结构一致", "跨风速", "消融", "留一", "support set", "ablation")):
            challenges.append(
                {
                    "challenge_type": "support_set_attack",
                    "title": "支持集鲁棒性未验证",
                    "summary": "尚未通过消融或留一重拟合确认跨风速结构是否稳定。",
                    "blocking": not approved,
                    "required_evidence": "执行留一重拟合或高阶项消融，并比较结构与系数变化。",
                    "blocking_reason": "缺少结构稳定性证据，无法确认当前支持集不是伪规律。",
                    "intervention_type": "support_ablation",
                    "target_claim": "跨风速结构已经稳定",
                    "execution_status": "not_executed",
                    "expected_artifacts": expected_artifacts,
                }
            )

        if not challenges and ("ood" in lower_summary or "泛化" in summary or "锁定区间外" in summary):
            challenges.append(
                {
                    "challenge_type": "ood_generalization_attack",
                    "title": "OOD 泛化尚未验证",
                    "summary": "还没有在锁定区外或额外工况上验证候选规律的稳健性。",
                    "blocking": not approved,
                    "required_evidence": "在 OOD 工况或额外切片上运行验证并落盘结果。",
                    "blocking_reason": "泛化风险仍未被排除。",
                    "intervention_type": "ood_slice_probe",
                    "target_claim": "当前候选具有工况外泛化能力",
                    "execution_status": "not_executed",
                    "expected_artifacts": expected_artifacts,
                }
            )

        if not challenges:
            challenges.append(
                {
                    "challenge_type": "evidence_gap_attack",
                    "title": "缺少关键验证证据",
                    "summary": compact_summary or "Critic finish message indicates unresolved validation gaps.",
                    "blocking": not approved,
                    "required_evidence": "补充真实结果文件并完成最小验证路径。",
                    "blocking_reason": compact_summary or "Critic finish message indicates unresolved validation gaps.",
                    "intervention_type": "",
                    "target_claim": "",
                    "execution_status": "not_executed",
                    "expected_artifacts": expected_artifacts,
                }
            )

        return challenges

    def _ensure_attack_log_entries_from_report(
        self,
        payload: dict[str, Any],
        attack_log: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if attack_log or not isinstance(payload, dict):
            return attack_log

        synthesized: list[dict[str, Any]] = []
        for challenge in payload.get("challenges", []):
            if not isinstance(challenge, dict):
                continue
            synthesized.append(
                {
                    "round": self.round_num,
                    "title": challenge.get("title", ""),
                    "challenge_type": challenge.get("challenge_type", "evidence_gap_attack"),
                    "status": challenge.get("execution_status", "not_executed") or "not_executed",
                    "intervention_type": challenge.get("intervention_type", ""),
                    "target_claim": challenge.get("target_claim", ""),
                    "required_evidence": challenge.get("required_evidence", ""),
                    "blocking_reason": challenge.get("blocking_reason", ""),
                    "expected_artifacts": self._normalize_expected_artifacts(
                        challenge.get("expected_artifacts", [])
                    ),
                }
            )
        return synthesized

    def _synthesize_critic_report_from_artifacts(
        self,
        *,
        critic_signal: dict[str, Any],
        report_md_text: str,
        attack_plan: dict[str, Any],
        attack_log: list[dict[str, Any]],
        attack_results: list[str],
    ) -> dict[str, Any]:
        summary = _truncate(
            (
                critic_signal.get("finish_message")
                or report_md_text
                or critic_signal.get("notes")
                or ""
            ).strip(),
            800,
        )
        interventions = attack_plan.get("interventions", []) if isinstance(attack_plan, dict) else []
        if not summary and not interventions and not attack_log and not attack_results:
            return {}

        executed_artifacts = list(dict.fromkeys(attack_results))
        approved = critic_signal.get("task_completed") == "true" or bool(critic_signal.get("approved", False))
        challenges: list[dict[str, Any]] = []

        for index, item in enumerate(interventions):
            challenge_type = str(item.get("challenge_type", "evidence_gap_attack")).strip()
            if challenge_type not in CRITIC_CHALLENGE_TYPES:
                challenge_type = "evidence_gap_attack"
            expected_artifacts = item.get("expected_artifacts") or item.get("artifacts") or executed_artifacts
            execution_status = self._normalize_execution_status(item.get("execution_status", ""))
            if execution_status == "not_executed" and executed_artifacts:
                execution_status = "executed"
            challenge = {
                "challenge_type": challenge_type,
                "title": item.get("title") or item.get("hypothesis") or f"Intervention {index + 1}",
                "summary": item.get("summary") or item.get("expected_outcome") or summary or "Critic generated an intervention but no structured report payload.",
                "blocking": not approved,
                "required_evidence": item.get("required_evidence") or "Resolve the intervention findings or document why the attack is inconclusive.",
                "blocking_reason": item.get("blocking_reason") or summary or "Critic intervention indicates unresolved risk.",
                "intervention_type": self._normalize_intervention_type(item.get("intervention_type", "")),
                "target_claim": item.get("target_claim") or item.get("hypothesis", ""),
                "execution_status": execution_status,
                "expected_artifacts": self._normalize_expected_artifacts(expected_artifacts),
            }
            challenges.append(challenge)

        if not challenges and (attack_log or executed_artifacts):
            challenges.append(
                {
                    "challenge_type": "evidence_gap_attack",
                    "title": "Critic partial attack evidence recovered",
                    "summary": summary or "Recovered critic attack artifacts but no structured challenge payload.",
                    "blocking": not approved,
                    "required_evidence": "Convert the recovered attack evidence into an explicit structured critic report.",
                    "blocking_reason": summary or "Critic artifacts exist, but the review was not fully structured.",
                    "intervention_type": "",
                    "target_claim": "",
                    "execution_status": "executed" if executed_artifacts else "planned",
                    "expected_artifacts": executed_artifacts,
                }
            )

        if not challenges and summary:
            challenges.extend(self._extract_recovered_critic_challenges(summary, approved=approved))

        if not challenges:
            return {}

        executed_interventions = self._count_executed_interventions(attack_log, executed_artifacts)

        return {
            "round": self.round_num,
            "approved": approved,
            "blocking": not approved,
            "summary": self._compact_text(summary or "Critic artifacts were recovered from partial outputs.", limit=500),
            "required_evidence": [] if approved else ["Resolve the recovered critic challenges or materialize a stricter structured report."],
            "attack_execution": {
                "execution_mode": self._critic_execution_mode(),
                "executed_interventions": executed_interventions,
                "status": "executed" if executed_interventions else ("planned" if interventions else "not_executed"),
                "attack_result_artifacts": executed_artifacts,
            },
            "challenges": challenges,
            "updated_at": _utc_now(),
        }

    def _materialize_critic_artifacts_from_report(
        self,
        payload: dict[str, Any],
        *,
        attack_plan: dict[str, Any],
        attack_log: list[dict[str, Any]],
    ) -> None:
        attack_plan_path, attack_log_path, _, _ = self._critic_attack_paths()
        challenges = payload.get("challenges", [])
        if not isinstance(challenges, list):
            return

        if (not attack_plan.get("interventions")) and challenges:
            interventions: list[dict[str, Any]] = []
            for challenge in challenges:
                if not isinstance(challenge, dict):
                    continue
                interventions.append(
                    {
                        "challenge_type": challenge.get("challenge_type", "evidence_gap_attack"),
                        "title": challenge.get("title", ""),
                        "summary": challenge.get("summary", ""),
                        "intervention_type": challenge.get("intervention_type", ""),
                        "target_claim": challenge.get("target_claim", ""),
                        "execution_status": challenge.get("execution_status", "not_executed"),
                        "expected_artifacts": self._normalize_expected_artifacts(
                            challenge.get("expected_artifacts", [])
                        ),
                        "required_evidence": challenge.get("required_evidence", ""),
                    }
                )
            synthesized_plan = {
                "round": self.round_num,
                "execution_mode": self._critic_execution_mode(),
                "max_interventions": self._critic_max_interventions(),
                "interventions": interventions,
            }
            _write_json(attack_plan_path, synthesized_plan)
            attack_plan = synthesized_plan

        normalized_plan = self._sync_attack_plan_with_execution(attack_plan, challenges, attack_log)
        if normalized_plan != attack_plan:
            _write_json(attack_plan_path, normalized_plan)

        normalized_attack_log = self._ensure_attack_log_entries_from_report(payload, attack_log)
        if normalized_attack_log != attack_log:
            _write_jsonl(attack_log_path, normalized_attack_log)
            attack_plan = normalized_plan

        if attack_log:
            return

        records: list[dict[str, Any]] = []
        for index, challenge in enumerate(challenges, start=1):
            if not isinstance(challenge, dict):
                continue
            expected_artifacts = self._normalize_expected_artifacts(challenge.get("expected_artifacts", []))
            records.append(
                {
                    "attack_id": f"critic_round{self.round_num}_{index}",
                    "challenge_type": challenge.get("challenge_type", "evidence_gap_attack"),
                    "status": challenge.get("execution_status", "not_executed"),
                    "summary": challenge.get("summary", ""),
                    "result_path": expected_artifacts[0] if expected_artifacts else "",
                    "blocking_reason": challenge.get("blocking_reason", ""),
                }
            )
        if records:
            _write_jsonl(attack_log_path, records)

    def _count_executed_interventions(self, attack_log: list[dict[str, Any]], attack_results: list[str]) -> int:
        executed = 0
        for item in attack_log:
            if not isinstance(item, dict):
                continue
            raw_status = str(item.get("status", "")).strip().lower()
            if raw_status in {"", "not_executed", "planned", "pending"}:
                continue
            executed += 1
        if executed == 0 and attack_log:
            executed = len([item for item in attack_log if isinstance(item, dict)])
        if executed == 0 and attack_results:
            executed = len(attack_results)
        return executed

    def _sync_attack_plan_with_execution(
        self,
        attack_plan: dict[str, Any],
        challenges: list[dict[str, Any]],
        attack_log: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(attack_plan, dict):
            return {
                "round": self.round_num,
                "execution_mode": self._critic_execution_mode(),
                "max_interventions": self._critic_max_interventions(),
                "interventions": [],
            }

        payload = dict(attack_plan)
        interventions = payload.get("interventions", [])
        if not isinstance(interventions, list):
            interventions = []

        normalized: list[dict[str, Any]] = []
        for item in interventions:
            if not isinstance(item, dict):
                continue
            updated = dict(item)
            current_status = str(updated.get("status", updated.get("execution_status", "planned"))).strip().lower()
            matched_executed = False

            for challenge in challenges:
                if not isinstance(challenge, dict):
                    continue
                if challenge.get("challenge_type") != updated.get("challenge_type"):
                    continue
                if updated.get("intervention_type") and challenge.get("intervention_type") != updated.get("intervention_type"):
                    continue
                if challenge.get("execution_status") == "executed":
                    matched_executed = True
                    break

            if not matched_executed:
                for record in attack_log:
                    if not isinstance(record, dict):
                        continue
                    record_type = record.get("challenge_type") or record.get("intervention_type") or ""
                    if updated.get("intervention_type") and updated.get("intervention_type") == record_type:
                        matched_executed = True
                        break
                    if updated.get("target") and updated.get("target") in str(record.get("target", "")):
                        matched_executed = True
                        break

            if matched_executed:
                updated["status"] = "executed"
            elif current_status not in {"executed", "planned", "not_executed"}:
                updated["status"] = "planned"
            else:
                updated["status"] = current_status or "planned"
            normalized.append(updated)

        payload["interventions"] = normalized
        payload.setdefault("round", self.round_num)
        payload.setdefault("execution_mode", self._critic_execution_mode())
        payload.setdefault("max_interventions", self._critic_max_interventions())
        return payload

    def _align_critic_signal_with_report(
        self,
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(critic_signal)
        normalized["approved"] = bool(critic_report.get("approved", False))
        normalized["satisfied"] = bool(critic_report.get("approved", False)) and not bool(critic_report.get("blocking", True))
        normalized["task_completed"] = "true" if normalized["satisfied"] else "false"
        normalized["notes"] = _truncate(
            critic_report.get("summary", "") or normalized.get("notes", ""),
            500,
        )
        normalized["summary"] = critic_report.get("summary", "")
        return normalized

    def _load_critic_attack_log(self) -> list[dict[str, Any]]:
        _, attack_log_path, _, _ = self._critic_attack_paths()
        return _read_jsonl(attack_log_path)

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
                        "intervention_type": "",
                        "target_claim": "",
                        "execution_status": "not_executed",
                        "expected_artifacts": [],
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
                    "intervention_type": self._normalize_intervention_type(item.get("intervention_type", "")),
                    "target_claim": item.get("target_claim", ""),
                    "execution_status": self._normalize_execution_status(item.get("execution_status", "")),
                    "expected_artifacts": self._normalize_expected_artifacts(item.get("expected_artifacts", [])),
                }
            )
        return normalized

    def _normalize_intervention_type(self, raw_value: Any) -> str:
        value = str(raw_value or "").strip()
        return value if value in CRITIC_INTERVENTION_TYPES else ""

    def _normalize_execution_status(self, raw_value: Any) -> str:
        value = str(raw_value or "").strip().lower()
        return value if value in {"planned", "executed", "not_executed"} else "not_executed"

    def _normalize_expected_artifacts(self, raw_value: Any) -> list[str]:
        if isinstance(raw_value, str):
            raw_value = [raw_value]
        if not isinstance(raw_value, list):
            return []
        normalized: list[str] = []
        for item in raw_value:
            normalized_item = str(item).strip()
            if normalized_item and normalized_item not in normalized:
                normalized.append(normalized_item)
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

    def _read_debate_state(self) -> dict[str, Any]:
        if not self.run_dir:
            return {}
        return _read_json(self.run_dir / DEBATE_STATE_FILE, {})

    def _read_scheduler_state(self) -> dict[str, Any]:
        return self._load_critic_scheduler_state()

    def _read_relevant_updated_text(self, updated: list[str]) -> str:
        return "\n".join(
            self._read_artifact_text(name)
            for name in updated
            if name in {"plan.md", "findings.md", "variable_memory.json", "hypothesis_archive.jsonl", "routing_state.json"}
        )

    def _detect_strong_conclusion_reasons(self, hamilton_signal: dict[str, Any]) -> list[str]:
        completion_gate = hamilton_signal.get("completion_gate") or {}
        updated = completion_gate.get("updated", [])
        claim_binding = completion_gate.get("claim_evidence_binding", {})
        reasons: list[str] = []

        if completion_gate.get("round_results"):
            reasons.append("new_results_artifact")
        if claim_binding.get("quantified_claims"):
            reasons.append("quantified_result_claim")

        updated_text = self._read_relevant_updated_text(updated)
        if any(re.search(pattern, updated_text, flags=re.IGNORECASE | re.DOTALL) for pattern in STRONG_CONCLUSION_PATTERNS):
            reasons.append("strong_conclusion_text")

        return list(dict.fromkeys(reasons))

    def _detect_unresolved_challenge_response_reasons(self, hamilton_signal: dict[str, Any]) -> list[str]:
        debate_state = self._read_debate_state()
        unresolved = debate_state.get("unresolved_challenges", []) if isinstance(debate_state, dict) else []
        if not unresolved:
            return []

        completion_gate = hamilton_signal.get("completion_gate") or {}
        updated_text = " ".join(
            [
                hamilton_signal.get("finish_message", ""),
                self._read_relevant_updated_text(completion_gate.get("updated", [])),
            ]
        ).lower()
        reasons: list[str] = []
        for challenge in unresolved:
            if not isinstance(challenge, dict):
                continue
            challenge_type = challenge.get("challenge_type", "evidence_gap_attack")
            keywords = CHALLENGE_KEYWORDS.get(challenge_type, ())
            if keywords and any(keyword.lower() in updated_text for keyword in keywords):
                reasons.append(f"response_to_{challenge_type}")

        return list(dict.fromkeys(reasons))

    def _has_new_round_results_since_last_critic(self, scheduler_state: dict[str, Any]) -> bool:
        last_critic_round = int(scheduler_state.get("last_critic_round", 0) or 0)
        if self.round_num <= 0 or not self.run_dir:
            return False
        for round_num in range(last_critic_round + 1, self.round_num + 1):
            if self._history_round_inventory_for(round_num, "results"):
                return True
        return False

    def _evaluate_critic_schedule(self, hamilton_signal: dict[str, Any]) -> dict[str, Any]:
        scheduler_state = self._read_scheduler_state()
        schedule_mode = self._critic_schedule_mode()
        period = self._critic_periodic_every_n_rounds()
        trigger_reasons: list[str] = []
        allowed_immediate_triggers = self._critic_immediate_triggers()
        periodic_due = False

        if schedule_mode in {"hybrid", "periodic"} and self.round_num >= int(scheduler_state.get("next_periodic_round", period) or period):
            trigger_reasons.append("periodic_round")
            periodic_due = True

        if (
            "hamilton_finish_true" in allowed_immediate_triggers
            and hamilton_signal.get("finish_called")
            and hamilton_signal.get("task_completed") == "true"
        ):
            trigger_reasons.append("hamilton_finish_true")

        for reason in self._detect_strong_conclusion_reasons(hamilton_signal):
            if reason in allowed_immediate_triggers:
                trigger_reasons.append(reason)

        if "challenge_response" in allowed_immediate_triggers:
            trigger_reasons.extend(self._detect_unresolved_challenge_response_reasons(hamilton_signal))

        run_critic = bool(trigger_reasons) if schedule_mode in {"hybrid", "periodic", "event_only"} else False
        has_new_results = "new_results_artifact" in trigger_reasons or self._has_new_round_results_since_last_critic(scheduler_state)
        has_challenge_response = any(reason.startswith("response_to_") for reason in trigger_reasons)
        has_finish_true = "hamilton_finish_true" in trigger_reasons
        if run_critic and not periodic_due and not has_new_results and not has_finish_true:
            run_critic = False
        pending_attack_backlog = []
        debate_state = self._read_debate_state()
        if isinstance(debate_state, dict):
            for challenge in debate_state.get("unresolved_challenges", []):
                if isinstance(challenge, dict):
                    pending_attack_backlog.append(
                        challenge.get("title")
                        or challenge.get("challenge_type")
                        or "unresolved_challenge"
                    )

        return {
            "mode": schedule_mode,
            "periodic_every_n_rounds": period,
            "run_critic": run_critic,
            "trigger_reasons": list(dict.fromkeys(trigger_reasons)),
            "periodic_due": periodic_due,
            "has_new_results_since_last_critic": has_new_results,
            "scheduler_state_before": scheduler_state,
            "pending_attack_backlog": pending_attack_backlog,
        }

    def _build_skipped_critic_signal(self, critic_schedule: dict[str, Any]) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "actor": "critic",
            "satisfied": False,
            "task_completed": "false",
            "finish_called": False,
            "finish_message": "Critic skipped this round by scheduler.",
            "notes": _truncate(
                "Critic skipped this round. Trigger reasons: "
                + ", ".join(critic_schedule.get("trigger_reasons", []) or ["none"]),
                300,
            ),
            "approved": False,
            "skipped": True,
        }

    def _build_skipped_critic_report(self, critic_schedule: dict[str, Any]) -> dict[str, Any]:
        return {
            "round": self.round_num,
            "approved": False,
            "blocking": False,
            "summary": "Critic skipped this round by scheduler.",
            "required_evidence": [],
            "challenges": [],
            "attack_execution": {
                "execution_mode": self._critic_execution_mode(),
                "executed_interventions": 0,
                "status": "not_triggered",
            },
            "schedule": critic_schedule,
        }

    def _update_critic_scheduler_state(
        self,
        critic_schedule: dict[str, Any],
        critic_report: dict[str, Any],
        aggregated_signal: dict[str, Any],
    ) -> None:
        state = dict(critic_schedule.get("scheduler_state_before", {}) or self._read_scheduler_state())
        state["pending_trigger_reasons"] = []
        state["pending_attack_backlog"] = critic_schedule.get("pending_attack_backlog", [])
        if critic_schedule.get("run_critic", False):
            state["last_critic_round"] = self.round_num
            state["next_periodic_round"] = self.round_num + self._critic_periodic_every_n_rounds()
            state["last_critic_outcome"] = {
                "round": self.round_num,
                "approved": critic_report.get("approved", False),
                "blocking": critic_report.get("blocking", True),
                "summary": critic_report.get("summary", ""),
                "blocked_by": aggregated_signal.get("blocked_by", []),
            }
        else:
            state["pending_trigger_reasons"] = critic_schedule.get("trigger_reasons", [])
        self._write_critic_scheduler_state(state)

    def _append_hcc_ledger_entries(
        self,
        *,
        hamilton_signal: dict[str, Any],
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
        aggregated_signal: dict[str, Any],
        critic_schedule: dict[str, Any],
    ) -> None:
        if not self.run_dir:
            return

        ledger_path = self.run_dir / HCC_LEDGER_FILE
        existing = _read_jsonl(ledger_path)
        existing_ids = {item.get("entry_id") for item in existing if isinstance(item, dict)}

        completion_gate = hamilton_signal.get("completion_gate") or {}
        round_results = completion_gate.get("round_results", [])
        round_scripts = completion_gate.get("round_scripts", [])
        entries: list[dict[str, Any]] = []
        canonical_hypothesis = self._build_canonical_hypothesis_record(
            hamilton_signal=hamilton_signal,
            aggregated_signal=aggregated_signal,
            critic_schedule=critic_schedule,
        )

        allow_positive_entry = bool(round_results) and canonical_hypothesis is not None and (
            not critic_schedule.get("run_critic", False) or aggregated_signal.get("critic_approved", False)
        )
        if allow_positive_entry:
            positive_entry = self._build_hcc_entry(
                polarity="positive",
                producer_role="hamilton",
                consumer_scope="both",
                title=canonical_hypothesis.get("title", f"Round {self.round_num} verified result update"),
                summary=self._build_verified_positive_hcc_summary(
                    hamilton_signal=hamilton_signal,
                    round_results=round_results,
                    round_scripts=round_scripts,
                ),
                card_type="operator_motif",
                evidence_paths=[*round_scripts, *round_results],
                evidence_strength=0.8,
                survived_attack=bool(critic_schedule.get("run_critic", False) and aggregated_signal.get("critic_approved", False)),
                attacked=bool(critic_schedule.get("run_critic", False)),
                source="system_round_summary",
            )
            entries.append(positive_entry)

        for challenge in critic_report.get("challenges", []):
            if not isinstance(challenge, dict):
                continue
            entries.append(
                self._build_hcc_entry(
                    polarity="negative",
                    producer_role="critic",
                    consumer_scope="both",
                    title=challenge.get("title", challenge.get("challenge_type", "critic_challenge")),
                    summary=challenge.get("summary", ""),
                    card_type="failure_card",
                    evidence_paths=self._critic_real_evidence_paths(challenge),
                    evidence_strength=0.75 if challenge.get("execution_status") == "executed" else 0.5,
                    survived_attack=False,
                    attacked=True,
                    source="critic_report",
                    negative_evidence=challenge.get("blocking_reason", ""),
                    metadata={
                        "challenge_type": challenge.get("challenge_type", "evidence_gap_attack"),
                        "intervention_type": challenge.get("intervention_type", ""),
                        "execution_status": challenge.get("execution_status", "not_executed"),
                    },
                )
            )

        if critic_schedule.get("run_critic", False) and aggregated_signal.get("critic_approved", False) and round_results:
            entries.append(
                self._build_hcc_entry(
                    polarity="positive",
                    producer_role="system",
                    consumer_scope="both",
                    title=f"Round {self.round_num} candidate survived critic attack",
                    summary=critic_report.get("summary", ""),
                    card_type="domain_prior",
                    evidence_paths=round_results,
                    evidence_strength=0.9,
                    survived_attack=True,
                    attacked=True,
                    source="critic_approval",
                )
            )

        merged = list(existing)
        for entry in entries:
            if entry["entry_id"] in existing_ids:
                continue
            merged.append(entry)
            existing_ids.add(entry["entry_id"])
        _write_jsonl(ledger_path, merged)

    def _sync_round_fact_ledgers(
        self,
        *,
        hamilton_signal: dict[str, Any],
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
        aggregated_signal: dict[str, Any],
        critic_schedule: dict[str, Any],
    ) -> None:
        if not self.run_dir:
            return

        self._canonicalize_routing_state()
        self._append_system_hypothesis_record(
            hamilton_signal=hamilton_signal,
            aggregated_signal=aggregated_signal,
            critic_schedule=critic_schedule,
        )
        self._append_system_falsification_records(
            critic_signal=critic_signal,
            critic_report=critic_report,
            critic_schedule=critic_schedule,
        )
        self._normalize_updated_workspace_state(
            ["findings.md", "plan.md", "variable_memory.json", "hypothesis_archive.jsonl", "falsification_log.jsonl"]
        )
        self._canonicalize_summary_documents()

    def _read_csv_header(self, relpath: str) -> list[str]:
        if not self.run_dir:
            return []
        path = self.run_dir / relpath
        if not path.exists() or path.suffix.lower() != ".csv":
            return []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                return next(reader, [])
        except Exception:
            return []

    def _support_set_from_result_files(self, result_paths: list[str], finish_message: str = "") -> list[str]:
        support = self._extract_support_set_from_text(finish_message)
        if support:
            return support

        inferred: list[str] = []
        column_to_support = {
            "c_x": "x",
            "c_x3": "x^3",
            "c_v": "v",
            "c_v3": "v^3",
            "c_v5": "v^5",
        }
        for relpath in result_paths:
            payload = self._load_json_result(relpath)
            for item in self._support_set_from_result_payload(payload):
                if item not in inferred:
                    inferred.append(item)
            referenced_coef = self._referenced_coef_table_paths(payload)
            for coef_path in referenced_coef:
                for item in self._support_set_from_result_files([coef_path], finish_message=""):
                    if item not in inferred:
                        inferred.append(item)
            for column in self._read_csv_header(relpath):
                label = column_to_support.get(column.strip())
                if label and label not in inferred:
                    inferred.append(label)
        return inferred

    def _canonical_equation_from_support_set(self, support_set: list[str]) -> str:
        render_map = {
            "x": "c_x*x",
            "x^3": "c_x3*x^3",
            "v": "c_v*v",
            "v^3": "c_v3*v^3",
            "v^5": "c_v5*v^5",
        }
        rendered = [render_map[item] for item in support_set if item in render_map]
        if not rendered:
            return ""
        return "a = " + " + ".join(rendered)

    def _normalize_support_set_items(self, items: list[Any]) -> list[str]:
        normalized: list[str] = []
        for item in items:
            raw = str(item or "").strip().strip("`")
            if not raw:
                continue
            canonical = SUPPORT_SET_TOKEN_ALIASES.get(raw.lower(), raw)
            if canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def _load_json_result(self, relpath: str) -> dict[str, Any]:
        if not self.run_dir:
            return {}
        path = self.run_dir / relpath
        if not path.exists() or path.suffix.lower() != ".json":
            return {}
        payload = _read_json(path, {})
        return payload if isinstance(payload, dict) else {}

    def _support_set_from_result_payload(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("support_set"), list):
            normalized = self._normalize_support_set_items(payload.get("support_set", []))
            if normalized:
                return normalized
        label = str(payload.get("support_set_label", "") or "").strip()
        if label:
            parts = [item.strip() for item in re.split(r"[，,;；]", label) if item.strip()]
            normalized = self._normalize_support_set_items(parts)
            if normalized:
                return normalized
        equation = str(payload.get("equation", "") or "").strip()
        if equation:
            normalized = self._extract_support_set_from_equation(equation)
            if normalized:
                return normalized
        return []

    def _referenced_coef_table_paths(self, payload: dict[str, Any]) -> list[str]:
        if not isinstance(payload, dict):
            return []
        candidates = []
        for key in ("coef_table", "coeff_table", "output"):
            value = str(payload.get(key, "") or "").strip()
            if value.endswith(".csv") and value not in candidates:
                candidates.append(value)
        return candidates

    def _equation_from_result_files(self, result_paths: list[str], finish_message: str = "") -> str:
        equation = self._extract_equation_candidate(finish_message)
        if equation:
            return equation
        for relpath in result_paths:
            payload = self._load_json_result(relpath)
            candidate = " ".join(str(payload.get("equation", "") or "").split())
            if candidate:
                return candidate[:220]
        return ""

    def _extract_support_set_from_equation(self, equation: str) -> list[str]:
        support: list[str] = []
        normalized = equation.replace(" ", "")
        patterns = (
            ("x^3", r"c_x3\*x\^3|x\^3"),
            ("v^5", r"c_v5\*v\^5|v\^5"),
            ("v^3", r"c_v3\*v\^3|v\^3"),
            ("x", r"c_x\*x(?!\^)|(?<![a-zA-Z0-9_])x(?![\^a-zA-Z0-9_])"),
            ("v", r"c_v\*v(?!\^)|(?<![a-zA-Z0-9_])v(?![\^a-zA-Z0-9_])"),
        )
        for label, pattern in patterns:
            if re.search(pattern, normalized):
                support.append(label)
        return support

    def _latest_hypothesis_before_round(self, round_num: int) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for item in self._hypothesis_records():
            try:
                item_round = int(item.get("round", 0))
            except Exception:
                item_round = 0
            if 0 < item_round < round_num:
                latest = item
        return latest

    def _result_summary_from_payloads(
        self,
        *,
        round_results: list[str],
        support_set: list[str],
        equation: str,
        executed_strategy: str,
    ) -> str:
        support_label = ", ".join(support_set) if support_set else "未显式落盘"
        for relpath in round_results:
            payload = self._load_json_result(relpath)
            if not payload:
                continue
            name = Path(relpath).name
            if name == "fit_summary.json":
                mean_r2 = payload.get("mean_r2")
                num_speeds = payload.get("num_speeds")
                return (
                    f"{executed_strategy or 'analytic_fit'} 已生成候选 `{equation}`；"
                    f"支持集 {{{support_label}}}；"
                    f"跨 {num_speeds or '?'} 个风速的平均 R²={mean_r2 if mean_r2 is not None else '?'}。"
                )
            if name == "amplitude_error.json":
                success = payload.get("num_successful_cases")
                failed = payload.get("num_failed_cases")
                mean_amp = payload.get("mean_amplitude_error")
                return (
                    f"{executed_strategy or 'integration_validation'} 基于候选 `{equation}` 做动力学积分；"
                    f"支持集 {{{support_label}}}；"
                    f"成功 {success if success is not None else '?'} 个工况，失败 {failed if failed is not None else '?'} 个，"
                    f"mean_amplitude_error={mean_amp if mean_amp is not None else '?'}。"
                )
            if name == "support_ablation.json":
                variant = str(payload.get("variant", "") or "ablation").strip()
                mean_r2 = payload.get("mean_r2")
                return (
                    f"{executed_strategy or 'support_ablation'} 完成 `{variant}` 消融；"
                    f"候选 `{equation}`；支持集 {{{support_label}}}；"
                    f"平均 R²={mean_r2 if mean_r2 is not None else '?'}。"
                )
        return ""

    def _build_canonical_hypothesis_record(
        self,
        *,
        hamilton_signal: dict[str, Any],
        aggregated_signal: dict[str, Any],
        critic_schedule: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.run_dir:
            return None
        completion_gate = hamilton_signal.get("completion_gate") or {}
        round_results = completion_gate.get("round_results", [])
        if not round_results:
            return None

        finish_message = hamilton_signal.get("finish_message", "")
        executed_strategy = str(hamilton_signal.get("executed_strategy", "") or "").strip()
        previous_hypothesis = self._latest_hypothesis_before_round(self.round_num)
        support_set = self._support_set_from_result_files(round_results, finish_message=finish_message)
        equation = self._equation_from_result_files(round_results, finish_message=finish_message)
        if not support_set and isinstance(previous_hypothesis, dict):
            prior_support = previous_hypothesis.get("support_set", [])
            if isinstance(prior_support, list):
                support_set = self._normalize_support_set_items(prior_support)
        if not equation and isinstance(previous_hypothesis, dict):
            equation = " ".join(str(previous_hypothesis.get("equation", "") or "").split())
        if not equation:
            equation = self._canonical_equation_from_support_set(support_set)
        if not equation and not support_set:
            return None

        status = (
            "survived_attack"
            if aggregated_signal.get("critic_approved", False)
            else ("candidate_blocked" if critic_schedule.get("run_critic", False) else "candidate_unreviewed")
        )
        evidence_paths = [*completion_gate.get("round_scripts", []), *round_results]
        summary = self._result_summary_from_payloads(
            round_results=round_results,
            support_set=support_set,
            equation=equation,
            executed_strategy=executed_strategy,
        ) or self._sanitize_candidate_summary(
            _truncate(finish_message, 500),
            evidence_paths=evidence_paths,
            status=status,
        )
        return {
            "round": self.round_num,
            "title": f"Round {self.round_num} candidate",
            "summary": summary,
            "equation": equation,
            "support_set": support_set,
            "tool_path": executed_strategy or str(previous_hypothesis.get("tool_path", "") or "").strip(),
            "status": status,
            "critic_status": "approved" if aggregated_signal.get("critic_approved", False) else "blocked",
            "evidence_paths": evidence_paths,
        }

    def _append_jsonl_record(
        self,
        path: Path,
        record: dict[str, Any],
        *,
        dedupe_key: tuple[Any, ...],
    ) -> None:
        existing = _read_jsonl(path)
        seen = set()
        merged: list[dict[str, Any]] = []
        for item in existing:
            if not isinstance(item, dict):
                continue
            raw_key = item.get("_system_key", []) if "_system_key" in item else tuple(
                item.get(part) for part in ("round", "title", "summary")
            )
            key = self._freeze_dedupe_value(raw_key)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        frozen_dedupe_key = self._freeze_dedupe_value(dedupe_key)
        if frozen_dedupe_key not in seen:
            record_copy = dict(record)
            record_copy["_system_key"] = self._json_safe_dedupe_value(dedupe_key)
            merged.append(record_copy)
        _write_jsonl(path, merged)

    def _format_evidence_paths(self, paths: list[str], *, limit: int = 3) -> str:
        normalized = [f"`{path}`" for path in paths if isinstance(path, str) and path.strip()]
        if not normalized:
            return "无"
        if len(normalized) <= limit:
            return ", ".join(normalized)
        return ", ".join(normalized[:limit]) + f" 等 {len(normalized)} 项"

    def _latest_hypothesis_record(self) -> dict[str, Any]:
        if not self.run_dir:
            return {}
        records = _read_jsonl(self.run_dir / "hypothesis_archive.jsonl")
        for item in reversed(records):
            if isinstance(item, dict):
                return item
        return {}

    def _falsification_records(self) -> list[dict[str, Any]]:
        if not self.run_dir:
            return []
        return [item for item in _read_jsonl(self.run_dir / "falsification_log.jsonl") if isinstance(item, dict)]

    def _positive_hcc_records(self) -> list[dict[str, Any]]:
        if not self.run_dir:
            return []
        records = []
        for item in _read_jsonl(self.run_dir / HCC_LEDGER_FILE):
            if not isinstance(item, dict):
                continue
            if item.get("polarity") != "positive":
                continue
            if not item.get("evidence_paths"):
                continue
            records.append(item)
        return records

    def _negative_hcc_records(self) -> list[dict[str, Any]]:
        if not self.run_dir:
            return []
        records = []
        for item in _read_jsonl(self.run_dir / HCC_LEDGER_FILE):
            if not isinstance(item, dict):
                continue
            if item.get("polarity") != "negative":
                continue
            records.append(item)
        return records

    def _history_round_dir_for(self, round_num: int) -> Path:
        if not self.run_dir:
            raise ValueError("RoundExp.run_dir is not set")
        return self.run_dir / "history" / f"round{round_num}"

    def _history_round_inventory_for(self, round_num: int, dirname: str) -> list[str]:
        if not self.run_dir:
            return []
        root = self._history_round_dir_for(round_num) / dirname
        return [str(path.relative_to(self.run_dir)) for path in _round_file_inventory(root)]

    def _history_round_numbers(self) -> list[int]:
        rounds: set[int] = set()
        if self.run_dir:
            history_root = self.run_dir / "history"
            if history_root.exists():
                for path in history_root.iterdir():
                    if not path.is_dir():
                        continue
                    match = re.fullmatch(r"round(\d+)", path.name)
                    if match:
                        rounds.add(int(match.group(1)))

        for record in [*self._hypothesis_records(), *self._falsification_records(), *self._positive_hcc_records(), *self._negative_hcc_records()]:
            try:
                round_num = int(record.get("round", 0))
            except Exception:
                round_num = 0
            if round_num > 0:
                rounds.add(round_num)

        if not rounds and self.round_num > 0:
            rounds.add(self.round_num)
        return sorted(rounds)

    def _latest_history_result_matching(self, filenames: set[str]) -> str:
        for round_num in reversed(self._history_round_numbers()):
            for item in self._history_round_inventory_for(round_num, "results"):
                if Path(item).name in filenames:
                    return item
        return ""

    def _latest_history_artifact_matching(
        self,
        *,
        dirname: str,
        predicate,
        include_critic_attacks: bool = False,
    ) -> str:
        for round_num in reversed(self._history_round_numbers()):
            candidates = list(reversed(self._history_round_inventory_for(round_num, dirname)))
            if include_critic_attacks and self.run_dir:
                attack_root = self._history_round_dir_for(round_num) / "critic_attacks" / dirname
                candidates.extend(
                    reversed(
                        [str(path.relative_to(self.run_dir)) for path in _round_file_inventory(attack_root)]
                    )
                )
            for item in candidates:
                if predicate(item):
                    return item
        return ""

    def _latest_support_ablation_artifact(self) -> str:
        return self._latest_history_artifact_matching(
            dirname="results",
            predicate=lambda item: Path(item).name in ABLATION_RESULT_FILENAMES
            or "support_ablation" in Path(item).name,
            include_critic_attacks=True,
        )

    def _latest_ablation_rollout_artifact(self) -> str:
        return self._latest_history_artifact_matching(
            dirname="results",
            predicate=lambda item: Path(item).name in ABLATION_ROLLOUT_RESULT_FILENAMES,
            include_critic_attacks=True,
        )

    def _primary_artifact_for_strategy(self, strategy: str) -> str:
        return {
            "analytic_fit": f"history/round{self.round_num}/results/coef_table.csv",
            "integration_validation": f"history/round{self.round_num}/results/amplitude_error.json",
            "ablation_validation": f"history/round{self.round_num}/results/ablation_amplitude_error.json",
            "support_ablation": f"history/round{self.round_num}/results/support_ablation.json",
        }.get(strategy, f"history/round{self.round_num}/results/")

    def _stage_input_artifact_for_strategy(self, strategy: str) -> str:
        if strategy == "integration_validation":
            return self._latest_history_result_matching(COEF_TABLE_FILENAMES)
        if strategy == "ablation_validation":
            return self._latest_support_ablation_artifact()
        if strategy == "support_ablation":
            latest_rollout = self._latest_history_result_matching(ROLLOUT_RESULT_FILENAMES)
            return latest_rollout or self._latest_history_result_matching(COEF_TABLE_FILENAMES)
        return ""

    def _recommended_command_for_strategy(self, strategy: str, next_tools: Any) -> str:
        if not isinstance(next_tools, list):
            return ""
        strategy_to_index = {
            "analytic_fit": 0,
            "integration_validation": 1,
            "support_ablation": 2,
            "ablation_validation": 3,
        }
        tool_index = strategy_to_index.get(strategy, 0)
        if not (0 <= tool_index < len(next_tools)):
            return ""
        template = next_tools[tool_index]
        if not isinstance(template, str):
            return ""
        command = template.replace("roundN", f"round{self.round_num}")
        if strategy in {"integration_validation", "ablation_validation"}:
            coef_table_path = self._stage_input_artifact_for_strategy(strategy)
            if coef_table_path:
                command = re.sub(r"(--coef-table\s+)(\S+)", rf"\1{coef_table_path}", command, count=1)
        return command

    def _hypothesis_records(self) -> list[dict[str, Any]]:
        if not self.run_dir:
            return []
        return [item for item in _read_jsonl(self.run_dir / "hypothesis_archive.jsonl") if isinstance(item, dict)]

    def _hypothesis_by_round(self) -> dict[int, dict[str, Any]]:
        latest_by_round: dict[int, dict[str, Any]] = {}
        for item in self._hypothesis_records():
            try:
                round_num = int(item.get("round", 0))
            except Exception:
                round_num = 0
            if round_num > 0:
                latest_by_round[round_num] = item
        return latest_by_round

    def _records_grouped_by_round(self, records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for item in records:
            try:
                round_num = int(item.get("round", 0))
            except Exception:
                round_num = 0
            if round_num <= 0:
                continue
            grouped.setdefault(round_num, []).append(item)
        return grouped

    def _critic_report_for_round(self, round_num: int) -> dict[str, Any]:
        if not self.run_dir:
            return {}
        payload = _read_json(self._history_round_dir_for(round_num) / CRITIC_REPORT_JSON, {})
        return payload if isinstance(payload, dict) else {}

    def _compact_artifact_list(
        self,
        paths: list[str],
        *,
        limit: int = 3,
        filenames_only: bool = True,
        markdown: bool = False,
    ) -> str:
        normalized: list[str] = []
        for item in paths:
            value = str(item or "").strip()
            if not value:
                continue
            rendered = Path(value).name if filenames_only else value
            if markdown:
                rendered = f"`{rendered}`"
            if rendered not in normalized:
                normalized.append(rendered)
        if not normalized:
            return "无"
        if len(normalized) <= limit:
            return ", ".join(normalized)
        return ", ".join(normalized[:limit]) + f" 等 {len(normalized)} 项"

    def _format_support_set(self, record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return "未显式落盘"
        support_set = record.get("support_set", [])
        if isinstance(support_set, list):
            normalized = [str(item).strip() for item in support_set if str(item).strip()]
        else:
            normalized = []
        if not normalized:
            normalized = self._extract_support_set_from_text(record.get("summary", ""))
        return ", ".join(normalized) if normalized else "未显式落盘"

    def _format_candidate_label(self, record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return "待定"
        equation = " ".join(str(record.get("equation", "") or "").split())
        if equation:
            return equation[:120]
        title = " ".join(str(record.get("title", "") or "").split())
        if title:
            return title[:120]
        support_set = self._format_support_set(record)
        if support_set != "未显式落盘":
            return f"基于支持集 {{{support_set}}} 的候选"
        return "待定"

    def _format_route_label(self, record: dict[str, Any] | None) -> str:
        if not isinstance(record, dict):
            return "待定"
        route = str(record.get("tool_path", "") or "").strip()
        return route or "待定"

    def _critic_status_label(self, critic_report: dict[str, Any]) -> str:
        if not critic_report:
            return "未触发"
        if critic_report.get("approved", False):
            return "approved"
        challenges = critic_report.get("challenges", [])
        if isinstance(challenges, list) and challenges:
            first = challenges[0] if isinstance(challenges[0], dict) else {}
            title = str(first.get("title", "") or "").strip()
            if title:
                return f"blocked: {title[:40]}"
        summary = str(critic_report.get("summary", "") or "").strip()
        return f"blocked: {summary[:40]}" if summary else "blocked"

    def _sanitize_table_cell(self, value: str) -> str:
        cell = " ".join(str(value or "").split())
        return cell.replace("|", "/")[:160] or "-"

    def _build_round_insight_lines(
        self,
        round_num: int,
        *,
        hypothesis: dict[str, Any] | None,
        critic_report: dict[str, Any],
        falsifications: list[dict[str, Any]],
        negative_hcc: list[dict[str, Any]],
        scripts: list[str],
        results: list[str],
    ) -> list[str]:
        support_set = self._format_support_set(hypothesis)
        candidate = self._format_candidate_label(hypothesis)

        if results and support_set != "未显式落盘":
            mechanism = f"当前候选已围绕 `{support_set}` 形成可审查结构，候选标签为 `{candidate}`。"
        elif results:
            mechanism = "本轮已产出真实结果工件，但显式候选方程与支持集仍未完整落盘。"
        else:
            mechanism = "本轮主要在补证、规划或修复 blocker，尚未形成新的可审查结果工件。"

        evidence = (
            f"真实脚本 {len(scripts)} 个，结果 {len(results)} 个；主要工件："
            f"{self._compact_artifact_list([*scripts, *results], limit=3, filenames_only=False, markdown=True)}。"
        )

        if critic_report:
            feedback = str(critic_report.get("summary", "") or "").strip()
            if not feedback:
                feedback = self._critic_status_label(critic_report)
        elif falsifications:
            feedback = str(falsifications[0].get("summary", "") or falsifications[0].get("title", "")).strip()
        elif negative_hcc:
            feedback = str(negative_hcc[0].get("summary", "") or negative_hcc[0].get("title", "")).strip()
        else:
            feedback = "本轮未触发 Critic，下一次对抗审查将取决于 scheduler 与新证据。"

        return [
            f"### Round {round_num}",
            f"- 机制判读：{mechanism}",
            f"- 证据状态：{evidence}",
            f"- 对抗反馈：{feedback or '暂无结构化对抗反馈。'}",
            "",
        ]

    def _round_result_conclusion(
        self,
        *,
        round_num: int,
        hypothesis: dict[str, Any] | None,
        critic_report: dict[str, Any],
        falsifications: list[dict[str, Any]],
        positive_hcc: list[dict[str, Any]],
        results: list[str],
    ) -> str:
        if positive_hcc or critic_report.get("approved", False):
            return "通过当前轮审查"
        if falsifications:
            title = str(falsifications[0].get("title", "") or "").strip()
            return title[:60] if title else "被 Critic 攻击阻断"
        if results:
            status = str((hypothesis or {}).get("status", "")).strip()
            return "已有真实结果，待进一步验证" if not status else status[:60]
        return "以分析/规划为主"

    def _build_candidate_equation_section(
        self,
        *,
        round_num: int | None,
        hypothesis: dict[str, Any] | None,
        critic_report: dict[str, Any],
        falsifications: list[dict[str, Any]],
        scripts: list[str],
        results: list[str],
    ) -> list[str]:
        if round_num is None:
            return [
                "## 候选方程解析",
                "### 1) 方程与物理解释",
                "- 当前暂无带真实结果工件的候选方程。",
                "### 2) 参数/系数敏感性",
                "- 待后续真实结果补充。",
                "### 3) 物理洞察",
                "- 待后续真实结果补充。",
                "### 4) 消融分析",
                "- 待后续真实结果补充。",
            ]

        support_set = self._format_support_set(hypothesis)
        candidate = self._format_candidate_label(hypothesis)
        status = str((hypothesis or {}).get("status", "") or "candidate_pending").strip()
        route = self._format_route_label(hypothesis)
        evidence_paths = list((hypothesis or {}).get("evidence_paths", []) or [])
        if not evidence_paths:
            evidence_paths = [*scripts, *results]

        physical_insight = "当前候选仍更像结构模板而非最终定律，需要继续用对抗实验确认必要项。"
        lower_support = support_set.lower()
        if any(token in lower_support for token in ("x", "x^3", "位移")) and any(
            token in lower_support for token in ("v", "v^3", "v^5", "速度")
        ):
            physical_insight = "候选同时包含位移相关项与速度相关项，符合恢复力/阻尼分解的常见动力学建模方式。"
        elif any(token in lower_support for token in ("v", "v^3", "v^5", "速度")):
            physical_insight = "当前候选主要由速度相关项驱动，更像在检验阻尼与高阶阻尼结构是否必要。"
        elif any(token in lower_support for token in ("x", "x^3", "位移")):
            physical_insight = "当前候选主要围绕位移相关项展开，更像在稳定恢复力结构。"

        lines = [
            f"## 候选方程解析（Round {round_num})",
            "### 1) 方程与物理解释",
            f"- 当前候选：`{candidate}`",
            f"- 支持集：{support_set}",
            f"- 当前状态：`{status}`；工具路径：`{route}`",
            f"- 现有证据：{self._format_evidence_paths(evidence_paths)}",
            "### 2) 参数/系数敏感性",
            f"- 本轮真实脚本：{self._compact_artifact_list(scripts, limit=3, filenames_only=False, markdown=True)}",
            f"- 本轮真实结果：{self._compact_artifact_list(results, limit=4, filenames_only=False, markdown=True)}",
        ]

        critic_summary = str(critic_report.get("summary", "") or "").strip()
        if critic_summary:
            lines.append(f"- Critic 审核：{critic_summary}")
        else:
            lines.append("- Critic 审核：本轮未触发或尚未形成结构化结论。")

        lines.extend(
            [
                "### 3) 物理洞察",
                f"- {physical_insight}",
            ]
        )
        if support_set != "未显式落盘":
            lines.append(f"- 当前支持集 `{support_set}` 需要继续通过跨工况一致性、消融和动力学积分来确认。")
        else:
            lines.append("- 当前轮还没有把封闭方程和支持集显式序列化到 hypothesis archive，需要后续补齐。")

        lines.append("### 4) 消融分析")
        if falsifications:
            for record in falsifications[:3]:
                title = str(record.get("title", "证伪记录")).strip()
                summary = str(record.get("summary", "") or record.get("result", "")).strip()
                lines.append(f"- {title}：{summary or '见对应证据文件。'}")
        else:
            required_evidence = critic_report.get("required_evidence", []) if isinstance(critic_report, dict) else []
            if isinstance(required_evidence, list) and required_evidence:
                lines.append(f"- 当前仍缺：{'; '.join(str(item).strip() for item in required_evidence[:3] if str(item).strip())}")
            else:
                lines.append("- 当前尚无已落盘的消融结果，需要下一轮补齐最小对抗验证路径。")
        return lines

    def _build_next_steps_section(
        self,
        *,
        rounds: list[int],
        hypotheses_by_round: dict[int, dict[str, Any]],
        falsifications_by_round: dict[int, list[dict[str, Any]]],
    ) -> list[str]:
        lines = ["## Worth Trying Next"]
        for round_num in rounds:
            hypothesis = hypotheses_by_round.get(round_num, {})
            critic_report = self._critic_report_for_round(round_num)
            falsifications = falsifications_by_round.get(round_num, [])
            results = self._history_round_inventory_for(round_num, "results")

            if falsifications:
                top_issue = falsifications[0]
                target = str(top_issue.get("title", "") or "补齐当前最强 blocker").strip()
                action = str(top_issue.get("rejection_reason", "") or top_issue.get("summary", "")).strip()
                expected = self._normalize_expected_artifacts(top_issue.get("expected_artifacts", []))
                acceptance = (
                    f"至少生成 {self._compact_artifact_list(expected, limit=2, filenames_only=False, markdown=True)}"
                    if expected
                    else "至少新增一份真实结果文件并把结论写入 trace。"
                )
            elif critic_report:
                required = critic_report.get("required_evidence", []) if isinstance(critic_report, dict) else []
                target = "把当前候选从结果工件升级为可审查结论"
                action = (
                    "；".join(str(item).strip() for item in required[:2] if str(item).strip())
                    if isinstance(required, list) and required
                    else "围绕当前 Critic summary 设计最小验证脚本与结果工件。"
                )
                acceptance = "Critic 触发后能读取到对应脚本、结果和结构化 hypothesis 记录。"
            elif results:
                target = "把当前轮结果转化为更清晰的结构候选"
                action = "补齐 hypothesis_archive 中的显式方程、支持集和结果摘要。"
                acceptance = "候选方程、支持集和证据路径三者同时落盘。"
            else:
                target = "让本轮从叙述推进到真实计算"
                action = "至少完成一次小而真的脚本执行，并把脚本与结果落到本轮目录。"
                acceptance = "history/roundN/scripts 与 results 都出现真实工件。"

            lines.extend(
                [
                    f"### Round {round_num} -> Next",
                    f"- 目标：{target}",
                    f"- 动作：{action or '补齐最小验证路径。'}",
                    f"- 验收：{acceptance}",
                    "",
                ]
            )
        return lines

    def _build_equation_evolution_section(self, hypotheses: list[dict[str, Any]]) -> list[str]:
        lines = ["## 最优方程演化"]
        if not hypotheses:
            lines.append("（当前暂无带真实结果工件的候选方程演化记录）")
            return lines

        for record in hypotheses:
            try:
                round_num = int(record.get("round", 0))
            except Exception:
                round_num = 0
            candidate = self._format_candidate_label(record)
            support_set = self._format_support_set(record)
            status = str(record.get("status", "") or "candidate_pending").strip()
            evidence = self._format_evidence_paths(record.get("evidence_paths", []))
            lines.append(
                f"- Round {round_num}：`{candidate}`；支持集：{support_set}；状态：`{status}`；证据：{evidence}"
            )
        return lines

    def _build_canonical_findings_text(self) -> str:
        hypotheses = self._hypothesis_records()
        latest_hypothesis = self._latest_hypothesis_record()
        falsifications = self._falsification_records()
        positive_hcc = self._positive_hcc_records()
        negative_hcc = self._negative_hcc_records()
        rounds = self._history_round_numbers()
        hypotheses_by_round = self._hypothesis_by_round()
        falsifications_by_round = self._records_grouped_by_round(falsifications)
        positive_hcc_by_round = self._records_grouped_by_round(positive_hcc)
        negative_hcc_by_round = self._records_grouped_by_round(negative_hcc)

        lines = ["# 研究发现", "", "## 关键洞察", "（逐轮记录机制判断、证据状态和 Critic 对抗反馈）", ""]
        for round_num in rounds:
            hypothesis = hypotheses_by_round.get(round_num)
            critic_report = self._critic_report_for_round(round_num)
            round_falsifications = falsifications_by_round.get(round_num, [])
            round_negative_hcc = negative_hcc_by_round.get(round_num, [])
            scripts = self._history_round_inventory_for(round_num, "scripts")
            results = self._history_round_inventory_for(round_num, "results")
            lines.extend(
                self._build_round_insight_lines(
                    round_num,
                    hypothesis=hypothesis,
                    critic_report=critic_report,
                    falsifications=round_falsifications,
                    negative_hcc=round_negative_hcc,
                    scripts=scripts,
                    results=results,
                )
            )

        lines.extend(
            [
                "## 实验结果",
                "| 轮次 | 方法 | 候选方程 | 支持集 | 结果工件 | Critic | 结论 |",
                "|------|------|----------|--------|----------|--------|------|",
            ]
        )
        for round_num in rounds:
            hypothesis = hypotheses_by_round.get(round_num)
            critic_report = self._critic_report_for_round(round_num)
            round_falsifications = falsifications_by_round.get(round_num, [])
            round_positive_hcc = positive_hcc_by_round.get(round_num, [])
            results = self._history_round_inventory_for(round_num, "results")
            lines.append(
                "| "
                + " | ".join(
                    [
                        self._sanitize_table_cell(str(round_num)),
                        self._sanitize_table_cell(self._format_route_label(hypothesis)),
                        self._sanitize_table_cell(self._format_candidate_label(hypothesis)),
                        self._sanitize_table_cell(self._format_support_set(hypothesis)),
                        self._sanitize_table_cell(self._compact_artifact_list(results, limit=3, filenames_only=False)),
                        self._sanitize_table_cell(self._critic_status_label(critic_report)),
                        self._sanitize_table_cell(
                            self._round_result_conclusion(
                                round_num=round_num,
                                hypothesis=hypothesis,
                                critic_report=critic_report,
                                falsifications=round_falsifications,
                                positive_hcc=round_positive_hcc,
                                results=results,
                            )
                        ),
                    ]
                )
                + " |"
            )

        analysis_round = None
        if latest_hypothesis:
            try:
                analysis_round = int(latest_hypothesis.get("round", 0)) or None
            except Exception:
                analysis_round = None
        if analysis_round is None and rounds:
            analysis_round = rounds[-1]

        analysis_hypothesis = hypotheses_by_round.get(analysis_round, latest_hypothesis if analysis_round else None)
        analysis_critic = self._critic_report_for_round(analysis_round) if analysis_round else {}
        analysis_falsifications = falsifications_by_round.get(analysis_round or -1, [])
        analysis_scripts = self._history_round_inventory_for(analysis_round, "scripts") if analysis_round else []
        analysis_results = self._history_round_inventory_for(analysis_round, "results") if analysis_round else []
        lines.extend(
            [
                "",
                *self._build_candidate_equation_section(
                    round_num=analysis_round,
                    hypothesis=analysis_hypothesis,
                    critic_report=analysis_critic,
                    falsifications=analysis_falsifications,
                    scripts=analysis_scripts,
                    results=analysis_results,
                ),
                "",
                *self._build_next_steps_section(
                    rounds=rounds,
                    hypotheses_by_round=hypotheses_by_round,
                    falsifications_by_round=falsifications_by_round,
                ),
                "",
                *self._build_equation_evolution_section(hypotheses),
            ]
        )
        return self._collapse_blank_lines("\n".join(lines))

    def _build_canonical_plan_text(self) -> str:
        latest_hypothesis = self._latest_hypothesis_record()
        scheduler_state = self._read_scheduler_state()
        debate_state = self._read_debate_state()
        routing_state = _read_json(self.run_dir / "routing_state.json", {}) if self.run_dir else {}

        best_round = latest_hypothesis.get("round", 0) if isinstance(latest_hypothesis, dict) else 0
        best_equation = latest_hypothesis.get("equation", "无") if isinstance(latest_hypothesis, dict) else "无"
        best_status = latest_hypothesis.get("status", "待定") if isinstance(latest_hypothesis, dict) else "待定"
        best_evidence = self._format_evidence_paths(latest_hypothesis.get("evidence_paths", [])) if isinstance(latest_hypothesis, dict) else "无"

        unresolved = debate_state.get("unresolved_challenges", []) if isinstance(debate_state, dict) else []
        lines = [
            "# 研究计划",
            "",
            "<!-- EVO_CURRENT_BEST_BEGIN -->",
            "## 当前最优",
            f"- 轮次：{best_round}",
            f"- 方程：{best_equation}",
            f"- 状态：{best_status}",
            f"- 证据：{best_evidence}",
            f"- 更新时间：{_utc_now()}",
            "<!-- EVO_CURRENT_BEST_END -->",
            "",
            "## 当前路由策略",
            f"- 当前阶段：{routing_state.get('current_strategy', '待定') if isinstance(routing_state, dict) else '待定'}",
            "- 选择理由：优先补齐 Critic 最新 blocker 对应的真实证据，而不是继续扩写叙述性结论。",
            f"- 下一轮优先工具：围绕未解决 challenge 生成真实 `scripts/` 与 `results/`，当前下一次周期审查轮次为 {scheduler_state.get('next_periodic_round', self.round_num + self._critic_periodic_every_n_rounds())}。",
            "",
            "## 当前假设",
        ]
        if latest_hypothesis:
            lines.append(f"1. {latest_hypothesis.get('equation') or latest_hypothesis.get('title') or '待定'}")
        else:
            lines.append("1. 待定")

        lines.extend(["", "## 当前 blocker"])
        if unresolved:
            for challenge in unresolved:
                if not isinstance(challenge, dict):
                    continue
                reason = self._compact_text(
                    challenge.get("blocking_reason") or challenge.get("summary") or "待补证据",
                    limit=220,
                )
                lines.append(
                    f"- [{challenge.get('challenge_type', 'challenge')}] {challenge.get('title', '未命名 blocker')}：{reason}"
                )
        else:
            lines.append("- 当前无未解决 blocker。")

        lines.extend(["", "## 下一步动作"])
        if unresolved:
            for idx, challenge in enumerate(unresolved, start=1):
                if not isinstance(challenge, dict):
                    continue
                required_evidence = challenge.get("required_evidence") or "补齐与该 challenge 对应的真实脚本和结果工件。"
                lines.append(f"{idx}. {required_evidence}")
        else:
            lines.append("1. 基于当前候选继续补齐真实验证证据。")

        return self._collapse_blank_lines("\n".join(lines))

    def _canonicalize_summary_documents(self) -> None:
        if not self.run_dir:
            return
        findings_path = self.run_dir / "findings.md"
        plan_path = self.run_dir / "plan.md"
        hypothesis_path = self.run_dir / "hypothesis_archive.jsonl"
        if hypothesis_path.exists():
            self._rewrite_text_if_changed(
                hypothesis_path,
                self._sanitize_hypothesis_archive_text(hypothesis_path.read_text(encoding="utf-8")),
            )
        findings_path.write_text(self._build_canonical_findings_text(), encoding="utf-8")
        plan_path.write_text(self._build_canonical_plan_text(), encoding="utf-8")

    def _freeze_dedupe_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return tuple(sorted((str(key), self._freeze_dedupe_value(val)) for key, val in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_dedupe_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_dedupe_value(item) for item in value))
        return value

    def _json_safe_dedupe_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._json_safe_dedupe_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe_dedupe_value(item) for item in value]
        return value

    def _extract_support_set_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        match = re.search(r"(支持集|support[\s_-]*set)[^{}]{0,40}\{([^}]*)\}", text, flags=re.IGNORECASE)
        if not match:
            return []
        items = [item.strip(" `") for item in re.split(r"[，,;；]", match.group(2)) if item.strip(" `")]
        return list(dict.fromkeys(items))

    def _extract_equation_candidate(self, text: str) -> str:
        if not text:
            return ""
        patterns = (
            r"(a\s*=\s*[^\n。；;]{6,220})",
            r"(x''\s*=\s*[^\n。；;]{6,220})",
            r"(方程[：:]\s*[^\n。；;]{6,220})",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return " ".join(match.group(1).split())[:220]
        return ""

    def _sanitize_candidate_summary(
        self,
        summary: str,
        *,
        evidence_paths: list[str] | None = None,
        status: str = "",
    ) -> str:
        text = self._collapse_blank_lines(summary or "").replace("\n", "\n")
        if not text.strip():
            return ""
        evidence_paths = evidence_paths or []
        candidate_reviewed = status in {"survived_attack", "critic_approved"}
        pending_markers = (
            r"(待验证|待后续验证|尚未|仍待|缺少|未完成|未验证|blocked|candidate|候选|下一步|后续|计划|建议)"
        )
        kept: list[str] = []
        skip_mode = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = re.sub(r"^[-*]\s*", "", line)
            normalized = re.sub(r"^\d+\.\s*", "", normalized)
            if normalized in {"当前成果：", "剩余缺口：", "下一步建议："}:
                skip_mode = normalized == "当前成果：" and not candidate_reviewed
                continue
            if skip_mode and (raw_line.startswith("- ") or re.match(r"^\d+\.", raw_line.strip())):
                continue
            if re.search(r"(下一步建议|当前成果|剩余缺口)", normalized):
                continue
            if not candidate_reviewed:
                refs = self._extract_claimed_artifact_refs(normalized)
                has_claims = bool(
                    self._extract_quantified_result_claims(normalized)
                    or self._extract_strong_conclusion_claims(normalized)
                )
                if has_claims and not refs:
                    continue
                if re.search(
                    r"(结构一致|恢复力系数|阻尼系数|符合自激|物理解释|稳态振幅可复现|跨风速稳定)",
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    continue
                if not re.search(pending_markers, normalized, flags=re.IGNORECASE):
                    continue
            kept.append(normalized)

        kept = list(dict.fromkeys(item for item in kept if item))
        if not kept:
            if evidence_paths:
                return "已生成候选并落盘当前轮次工件，仍待后续验证。"
            return "当前候选仍待后续验证。"
        return "；".join(kept[:4])[:800]

    def _sanitize_variable_memory_evidence(self, text: str) -> str:
        value = " ".join(str(text or "").split())
        if not value:
            return ""
        clauses = [piece.strip() for piece in re.split(r"[，,；;]", value) if piece.strip()]
        for clause in clauses:
            if self._extract_quantified_result_claims(clause) or self._extract_strong_conclusion_claims(clause):
                continue
            if any(re.search(pattern, clause, flags=re.IGNORECASE) for pattern in UNSAFE_VARIABLE_MEMORY_EVIDENCE_PATTERNS):
                continue
            return clause[:120]
        return ""

    def _append_system_hypothesis_record(
        self,
        *,
        hamilton_signal: dict[str, Any],
        aggregated_signal: dict[str, Any],
        critic_schedule: dict[str, Any],
    ) -> None:
        if not self.run_dir:
            return
        record = self._build_canonical_hypothesis_record(
            hamilton_signal=hamilton_signal,
            aggregated_signal=aggregated_signal,
            critic_schedule=critic_schedule,
        )
        if not record:
            return
        self._append_jsonl_record(
            self.run_dir / "hypothesis_archive.jsonl",
            record,
            dedupe_key=("hypothesis", self.round_num, tuple(record["evidence_paths"])),
        )

    def _append_system_falsification_records(
        self,
        *,
        critic_signal: dict[str, Any],
        critic_report: dict[str, Any],
        critic_schedule: dict[str, Any],
    ) -> None:
        if not self.run_dir or not critic_schedule.get("run_critic", False):
            return

        path = self.run_dir / "falsification_log.jsonl"
        for challenge in critic_report.get("challenges", []):
            if not isinstance(challenge, dict):
                continue
            if challenge.get("execution_status") != "executed":
                continue
            record = {
                "round": self.round_num,
                "title": challenge.get("title") or challenge.get("challenge_type") or "critic_challenge",
                "challenge_type": challenge.get("challenge_type", "evidence_gap_attack"),
                "summary": challenge.get("summary", ""),
                "result": critic_report.get("summary", "") or critic_signal.get("summary", ""),
                "rejection_reason": challenge.get("blocking_reason", ""),
                "evidence_paths": self._critic_real_evidence_paths(challenge),
                "expected_artifacts": self._normalize_expected_artifacts(challenge.get("expected_artifacts", [])),
            }
            self._append_jsonl_record(
                path,
                record,
                dedupe_key=("falsification", self.round_num, record["title"]),
            )

    def _canonicalize_routing_state(self) -> None:
        if not self.run_dir:
            return
        path = self.run_dir / "routing_state.json"
        original_text = path.read_text(encoding="utf-8") if path.exists() else ""
        original_valid = True
        try:
            payload = json.loads(original_text) if original_text.strip() else {}
        except Exception:
            original_valid = False
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        history_rounds = self._history_round_numbers()
        result_paths: list[str] = []
        for round_num in history_rounds:
            result_paths.extend(self._history_round_inventory_for(round_num, "results"))

        latest_coef_table = self._latest_history_result_matching(COEF_TABLE_FILENAMES)
        latest_rollout_result = self._latest_history_result_matching(ROLLOUT_RESULT_FILENAMES)
        latest_support_ablation = self._latest_support_ablation_artifact()
        latest_ablation_rollout = self._latest_ablation_rollout_artifact()
        has_coef_table = bool(latest_coef_table)
        has_rollout = bool(latest_rollout_result)
        has_ablation = bool(latest_support_ablation)
        has_ablation_rollout = bool(latest_ablation_rollout)

        existing_strategy = str(payload.get("current_strategy", "") or "").strip()

        if not has_coef_table:
            current_strategy = existing_strategy or "analytic_fit"
            current_focus = "先生成跨风速固定结构的真实系数表"
            next_stage = "integration_validation"
            next_action = "运行 lib/fit_viv_analytic.py 生成 coef_table.csv"
        elif not has_rollout:
            current_strategy = "integration_validation"
            current_focus = "基于最近可用的系数表做长时间积分验证"
            next_stage = "support_ablation"
            next_action = (
                f"运行 lib/validate_viv_rollout.py，优先复用最近可用的系数表 "
                f"({latest_coef_table or '待生成'}) 生成 amplitude_error.json"
            )
        elif not has_ablation:
            current_strategy = "support_ablation"
            current_focus = "验证支持集是否稳定并拆穿伪规律"
            next_stage = "ablation_validation"
            next_action = "运行 lib/support_ablation.py 生成 support_ablation.json"
        elif not has_ablation_rollout:
            current_strategy = "ablation_validation"
            current_focus = "对消融后的 3 项模型做长时间积分验证"
            next_stage = "support_ablation"
            next_action = (
                "运行 lib/validate_viv_rollout.py，直接复用最近的 support_ablation.json "
                f"({latest_support_ablation or '待生成'}) 生成 ablation_amplitude_error.json"
            )
        else:
            current_strategy = "support_ablation"
            current_focus = "继续围绕已生成结果做更强的结构攻击与补证"
            next_stage = "support_ablation"
            next_action = "围绕 Critic 最新 blocker 做 targeted ablation 或额外积分验证"

        history = payload.get("strategy_history", [])
        normalized_history: list[str] = []
        if isinstance(history, list):
            for item in history:
                value = str(item).strip()
                if value:
                    normalized_history.append(value)
        if not normalized_history or normalized_history[-1] != current_strategy:
            normalized_history.append(current_strategy)

        debate_state = self._read_debate_state()
        unresolved = debate_state.get("unresolved_challenges", []) if isinstance(debate_state, dict) else []
        blockers = []
        for challenge in unresolved:
            if not isinstance(challenge, dict):
                continue
            title = str(challenge.get("title", "") or challenge.get("challenge_type", "")).strip()
            if title and title not in blockers:
                blockers.append(title)

        payload.update(
            {
                "current_strategy": current_strategy,
                "strategy_history": normalized_history,
                "last_updated_round": self.round_num,
                "current_focus": current_focus,
                "priority_reason": "优先补齐真实结果链：系数表 -> 积分验证 -> 支持集消融。",
                "next_stage": next_stage,
                "next_action": next_action,
                "latest_coef_table": latest_coef_table,
                "latest_rollout_result": latest_rollout_result,
                "latest_support_ablation_result": latest_support_ablation,
                "latest_ablation_rollout_result": latest_ablation_rollout,
                "next_tools": [
                    "python lib/fit_viv_analytic.py --input-dir input --output history/roundN/results/coef_table.csv --summary-json history/roundN/results/fit_summary.json",
                    "python lib/validate_viv_rollout.py --input-dir input --coef-table history/roundN/results/coef_table.csv --output-json history/roundN/results/amplitude_error.json --output-csv history/roundN/results/rollout_metrics.csv --duration 200 --method Radau --step 0.01 --max-step 0.01",
                    "python lib/support_ablation.py --input-dir input --output history/roundN/results/support_ablation.json --variant no_v3v5",
                    "python lib/validate_viv_rollout.py --input-dir input --coef-table history/roundN/results/support_ablation.json --output-json history/roundN/results/ablation_amplitude_error.json --output-csv history/roundN/results/ablation_rollout_metrics.csv --duration 200 --method BDF --step 0.01 --max-step 0.005",
                ],
                "blockers": blockers,
            }
        )
        new_text = json.dumps(payload, ensure_ascii=False, indent=2)
        if (not original_valid) or (not path.exists()) or new_text.strip() != original_text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text, encoding="utf-8")

    def _build_verified_positive_hcc_summary(
        self,
        *,
        hamilton_signal: dict[str, Any],
        round_results: list[str],
        round_scripts: list[str],
    ) -> str:
        claim_binding = (hamilton_signal.get("completion_gate") or {}).get("claim_evidence_binding", {})
        verified_claims = [*claim_binding.get("quantified_claims", [])[:3]]
        summary_parts = [f"round_results={', '.join(round_results[:4])}"]
        if round_scripts:
            summary_parts.append(f"round_scripts={', '.join(round_scripts[:3])}")
        if verified_claims:
            summary_parts.append("verified_claims=" + " | ".join(verified_claims))
        return "；".join(summary_parts)[:1200]

    def _build_hcc_entry(
        self,
        *,
        polarity: str,
        producer_role: str,
        consumer_scope: str,
        title: str,
        summary: str,
        card_type: str,
        evidence_paths: list[str],
        evidence_strength: float,
        survived_attack: bool,
        attacked: bool,
        source: str,
        negative_evidence: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = "|".join(
            [
                str(self.round_num),
                polarity,
                producer_role,
                consumer_scope,
                card_type,
                title,
                summary,
                source,
            ]
        )
        entry_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
        return {
            "entry_id": entry_id,
            "round": self.round_num,
            "polarity": polarity,
            "producer_role": producer_role,
            "consumer_scope": consumer_scope,
            "title": title[:160],
            "summary": summary[:1200],
            "card_type": card_type,
            "evidence_paths": list(dict.fromkeys(path for path in evidence_paths if path)),
            "evidence_strength": round(max(0.0, min(float(evidence_strength), 1.0)), 3),
            "survived_attack": bool(survived_attack),
            "attacked": bool(attacked),
            "source": source,
            "negative_evidence": negative_evidence[:400],
            "metadata": metadata or {},
            "created_at": _utc_now(),
        }

    def _critic_real_evidence_paths(self, challenge: dict[str, Any] | None = None) -> list[str]:
        if not self.run_dir:
            return []
        report_md, report_json = self._critic_report_paths()
        _, attack_log_path, _, _ = self._critic_attack_paths()
        evidence: list[str] = []
        for path in (report_json, attack_log_path):
            if path.exists() and path.stat().st_size > 0:
                evidence.append(str(path.relative_to(self.run_dir)))
        evidence.extend(self._round_file_inventory(f"{CRITIC_ATTACKS_DIR}/results"))
        if isinstance(challenge, dict):
            for artifact in self._normalize_expected_artifacts(challenge.get("expected_artifacts", [])):
                path = self.run_dir / artifact
                if path.exists():
                    evidence.append(artifact)
        return list(dict.fromkeys(evidence))

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
                if self._line_has_unverified_result_claim(line):
                    continue
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

    def _normalize_markdown_table_row(self, line: str, expected_cells: int | None = None) -> str:
        stripped = line.strip()
        if not stripped.startswith("|"):
            return line.rstrip()
        if not stripped.endswith("|"):
            stripped = stripped + " |"
        cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
        if expected_cells and len(cells) < expected_cells:
            cells.extend([""] * (expected_cells - len(cells)))
        return "| " + " | ".join(cells) + " |"

    def _should_keep_findings_experiment_row(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("|"):
            return True
        if re.search(r"(待生成|待验证|尚未验证|未验证|planned)", stripped, flags=re.IGNORECASE):
            return False
        artifact_refs = self._extract_claimed_artifact_refs(stripped)
        if artifact_refs and self.run_dir:
            if any(not (self.run_dir / ref).exists() for ref in artifact_refs):
                return False
        has_claims = bool(
            self._extract_quantified_result_claims(stripped)
            or self._extract_strong_conclusion_claims(stripped)
        )
        if has_claims and not artifact_refs:
            return False
        return True

    def _should_keep_findings_bullet_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("- "):
            return True
        content = stripped[2:].strip()
        if re.match(r"(模板|当前候选|候选方程|方程草案|方程模板|候选)\s*[:：]", content, flags=re.IGNORECASE):
            return True
        if re.search(r"(待验证|待生成|未验证|计划|下一步|建议|候选|假设|blocker)", content, flags=re.IGNORECASE):
            return True
        artifact_refs = self._extract_claimed_artifact_refs(content)
        if artifact_refs and self.run_dir:
            if any(not (self.run_dir / ref).exists() for ref in artifact_refs):
                return False
        has_claims = bool(
            self._extract_quantified_result_claims(content)
            or self._extract_strong_conclusion_claims(content)
        )
        if has_claims and not artifact_refs:
            return False
        return True

    def _sanitize_findings_bullets(self, text: str) -> str:
        kept_lines: list[str] = []
        for raw_line in text.splitlines():
            if self._should_keep_findings_bullet_line(raw_line):
                kept_lines.append(raw_line.rstrip())
        return self._collapse_blank_lines("\n".join(kept_lines))

    def _sanitize_markdown_table_section(
        self,
        text: str,
        start_heading: str,
        end_heading: str,
        *,
        expected_cells: int | None = None,
        row_filter: Any | None = None,
    ) -> str:
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
                    header_lines.append(self._normalize_markdown_table_row(line, expected_cells=expected_cells))
                    continue
                normalized_line = self._normalize_markdown_table_row(line, expected_cells=expected_cells)
                if callable(row_filter) and not row_filter(normalized_line):
                    continue
                cells = [cell.strip() for cell in normalized_line.split("|")[1:-1]]
                key = cells[0] if cells else stripped
                if key not in row_order:
                    row_order.append(key)
                row_map[key] = normalized_line
            else:
                prefix_lines.append(line)

        rebuilt_lines = [*prefix_lines, *header_lines, *(row_map[key] for key in row_order)]
        rebuilt_section = ("\n".join(rebuilt_lines).strip() + "\n") if rebuilt_lines else ""
        rebuilt = before + start_heading + "\n" + rebuilt_section + "\n" + end_heading + after
        return self._collapse_blank_lines(rebuilt)

    def _sanitize_markdown_table_section_with_end_candidates(
        self,
        text: str,
        start_heading: str,
        end_headings: list[str],
        *,
        expected_cells: int | None = None,
        row_filter: Any | None = None,
    ) -> str:
        if start_heading not in text:
            return text

        start_index = text.find(start_heading)
        search_from = start_index + len(start_heading)
        end_index = None
        end_heading = None
        for candidate in end_headings:
            candidate_index = text.find(candidate, search_from)
            if candidate_index == -1:
                continue
            if end_index is None or candidate_index < end_index:
                end_index = candidate_index
                end_heading = candidate

        if end_index is None or end_heading is None:
            return text

        before = text[:start_index]
        section = text[search_from:end_index]
        after = text[end_index + len(end_heading) :]
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
                    header_lines.append(self._normalize_markdown_table_row(line, expected_cells=expected_cells))
                    continue
                normalized_line = self._normalize_markdown_table_row(line, expected_cells=expected_cells)
                if callable(row_filter) and not row_filter(normalized_line):
                    continue
                cells = [cell.strip() for cell in normalized_line.split("|")[1:-1]]
                key = cells[0] if cells else stripped
                if key not in row_order:
                    row_order.append(key)
                row_map[key] = normalized_line
            else:
                prefix_lines.append(line)

        rebuilt_lines = [*prefix_lines, *header_lines, *(row_map[key] for key in row_order)]
        rebuilt_section = ("\n".join(rebuilt_lines).strip() + "\n") if rebuilt_lines else ""
        rebuilt = before + start_heading + "\n" + rebuilt_section + "\n" + end_heading + after
        return self._collapse_blank_lines(rebuilt)

    def _sanitize_findings_text(self, text: str) -> str:
        text = self._collapse_blank_lines(text)
        text = self._sanitize_findings_bullets(text)
        text = self._sanitize_markdown_table_section(
            text,
            "## 变量角色结论",
            "## 实验结果",
            expected_cells=4,
        )
        text = self._sanitize_markdown_table_section(
            text,
            "## 实验结果",
            "## 证伪记录",
            expected_cells=8,
            row_filter=self._should_keep_findings_experiment_row,
        )
        text = self._sanitize_markdown_table_section(
            text,
            "## 证伪记录",
            "## 最优方程演化",
            expected_cells=5,
        )
        text = self._sanitize_markdown_table_section_with_end_candidates(
            text,
            "## 实验结果",
            ["## 候选方程解析", "## 候选方程解析（", "## Worth Trying Next", "## 最优方程演化"],
            expected_cells=7,
            row_filter=self._should_keep_findings_experiment_row,
        )
        return self._collapse_blank_lines(text)

    def _sanitize_variable_memory_json_text(self, text: str) -> str:
        try:
            payload = json.loads(text)
        except Exception:
            return text
        if not isinstance(payload, dict):
            return text

        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in SAFE_VARIABLE_MEMORY_TOP_LEVEL_KEYS:
                continue
            if key != "variables":
                sanitized[key] = value
                continue
            variables: dict[str, Any] = {}
            if isinstance(value, dict):
                for var_name, var_payload in value.items():
                    if not isinstance(var_payload, dict):
                        continue
                    filtered: dict[str, Any] = {}
                    for field, field_value in var_payload.items():
                        if field not in SAFE_VARIABLE_MEMORY_VARIABLE_KEYS:
                            continue
                        if field == "evidence":
                            sanitized_evidence = self._sanitize_variable_memory_evidence(str(field_value))
                            if sanitized_evidence:
                                filtered[field] = sanitized_evidence
                            continue
                        filtered[field] = field_value
                    variables[str(var_name)] = filtered
            sanitized["variables"] = variables

        sanitized.setdefault("variables", {})
        sanitized.setdefault("last_updated_round", payload.get("last_updated_round", 0))
        notes = payload.get("notes", [])
        if not isinstance(notes, list):
            notes = []
        sanitized["notes"] = [
            note.strip()
            for note in notes
            if isinstance(note, str) and note.strip() and self._is_safe_variable_memory_note(note)
        ]
        return json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n"

    def _is_safe_variable_memory_note(self, note: str) -> bool:
        text = str(note or "").strip()
        if not text:
            return False
        if self._extract_quantified_result_claims(text) or self._extract_strong_conclusion_claims(text):
            return False
        return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in UNSAFE_VARIABLE_MEMORY_NOTE_PATTERNS)

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

    def _sanitize_hypothesis_archive_text(self, text: str) -> str:
        sanitized_records: list[dict[str, Any]] = []
        seen: set[Any] = set()
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            evidence_paths = self._normalize_expected_artifacts(record.get("evidence_paths", []))
            record["evidence_paths"] = evidence_paths
            record["summary"] = self._sanitize_candidate_summary(
                str(record.get("summary", "")),
                evidence_paths=evidence_paths,
                status=str(record.get("status", "")),
            )
            frozen = self._freeze_dedupe_value(
                record.get("_system_key")
                or (
                    "hypothesis",
                    record.get("round"),
                    tuple(evidence_paths),
                )
            )
            if frozen in seen:
                continue
            seen.add(frozen)
            sanitized_records.append(record)
        return "\n".join(json.dumps(record, ensure_ascii=False) for record in sanitized_records) + (
            "\n" if sanitized_records else ""
        )

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
            elif relpath == "variable_memory.json":
                self._rewrite_text_if_changed(path, self._sanitize_variable_memory_json_text(path.read_text(encoding="utf-8")))
            elif relpath == "hypothesis_archive.jsonl":
                self._rewrite_text_if_changed(path, self._sanitize_hypothesis_archive_text(path.read_text(encoding="utf-8")))
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
        raw = raw_path.strip().strip("`'\"),.;:，。；：（）()[]【】")
        if not raw:
            return None
        normalized = raw.lstrip("./")
        if normalized.startswith("history/round"):
            return normalized if Path(normalized).suffix else None
        if normalized.startswith("scripts/") or normalized.startswith("results/"):
            candidate = Path("history") / f"round{self.round_num}" / normalized
            return str(candidate) if candidate.suffix else None
        return normalized if Path(normalized).suffix else None

    def _is_planned_artifact_ref(self, ref: str, context: str) -> bool:
        match = re.search(r"history/round(\d+)/", ref)
        if match:
            try:
                round_num = int(match.group(1))
            except Exception:
                round_num = self.round_num
            if round_num > self.round_num:
                return True
        return any(re.search(pattern, context, flags=re.IGNORECASE) for pattern in PLANNED_ARTIFACT_CONTEXT_PATTERNS)

    def _extract_claimed_artifact_refs(self, *texts: str) -> list[str]:
        refs: list[str] = []
        for text in texts:
            if not text:
                continue
            for match in CLAIMED_ROUND_ARTIFACT_RE.finditer(text):
                normalized = self._normalize_claimed_artifact_ref(match.group("path"))
                if normalized:
                    line_start = text.rfind("\n", 0, match.start()) + 1
                    line_end = text.find("\n", match.end())
                    if line_end == -1:
                        line_end = len(text)
                    context_line = text[line_start:line_end]
                    window_start = max(0, match.start() - 1200)
                    window_end = min(len(text), match.end() + 1200)
                    context_window = text[window_start:window_end]
                    combined_context = "\n".join({context_line, context_window})
                    if self._is_planned_artifact_ref(normalized, combined_context):
                        continue
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

    def _extract_strong_conclusion_claims(self, *texts: str) -> list[str]:
        claims: list[str] = []
        for text in texts:
            if not text:
                continue
            for pattern in STRONG_CONCLUSION_PATTERNS:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 60)
                    snippet = " ".join(text[start:end].split())
                    if snippet:
                        claims.append(snippet[:220])
        return list(dict.fromkeys(claims))

    def _line_has_unverified_result_claim(self, line: str) -> bool:
        return bool(
            self._extract_claimed_artifact_refs(line)
            or self._extract_quantified_result_claims(line)
            or self._extract_strong_conclusion_claims(line)
        )

    def _demote_unverified_round_findings(self) -> None:
        if not self.run_dir:
            return
        findings_path = self.run_dir / "findings.md"
        if not findings_path.exists():
            return
        text = findings_path.read_text(encoding="utf-8")
        heading = f"### Round {self.round_num}"
        if heading not in text:
            return
        lines = text.splitlines()
        rebuilt: list[str] = []
        replaced = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(heading):
                replaced = True
                rebuilt.append(f"### Round {self.round_num} 待验证假设")
                rebuilt.append(f"- 本轮尚无 `history/round{self.round_num}/results/*` 真实结果工件；原先强结论已降级为待验证假设。")
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.startswith("### ") or next_line.startswith("## "):
                        break
                    i += 1
                continue
            rebuilt.append(line)
            i += 1
        if replaced:
            findings_path.write_text(self._sanitize_findings_text("\n".join(rebuilt)), encoding="utf-8")

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
        strong_conclusion_claims = self._extract_strong_conclusion_claims(*all_texts)

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
        if strong_conclusion_claims and not round_results:
            missing_artifacts.append(self._round_results_relpath())
            warnings.append(
                "Strong conclusion claims were recorded without any round result artifacts: "
                + " | ".join(strong_conclusion_claims[:3])
            )

        return {
            "claimed_artifact_refs": claimed_artifact_refs,
            "missing_claimed_artifacts": list(dict.fromkeys(missing_claimed_artifacts)),
            "quantified_claims": quantified_claims,
            "strong_conclusion_claims": strong_conclusion_claims,
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
            if not round_results and (
                evidence_binding["quantified_claims"] or evidence_binding.get("strong_conclusion_claims")
            ):
                self._demote_unverified_round_findings()
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
                "strong_conclusion_claims": evidence_binding.get("strong_conclusion_claims", []),
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
        *,
        critic_schedule: dict[str, Any],
    ) -> dict[str, Any]:
        critic_ran = bool(critic_schedule.get("run_critic", False))
        critic_blocking = bool(critic_report.get("blocking", True)) if critic_ran else False
        critic_approved = bool(critic_report.get("approved", False)) and not critic_blocking if critic_ran else False
        unresolved = []
        if critic_ran:
            unresolved = [
                challenge["challenge_type"]
                for challenge in critic_report.get("challenges", [])
                if challenge.get("blocking", True)
            ]
        completion_gate = hamilton_signal.get("completion_gate") or {}
        if not completion_gate.get("accepted", True):
            unresolved = ["hamilton_artifact_gate", *unresolved]
        if not critic_ran and bool(hamilton_signal.get("satisfied", False)):
            unresolved = ["critic_pending_review", *unresolved]
        satisfied = bool(hamilton_signal.get("satisfied", False)) and critic_approved and not unresolved
        return {
            "round": self.round_num,
            "satisfied": satisfied,
            "task_completed": "true" if satisfied else "false",
            "hamilton_satisfied": bool(hamilton_signal.get("satisfied", False)),
            "critic_approved": critic_approved,
            "critic_ran": critic_ran,
            "critic_trigger_reasons": critic_schedule.get("trigger_reasons", []),
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
        *,
        critic_schedule: dict[str, Any],
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

        previous_unresolved = state.get("unresolved_challenges", [])
        challenges = critic_report.get("challenges", []) if critic_schedule.get("run_critic", False) else []
        unresolved = []
        resolved = []
        for challenge in challenges:
            challenge_copy = dict(challenge)
            challenge_copy["round"] = self.round_num
            if challenge_copy.get("blocking", True):
                unresolved.append(challenge_copy)
            else:
                resolved.append(challenge_copy)

        if not critic_schedule.get("run_critic", False):
            state["unresolved_challenges"] = previous_unresolved
        elif aggregated_signal.get("satisfied", False):
            state["resolved_challenges"] = state.get("resolved_challenges", []) + unresolved + resolved
            state["unresolved_challenges"] = []
        else:
            state["resolved_challenges"] = state.get("resolved_challenges", []) + resolved
            merged_unresolved: list[dict[str, Any]] = []
            seen_keys: set[tuple[str, str]] = set()
            resolved_keys = {
                (item.get("challenge_type", ""), item.get("title", ""))
                for item in resolved
                if isinstance(item, dict)
            }
            for item in [*previous_unresolved, *unresolved]:
                if not isinstance(item, dict):
                    continue
                key = (item.get("challenge_type", ""), item.get("title", ""))
                if key in resolved_keys or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_unresolved.append(item)
            state["unresolved_challenges"] = merged_unresolved

        state.setdefault("rounds", [])
        state["rounds"].append(
            {
                "round": self.round_num,
                "hamilton_signal": hamilton_signal,
                "critic_signal": critic_signal,
                "aggregated_signal": aggregated_signal,
                "critic_schedule": critic_schedule,
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
