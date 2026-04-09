"""Tests for Hamilton proposer/critic round orchestration."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from playground.hamilton.core.constants import (
    CRITIC_ATTACK_LOG_JSONL,
    CRITIC_ATTACK_PLAN_JSON,
    CRITIC_REPORT_JSON,
    CRITIC_SCHEDULER_STATE_FILE,
)
from playground.hamilton.core.exp import RoundExp


def _make_finish_trajectory(task_completed: str, message: str):
    tool_call = SimpleNamespace(
        function=SimpleNamespace(
            name="finish",
            arguments=json.dumps({"message": message, "task_completed": task_completed}, ensure_ascii=False),
        )
    )
    assistant_message = SimpleNamespace(role="assistant", content=message, tool_calls=[tool_call])
    dialog = SimpleNamespace(messages=[assistant_message])
    step = SimpleNamespace(assistant_message=assistant_message)
    return SimpleNamespace(status="completed", steps=[step], dialogs=[dialog])


class DummyHamiltonAgent:
    def __init__(
        self,
        workspace: Path,
        task_completed: str = "true",
        updates_per_call: list[set[str]] | None = None,
        messages_per_call: list[str] | None = None,
    ):
        self.workspace = workspace
        self.task_completed = task_completed
        self.updates_per_call = updates_per_call or [{"plan", "findings", "trace", "routing"}]
        self.messages_per_call = messages_per_call or ["Hamilton round complete"]
        self.task_descriptions: list[str] = []
        self.input_data_history: list[dict[str, object]] = []
        self.calls = 0

    def run(self, task):
        self.calls += 1
        self.task_descriptions.append(task.description)
        self.input_data_history.append(dict(getattr(task, "input_data", {}) or {}))
        update_keys = self.updates_per_call[min(self.calls - 1, len(self.updates_per_call) - 1)]
        message = self.messages_per_call[min(self.calls - 1, len(self.messages_per_call) - 1)]
        round_num = int(getattr(task, "input_data", {}).get("round", 1))
        round_dir = self.workspace / "history" / f"round{round_num}"

        if "plan" in update_keys:
            (self.workspace / "plan.md").write_text(
                "# 研究计划\n\n- 当前路由：support_then_verify\n",
                encoding="utf-8",
            )
        if "findings" in update_keys:
            (self.workspace / "findings.md").write_text(
                "# 研究发现\n\n- Hamilton 完成了本轮更新\n",
                encoding="utf-8",
            )
        if "trace" in update_keys:
            metrics_row = (
                "| fit_joint | linear+nonlinear | x,x^3,v,v^3,v^5 | shared-structure | 0.98 | stable | shared | ok |\n"
                if "trace_metrics" in update_keys
                else ""
            )
            falsification_row = (
                "| ablate_v5 | remove v^5 | ablation | diverged | v^5 necessary |\n"
                if "trace_falsification" in update_keys
                else ""
            )
            trace_text = (
                "# 执行日志 — 第 1 轮\n\n"
                "### 本轮策略\n"
                "- 路由决策：解析拟合 + 证伪\n"
                "- 目标支持集：x, x^3, v, v^3, v^5\n"
                "- 主要证伪对象：v^5\n\n"
                "### 操作记录\n"
                "- 已更新本轮日志\n\n"
                "### 指标记录\n"
                "| 实验 | 方法 | 支持集 | 关键参数/模板 | Fit | Support Stability | Structure | 备注 |\n"
                "|------|------|--------|----------------|-----|-------------------|-----------|------|\n"
                f"{metrics_row}\n"
                "### 证伪实验\n"
                "| 实验 | 比较候选 | 证伪方式 | 结果 | 结论 |\n"
                "|------|----------|----------|------|------|\n"
                f"{falsification_row}\n"
                "### 工作笔记\n"
                "- 已记录本轮观察\n"
            )
            (round_dir / "trace.md").write_text(trace_text, encoding="utf-8")
        if "routing" in update_keys:
            (self.workspace / "routing_state.json").write_text(
                json.dumps(
                    {
                        "current_strategy": "support_then_verify",
                        "strategy_history": [{"round": 1, "strategy": "support_then_verify"}],
                        "last_updated_round": 1,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if "routing_invalid" in update_keys:
            (self.workspace / "routing_state.json").write_text(
                '{"current_strategy": "support_then_verify"',
                encoding="utf-8",
            )
        if "variable_memory" in update_keys:
            (self.workspace / "variable_memory.json").write_text(
                json.dumps(
                    {"variables": {"v^5": {"role": "high-order damping", "evidence": "ablation"}}},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if "hypothesis" in update_keys:
            (self.workspace / "hypothesis_archive.jsonl").write_text(
                '{"round": 1, "structure": "-k1*x - c1*v"}\n',
                encoding="utf-8",
            )
        if "hypothesis_invalid" in update_keys:
            (self.workspace / "hypothesis_archive.jsonl").write_text(
                '{"round": 1, "structure": "-k1*x - c1*v"}\nnot-json\n',
                encoding="utf-8",
            )
        if "falsification_log" in update_keys:
            (self.workspace / "falsification_log.jsonl").write_text(
                '{"round": 1, "experiment": "ablate_v5", "result": "diverged"}\n',
                encoding="utf-8",
            )
        if "script" in update_keys:
            (round_dir / "scripts" / "fit_and_validate.py").write_text(
                "print('fit and validate')\n",
                encoding="utf-8",
            )
        if "result" in update_keys:
            (round_dir / "results" / "metrics.csv").write_text(
                "metric,value\nsupport_stability,0.9\n",
                encoding="utf-8",
            )
        if "coef_table" in update_keys:
            (round_dir / "results" / "coef_table.csv").write_text(
                "wind_speed,train_file,c_x,c_x3,c_v,c_v3,c_v5,r2,num_samples\n"
                "2.48,U248_train.csv,-163.0,0.2,-0.5,0.03,-0.002,0.98,200\n",
                encoding="utf-8",
            )

        return _make_finish_trajectory(self.task_completed, message)


class DummyCriticAgent:
    def __init__(
        self,
        workspace: Path,
        approved: bool,
        *,
        write_attack_artifacts: bool = True,
        write_report: bool = True,
        attack_plan_key: str = "interventions",
        finish_message: str | None = None,
    ):
        self.workspace = workspace
        self.approved = approved
        self.write_attack_artifacts = write_attack_artifacts
        self.write_report = write_report
        self.attack_plan_key = attack_plan_key
        self.finish_message = "Critic review complete" if finish_message is None else finish_message
        self.calls = 0

    def run(self, task):
        self.calls += 1
        round_num = int(getattr(task, "input_data", {}).get("round", 1))
        round_dir = self.workspace / "history" / f"round{round_num}"
        report_path = round_dir / CRITIC_REPORT_JSON
        payload = {
            "round": round_num,
            "approved": self.approved,
            "blocking": not self.approved,
            "summary": "Critic reviewed the candidate.",
            "required_evidence": [] if self.approved else ["Need stronger validation"],
            "attack_execution": {
                "execution_mode": "light_self_execute",
                "executed_interventions": 1 if self.write_attack_artifacts and not self.approved else 0,
                "status": "executed" if self.write_attack_artifacts and not self.approved else "not_executed",
            },
            "challenges": []
            if self.approved
            else [
                {
                    "challenge_type": "evidence_gap_attack",
                    "title": "Evidence is insufficient",
                    "summary": "Hamilton has not closed the main evidence gap.",
                    "blocking": True,
                    "required_evidence": "Provide stronger evidence",
                    "blocking_reason": "Current completion claim lacks support",
                    "intervention_type": "ood_slice_probe",
                        "target_claim": "support-set is stable",
                        "execution_status": "executed" if self.write_attack_artifacts else "not_executed",
                        "expected_artifacts": [
                        f"history/round{round_num}/critic_attacks/results/ood_probe.json",
                    ],
                }
            ],
        }
        if self.write_report:
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.write_attack_artifacts:
            (round_dir / CRITIC_ATTACK_PLAN_JSON).write_text(
                json.dumps(
                    {
                        "round": round_num,
                        self.attack_plan_key: [
                            {
                                "challenge_type": "ood_generalization_attack",
                                "title": "OOD probe challenge",
                                "intervention_type": "ood_slice_probe",
                                "target_claim": "support-set is stable",
                                "expected_artifacts": [
                                    f"history/round{round_num}/critic_attacks/results/ood_probe.json",
                                ],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (round_dir / CRITIC_ATTACK_LOG_JSONL).write_text(
                json.dumps(
                    {
                        "round": round_num,
                        "status": "executed",
                        "intervention_type": "ood_slice_probe",
                        "result_artifact": f"history/round{round_num}/critic_attacks/results/ood_probe.json",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            attack_results_dir = round_dir / "critic_attacks" / "results"
            attack_results_dir.mkdir(parents=True, exist_ok=True)
            (attack_results_dir / "ood_probe.json").write_text(
                json.dumps({"status": "failed_under_ood"}, ensure_ascii=False),
                encoding="utf-8",
            )
        return _make_finish_trajectory("true" if self.approved else "false", self.finish_message)


class TestRoundExp(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="hamilton_round_test_"))
        (self.tmpdir / "plan.md").write_text("# 初始计划\n", encoding="utf-8")
        (self.tmpdir / "findings.md").write_text("# 初始发现\n", encoding="utf-8")
        (self.tmpdir / "variable_memory.json").write_text(json.dumps({"variables": {}}, ensure_ascii=False), encoding="utf-8")
        (self.tmpdir / "routing_state.json").write_text(json.dumps({"strategy_history": []}, ensure_ascii=False), encoding="utf-8")
        (self.tmpdir / "hypothesis_archive.jsonl").write_text("", encoding="utf-8")
        (self.tmpdir / "falsification_log.jsonl").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_round(
        self,
        *,
        critic_approved: bool,
        updates_per_call: list[set[str]] | None = None,
        completion_policy: dict | None = None,
        hamilton_task_completed: str = "true",
        messages_per_call: list[str] | None = None,
        task_contract: dict | None = None,
        round_num: int = 1,
        write_attack_artifacts: bool = True,
        write_critic_report: bool = True,
        attack_plan_key: str = "interventions",
        critic_finish_message: str | None = None,
    ) -> tuple[dict, DummyHamiltonAgent, DummyCriticAgent]:
        hamilton_agent = DummyHamiltonAgent(
            self.tmpdir,
            task_completed=hamilton_task_completed,
            updates_per_call=updates_per_call,
            messages_per_call=messages_per_call,
        )
        critic_agent = DummyCriticAgent(
            self.tmpdir,
            approved=critic_approved,
            write_attack_artifacts=write_attack_artifacts,
            write_report=write_critic_report,
            attack_plan_key=attack_plan_key,
            finish_message=critic_finish_message,
        )
        exp = RoundExp(
            hamilton_agent=hamilton_agent,
            critic_agent=critic_agent,
            config={},
            round_num=round_num,
            completion_policy=completion_policy or {},
            task_contract=task_contract,
        )
        exp.set_run_dir(self.tmpdir)
        return exp.run("Test task"), hamilton_agent, critic_agent

    def test_critic_can_block_completion(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
        )
        self.assertFalse(result["signal"]["satisfied"])
        self.assertIn("evidence_gap_attack", result["signal"]["blocked_by"])

        debate_state = json.loads((self.tmpdir / "debate_state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(debate_state["unresolved_challenges"]), 1)

    def test_scheduler_skips_critic_without_new_results_on_non_periodic_round(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮只完成普通进展，没有新的结果文件"],
            round_num=1,
        )
        self.assertEqual(critic_agent.calls, 0)
        self.assertFalse(result["signal"]["satisfied"])
        self.assertFalse(result["critic_schedule"]["run_critic"])

    def test_scheduler_triggers_critic_when_new_results_exist(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "coef_table"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮已完成拟合并生成 history/round1/results/coef_table.csv"],
            round_num=1,
        )
        self.assertEqual(critic_agent.calls, 1)
        self.assertTrue(result["critic_schedule"]["run_critic"])

    def test_scheduler_does_not_trigger_on_challenge_response_without_new_results(self) -> None:
        (self.tmpdir / "debate_state.json").write_text(
            json.dumps(
                {
                    "unresolved_challenges": [
                        {
                            "challenge_type": "support_set_attack",
                            "title": "支持集仍不稳",
                            "blocking": True,
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮先回应上轮 blocker，但还没有新的结果文件"],
            round_num=2,
        )
        self.assertEqual(critic_agent.calls, 0)
        self.assertFalse(result["critic_schedule"]["run_critic"])

    def test_integration_validation_reuses_latest_coef_table_from_previous_round(self) -> None:
        round1_results = self.tmpdir / "history" / "round1" / "results"
        round1_results.mkdir(parents=True, exist_ok=True)
        (round1_results / "coef_table.csv").write_text(
            "wind_speed,train_file,c_x,c_x3,c_v,c_v3,c_v5,r2,num_samples\n"
            "2.48,U248_train.csv,-163.0,0.2,-0.5,0.03,-0.002,0.98,200\n",
            encoding="utf-8",
        )
        (self.tmpdir / "routing_state.json").write_text(
            json.dumps(
                {
                    "current_strategy": "integration_validation",
                    "strategy_history": ["analytic_fit", "integration_validation"],
                    "last_updated_round": 1,
                    "current_focus": "基于现有系数表做长时间积分验证",
                    "next_stage": "support_ablation",
                    "next_tools": [
                        "python lib/fit_viv_analytic.py --input-dir input --output history/roundN/results/coef_table.csv --summary-json history/roundN/results/fit_summary.json",
                        "python lib/validate_viv_rollout.py --input-dir input --coef-table history/roundN/results/coef_table.csv --output-json history/roundN/results/amplitude_error.json --output-csv history/roundN/results/rollout_metrics.csv --duration 200 --method Radau --step 0.01 --max-step 0.01",
                        "python lib/support_ablation.py --input-dir input --output history/roundN/results/support_ablation.json --variant no_v3v5",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        _, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮先复用上一轮的系数表做积分验证"],
            round_num=2,
        )

        first_input = hamilton_agent.input_data_history[0]
        self.assertEqual(first_input["current_strategy"], "integration_validation")
        self.assertEqual(first_input["stage_input_artifact"], "history/round1/results/coef_table.csv")
        self.assertIn("history/round1/results/coef_table.csv", str(first_input["recommended_command"]))
        self.assertIn("history/round2/results/amplitude_error.json", str(first_input["recommended_command"]))
        self.assertIn("--method Radau", str(first_input["recommended_command"]))

    def test_minimal_gate_requires_core_round_artifacts(self) -> None:
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[set(), set()],
            completion_policy={"max_hamilton_repair_attempts": 1},
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertIn("System Completion Gate Feedback", hamilton_agent.task_descriptions[1])
        self.assertFalse(result["signal"]["satisfied"])
        self.assertIn("hamilton_artifact_gate", result["signal"]["blocked_by"])
        self.assertIn("plan.md", result["signal"]["missing_artifacts"])

    def test_init_critic_artifacts_materializes_placeholders(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=2,
        )
        exp.set_run_dir(self.tmpdir)

        exp._init_critic_artifacts()

        round_dir = self.tmpdir / "history" / "round2"
        self.assertTrue((round_dir / CRITIC_REPORT_JSON).exists())
        self.assertTrue((round_dir / CRITIC_ATTACK_PLAN_JSON).exists())
        self.assertTrue((round_dir / CRITIC_ATTACK_LOG_JSONL).exists())
        report_payload = json.loads((round_dir / CRITIC_REPORT_JSON).read_text(encoding="utf-8"))
        self.assertTrue(report_payload["__placeholder__"])
        attack_payload = json.loads((round_dir / CRITIC_ATTACK_PLAN_JSON).read_text(encoding="utf-8"))
        self.assertTrue(attack_payload["__placeholder__"])
        self.assertEqual((round_dir / CRITIC_ATTACK_LOG_JSONL).read_text(encoding="utf-8"), "")

    def test_variable_memory_sanitizer_drops_unverified_scientific_notes(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=1,
        )
        exp.set_run_dir(self.tmpdir)

        payload = {
            "variables": {
                "x": {"role": "displacement", "unit": "mm", "evidence": "来自列定义"},
            },
            "last_updated_round": 1,
            "notes": [
                "核心变量已根据数据文件确认角色和单位",
                "本轮拟合得到的恢复力系数 k 跨风速一致，阻尼/自激系数随风速变化且符号符合物理预期",
            ],
        }
        (self.tmpdir / "variable_memory.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        exp._normalize_updated_workspace_state(["variable_memory.json"])
        sanitized = json.loads((self.tmpdir / "variable_memory.json").read_text(encoding="utf-8"))

        self.assertEqual(sanitized["notes"], ["核心变量已根据数据文件确认角色和单位"])

    def test_variable_memory_sanitizer_compacts_speculative_evidence_fields(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)

        payload = {
            "variables": {
                "x": {"role": "state_position", "unit": "mm", "evidence": "记录位移，恢复力项相关，跨风速系数稳定预期"},
                "v": {"role": "state_velocity", "unit": "mm/s", "evidence": "记录速度，阻尼项相关，系数随风速变化预期"},
                "a": {"role": "acceleration_target", "unit": "mm/s^2", "evidence": "目标变量，用于拟合候选方程"},
            },
            "last_updated_round": 3,
            "notes": [],
        }
        (self.tmpdir / "variable_memory.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        exp._normalize_updated_workspace_state(["variable_memory.json"])
        sanitized = json.loads((self.tmpdir / "variable_memory.json").read_text(encoding="utf-8"))

        self.assertEqual(sanitized["variables"]["x"]["evidence"], "记录位移")
        self.assertEqual(sanitized["variables"]["v"]["evidence"], "记录速度")
        self.assertEqual(sanitized["variables"]["a"]["evidence"], "目标变量")

    def test_variable_memory_sanitizer_drops_english_physics_interpretation_evidence(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=2,
        )
        exp.set_run_dir(self.tmpdir)
        (self.tmpdir / "variable_memory.json").write_text(
            json.dumps(
                {
                    "variables": {
                        "x": {"role": "state", "evidence": "Basis variable for restoring force term."},
                        "v": {"role": "state", "evidence": "Negative damping at small amplitudes."},
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        exp._normalize_updated_workspace_state(["variable_memory.json"])
        sanitized = json.loads((self.tmpdir / "variable_memory.json").read_text(encoding="utf-8"))
        self.assertNotIn("evidence", sanitized["variables"]["x"])
        self.assertNotIn("evidence", sanitized["variables"]["v"])

    def test_round_end_rebuilds_findings_and_plan_from_fact_ledgers(self) -> None:
        self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "script", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并生成 history/round1/results/metrics.csv"],
            round_num=1,
            write_attack_artifacts=True,
        )

        findings_text = (self.tmpdir / "findings.md").read_text(encoding="utf-8")
        plan_text = (self.tmpdir / "plan.md").read_text(encoding="utf-8")

        self.assertIn("## 关键洞察", findings_text)
        self.assertIn("### Round 1", findings_text)
        self.assertIn("## 实验结果", findings_text)
        self.assertIn("## 候选方程解析（Round 1)", findings_text)
        self.assertIn("## Worth Trying Next", findings_text)
        self.assertIn("## 最优方程演化", findings_text)
        self.assertIn("history/round1/results/metrics.csv", findings_text)
        self.assertIn("<!-- EVO_CURRENT_BEST_BEGIN -->", plan_text)
        self.assertIn("## 当前 blocker", plan_text)
        self.assertIn("Provide stronger evidence", plan_text)

    def test_round_end_auto_materializes_hypothesis_from_coef_table(self) -> None:
        self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "coef_table"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成拟合并生成 history/round1/results/coef_table.csv"],
            round_num=1,
            write_attack_artifacts=False,
        )
        records = [
            json.loads(line)
            for line in (self.tmpdir / "hypothesis_archive.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(records)
        self.assertEqual(records[0]["support_set"], ["x", "x^3", "v", "v^3", "v^5"])
        self.assertIn("c_x*x", records[0]["equation"])

    def test_integration_validation_result_inherits_previous_candidate(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=2,
        )
        exp.set_run_dir(self.tmpdir)
        (self.tmpdir / "history" / "round1" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round2" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round1" / "results" / "coef_table.csv").write_text(
            "wind_speed,train_file,c_x,c_x3,c_v,c_v3,c_v5,r2,num_samples\n"
            "2.48,U248_train.csv,-163.0,0.2,-0.5,0.03,-0.002,0.98,200\n",
            encoding="utf-8",
        )
        (self.tmpdir / "hypothesis_archive.jsonl").write_text(
            json.dumps(
                {
                    "round": 1,
                    "equation": "a = c_x*x + c_x3*x^3 + c_v*v + c_v3*v^3 + c_v5*v^5",
                    "support_set": ["x", "x^3", "v", "v^3", "v^5"],
                    "tool_path": "analytic_fit",
                    "evidence_paths": [
                        "history/round1/results/coef_table.csv",
                        "history/round1/results/fit_summary.json",
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.tmpdir / "history" / "round2" / "results" / "amplitude_error.json").write_text(
            json.dumps(
                {
                    "status": "partial_success",
                    "coef_table": "history/round1/results/coef_table.csv",
                    "num_successful_cases": 6,
                    "num_failed_cases": 4,
                    "mean_amplitude_error": 0.40,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        record = exp._build_canonical_hypothesis_record(
            hamilton_signal={
                "finish_message": "本轮完成积分验证。",
                "executed_strategy": "integration_validation",
                "completion_gate": {
                    "round_results": ["history/round2/results/amplitude_error.json"],
                    "round_scripts": [],
                },
            },
            aggregated_signal={"critic_approved": False},
            critic_schedule={"run_critic": True},
        )

        self.assertIsNotNone(record)
        self.assertEqual(record["tool_path"], "integration_validation")
        self.assertEqual(record["support_set"], ["x", "x^3", "v", "v^3", "v^5"])
        self.assertIn("c_x*x", record["equation"])
        self.assertIn("mean_amplitude_error=0.4", record["summary"])

    def test_support_ablation_result_materializes_candidate_and_fills_findings_rows(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)
        (self.tmpdir / "history" / "round2" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round3" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round2" / "results" / "amplitude_error.json").write_text(
            json.dumps(
                {
                    "status": "partial_success",
                    "coef_table": "history/round1/results/coef_table.csv",
                    "num_successful_cases": 6,
                    "num_failed_cases": 4,
                    "mean_amplitude_error": 0.40,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.tmpdir / "history" / "round3" / "results" / "support_ablation.json").write_text(
            json.dumps(
                {
                    "variant": "no_v3v5",
                    "support_set": ["x", "x3", "v"],
                    "equation": "a = c_x*x + c_x3*x^3 + c_v*v",
                    "mean_r2": 0.445,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        round2_record = {
            "round": 2,
            "title": "Round 2 candidate",
            "summary": "integration_validation 基于候选做动力学积分。",
            "equation": "a = c_x*x + c_x3*x^3 + c_v*v + c_v3*v^3 + c_v5*v^5",
            "support_set": ["x", "x^3", "v", "v^3", "v^5"],
            "tool_path": "integration_validation",
            "status": "candidate_blocked",
            "critic_status": "blocked",
            "evidence_paths": ["history/round2/results/amplitude_error.json"],
        }
        exp._append_jsonl_record(
            self.tmpdir / "hypothesis_archive.jsonl",
            round2_record,
            dedupe_key=("hypothesis", 2, tuple(round2_record["evidence_paths"])),
        )
        round3_record = exp._build_canonical_hypothesis_record(
            hamilton_signal={
                "finish_message": "本轮完成 no_v3v5 消融。",
                "executed_strategy": "support_ablation",
                "completion_gate": {
                    "round_results": ["history/round3/results/support_ablation.json"],
                    "round_scripts": [],
                },
            },
            aggregated_signal={"critic_approved": False},
            critic_schedule={"run_critic": True},
        )
        self.assertIsNotNone(round3_record)
        exp._append_jsonl_record(
            self.tmpdir / "hypothesis_archive.jsonl",
            round3_record,
            dedupe_key=("hypothesis", 3, tuple(round3_record["evidence_paths"])),
        )

        findings_text = exp._build_canonical_findings_text()

        self.assertIn("| 2 | integration_validation | a = c_x*x + c_x3*x^3 + c_v*v + c_v3*v^3 + c_v5*v^5 | x, x^3, v, v^3, v^5 |", findings_text)
        self.assertIn("| 3 | support_ablation | a = c_x*x + c_x3*x^3 + c_v*v | x, x^3, v |", findings_text)

    def test_routing_state_switches_to_ablation_validation_after_support_ablation(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=4,
        )
        exp.set_run_dir(self.tmpdir)
        (self.tmpdir / "history" / "round1" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round2" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round3" / "results").mkdir(parents=True, exist_ok=True)
        (self.tmpdir / "history" / "round1" / "results" / "coef_table.csv").write_text(
            "wind_speed,train_file,c_x,c_x3,c_v,c_v3,c_v5,r2,num_samples\n"
            "2.48,U248_train.csv,-163.0,0.2,-0.5,0.03,-0.002,0.98,200\n",
            encoding="utf-8",
        )
        (self.tmpdir / "history" / "round2" / "results" / "amplitude_error.json").write_text(
            json.dumps(
                {
                    "status": "partial_success",
                    "coef_table": "history/round1/results/coef_table.csv",
                    "num_successful_cases": 6,
                    "num_failed_cases": 4,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.tmpdir / "history" / "round3" / "results" / "support_ablation.json").write_text(
            json.dumps(
                {
                    "variant": "no_v3v5",
                    "support_set": ["x", "x3", "v"],
                    "equation": "a = c_x*x + c_x3*x^3 + c_v*v",
                    "mean_r2": 0.445,
                    "per_speed": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        exp._canonicalize_routing_state()
        state = json.loads((self.tmpdir / "routing_state.json").read_text(encoding="utf-8"))

        self.assertEqual(state["current_strategy"], "ablation_validation")
        self.assertEqual(state["latest_support_ablation_result"], "history/round3/results/support_ablation.json")
        self.assertIn("support_ablation.json", state["next_tools"][3])
        self.assertIn("ablation_amplitude_error.json", state["next_tools"][3])

    def test_planned_critic_challenges_do_not_write_falsification_log(self) -> None:
        self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "coef_table"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成拟合并生成 history/round1/results/coef_table.csv"],
            round_num=1,
            write_attack_artifacts=False,
        )
        self.assertEqual((self.tmpdir / "falsification_log.jsonl").read_text(encoding="utf-8").strip(), "")

    def test_negative_records_use_real_critic_artifacts_as_evidence(self) -> None:
        self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "script", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并生成 history/round1/results/metrics.csv"],
            round_num=1,
            write_attack_artifacts=True,
        )

        falsifications = [
            json.loads(line)
            for line in (self.tmpdir / "falsification_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ledger = [
            json.loads(line)
            for line in (self.tmpdir / "hcc_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertTrue(falsifications)
        self.assertIn("history/round1/critic_report.json", falsifications[0]["evidence_paths"])
        self.assertIn("history/round1/critic_attack_log.jsonl", falsifications[0]["evidence_paths"])
        self.assertTrue(any("history/round1/critic_report.json" in item.get("evidence_paths", []) for item in ledger))

    def test_attack_log_missing_status_counts_as_executed_and_syncs_plan(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)
        exp._ensure_round_dirs()
        exp._init_critic_artifacts()

        round_dir = self.tmpdir / "history" / "round3"
        (round_dir / CRITIC_REPORT_JSON).write_text(
            json.dumps(
                {
                    "round": 3,
                    "approved": False,
                    "blocking": True,
                    "summary": "缺失关键结果文件",
                    "required_evidence": ["生成 coefficients.csv"],
                    "challenges": [
                        {
                            "challenge_type": "evidence_gap_attack",
                            "title": "缺失 coefficients.csv",
                            "summary": "结果文件缺失",
                            "blocking": True,
                            "required_evidence": "生成 coefficients.csv",
                            "blocking_reason": "无法验证拟合结果",
                            "intervention_type": "result_file_check",
                            "target_claim": "拟合已完成",
                            "execution_status": "executed",
                            "expected_artifacts": ["history/round3/results/coefficients.csv"],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (round_dir / CRITIC_ATTACK_PLAN_JSON).write_text(
            json.dumps(
                {
                    "round": 3,
                    "execution_mode": "light_self_execute",
                    "max_interventions": 2,
                    "interventions": [
                        {
                            "challenge_type": "evidence_gap_attack",
                            "title": "缺失 coefficients.csv",
                            "intervention_type": "result_file_check",
                            "target": "history/round3/results/coefficients.csv",
                            "status": "pending",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (round_dir / CRITIC_ATTACK_LOG_JSONL).write_text(
            json.dumps(
                {
                    "round": 3,
                    "status": "missing",
                    "intervention_type": "result_file_check",
                    "target": "history/round3/results/coefficients.csv",
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        payload = exp._load_critic_report(critic_signal={})
        synced_plan = json.loads((round_dir / CRITIC_ATTACK_PLAN_JSON).read_text(encoding="utf-8"))

        self.assertEqual(payload["attack_execution"]["status"], "executed")
        self.assertEqual(payload["attack_execution"]["executed_interventions"], 1)
        self.assertEqual(synced_plan["interventions"][0]["status"], "executed")

    def test_contractless_finish_does_not_block_on_claimed_experiment_text(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            messages_per_call=["已完成拟合、仿真和证伪，等待下一轮"],
            task_contract=None,
        )
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        self.assertTrue(result["signal"]["critic_approved"])
        self.assertTrue(result["signal"]["satisfied"])
        self.assertEqual(result["signal"]["advisory_warnings"], [])

    def test_advisory_contract_warns_but_does_not_block_missing_evidence(self) -> None:
        task_contract = {
            "protocol": {
                "evidence_policy": "advisory",
                "review_focus": ["support_set", "physics_consistency"],
                "required_evidence": {
                    "fit": ["script", "result", "trace_metrics"],
                },
            },
            "meta": {"has_front_matter": True, "parse_errors": []},
        }
        result, _, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            task_contract=task_contract,
        )
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        self.assertTrue(result["signal"]["satisfied"])
        self.assertTrue(result["signal"]["advisory_warnings"])
        self.assertEqual(result["signal"]["task_contract_evidence_policy"], "advisory")

    def test_blocking_contract_blocks_missing_declared_evidence(self) -> None:
        task_contract = {
            "protocol": {
                "evidence_policy": "blocking",
                "review_focus": ["support_set"],
                "required_evidence": {
                    "fit": ["script", "result", "trace_metrics"],
                },
            },
            "meta": {"has_front_matter": True, "parse_errors": []},
        }
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}, {"plan", "findings", "trace", "routing"}],
            completion_policy={"max_hamilton_repair_attempts": 1},
            task_contract=task_contract,
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertFalse(result["hamilton_signal"]["completion_gate"]["accepted"])
        self.assertIn("history/round1/scripts/*", result["signal"]["missing_artifacts"])
        self.assertIn("history/round1/results/*", result["signal"]["missing_artifacts"])
        self.assertIn("history/round1/trace.md#指标记录", result["signal"]["missing_artifacts"])

    def test_blocking_contract_can_recover_after_required_evidence_is_written(self) -> None:
        task_contract = {
            "protocol": {
                "evidence_policy": "blocking",
                "review_focus": ["support_set"],
                "required_evidence": {
                    "fit": ["script", "result", "trace_metrics"],
                },
            },
            "meta": {"has_front_matter": True, "parse_errors": []},
        }
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[
                {"plan", "findings", "trace", "routing"},
                {"plan", "findings", "trace", "trace_metrics", "routing", "script", "result"},
            ],
            completion_policy={"max_hamilton_repair_attempts": 1},
            task_contract=task_contract,
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertIn("System Completion Gate Feedback", hamilton_agent.task_descriptions[1])
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        self.assertTrue(result["signal"]["critic_approved"])
        self.assertTrue(result["signal"]["satisfied"])

    def test_invalid_json_machine_state_blocks_finish_until_repaired(self) -> None:
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[
                {"plan", "findings", "trace", "routing_invalid"},
                {"plan", "findings", "trace", "routing"},
            ],
            messages_per_call=["本轮已更新状态", "已修复 routing_state 并完成"],
            completion_policy={"max_hamilton_repair_attempts": 1},
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertIn("routing_state.json#valid_json", hamilton_agent.task_descriptions[1])
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        routing_state = json.loads((self.tmpdir / "routing_state.json").read_text(encoding="utf-8"))
        self.assertEqual(routing_state["current_strategy"], "support_then_verify")

    def test_invalid_routing_state_is_canonicalized_before_hamilton_attempt(self) -> None:
        (self.tmpdir / "routing_state.json").write_text('{"bad":1}\n{"bad":2}', encoding="utf-8")
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "result"}],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        self.assertEqual(hamilton_agent.calls, 1)
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        routing_state = json.loads((self.tmpdir / "routing_state.json").read_text(encoding="utf-8"))
        self.assertIn("current_strategy", routing_state)

    def test_invalid_jsonl_machine_state_is_sanitized_before_completion(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "hypothesis_invalid"}],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        gate = result["hamilton_signal"]["completion_gate"]
        self.assertTrue(gate["accepted"])
        records = [
            json.loads(line)
            for line in (self.tmpdir / "hypothesis_archive.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 1)

    def test_quantified_result_claim_without_result_artifact_is_rejected(self) -> None:
        result, hamilton_agent, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[
                {"plan", "findings", "trace", "routing"},
                {"plan", "findings", "trace", "routing", "result"},
            ],
            messages_per_call=[
                "联合拟合已完成，R²约0.95，下一轮准备做更强证伪",
                "联合拟合已完成，结果已落到 history/round1/results/metrics.csv",
            ],
            completion_policy={"max_hamilton_repair_attempts": 1},
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertIn("history/round1/results/*", hamilton_agent.task_descriptions[1])
        self.assertTrue(result["hamilton_signal"]["completion_gate"]["accepted"])
        self.assertTrue(result["signal"]["critic_approved"])
        self.assertTrue(result["signal"]["satisfied"])

    def test_missing_claimed_result_path_blocks_finish(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            messages_per_call=["结果已写入 history/round1/results/coef_table.json，支持集稳定"],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        gate = result["hamilton_signal"]["completion_gate"]
        self.assertFalse(gate["accepted"])
        self.assertIn("history/round1/results/coef_table.json", gate["missing_artifacts"])

    def test_strong_conclusion_without_result_artifact_is_rejected(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            messages_per_call=["支持集已稳定为 {x, v, v^3, v^5}，方程结构已经确定"],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        gate = result["hamilton_signal"]["completion_gate"]
        self.assertFalse(gate["accepted"])
        self.assertIn("history/round1/results/*", gate["missing_artifacts"])
        self.assertTrue(gate["claim_evidence_binding"]["strong_conclusion_claims"])

    def test_critic_is_skipped_before_periodic_round_without_high_risk_trigger(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮只完成探索，不声称强结论"],
            round_num=1,
        )
        self.assertEqual(critic_agent.calls, 0)
        self.assertFalse(result["signal"]["critic_ran"])
        self.assertEqual(result["critic_report"]["summary"], "Critic skipped this round by scheduler.")
        scheduler_state = json.loads((self.tmpdir / CRITIC_SCHEDULER_STATE_FILE).read_text(encoding="utf-8"))
        self.assertEqual(scheduler_state["last_critic_round"], 0)

    def test_variable_memory_update_alone_does_not_trigger_critic(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "variable_memory"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮只完成变量角色整理，不声称结果"],
            round_num=1,
        )
        self.assertEqual(critic_agent.calls, 0)
        self.assertFalse(result["signal"]["critic_ran"])

    def test_periodic_round_triggers_critic(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮继续探索"],
            round_num=3,
        )
        self.assertEqual(critic_agent.calls, 1)
        self.assertTrue(result["signal"]["critic_ran"])
        self.assertIn("periodic_round", result["critic_schedule"]["trigger_reasons"])

    def test_finish_claim_triggers_critic_early(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="true",
            messages_per_call=["本轮已经完成主要目标"],
            round_num=1,
        )
        self.assertEqual(critic_agent.calls, 1)
        self.assertTrue(result["signal"]["critic_ran"])
        self.assertIn("hamilton_finish_true", result["critic_schedule"]["trigger_reasons"])

    def test_blocking_challenge_without_attack_artifacts_is_marked_not_executed(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="true",
            messages_per_call=["本轮已经完成主要目标"],
            round_num=1,
            write_attack_artifacts=False,
        )
        self.assertEqual(critic_agent.calls, 1)
        self.assertEqual(result["critic_report"]["attack_execution"]["status"], "not_executed")

    def test_missing_structured_critic_report_becomes_blocking_gap(self) -> None:
        result, _, critic_agent = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            hamilton_task_completed="true",
            messages_per_call=["本轮已经完成主要目标"],
            round_num=1,
            write_critic_report=False,
            write_attack_artifacts=False,
            critic_finish_message="",
        )
        self.assertEqual(critic_agent.calls, 1)
        self.assertTrue(result["signal"]["critic_ran"])
        self.assertFalse(result["signal"]["critic_approved"])
        self.assertEqual(result["critic_report"]["challenges"][0]["challenge_type"], "evidence_gap_attack")

    def test_missing_structured_critic_report_can_be_synthesized_from_attack_artifacts(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并落结果到 history/round1/results/metrics.csv"],
            round_num=1,
            write_critic_report=False,
            write_attack_artifacts=True,
            attack_plan_key="attacks",
        )
        self.assertTrue(result["signal"]["critic_ran"])
        self.assertEqual(result["critic_report"]["attack_execution"]["status"], "executed")
        self.assertNotEqual(result["critic_report"]["summary"], "Critic review is incomplete: critic did not materialize a structured report.")
        self.assertNotEqual(result["critic_report"]["challenges"][0]["title"], "Critic structured report missing")
        self.assertEqual(result["critic_report"]["challenges"][0]["execution_status"], "executed")

    def test_blocked_round_does_not_append_positive_hcc_entry(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并落结果到 history/round1/results/metrics.csv"],
            round_num=1,
        )
        self.assertTrue(result["signal"]["critic_ran"])
        ledger = [
            json.loads(line)
            for line in (self.tmpdir / "hcc_ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(ledger)
        self.assertFalse(any(item.get("polarity") == "positive" for item in ledger))
        self.assertTrue(any(item.get("polarity") == "negative" for item in ledger))

    def test_executed_critic_attack_is_synced_into_falsification_log(self) -> None:
        self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并落结果到 history/round1/results/metrics.csv"],
            round_num=1,
        )
        records = [
            json.loads(line)
            for line in (self.tmpdir / "falsification_log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(records)
        self.assertEqual(records[0]["challenge_type"], "evidence_gap_attack")
        self.assertTrue(records[0]["evidence_paths"])

    def test_hypothesis_archive_sync_handles_nested_system_keys(self) -> None:
        (self.tmpdir / "hypothesis_archive.jsonl").write_text(
            json.dumps(
                {
                    "round": 1,
                    "title": "Round 1 candidate",
                    "summary": "已有候选",
                    "_system_key": ["hypothesis", 1, ["history/round1/results/coef_table.csv"]],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        result, _, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "result"}],
            hamilton_task_completed="false",
            messages_per_call=[
                "本轮完成跨风速拟合并落结果到 history/round2/results/coef_table.csv；支持集 {x, x^3, v, v^3, v^5}"
            ],
            round_num=2,
        )
        self.assertTrue(result["signal"]["critic_ran"])
        records = [
            json.loads(line)
            for line in (self.tmpdir / "hypothesis_archive.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["round"], 2)
        self.assertTrue(records[-1]["evidence_paths"])

    def test_findings_sanitizer_drops_unbacked_quantified_bullets(self) -> None:
        findings_path = self.tmpdir / "findings.md"
        findings_path.write_text(
            "# 研究发现\n\n"
            "## 关键洞察\n"
            "### Round 1\n"
            "- 机制判读：阻尼项系数随风速单调变化，振幅误差约5%~10%\n"
            "- 对抗反馈：当前候选，待验证\n\n"
            "## 实验结果\n"
            "| 轮次 | 方法 | 候选方程 | 支持集 | 结果工件 | Critic | 结论 |\n"
            "|------|------|----------|--------|----------|--------|------|\n"
            "| 1 | analytic_fit | x'' + ω²x + αx³ + βv + γv³ + δv⁵ = 0 | x,v | 无 | 未触发 | 待验证 |\n\n"
            "## 候选方程解析（Round 1)\n"
            "### 1) 方程与物理解释\n"
            "- 模板：x'' + ω²x + αx³ + βv + γv³ + δv⁵ = 0\n"
            "- 瞬时拟合R²约0.85~0.9，振幅误差约5%~10%\n"
            "## Worth Trying Next\n"
            "### Round 1 -> Next\n"
            "- 目标：当前候选，待验证\n\n"
            "## 最优方程演化\n",
            encoding="utf-8",
        )
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=1,
        )
        exp.set_run_dir(self.tmpdir)
        exp._normalize_updated_workspace_state(["findings.md"])
        sanitized = findings_path.read_text(encoding="utf-8")
        self.assertNotIn("振幅误差约5%~10%", sanitized)
        self.assertIn("当前候选，待验证", sanitized)
        self.assertIn("模板：x'' + ω²x + αx³ + βv + γv³ + δv⁵ = 0", sanitized)

    def test_hypothesis_archive_sanitizer_removes_unverified_result_summary(self) -> None:
        hypothesis_path = self.tmpdir / "hypothesis_archive.jsonl"
        hypothesis_path.write_text(
            json.dumps(
                {
                    "round": 2,
                    "title": "Round 2 candidate",
                    "summary": (
                        "本轮已完成解析拟合路线。\n\n"
                        "当前成果：\n"
                        "- 拟合得到结构一致的方程族，恢复力系数跨风速稳定。\n"
                        "- 瞬时拟合精度中等（R²≈0.33~0.51）。\n\n"
                        "剩余缺口：\n"
                        "- 尚未进行 solve_ivp 积分验证极限环及稳态振幅误差。"
                    ),
                    "status": "candidate_blocked",
                    "evidence_paths": ["history/round2/results/poly_fit_results.json"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)
        exp._canonicalize_summary_documents()
        records = [
            json.loads(line)
            for line in hypothesis_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 1)
        self.assertIn("尚未进行 solve_ivp", records[0]["summary"])
        self.assertNotIn("恢复力系数跨风速稳定", records[0]["summary"])
        self.assertNotIn("R²≈0.33~0.51", records[0]["summary"])

    def test_hypothesis_archive_sanitizer_collapses_unreviewed_physics_interpretation(self) -> None:
        hypothesis_path = self.tmpdir / "hypothesis_archive.jsonl"
        hypothesis_path.write_text(
            json.dumps(
                {
                    "round": 2,
                    "title": "Round 2 candidate",
                    "summary": (
                        "U248风速拟合结果表明a/b项提供恢复力，c项在小振幅时为负阻尼，"
                        "大振幅正阻尼由d/e项提供，高次项虽小但对极限环稳定性重要。"
                    ),
                    "status": "candidate_blocked",
                    "evidence_paths": ["history/round2/results/poly_fit_U248_coeffs.csv"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)
        exp._canonicalize_summary_documents()
        records = [
            json.loads(line)
            for line in hypothesis_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(records[0]["summary"], "已生成候选并落盘当前轮次工件，仍待后续验证。")

    def test_missing_structured_critic_report_can_be_synthesized_from_finish_message(self) -> None:
        result, _, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing", "result"}],
            hamilton_task_completed="false",
            messages_per_call=["本轮完成解析拟合并落结果到 history/round1/results/metrics.csv"],
            round_num=1,
            write_attack_artifacts=False,
            write_critic_report=False,
            critic_finish_message=(
                "关键结果文件 history/round1/results/metrics.csv 无法支撑当前结论，"
                "需补充真实结果文件并完成 OOD 验证，因此 blocking=true。"
            ),
        )
        self.assertTrue(result["signal"]["critic_ran"])
        self.assertNotEqual(result["critic_report"]["summary"], "Critic review is incomplete: critic did not materialize a structured report.")
        self.assertEqual(result["critic_report"]["challenges"][0]["title"], "OOD 泛化尚未验证")
        self.assertEqual(result["critic_report"]["challenges"][0]["challenge_type"], "ood_generalization_attack")
        self.assertIn("history/round1/results/metrics.csv", result["critic_report"]["challenges"][0]["expected_artifacts"])
        self.assertTrue((self.tmpdir / "history" / "round1" / CRITIC_ATTACK_PLAN_JSON).exists())
        self.assertTrue((self.tmpdir / "history" / "round1" / CRITIC_ATTACK_LOG_JSONL).exists())
        attack_log = [
            json.loads(line)
            for line in (self.tmpdir / "history" / "round1" / CRITIC_ATTACK_LOG_JSONL).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue(attack_log)
        self.assertEqual(attack_log[0]["status"], "not_executed")

    def test_claimed_artifact_refs_ignore_future_round_plan_paths(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=3,
        )
        exp.set_run_dir(self.tmpdir)
        refs = exp._extract_claimed_artifact_refs(
            "下一轮建议：编写并运行 history/round4/scripts/integrate_viv.py，并把结果写入 history/round4/results/amplitude.csv"
        )
        self.assertEqual(refs, [])
        current_refs = exp._extract_claimed_artifact_refs("本轮结果已落盘到 history/round3/results/metrics.csv")
        self.assertEqual(current_refs, ["history/round3/results/metrics.csv"])

    def test_claimed_artifact_refs_ignore_routing_next_tools_outputs(self) -> None:
        exp = RoundExp(
            hamilton_agent=DummyHamiltonAgent(self.tmpdir),
            critic_agent=DummyCriticAgent(self.tmpdir, approved=False),
            config={},
            round_num=1,
        )
        exp.set_run_dir(self.tmpdir)
        refs = exp._extract_claimed_artifact_refs(
            json.dumps(
                {
                    "current_strategy": "analytic_fit",
                    "next_tools": [
                        "python lib/validate_viv_rollout.py --input-dir input --coef-table history/round1/results/coef_table.csv "
                        "--output-json history/round1/results/amplitude_error.json --output-csv history/round1/results/rollout_metrics.csv",
                        "python lib/support_ablation.py --input-dir input --output history/round1/results/support_ablation.json --variant no_v3v5",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        self.assertEqual(refs, [])
