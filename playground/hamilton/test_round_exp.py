"""Tests for Hamilton proposer/critic round orchestration."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from playground.hamilton.core.constants import CRITIC_REPORT_JSON
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
        self.calls = 0

    def run(self, task):
        self.calls += 1
        self.task_descriptions.append(task.description)
        update_keys = self.updates_per_call[min(self.calls - 1, len(self.updates_per_call) - 1)]
        message = self.messages_per_call[min(self.calls - 1, len(self.messages_per_call) - 1)]

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
            (self.workspace / "history" / "round1" / "trace.md").write_text(trace_text, encoding="utf-8")
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
            (self.workspace / "history" / "round1" / "scripts" / "fit_and_validate.py").write_text(
                "print('fit and validate')\n",
                encoding="utf-8",
            )
        if "result" in update_keys:
            (self.workspace / "history" / "round1" / "results" / "metrics.csv").write_text(
                "metric,value\nsupport_stability,0.9\n",
                encoding="utf-8",
            )

        return _make_finish_trajectory(self.task_completed, message)


class DummyCriticAgent:
    def __init__(self, workspace: Path, approved: bool):
        self.workspace = workspace
        self.approved = approved

    def run(self, task):
        round_dir = self.workspace / "history" / "round1"
        report_path = round_dir / CRITIC_REPORT_JSON
        payload = {
            "round": 1,
            "approved": self.approved,
            "blocking": not self.approved,
            "summary": "Critic reviewed the candidate.",
            "required_evidence": [] if self.approved else ["Need stronger validation"],
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
                }
            ],
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return _make_finish_trajectory("true" if self.approved else "false", "Critic review complete")


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
    ) -> tuple[dict, DummyHamiltonAgent]:
        hamilton_agent = DummyHamiltonAgent(
            self.tmpdir,
            task_completed=hamilton_task_completed,
            updates_per_call=updates_per_call,
            messages_per_call=messages_per_call,
        )
        exp = RoundExp(
            hamilton_agent=hamilton_agent,
            critic_agent=DummyCriticAgent(self.tmpdir, approved=critic_approved),
            config={},
            round_num=1,
            completion_policy=completion_policy or {},
            task_contract=task_contract,
        )
        exp.set_run_dir(self.tmpdir)
        return exp.run("Test task"), hamilton_agent

    def test_critic_can_block_completion(self) -> None:
        result, _ = self._run_round(
            critic_approved=False,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
        )
        self.assertFalse(result["signal"]["satisfied"])
        self.assertIn("evidence_gap_attack", result["signal"]["blocked_by"])

        debate_state = json.loads((self.tmpdir / "debate_state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(debate_state["unresolved_challenges"]), 1)

    def test_minimal_gate_requires_core_round_artifacts(self) -> None:
        result, hamilton_agent = self._run_round(
            critic_approved=True,
            updates_per_call=[set(), set()],
            completion_policy={"max_hamilton_repair_attempts": 1},
        )
        self.assertEqual(hamilton_agent.calls, 2)
        self.assertIn("System Completion Gate Feedback", hamilton_agent.task_descriptions[1])
        self.assertFalse(result["signal"]["satisfied"])
        self.assertIn("hamilton_artifact_gate", result["signal"]["blocked_by"])
        self.assertIn("plan.md", result["signal"]["missing_artifacts"])

    def test_contractless_finish_does_not_block_on_claimed_experiment_text(self) -> None:
        result, _ = self._run_round(
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
        result, _ = self._run_round(
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
        result, hamilton_agent = self._run_round(
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
        result, hamilton_agent = self._run_round(
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
        result, hamilton_agent = self._run_round(
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

    def test_invalid_jsonl_machine_state_rejects_completion(self) -> None:
        result, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "hypothesis_invalid"}],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        gate = result["hamilton_signal"]["completion_gate"]
        self.assertFalse(gate["accepted"])
        self.assertIn("machine_state_any_of", gate["missing_artifacts"])
        self.assertIn("hypothesis_archive.jsonl#valid_jsonl", gate["missing_artifacts"])

    def test_quantified_result_claim_without_result_artifact_is_rejected(self) -> None:
        result, hamilton_agent = self._run_round(
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
        result, _ = self._run_round(
            critic_approved=True,
            updates_per_call=[{"plan", "findings", "trace", "routing"}],
            messages_per_call=["结果已写入 history/round1/results/coef_table.json，支持集稳定"],
            completion_policy={"max_hamilton_repair_attempts": 0},
        )
        gate = result["hamilton_signal"]["completion_gate"]
        self.assertFalse(gate["accepted"])
        self.assertIn("history/round1/results/coef_table.json", gate["missing_artifacts"])
