"""Tests for Hamilton L3 memory store."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from playground.hamilton.core.constants import HCC_LEDGER_FILE
from playground.hamilton.core.l3 import L3MemoryStore


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestL3MemoryStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="hamilton_l3_test_"))
        self.store = L3MemoryStore(self.tmpdir / "l3_store", top_k=5)
        self.workspace_a = self.tmpdir / "workspace_a"
        self.workspace_b = self.tmpdir / "workspace_b"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _seed_workspace(self, workspace: Path, *, with_verified_evidence: bool = False) -> None:
        _write(
            workspace / "plan.md",
            """# 研究计划

<!-- EVO_CURRENT_BEST_BEGIN -->
## 当前最优
- 方程：x'' + 163*x + 0.2*x^3 + c(v)*x' = 0
- 支持集：x, v
<!-- EVO_CURRENT_BEST_END -->
""",
        )
        _write(
            workspace / "findings.md",
            """# 研究发现

## 关键洞察
- 变量 x 是核心恢复力变量
- 变量 v 同时承担阻尼与自激信息
""",
        )
        _write(
            workspace / "variable_memory.json",
            json.dumps(
                {
                    "variables": {
                        "x": {"role": "core", "evidence": {"leave_one_out_delta": 0.8}, "notes": ["恢复力主导"]},
                        "v": {"role": "core", "evidence": {"ood_stability": 0.7}, "notes": ["阻尼相关"]},
                        "proxy_u": {"role": "proxy", "evidence": {"swap_sensitivity": 0.9}, "notes": ["易诱导伪解"]},
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _write(
            workspace / "routing_state.json",
            json.dumps(
                {
                    "strategy_history": [
                        {"strategy": "redundancy_profile", "reason": "先筛支持集"},
                        {"strategy": "pysr_template_search", "reason": "支持集趋于稳定后再搜索结构"},
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        _write(
            workspace / "hypothesis_archive.jsonl",
            json.dumps({"equation": "a*x + b*x^3 + c*v", "support_set": ["x", "v"], "tool_path": "pysr"}, ensure_ascii=False)
            + "\n",
        )
        _write(
            workspace / "falsification_log.jsonl",
            json.dumps(
                {
                    "title": "proxy_u 替换失败",
                    "summary": "把 proxy_u 换入支持集后，OOD 振幅立即失稳",
                    "rejection_reason": "proxy variable caused unstable trajectory",
                },
                ensure_ascii=False,
            )
            + "\n",
        )
        if with_verified_evidence:
            _write(
                workspace / "history" / "round1" / "results" / "metrics.json",
                json.dumps({"fit_r2": 0.95, "support_stability": 0.9}, ensure_ascii=False),
            )
            _write(
                workspace / HCC_LEDGER_FILE,
                json.dumps(
                    {
                        "round": 1,
                        "polarity": "positive",
                        "producer_role": "hamilton",
                        "consumer_scope": "both",
                        "title": "结构在攻击后仍成立",
                        "summary": "共享结构在跨工况验证后仍成立",
                        "card_type": "operator_motif",
                        "evidence_paths": ["history/round1/results/metrics.json"],
                        "evidence_strength": 0.9,
                        "survived_attack": True,
                        "attacked": True,
                        "source": "critic_approval",
                    },
                    ensure_ascii=False,
                )
                + "\n"
                + json.dumps(
                    {
                        "round": 1,
                        "polarity": "negative",
                        "producer_role": "critic",
                        "consumer_scope": "both",
                        "title": "proxy_u 会诱导伪规律",
                        "summary": "引入 proxy_u 后 OOD 立即失稳",
                        "card_type": "failure_card",
                        "evidence_paths": ["history/round1/results/metrics.json"],
                        "evidence_strength": 0.8,
                        "survived_attack": False,
                        "attacked": True,
                        "source": "critic_report",
                        "negative_evidence": "OOD failure",
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )

    def test_promote_and_retrieve_cross_task_cards(self) -> None:
        self._seed_workspace(self.workspace_a, with_verified_evidence=True)

        signature_a = self.store.build_task_signature(
            "VIV equation discovery with support set stability, physics consistency and OOD validation",
            task_id="viv_task_a",
        )
        self.store.materialize_runtime_context(self.workspace_a, signature_a)
        summary = self.store.promote_task(
            self.workspace_a,
            "VIV equation discovery with support set stability, physics consistency and OOD validation",
            task_id="viv_task_a",
            experiment_record={"rounds": [{"signal": {"critic_approved": True}}]},
        )

        self.assertGreaterEqual(summary["card_count"], 4)

        signature_b = self.store.build_task_signature(
            "VIV support set recovery under redundant variables with OOD trajectory validation",
            task_id="viv_task_b",
        )
        hits = self.store.retrieve(signature_b)
        card_types = {card["card_type"] for card in hits["cards"]}

        self.assertTrue(hits["cards"])
        self.assertIn("support_set_prior", card_types)
        self.assertIn("failure_card", card_types)

    def test_promote_skips_positive_transfer_cards_without_verified_evidence(self) -> None:
        self._seed_workspace(self.workspace_a, with_verified_evidence=False)

        signature_a = self.store.build_task_signature(
            "VIV equation discovery with support set stability, physics consistency and OOD validation",
            task_id="viv_task_a",
        )
        self.store.materialize_runtime_context(self.workspace_a, signature_a)
        summary = self.store.promote_task(
            self.workspace_a,
            "VIV equation discovery with support set stability, physics consistency and OOD validation",
            task_id="viv_task_a",
            experiment_record={"rounds": [{"signal": {"critic_approved": False}}]},
        )

        self.assertGreaterEqual(summary["card_count"], 1)
        signature_b = self.store.build_task_signature(
            "VIV support set recovery under redundant variables with OOD trajectory validation",
            task_id="viv_task_b",
        )
        cards = self.store.retrieve(signature_b)["cards"]
        card_types = {card["card_type"] for card in cards}
        self.assertNotIn("support_set_prior", card_types)
        self.assertNotIn("operator_motif", card_types)
        self.assertNotIn("domain_prior", card_types)
        self.assertIn("failure_card", card_types)
        self.assertFalse(any(card["polarity"] == "positive" for card in cards))

    def test_hcc_ledger_cards_preserve_polarity_metadata(self) -> None:
        self._seed_workspace(self.workspace_a, with_verified_evidence=True)
        signature_a = self.store.build_task_signature(
            "VIV equation discovery with support set stability and attack-based validation",
            task_id="viv_task_a",
        )
        self.store.materialize_runtime_context(self.workspace_a, signature_a)
        self.store.promote_task(
            self.workspace_a,
            "VIV equation discovery with support set stability and attack-based validation",
            task_id="viv_task_a",
            experiment_record={"rounds": [{"signal": {"critic_approved": True}}]},
        )

        signature_b = self.store.build_task_signature(
            "VIV support set recovery under redundant variables with OOD trajectory validation",
            task_id="viv_task_b",
        )
        cards = self.store.retrieve(signature_b)["cards"]
        operator_cards = [card for card in cards if card["card_type"] == "operator_motif"]
        failure_cards = [card for card in cards if card["card_type"] == "failure_card"]
        self.assertTrue(operator_cards)
        self.assertTrue(failure_cards)
        self.assertTrue(any(card["polarity"] == "positive" and card["survived_attack"] for card in operator_cards))
        self.assertTrue(any(card["polarity"] == "negative" and card["producer_role"] == "critic" for card in failure_cards))


if __name__ == "__main__":
    unittest.main()
