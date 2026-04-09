"""Hamilton Playground implementation with L3 memory and proposer/critic flow."""

from __future__ import annotations

import importlib.util
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure evomaster is importable when playground is run directly.
_module_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_module_root) not in sys.path:
    sys.path.insert(0, str(_module_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.core.task_contract import TASK_CONTRACT_FILE

from .constants import (
    CRITIC_CONTEXT_FILE,
    CRITIC_SCHEDULER_STATE_FILE,
    CURRENT_BEST_BEGIN,
    CURRENT_BEST_END,
    DEBATE_STATE_FILE,
    ENV_CAPABILITIES_FILE,
    EVALUATION_CONTEXT_FILE,
    EVALUATION_CONTEXT_MD,
    HCC_LEDGER_FILE,
    L3_CONTEXT_FILE,
    STRATEGY_QUEUE_BEGIN,
    STRATEGY_QUEUE_END,
)
from .evaluation import build_evaluation_context, materialize_evaluation_context
from .exp import RoundExp
from .l3 import L3MemoryStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@register_playground("hamilton")
class HamiltonPlayground(BasePlayground):
    """Hamilton Playground - redundancy-aware symbolic regression agent."""

    DEFAULT_WORKSPACE_ASSETS = ("input", "lib")

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        self._project_root = Path(__file__).resolve().parent.parent.parent.parent
        if config_path is None and config_dir is None:
            config_dir = self._project_root / "configs" / "hamilton"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.workspace_dir: Path | None = None
        self.hamilton_agent = None
        self.critic_agent = None
        self.l3_store: L3MemoryStore | None = None
        self.l3_config: dict[str, Any] = {}
        self.critic_policy: dict[str, Any] = {}
        self.completion_policy: dict[str, Any] = {}

        self.experiment_record = {
            "task": "",
            "rounds": [],
            "start_time": datetime.now().isoformat(),
        }

    def set_run_dir(self, run_dir: str | Path, task_id: str | None = None) -> None:
        super().set_run_dir(run_dir, task_id=task_id)

    def _resolve_project_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        return (self._project_root / path).resolve()

    def _get_extra_section(self, name: str) -> dict[str, Any]:
        section = getattr(self.config, name, {})
        return section if isinstance(section, dict) else {}

    def _get_l3_config(self) -> dict[str, Any]:
        memory_cfg = self._get_extra_section("memory")
        l3_cfg = memory_cfg.get("l3", {}) if isinstance(memory_cfg, dict) else {}
        if not isinstance(l3_cfg, dict):
            l3_cfg = {}
        return {
            "enabled": bool(l3_cfg.get("enabled", True)),
            "root": l3_cfg.get("root", "./runs/hamilton_l3"),
            "top_k": int(l3_cfg.get("top_k", 6) or 6),
            "retrieval_mode": l3_cfg.get("retrieval_mode", "hybrid"),
            "promotion_policy": l3_cfg.get("promotion_policy", "task_end_only"),
        }

    def _workspace_template_dir(self) -> Path:
        return self._project_root / "playground" / "hamilton" / "workspace"

    def _copy_asset_path(self, source: Path, destination: Path) -> int:
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return 1

        if not source.is_dir():
            return 0

        copied_files = 0
        for source_path in sorted(source.rglob("*")):
            if source_path.is_dir():
                continue
            relative_path = source_path.relative_to(source)
            target_path = destination / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_files += 1
        return copied_files

    def _materialize_workspace_assets(self, workspace: Path) -> dict[str, Any]:
        template_dir = self._workspace_template_dir()
        asset_summary: dict[str, Any] = {}

        for relative_name in self.DEFAULT_WORKSPACE_ASSETS:
            source = template_dir / relative_name
            if not source.exists():
                self.logger.warning("Workspace asset source missing: %s", source)
                continue

            destination = workspace / relative_name
            copied_files = self._copy_asset_path(source, destination)
            asset_summary[relative_name] = {
                "source": str(source),
                "destination": str(destination),
                "files_copied": copied_files,
            }
            self.logger.info(
                "Materialized workspace asset `%s` -> %s (%s files)",
                relative_name,
                destination,
                copied_files,
            )

        return asset_summary

    def _detect_environment_capabilities(self) -> dict[str, Any]:
        packages = {}
        for package_name in ("numpy", "pandas", "scipy", "sklearn", "sympy"):
            packages[package_name] = bool(importlib.util.find_spec(package_name))

        available = [name for name, installed in packages.items() if installed]
        missing = [name for name, installed in packages.items() if not installed]
        return {
            "detected_at": _utc_now(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "packages": packages,
            "available_packages": available,
            "missing_packages": missing,
            "notes": [
                "Only packages marked available should be assumed by Hamilton scripts.",
                "If a required package is missing, prefer stdlib alternatives or record the blocker explicitly.",
            ],
        }

    def _materialize_environment_capabilities(self, workspace: Path) -> Path:
        capabilities_path = workspace / ENV_CAPABILITIES_FILE
        capabilities = self._detect_environment_capabilities()
        capabilities_path.write_text(
            json.dumps(capabilities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info("Materialized %s", capabilities_path)
        return capabilities_path

    def _materialize_evaluation_context(
        self,
        workspace: Path,
        task_description: str,
        task_contract: dict[str, Any],
    ) -> tuple[Path, Path]:
        context = build_evaluation_context(
            task_description=task_description,
            task_contract=task_contract,
            project_root=self._project_root,
        )
        json_path, md_path = materialize_evaluation_context(workspace, context)
        self.logger.info("Materialized %s and %s", json_path, md_path)
        return json_path, md_path

    def _init_workspace(
        self,
        task_description: str | None = None,
        task_contract: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace = self.workspace_dir
        if not workspace:
            return {}

        workspace.mkdir(parents=True, exist_ok=True)
        workspace_assets = self._materialize_workspace_assets(workspace)
        environment_capabilities_path = self._materialize_environment_capabilities(workspace)

        task_file = workspace / "task.md"
        if task_description and (
            not task_file.exists() or task_file.read_text(encoding="utf-8") != task_description
        ):
            task_file.write_text(task_description, encoding="utf-8")
            self.logger.info("Created %s", task_file)
        if task_contract is not None:
            contract_path = self.materialize_task_contract(workspace, task_contract)
            self.logger.info("Materialized %s", contract_path)
            evaluation_json_path, evaluation_md_path = self._materialize_evaluation_context(
                workspace,
                task_description or "",
                task_contract,
            )
            workspace_assets["evaluation_context"] = {
                "json": str(evaluation_json_path),
                "markdown": str(evaluation_md_path),
            }

        findings_file = workspace / "findings.md"
        if not findings_file.exists():
            findings_file.write_text(
                "# 研究发现\n\n"
                "## 关键洞察\n"
                "（按轮次记录机制判断、证据状态和 Critic 对抗反馈）\n\n"
                "<!-- APPEND_FINDINGS -->\n\n"
                "## 实验结果\n"
                "| 轮次 | 方法 | 候选方程 | 支持集 | 结果工件 | Critic | 结论 |\n"
                "|------|------|----------|--------|----------|--------|------|\n\n"
                "<!-- APPEND_RESULTS -->\n\n"
                "## 候选方程解析\n"
                "### 1) 方程与物理解释\n"
                "- 当前暂无带真实结果工件的候选方程。\n"
                "### 2) 参数/系数敏感性\n"
                "- 待后续真实结果补充。\n"
                "### 3) 物理洞察\n"
                "- 待后续真实结果补充。\n"
                "### 4) 消融分析\n"
                "- 待后续真实结果补充。\n\n"
                "## Worth Trying Next\n"
                "（按轮次记录下一步验证目标、动作和验收标准）\n\n"
                "<!-- APPEND_NEXT -->\n\n"
                "## 最优方程演化\n"
                "（记录最优方程在各轮中的变化过程）\n",
                encoding="utf-8",
            )
            self.logger.info("Created %s", findings_file)

        lib_dir = workspace / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        lib_readme = lib_dir / "README.md"
        if not lib_readme.exists():
            lib_readme.write_text("# lib/ 可复用脚本索引\n\n（每次新增脚本时更新）\n", encoding="utf-8")

        state_files = {
            "variable_memory.json": {
                "variables": {},
                "last_updated_round": 0,
                "notes": [],
            },
            "routing_state.json": {
                "current_strategy": "analytic_fit",
                "strategy_history": ["analytic_fit"],
                "last_updated_round": 0,
                "current_focus": "先生成跨风速固定结构的真实系数表",
                "priority_reason": "当前 no-PySR 主线优先完成真实结果链，而不是扩写叙述性结论。",
                "next_stage": "integration_validation",
                "latest_coef_table": "",
                "latest_rollout_result": "",
                "next_tools": [
                    "python lib/fit_viv_analytic.py --input-dir input --output history/roundN/results/coef_table.csv --summary-json history/roundN/results/fit_summary.json",
                    "python lib/validate_viv_rollout.py --input-dir input --coef-table history/roundN/results/coef_table.csv --output-json history/roundN/results/amplitude_error.json --output-csv history/roundN/results/rollout_metrics.csv --duration 200",
                    "python lib/support_ablation.py --input-dir input --output history/roundN/results/support_ablation.json --variant no_v3v5",
                ],
                "blockers": [],
            },
        }
        for filename, payload in state_files.items():
            state_path = workspace / filename
            if not state_path.exists():
                state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                self.logger.info("Created %s", state_path)

        for filename in ("hypothesis_archive.jsonl", "falsification_log.jsonl"):
            state_path = workspace / filename
            if not state_path.exists():
                state_path.write_text("", encoding="utf-8")
                self.logger.info("Created %s", state_path)

        hcc_ledger = workspace / HCC_LEDGER_FILE
        if not hcc_ledger.exists():
            hcc_ledger.write_text("", encoding="utf-8")
            self.logger.info("Created %s", hcc_ledger)

        if not (workspace / L3_CONTEXT_FILE).exists():
            (workspace / L3_CONTEXT_FILE).write_text("# L3 跨任务经验\n\n（待系统填充）\n", encoding="utf-8")
        if not (workspace / CRITIC_CONTEXT_FILE).exists():
            (workspace / CRITIC_CONTEXT_FILE).write_text("# Critic 审核参考\n\n（待系统填充）\n", encoding="utf-8")
        if not (workspace / DEBATE_STATE_FILE).exists():
            (workspace / DEBATE_STATE_FILE).write_text(
                json.dumps(
                    {
                        "task_id": getattr(self, "task_id", ""),
                        "task_hash": "",
                        "unresolved_challenges": [],
                        "resolved_challenges": [],
                        "rounds": [],
                        "updated_at": _utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not (workspace / CRITIC_SCHEDULER_STATE_FILE).exists():
            (workspace / CRITIC_SCHEDULER_STATE_FILE).write_text(
                json.dumps(
                    {
                        "last_critic_round": 0,
                        "next_periodic_round": int(self.critic_policy.get("periodic_every_n_rounds", 3) or 3),
                        "pending_trigger_reasons": [],
                        "pending_attack_backlog": [],
                        "last_critic_outcome": {},
                        "updated_at": _utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        plan_file = workspace / "plan.md"
        if not plan_file.exists():
            self._create_plan_file(plan_file)
            self.logger.info("Created %s", plan_file)

        if environment_capabilities_path:
            workspace_assets["environment_capabilities"] = {
                "path": str(environment_capabilities_path),
            }
        evaluation_json = workspace / EVALUATION_CONTEXT_FILE
        evaluation_md = workspace / EVALUATION_CONTEXT_MD
        if evaluation_json.exists() and evaluation_md.exists():
            workspace_assets.setdefault(
                "evaluation_context",
                {"json": str(evaluation_json), "markdown": str(evaluation_md)},
            )

        return workspace_assets

    def setup(self) -> None:
        self.logger.info("Setting up Hamilton playground...")
        super().setup()

        if self.session is not None:
            try:
                self.workspace_dir = Path(self.session.config.workspace_path)
            except Exception:
                self.workspace_dir = None

        self.hamilton_agent = self.agents.get("hamilton_agent")
        self.critic_agent = self.agents.get("critic_agent")
        self.agent = self.hamilton_agent

        if self.hamilton_agent is None:
            raise ValueError("Hamilton requires 'agents.hamilton' in config.yaml")
        if self.critic_agent is None:
            raise ValueError("Hamilton proposer-critic flow requires 'agents.critic' in config.yaml")

        self.l3_config = self._get_l3_config()
        self.critic_policy = self._get_extra_section("critic_policy")
        self.completion_policy = self._get_extra_section("completion_policy")

        if self.l3_config.get("enabled", True):
            self.l3_store = L3MemoryStore(
                root=self._resolve_project_path(self.l3_config["root"]),
                top_k=self.l3_config["top_k"],
                retrieval_mode=self.l3_config["retrieval_mode"],
            )
        else:
            self.l3_store = None

        self.logger.info("Hamilton playground setup complete")

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()
            self._setup_trajectory_file(output_file)

            task_bundle = self.parse_task_contract(task_description)
            normalized_task = task_bundle.task_body
            task_contract = task_bundle.contract

            self.experiment_record["task"] = normalized_task
            self.experiment_record["task_contract"] = task_contract
            if normalized_task != task_description:
                self.experiment_record["task_raw"] = task_description

            experiment_cfg = getattr(self.config, "experiment", {})
            if not isinstance(experiment_cfg, dict):
                experiment_cfg = {}
            max_rounds = int(experiment_cfg.get("max_rounds", 5) or 5)

            self.logger.info("Starting Hamilton experiment with %s max rounds", max_rounds)
            self.logger.info("Task: %s", normalized_task)

            workspace_assets = self._init_workspace(normalized_task, task_contract=task_contract)
            if workspace_assets:
                self.experiment_record["workspace_assets"] = workspace_assets

            task_id = getattr(self, "task_id", None) or "hamilton_task"
            if self.workspace_dir:
                self.experiment_record["task_contract_path"] = str(self.workspace_dir / TASK_CONTRACT_FILE)

            if self.l3_store and self.workspace_dir:
                signature = self.l3_store.build_task_signature(normalized_task, task_id=task_id)
                hits = self.l3_store.materialize_runtime_context(self.workspace_dir, signature)
                self.experiment_record["task_signature"] = signature.to_dict()
                self.experiment_record["l3_hits_count"] = len(hits.get("cards", []))

            for round_num in range(1, max_rounds + 1):
                self.logger.info("=" * 60)
                self.logger.info("Round %s/%s", round_num, max_rounds)
                self.logger.info("=" * 60)

                exp = RoundExp(
                    hamilton_agent=self.hamilton_agent,
                    critic_agent=self.critic_agent,
                    config=self.config,
                    round_num=round_num,
                    critic_policy=self.critic_policy,
                    completion_policy=self.completion_policy,
                    task_contract=task_contract,
                )
                if self.workspace_dir:
                    exp.set_run_dir(self.workspace_dir)

                result = exp.run(normalized_task, task_id=task_id)
                signal = result.get("signal") or {}

                round_record = {
                    "round": result.get("round", round_num),
                    "hamilton_result": result.get("hamilton_result", ""),
                    "critic_result": result.get("critic_result", ""),
                    "signal": signal,
                    "hamilton_signal": result.get("hamilton_signal", {}),
                    "critic_signal": result.get("critic_signal", {}),
                    "critic_report": result.get("critic_report", {}),
                    "trajectory": self._summarize_trajectory(result.get("trajectory")),
                }
                self.experiment_record["rounds"].append(round_record)

                if self._is_satisfied(signal):
                    self.logger.info("Found critic-approved satisfactory result.")
                    break

            if self.l3_store and self.workspace_dir and self.l3_config.get("promotion_policy") == "task_end_only":
                self.experiment_record["l3_promotion"] = self.l3_store.promote_task(
                    self.workspace_dir,
                    task_description=normalized_task,
                    task_id=task_id,
                    experiment_record=self.experiment_record,
                )

            self._save_experiment_record()

            return {
                "status": "completed",
                "total_rounds": len(self.experiment_record["rounds"]),
                "experiment_record": self.experiment_record,
            }

        except Exception as e:
            self.logger.error("Hamilton experiment failed: %s", e, exc_info=True)
            return {"status": "failed", "error": str(e)}
        finally:
            self.cleanup()

    def _create_plan_file(self, plan_file: Path) -> None:
        plan_content = f"""# 研究计划

{CURRENT_BEST_BEGIN}
## 当前最优
- 轮次：0
- 方程：无
- 支持集：待定
- Fit：未知
- Support Stability：未知
- Structure：未知
- 更新时间：{datetime.now().isoformat()}
{CURRENT_BEST_END}

## 数据概览
（首轮填写：变量列表、冗余风险、初步支持集判断）

## 当前路由策略
- 当前阶段：待定
- 选择理由：待定
- 下一轮优先工具：待定

## 当前假设
1. 待定

## 已确认知识
- 核心变量：待定
- 冗余/代理变量：待定
- 已发现的关键关系：无

## 证伪优先级
1. 待定

## 策略队列
{STRATEGY_QUEUE_BEGIN}
（Agent 自行制定）
{STRATEGY_QUEUE_END}

## 失败方法
| 轮次 | 策略 | 支持集 | 模板/参数 | 失败类型 | 失败原因 |
|------|------|--------|-----------|----------|----------|
"""
        plan_file.write_text(plan_content, encoding="utf-8")

    def _summarize_trajectory(self, trajectory: Any) -> dict[str, Any]:
        if isinstance(trajectory, dict):
            return {key: self._summarize_single_trajectory(value) for key, value in trajectory.items()}
        return self._summarize_single_trajectory(trajectory)

    def _summarize_single_trajectory(self, trajectory: Any) -> dict[str, Any]:
        try:
            if trajectory is None:
                return {}
            status = getattr(trajectory, "status", None)
            steps = getattr(trajectory, "steps", None)
            steps_n = len(steps) if isinstance(steps, list) else None
            return {"status": status, "steps": steps_n}
        except Exception:
            return {}

    def _is_satisfied(self, signal: dict[str, Any]) -> bool:
        try:
            if isinstance(signal, dict):
                return bool(signal.get("satisfied", False))
        except Exception:
            pass
        return False

    def _save_experiment_record(self) -> None:
        try:
            if self.run_dir:
                record_dir = Path(self.run_dir) / "records"
            else:
                record_dir = Path("./runs") / "hamilton" / "records"
            record_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_file = record_dir / f"experiment_{timestamp}.json"
            self.experiment_record["end_time"] = datetime.now().isoformat()

            with record_file.open("w", encoding="utf-8") as f:
                json.dump(self.experiment_record, f, ensure_ascii=False, indent=2)

            self.logger.info("Experiment record saved to %s", record_file)
        except Exception as e:
            self.logger.error("Failed to save experiment record: %s", e)
