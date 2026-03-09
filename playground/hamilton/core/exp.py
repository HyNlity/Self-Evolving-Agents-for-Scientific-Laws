"""Hamilton Round Exp - 单轮执行

负责单轮的 Agent 迭代。

每轮流程：
- 系统: 创建 history/round{N}/trace.md（L1 工作记忆）
- Agent: 读 plan.md + findings.md（L2），发现方程 → 验证 → 提炼到 L2，调用 finish
- 系统: 从 finish(task_completed) 判断是否继续

文件规范：
- history/round{N}/trace.md: L1 工作记忆，每轮独立
- findings.md: L2 知识积累，Agent 追加
- plan.md: L2 战略计划，Agent 全权维护（含 Current Best）
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from evomaster.core.exp import BaseExp
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance


FINAL_LAW_BLOCK_RE = re.compile(r"<final_law>\s*(.*?)\s*</final_law>", re.S)
DISCOVERED_LAW_SIGNATURE_RE = re.compile(r"<final_law>\s*def\s+discovered_law\s*\(", re.S)
DISCOVERED_LAW_DEF_RE = re.compile(r"def\s+discovered_law\s*\((.*?)\)\s*:", re.S)


class RoundExp(BaseExp):
    """单轮实验

    负责：
    1. 创建 L1 工作记忆（history/round{N}/trace.md）
    2. Agent 自主执行发现 → 验证 → 提炼闭环
    3. 从 finish(task_completed) 解析停止信号
    """

    def __init__(self, agent, config, round_num):
        super().__init__(agent, config)
        self.round_num = round_num
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return f"Round_{self.round_num}"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """执行单轮实验"""
        self.logger.info(f"Starting Round {self.round_num}")

        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=self.round_num)

        # 初始化 L1 + 轮次目录
        self._ensure_round_dirs()
        self._init_trace()

        # 记录 L2 文件状态（用于 post-check）
        l2_snapshot = self._snapshot_l2()

        # ========== Agent ==========
        self.logger.info(f"[Round {self.round_num}] Running Agent...")
        task = TaskInstance(
            task_id=f"{task_id}_round{self.round_num}",
            task_type="hamilton",
            description=task_description,
            input_data={"round": self.round_num},
        )
        trajectory = self.agent.run(task)
        agent_result = self._extract_agent_response(trajectory)
        self.logger.info(f"[Round {self.round_num}] Agent completed")

        # 解析 satisfied 信号（系统唯一职责：决定是否继续迭代）
        signal = self._parse_signal(agent_result, trajectory, task_description=task_description)

        # L2 post-check: Agent 是否更新了 L2？
        self._check_l2_promotion(l2_snapshot)

        findings_content = self._read_findings()

        self.logger.info(f"Round {self.round_num} completed")

        return {
            "round": self.round_num,
            "agent_result": agent_result,
            "signal": signal,
            "findings": findings_content,
            "trajectory": trajectory,
        }

    def _ensure_round_dirs(self):
        """确保本轮目录存在"""
        if not self.run_dir:
            return
        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        (round_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (round_dir / "results").mkdir(parents=True, exist_ok=True)

    def _init_trace(self):
        """创建 L1 工作记忆 — history/round{N}/trace.md"""
        if not self.run_dir:
            return

        trace_file = self.run_dir / "history" / f"round{self.round_num}" / "trace.md"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        l1_template = f"""# 执行日志 — 第 {self.round_num} 轮

### 操作记录
（记录执行的脚本、使用的参数、观察到的现象）

### 指标记录
| 实验 | 方法 | 目标 | 关键参数 | MSE (训练) | MSE (OOD) | 方程 | 备注 |
|------|------|------|---------|-----------|-----------|------|------|

### 工作笔记
（当前假设、中间观察、思考过程）
"""
        trace_file.write_text(l1_template, encoding="utf-8")
        self.logger.info(f"创建 history/round{self.round_num}/trace.md（L1）")

    def _read_findings(self) -> str:
        """读取 findings.md（L2 知识）"""
        if not self.run_dir:
            return ""
        findings_file = self.run_dir / "findings.md"
        if findings_file.exists():
            return findings_file.read_text(encoding="utf-8")
        return ""

    def _snapshot_l2(self) -> dict:
        """记录 L2 文件的修改时间，用于 post-check"""
        if not self.run_dir:
            return {}
        snapshot = {}
        for name in ("findings.md", "plan.md"):
            path = self.run_dir / name
            if path.exists():
                snapshot[name] = path.stat().st_mtime
            else:
                snapshot[name] = 0
        return snapshot

    def _check_l2_promotion(self, before: dict) -> None:
        """检查 Agent 是否更新了 L2 文件（Phase 3 Promotion）"""
        if not self.run_dir or not before:
            return
        unchanged = []
        for name in ("findings.md", "plan.md"):
            path = self.run_dir / name
            if path.exists():
                after_mtime = path.stat().st_mtime
                if after_mtime <= before.get(name, 0):
                    unchanged.append(name)
        if unchanged:
            self.logger.warning(
                f"[Round {self.round_num}] L2 files not updated: {unchanged}. "
                "Agent may have skipped Phase 3 (Promotion). "
                "Next round will read stale L2 data."
            )

    def _extract_agent_response(self, trajectory) -> str:
        return super()._extract_agent_response(trajectory)

    # =========================
    # Signal parsing
    # =========================

    def _parse_signal(self, agent_message: str, trajectory, task_description: str | None = None) -> dict:
        """Parse signal from finish tool's task_completed parameter.

        The Agent calls finish(message=..., task_completed="true"/"false").
        task_completed="true" → satisfied=True (stop iterating).
        task_completed="false" → satisfied=False (continue).
        """
        task_completed = self._extract_task_completed(trajectory)

        if task_completed is None:
            self.logger.warning(
                f"[Round {self.round_num}] Could not extract task_completed from trajectory. "
                "Agent may not have called finish(). Defaulting to satisfied=False."
            )
            return {"round": self.round_num, "satisfied": False}

        satisfied = task_completed == "true"
        finish_message = self._extract_finish_message_from_trajectory(trajectory)

        signal: dict[str, Any] = {
            "round": self.round_num,
            "satisfied": satisfied,
            "task_completed": task_completed,
            "notes": finish_message[:500] if finish_message else "",
        }

        if self._is_newtonbench_task(task_description):
            protocol = self._evaluate_newtonbench_protocol(trajectory, finish_message)
            signal["protocol"] = protocol

            if satisfied and protocol["violations"]:
                self.logger.warning(
                    f"[Round {self.round_num}] NewtonBench protocol guard blocked completion: "
                    f"{protocol['violations']}"
                )
                signal["satisfied"] = False
                signal["protocol_guard_blocked"] = True

        return signal

    def _extract_task_completed(self, trajectory) -> str | None:
        """Extract task_completed value from the finish tool call in trajectory."""
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

    def _extract_finish_message_from_trajectory(self, trajectory) -> str:
        """Extract finish.message from trajectory (robust fallback)."""
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

    def _is_newtonbench_task(self, task_description: str | None) -> bool:
        """Best-effort detection for NewtonBench profile tasks."""
        text = (task_description or "").lower()
        if "profile: newtonbench" in text:
            return True
        if all(token in text for token in ("module:", "difficulty:", "law_version:", "noise:")):
            return True

        try:
            agents_cfg = getattr(self.config, "agents", {})
            if isinstance(agents_cfg, dict):
                hamilton_cfg = agents_cfg.get("hamilton", {})
                if isinstance(hamilton_cfg, dict):
                    skills = hamilton_cfg.get("skills", [])
                    if isinstance(skills, list):
                        for skill in skills:
                            if str(skill).strip().lower() == "newtonbench":
                                return True
        except Exception:
            pass
        return False

    def _evaluate_newtonbench_protocol(self, trajectory, finish_message: str) -> dict[str, Any]:
        """Validate completion protocol for NewtonBench tasks."""
        script_records = self._collect_skill_script_records(trajectory)
        script_stats = self._collect_skill_script_stats(script_records)
        run_stats = script_stats.get("run_experiment.py", {"calls": 0, "success_calls": 0})
        eval_stats = script_stats.get("evaluate_submission.py", {"calls": 0, "success_calls": 0})

        has_final_law_block = bool(FINAL_LAW_BLOCK_RE.search(finish_message or ""))
        has_discovered_law_signature = bool(DISCOVERED_LAW_SIGNATURE_RE.search(finish_message or ""))
        expected_signature = self._extract_expected_function_signature(script_records)
        final_law_signature = self._extract_final_law_signature(finish_message)
        signature_match = (
            expected_signature == final_law_signature
            if expected_signature and final_law_signature
            else None
        )
        last_eval_metrics = self._extract_last_successful_evaluation(script_records)

        protocol_cfg = self._get_newtonbench_protocol_cfg()
        require_signature_match = bool(protocol_cfg.get("require_signature_match", True))
        require_finite_rmsle = bool(protocol_cfg.get("require_finite_rmsle", True))
        max_rmsle = protocol_cfg.get("max_rmsle")
        max_rmsle_value: float | None = None
        if max_rmsle is not None:
            try:
                parsed = float(max_rmsle)
                if math.isfinite(parsed):
                    max_rmsle_value = parsed
            except Exception:
                max_rmsle_value = None

        violations: list[str] = []
        if int(run_stats.get("success_calls", 0)) < 1:
            violations.append("missing_successful_run_experiment")
        if int(eval_stats.get("success_calls", 0)) < 1:
            violations.append("missing_successful_evaluate_submission")
        if not has_final_law_block:
            violations.append("missing_final_law_block")
        if not has_discovered_law_signature:
            violations.append("final_law_missing_discovered_law_signature")
        if require_signature_match:
            if not expected_signature:
                violations.append("missing_expected_function_signature")
            elif not final_law_signature:
                violations.append("missing_final_law_signature")
            elif signature_match is False:
                violations.append("final_law_signature_mismatch")

        if int(eval_stats.get("success_calls", 0)) > 0:
            eval_rmsle = last_eval_metrics.get("rmsle")
            if require_finite_rmsle and eval_rmsle is None:
                violations.append("non_finite_rmsle")
            if max_rmsle_value is not None and (
                eval_rmsle is None or eval_rmsle > max_rmsle_value
            ):
                violations.append("rmsle_above_threshold")

        return {
            "run_experiment_calls": int(run_stats.get("calls", 0)),
            "run_experiment_success_calls": int(run_stats.get("success_calls", 0)),
            "evaluate_submission_calls": int(eval_stats.get("calls", 0)),
            "evaluate_submission_success_calls": int(eval_stats.get("success_calls", 0)),
            "has_final_law_block": has_final_law_block,
            "has_discovered_law_signature": has_discovered_law_signature,
            "expected_function_signature": expected_signature,
            "final_law_signature": final_law_signature,
            "signature_match": signature_match,
            "last_evaluation": last_eval_metrics,
            "quality_guard": {
                "require_signature_match": require_signature_match,
                "require_finite_rmsle": require_finite_rmsle,
                "max_rmsle": max_rmsle_value,
            },
            "violations": violations,
        }

    def _collect_skill_script_stats(self, script_records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        """Count per-script run_script calls and success counts from trajectory steps."""
        stats: dict[str, dict[str, int]] = {}
        for rec in script_records:
            script_name = str(rec.get("script_name", "") or "").strip()
            if not script_name:
                continue
            stat = stats.setdefault(script_name, {"calls": 0, "success_calls": 0})
            stat["calls"] += 1
            if bool(rec.get("success")):
                stat["success_calls"] += 1

        return stats

    def _collect_skill_script_records(self, trajectory) -> list[dict[str, Any]]:
        """Collect run_script call records including output and success info."""
        records: list[dict[str, Any]] = []
        steps = getattr(trajectory, "steps", None)
        if not isinstance(steps, list):
            return records

        for step in steps:
            assistant_message = getattr(step, "assistant_message", None)
            tool_calls = getattr(assistant_message, "tool_calls", None)
            if not tool_calls:
                continue

            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if not fn or getattr(fn, "name", None) != "use_skill":
                    continue

                args = self._safe_json_loads(getattr(fn, "arguments", "") or "")
                if not isinstance(args, dict) or args.get("action") != "run_script":
                    continue

                script_name = str(args.get("script_name", "") or "").strip()
                if not script_name:
                    continue

                tool_call_id = getattr(tc, "id", "") or ""
                tool_response = self._get_tool_response(step, tool_call_id)
                output = getattr(tool_response, "content", "") if tool_response is not None else ""
                meta = getattr(tool_response, "meta", {}) if tool_response is not None else {}
                info = meta.get("info", {}) if isinstance(meta, dict) else {}

                records.append(
                    {
                        "script_name": script_name,
                        "args": args,
                        "tool_call_id": tool_call_id,
                        "success": self._is_tool_call_success(step, tool_call_id),
                        "output": output if isinstance(output, str) else str(output),
                        "info": info if isinstance(info, dict) else {},
                    }
                )

        return records

    def _get_tool_response(self, step, tool_call_id: str):
        if not tool_call_id:
            return None
        tool_responses = getattr(step, "tool_responses", None)
        if not isinstance(tool_responses, list):
            return None
        for tr in tool_responses:
            if getattr(tr, "tool_call_id", None) == tool_call_id:
                return tr
        return None

    def _extract_expected_function_signature(
        self,
        script_records: list[dict[str, Any]],
    ) -> str | None:
        expected_sig: str | None = None
        for rec in script_records:
            if rec.get("script_name") != "generate_task_prompt.py" or not rec.get("success"):
                continue
            payload = self._extract_last_json_object(str(rec.get("output", "") or ""))
            if not isinstance(payload, dict):
                continue
            function_signature = payload.get("function_signature")
            if not isinstance(function_signature, str):
                continue
            canonical = self._canonical_discovered_law_signature(function_signature)
            if canonical:
                expected_sig = canonical
        return expected_sig

    def _extract_final_law_signature(self, finish_message: str) -> str | None:
        source = finish_message or ""
        match = FINAL_LAW_BLOCK_RE.search(source)
        if match:
            source = match.group(1).strip()
        return self._canonical_discovered_law_signature(source)

    def _extract_last_successful_evaluation(
        self,
        script_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for rec in script_records:
            if rec.get("script_name") != "evaluate_submission.py" or not rec.get("success"):
                continue
            payload = self._extract_last_json_object(str(rec.get("output", "") or ""))
            if not isinstance(payload, dict):
                continue
            evaluation = payload.get("evaluation")
            if not isinstance(evaluation, dict):
                continue

            parsed: dict[str, Any] = {}
            rmsle = self._safe_float(evaluation.get("rmsle"))
            exact_accuracy = self._safe_float(evaluation.get("exact_accuracy"))
            if rmsle is not None:
                parsed["rmsle"] = rmsle
            else:
                parsed["rmsle"] = None
            if exact_accuracy is not None:
                parsed["exact_accuracy"] = exact_accuracy
            else:
                parsed["exact_accuracy"] = None

            symbolic_equivalent = evaluation.get("symbolic_equivalent")
            if isinstance(symbolic_equivalent, bool):
                parsed["symbolic_equivalent"] = symbolic_equivalent
            symbolic_msg = evaluation.get("symbolic_msg")
            if isinstance(symbolic_msg, str):
                parsed["symbolic_msg"] = symbolic_msg
            eval_error = evaluation.get("error")
            if isinstance(eval_error, str) and eval_error.strip():
                parsed["error"] = eval_error.strip()

            metrics = parsed
        return metrics

    def _canonical_discovered_law_signature(self, text: str) -> str | None:
        if not isinstance(text, str):
            return None
        match = DISCOVERED_LAW_DEF_RE.search(text)
        if not match:
            return None
        raw_params = match.group(1).strip()
        if not raw_params:
            return "def discovered_law():"
        names: list[str] = []
        for token in raw_params.split(","):
            part = token.strip()
            if not part:
                continue
            if "=" in part:
                part = part.split("=", 1)[0].strip()
            if ":" in part:
                part = part.split(":", 1)[0].strip()
            if not part:
                return None
            names.append(part)
        return f"def discovered_law({', '.join(names)}):"

    def _extract_last_json_object(self, text: str) -> dict[str, Any] | None:
        if not isinstance(text, str) or not text:
            return None
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        last_obj: dict[str, Any] | None = None
        while idx < n:
            if text[idx] != "{":
                idx += 1
                continue
            try:
                value, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(value, dict):
                last_obj = value
            idx = end
        return last_obj

    def _safe_float(self, value: Any) -> float | None:
        try:
            f = float(value)
            if math.isfinite(f):
                return f
            return None
        except Exception:
            return None

    def _get_newtonbench_protocol_cfg(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "require_signature_match": True,
            "require_finite_rmsle": True,
            "max_rmsle": None,
        }
        try:
            experiment_cfg = getattr(self.config, "experiment", {})
            if not isinstance(experiment_cfg, dict):
                return defaults
            cfg = experiment_cfg.get("newtonbench_protocol", {})
            if not isinstance(cfg, dict):
                return defaults
            out = dict(defaults)
            out.update(cfg)
            return out
        except Exception:
            return defaults

    def _is_tool_call_success(self, step, tool_call_id: str) -> bool:
        """Judge whether a tool call succeeded using tool response metadata."""
        if not tool_call_id:
            return False

        tool_responses = getattr(step, "tool_responses", None)
        if not isinstance(tool_responses, list):
            return False

        for tr in tool_responses:
            if getattr(tr, "tool_call_id", None) != tool_call_id:
                continue
            meta = getattr(tr, "meta", {})
            info = meta.get("info", {}) if isinstance(meta, dict) else {}
            exit_code = info.get("exit_code") if isinstance(info, dict) else None
            if exit_code is None:
                return True
            try:
                return int(exit_code) == 0
            except Exception:
                return str(exit_code).strip() == "0"
        return False

    def _safe_json_loads(self, text: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(text) if isinstance(text, str) and text.strip() else {}
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None
