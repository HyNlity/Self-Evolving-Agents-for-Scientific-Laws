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
FINAL_LAW_STANDALONE_BLOCK_RE = re.compile(
    r"(?ms)^[ \t]*<final_law>[ \t]*\n(.*?)\n[ \t]*</final_law>[ \t]*$"
)
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
        raw_finish_message = self._extract_finish_message_from_trajectory(trajectory)
        finish_message = self._resolve_finish_message_with_eval_fallback(
            trajectory=trajectory,
            finish_message=raw_finish_message,
        )
        signal = self._parse_signal(
            agent_result,
            trajectory,
            task_description=task_description,
            finish_message_override=finish_message,
        )

        # L2 post-check: Agent 是否更新了 L2？
        unchanged = self._check_l2_promotion(l2_snapshot)
        is_newtonbench = self._is_newtonbench_task(task_description)

        # Lightweight mode: keep only best-memory sync by default for NewtonBench.
        if is_newtonbench:
            self._sync_current_best_from_protocol(signal=signal, finish_message=finish_message)
            self._sync_findings_eval_from_protocol(signal=signal, finish_message=finish_message)

        # Optional fallback when Agent skipped L2 promotion.
        if unchanged and self._should_system_backfill(is_newtonbench=is_newtonbench):
            self._system_backfill_l2(signal=signal, finish_message=finish_message, unchanged=unchanged)

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

    def _check_l2_promotion(self, before: dict) -> list[str]:
        """检查 Agent 是否更新了 L2 文件（Phase 3 Promotion）"""
        if not self.run_dir or not before:
            return []
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
        return unchanged

    def _should_system_backfill(self, is_newtonbench: bool) -> bool:
        """Whether system should auto-write findings/plan when agent skipped promotion.

        Default behavior:
        - NewtonBench: disabled (keep single-agent ownership of findings/plan).
        - Others: enabled as a safety fallback.
        """
        try:
            experiment_cfg = getattr(self.config, "experiment", {})
            if isinstance(experiment_cfg, dict):
                configured = experiment_cfg.get("auto_backfill_l2")
                if isinstance(configured, bool):
                    return configured
        except Exception:
            pass
        return not is_newtonbench

    def _system_backfill_l2(self, signal: dict[str, Any], finish_message: str, unchanged: list[str]) -> None:
        """Auto-backfill L2 files when agent skipped promotion.

        Keep the result loop closed in the current run workspace:
        - findings.md: append round-level result summary (metrics/final_law/protocol)
        - plan.md: append round-level continuation note (when missing)
        """
        if not self.run_dir or not unchanged:
            return
        if not isinstance(signal, dict):
            return

        if "findings.md" in unchanged:
            self._append_system_round_to_findings(signal=signal, finish_message=finish_message)
        if "plan.md" in unchanged:
            self._append_system_round_to_plan(signal=signal)

    def _append_system_round_to_findings(self, signal: dict[str, Any], finish_message: str) -> None:
        if not self.run_dir:
            return
        findings_file = self.run_dir / "findings.md"
        if not findings_file.exists():
            return

        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
        last_eval = protocol.get("last_evaluation", {})
        if not isinstance(last_eval, dict):
            last_eval = {}

        violations = protocol.get("violations", [])
        if not isinstance(violations, list):
            violations = []

        final_law_code = self._extract_final_law_code(finish_message)
        self._upsert_findings_experiment_row(
            findings_file=findings_file,
            signal=signal,
            final_law_code=final_law_code,
        )
        self._normalize_findings_experiment_table(findings_file)
        self._normalize_findings_body_sections(findings_file)

        marker = f"<!-- HAM_SYS_BACKFILL_ROUND_{self.round_num} -->"
        existing_after_table_sync = findings_file.read_text(encoding="utf-8")
        if marker in existing_after_table_sync:
            return

        lines = [
            "",
            marker,
            f"## 系统回填 Round {self.round_num}",
            f"- task_completed: {signal.get('task_completed')}",
            f"- satisfied: {signal.get('satisfied')}",
            f"- run_experiment_success_calls: {protocol.get('run_experiment_success_calls')}",
            f"- evaluate_submission_success_calls: {protocol.get('evaluate_submission_success_calls')}",
            (
                "- 评测指标: "
                f"rmsle={self._format_metric(last_eval.get('rmsle'))}, "
                f"exact_accuracy={self._format_metric(last_eval.get('exact_accuracy'))}, "
                f"symbolic_equivalent={self._format_metric(last_eval.get('symbolic_equivalent'))}"
            ),
        ]

        eval_error = last_eval.get("error")
        if isinstance(eval_error, str) and eval_error.strip():
            lines.append(f"- evaluation_error: {eval_error.strip()}")

        if violations:
            lines.append(f"- protocol_violations: {', '.join(str(v) for v in violations)}")

        if final_law_code:
            lines.append("- 最终方程:")
            lines.append("```python")
            lines.extend(final_law_code.splitlines())
            lines.append("```")

        with findings_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.logger.info(
            "[Round %s] Auto-backfilled findings.md in workspace: %s",
            self.round_num,
            findings_file,
        )

    def _sync_findings_eval_from_protocol(self, signal: dict[str, Any], finish_message: str) -> None:
        """Force authoritative per-round eval row into findings.md.

        Values are parsed from protocol.last_evaluation (tool outputs), not agent prose.
        """
        if not self.run_dir or not isinstance(signal, dict):
            return
        findings_file = self.run_dir / "findings.md"
        if not findings_file.exists():
            return

        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            return
        eval_calls = int(protocol.get("evaluate_submission_calls", 0) or 0)
        if eval_calls < 1:
            return

        final_law_code = self._extract_final_law_code(finish_message)
        self._upsert_findings_experiment_row(
            findings_file=findings_file,
            signal=signal,
            final_law_code=final_law_code,
        )
        self._normalize_findings_experiment_table(findings_file)
        self._normalize_findings_body_sections(findings_file)
        self.logger.info(
            "[Round %s] Synced authoritative eval metrics to findings.md",
            self.round_num,
        )

    def _upsert_findings_experiment_row(
        self,
        findings_file: Path,
        signal: dict[str, Any],
        final_law_code: str,
    ) -> None:
        """Insert/update one row in findings.md experiment table for this round."""
        text = findings_file.read_text(encoding="utf-8")
        lines = text.splitlines()

        header_idx = -1
        sep_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("| 轮次 | 方法 | 方程 |"):
                header_idx = i
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|------|"):
                    sep_idx = i + 1
                break
        if header_idx < 0 or sep_idx < 0:
            return

        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
        last_eval = protocol.get("last_evaluation", {})
        if not isinstance(last_eval, dict):
            last_eval = {}

        # Prefer the actually evaluated expression from tool output.
        submitted_expr = str(last_eval.get("submitted_return_expr") or "").strip()
        equation = submitted_expr or self._extract_return_expression(final_law_code)
        if not equation:
            equation = self._extract_final_law_signature(final_law_code) or "N/A"
        equation = self._compact_text(equation, limit=96).replace("|", "/")

        task_completed = str(signal.get("task_completed") or "").strip().lower() == "true"
        rmsle_text = self._format_metric(last_eval.get("rmsle"))
        exact_text = self._format_metric(last_eval.get("exact_accuracy"))
        symbolic_text = self._format_metric(last_eval.get("symbolic_equivalent"))
        eval_error = str(last_eval.get("error") or "").strip()

        mse_train = f"RMSLE={rmsle_text}"
        mse_ood = f"Exact={exact_text}"
        conclusion = (
            f"{'完成' if task_completed else '未完成'}; "
            f"symbolic={symbolic_text}; source=protocol_eval"
        )
        if eval_error:
            conclusion += f"; error={self._compact_text(eval_error, limit=48)}"
        conclusion = conclusion.replace("|", "/")

        row = f"| {self.round_num} | protocol_eval | `{equation}` | {mse_train} | {mse_ood} | {conclusion} |"

        existing_indices: list[int] = []
        for idx in range(sep_idx + 1, len(lines)):
            line = lines[idx]
            if not line.lstrip().startswith("|"):
                continue
            existing_round = self._extract_round_num_from_table_row(line)
            if existing_round == self.round_num:
                existing_indices.append(idx)

        if existing_indices:
            lines[existing_indices[0]] = row
            for idx in reversed(existing_indices[1:]):
                del lines[idx]
            findings_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            return

        insert_at = sep_idx + 1
        while insert_at < len(lines) and lines[insert_at].lstrip().startswith("|"):
            existing_round = self._extract_round_num_from_table_row(lines[insert_at])
            if existing_round is not None and existing_round > self.round_num:
                break
            insert_at += 1
        lines.insert(insert_at, row)

        findings_file.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def _normalize_findings_experiment_table(self, findings_file: Path) -> None:
        """Normalize findings experiment table layout and ordering.

        Some rounds may insert free-text analysis into the table region. This
        method rebuilds the result section from authoritative round rows only.
        """
        text = findings_file.read_text(encoding="utf-8")
        lines = text.splitlines()

        header_idx = -1
        sep_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("| 轮次 | 方法 | 方程 |"):
                header_idx = i
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|------|"):
                    sep_idx = i + 1
                break
        if header_idx < 0 or sep_idx < 0:
            return

        section_end = len(lines)
        for i in range(header_idx + 1, len(lines)):
            if i > header_idx and lines[i].startswith("## "):
                section_end = i
                break

        body_lines = lines[sep_idx + 1 : section_end]
        rows_by_round: dict[int, str] = {}
        non_round_table_rows: list[str] = []

        for line in body_lines:
            stripped = line.strip()
            if not stripped or "<!-- APPEND_RESULTS -->" in stripped:
                continue
            if stripped.startswith("|"):
                round_num = self._extract_round_num_from_table_row(stripped)
                if round_num is None:
                    non_round_table_rows.append(stripped)
                else:
                    rows_by_round[round_num] = stripped

        sorted_round_rows = [rows_by_round[k] for k in sorted(rows_by_round)]
        rebuilt_block = [lines[header_idx], lines[sep_idx], *sorted_round_rows, *non_round_table_rows]
        if rebuilt_block[-1].strip():
            rebuilt_block.append("")
        rebuilt_block.append("<!-- APPEND_RESULTS -->")

        new_lines = lines[:header_idx] + rebuilt_block + lines[section_end:]

        compact_lines: list[str] = []
        prev_blank = False
        for line in new_lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            compact_lines.append(line)
            prev_blank = is_blank

        findings_file.write_text("\n".join(compact_lines).rstrip() + "\n", encoding="utf-8")

    def _normalize_findings_body_sections(self, findings_file: Path) -> None:
        """Normalize free-text sections and keep only the latest round analysis block."""
        text = findings_file.read_text(encoding="utf-8")
        lines = text.splitlines()

        lines = self._normalize_key_findings_section(lines)
        lines = self._normalize_candidate_analysis_section(lines)
        lines = self._normalize_worth_trying_next_section(lines)

        compact_lines: list[str] = []
        prev_blank = False
        for line in lines:
            is_blank = line.strip() == ""
            if is_blank and prev_blank:
                continue
            compact_lines.append(line)
            prev_blank = is_blank

        findings_file.write_text("\n".join(compact_lines).rstrip() + "\n", encoding="utf-8")

    def _normalize_key_findings_section(self, lines: list[str]) -> list[str]:
        """Render key findings as per-round physical insights."""
        header_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "## 关键洞察":
                header_idx = i
                break
        if header_idx < 0:
            return lines

        section_end = len(lines)
        for i in range(header_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                section_end = i
                break

        body = lines[header_idx + 1 : section_end]
        placeholder = "（经验证的数据观察和物理关系）"
        for line in body:
            stripped = line.strip()
            if stripped.startswith("（") and stripped.endswith("）"):
                placeholder = stripped
                break

        round_summaries = self._extract_round_summaries_from_experiment_table(lines)
        rebuilt_body: list[str] = [placeholder, ""]
        for round_num in sorted(round_summaries):
            summary = round_summaries[round_num]
            rebuilt_body.append(f"### Round {round_num}")
            for insight in self._build_physical_insight_for_round(summary):
                rebuilt_body.append(f"- {insight}")
            rebuilt_body.append("")
        rebuilt_body.append("<!-- APPEND_FINDINGS -->")

        return lines[: header_idx + 1] + rebuilt_body + lines[section_end:]

    def _normalize_worth_trying_next_section(self, lines: list[str]) -> list[str]:
        """Render next-step section as per-round structured actions."""
        header_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "## Worth Trying Next":
                header_idx = i
                break
        if header_idx < 0:
            return lines

        section_end = len(lines)
        for i in range(header_idx + 1, len(lines)):
            if lines[i].startswith("## "):
                section_end = i
                break

        round_summaries = self._extract_round_summaries_from_experiment_table(lines)
        best_rmsle: float | None = None
        for summary in round_summaries.values():
            val = summary.get("rmsle")
            if isinstance(val, float) and math.isfinite(val):
                best_rmsle = val if best_rmsle is None else min(best_rmsle, val)

        rebuilt_body: list[str] = []
        for round_num in sorted(round_summaries):
            summary = round_summaries[round_num]
            rebuilt_body.append(f"### Round {round_num} -> Next")
            for item in self._build_next_actions_for_round(summary, best_rmsle):
                rebuilt_body.append(f"- {item}")
            rebuilt_body.append("")
        rebuilt_body.append("<!-- APPEND_NEXT -->")

        prefix = lines[: header_idx + 1]
        suffix = lines[section_end:]
        return prefix + rebuilt_body + suffix

    def _extract_round_summaries_from_experiment_table(
        self,
        lines: list[str],
    ) -> dict[int, dict[str, str]]:
        """Extract per-round metrics/conclusion from findings experiment table."""
        header_idx = -1
        sep_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("| 轮次 | 方法 | 方程 |"):
                header_idx = i
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|------|"):
                    sep_idx = i + 1
                break
        if header_idx < 0 or sep_idx < 0:
            return {}

        marker_idx = -1
        for i in range(sep_idx + 1, len(lines)):
            if "<!-- APPEND_RESULTS -->" in lines[i]:
                marker_idx = i
                break
        if marker_idx < 0:
            marker_idx = len(lines)

        summaries: dict[int, dict[str, Any]] = {}
        for line in lines[sep_idx + 1 : marker_idx]:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            round_num = self._extract_round_num_from_table_row(stripped)
            if round_num is None:
                continue
            parts = [cell.strip() for cell in stripped.split("|")[1:-1]]
            if len(parts) < 6:
                continue
            equation = parts[2].strip().strip("`")
            rmsle = self._parse_metric_value(parts[3], "RMSLE")
            exact = self._parse_metric_value(parts[4], "Exact")
            summaries[round_num] = {
                "round": round_num,
                "equation": equation,
                "mse_train": parts[3],
                "mse_ood": parts[4],
                "conclusion": parts[5],
                "rmsle": rmsle,
                "exact": exact,
            }
        return summaries

    def _extract_equation_features(self, equation: str) -> dict[str, Any]:
        raw = equation if isinstance(equation, str) else ""
        expr = re.sub(r"\s+", "", raw.lower())
        numeric_literals = [
            abs(float(match.group(0)))
            for match in re.finditer(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:e[-+]?\d+)?", expr)
            if self._safe_float(match.group(0)) is not None
        ]
        positive_literals = [x for x in numeric_literals if x > 0]
        has_mixed_scales = False
        if positive_literals:
            has_mixed_scales = max(positive_literals) / min(positive_literals) > 1e6

        ratio_like = (
            "omega/t" in expr
            or "omega)/(t" in expr
            or ("omega**1.5" in expr and "t**3" in expr)
            or ("omega**(3/2)" in expr and "t**3" in expr)
            or ("sqrt(omega)" in expr and "/t" in expr)
        )
        direct_sum = (
            "t+omega" in expr
            or "omega+t" in expr
            or "sqrt(t+omega)" in expr
            or "sqrt(omega+t)" in expr
        )
        denominator_sensitive = (
            "/(t-" in expr
            or "/(omega-" in expr
            or ")/(t-" in expr
            or ")/(omega-" in expr
        )
        additive_bias = bool(
            re.search(
                r"(?:\+|-)(?:\d+\.\d*|\d*\.\d+|\d+)(?:e[-+]?\d+)?\)?$",
                expr,
            )
        )

        return {
            "expr": expr,
            "has_exp": "exp(" in expr,
            "has_log": "log(" in expr,
            "has_sqrt": "sqrt(" in expr,
            "has_poly": "**" in expr,
            "has_ratio_like": ratio_like,
            "has_direct_sum": direct_sum,
            "has_denominator_sensitive": denominator_sensitive,
            "has_additive_bias": additive_bias,
            "has_mixed_scales": has_mixed_scales,
        }

    def _parse_metric_value(self, text: str, metric_name: str) -> float | None:
        if not isinstance(text, str):
            return None
        pattern = rf"{re.escape(metric_name)}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        m = re.search(pattern, text)
        if not m:
            return None
        return self._safe_float(m.group(1))

    def _build_physical_insight_for_round(self, summary: dict[str, Any]) -> list[str]:
        equation = str(summary.get("equation", "") or "")
        features = self._extract_equation_features(equation)

        if features["has_log"] and features["has_ratio_like"]:
            mechanism = (
                "机制判读：方程出现 `log(频率/温度尺度组合)`，"
                "更像是在描述乘法尺度被压缩后的响应，而不是直接的热指数抑制。"
            )
        elif features["has_exp"] and features["has_ratio_like"]:
            mechanism = (
                "机制判读：方程包含以频率/温度尺度组合作为输入的指数项，"
                "属于典型的热激发抑制家族。"
            )
        elif features["has_sqrt"] and features["has_direct_sum"]:
            mechanism = (
                "机制判读：出现 `sqrt(T+omega)` 一类直接混合项，"
                "说明模型在用代数 surrogate 吸收窄带 proxy 的局部曲率。"
            )
        elif features["has_sqrt"] and features["has_denominator_sensitive"]:
            mechanism = (
                "机制判读：当前是“根号 + 敏感分母”的有理式，"
                "更像数值近似器而非稳定的机制方程。"
            )
        else:
            mechanism = (
                "机制判读：当前主要依赖代数补丁项（根号/多项式/有理式），"
                "物理机制仍弱于经验拟合特征。"
            )

        if features["has_ratio_like"]:
            scene = (
                "情境关联：式子至少显式耦合了频率与温度的尺度比/尺度组合，"
                "这比单独堆叠 `omega`、`T` 更接近该类热辐射任务的物理结构。"
            )
        elif features["has_direct_sum"]:
            scene = (
                "情境关联：当前直接把 `T` 与 `omega` 相加后再开方，"
                "缺少明确的无量纲化，物理上更像 proxy 插值而非本征规律。"
            )
        else:
            scene = (
                "情境关联：当前形式未稳定体现无量纲主变量，"
                "与黑体腔 + 滤波 + 量热情境的核心尺度关系仍偏弱。"
            )

        if features["has_log"]:
            literature = (
                "文献对照：它与常见的 `exp` 型 Planck/Bose-Einstein 基线不同；"
                "这种差异本身不等于错误，也不等于新规律，需要看简化后是否仍保持低误差。"
            )
        elif features["has_exp"] and features["has_ratio_like"] and not (
            features["has_sqrt"] or features["has_poly"]
        ):
            literature = (
                "文献对照：该式更接近常见热占据数家族；"
                "接下来应检验是否还能用更少项保持同等误差。"
            )
        elif features["has_sqrt"] or features["has_poly"]:
            literature = (
                "文献对照：存在 `sqrt`/高阶幂/有理补丁项，"
                "更像经验 surrogate；只有在消融后仍保持低误差，才可讨论其必要性。"
            )
        else:
            literature = (
                "文献对照：当前结构既未贴近经典热占据数基线，也未形成清晰的新机制证据，"
                "需要继续压缩到更简洁的主导变量形式。"
            )

        return [mechanism, scene, literature]

    def _build_next_actions_for_round(
        self,
        summary: dict[str, Any],
        best_rmsle: float | None,
    ) -> list[str]:
        equation = str(summary.get("equation", "") or "")
        features = self._extract_equation_features(equation)
        rmsle = summary.get("rmsle")

        if isinstance(rmsle, float) and math.isfinite(rmsle):
            if rmsle > 1.0:
                target = "目标：先把误差降到 `RMSLE < 1`，再讨论符号等价。"
            elif rmsle > 0.01:
                target = "目标：当前仅达到数值可用，继续把误差压到 `RMSLE <= 1e-2` 再讨论收敛。"
            else:
                target = "目标：在保持低 RMSLE 的同时提升符号一致性与物理可解释性。"
        else:
            target = "目标：先获得稳定可评测结果（避免 NaN / overflow）。"

        if features["has_log"] and not features["has_ratio_like"]:
            action = "动作：保留 `log` 家族作为对照，但先改写成单一无量纲主变量后再比较 `log` 与 `exp` 基线。"
        elif not features["has_exp"] and not features["has_log"]:
            action = "动作：先把当前 surrogate 压缩成 `f(omega**a / T**b)` 单变量结构，再比较 `exp` / `log` / 代数基线。"
        elif features["has_exp"] and not features["has_ratio_like"]:
            action = "动作：重写为 `omega/T` 无量纲输入，并重新做常数扫描。"
        elif features["has_log"] or features["has_sqrt"] or features["has_poly"]:
            action = "动作：做消融（先去 log/sqrt/高阶项），并与更简洁的 `exp/log` 主变量版本做并行比较。"
        else:
            action = "动作：固定当前结构族，仅微调常数并补充极端温频样本验证稳健性。"

        if best_rmsle is not None and math.isfinite(best_rmsle):
            criterion = (
                f"验收：新候选需优于当前最优（RMSLE < {best_rmsle:.6g}），"
                "且 `evaluate_submission` 无 `math range error`。"
            )
        else:
            criterion = "验收：`evaluate_submission` 成功，且 RMSLE 持续下降。"

        return [target, action, criterion]

    def _normalize_candidate_analysis_section(self, lines: list[str]) -> list[str]:
        best_candidate = self._load_current_best_from_plan()
        if not isinstance(best_candidate, dict):
            return self._keep_latest_candidate_analysis_section(lines)

        round_num = best_candidate.get("round")
        equation = str(best_candidate.get("equation", "") or "").strip()
        if not isinstance(round_num, int) or not equation:
            return self._keep_latest_candidate_analysis_section(lines)

        round_summaries = self._extract_round_summaries_from_experiment_table(lines)
        summary = dict(round_summaries.get(round_num, {}))
        summary.setdefault("round", round_num)
        summary.setdefault("equation", equation)
        summary.setdefault("rmsle", best_candidate.get("rmsle"))
        summary.setdefault("exact", best_candidate.get("exact_accuracy"))
        summary.setdefault(
            "conclusion",
            (
                "symbolic="
                f"{self._format_metric(best_candidate.get('symbolic_equivalent'))}; "
                "source=current_best"
            ),
        )

        pattern = re.compile(r"^##\s*候选方程解析[（(]\s*Round\s*(\d+)\s*[)）]\s*$")
        ranges_to_remove: list[tuple[int, int]] = []
        for i, line in enumerate(lines):
            if not pattern.match(line.strip()):
                continue
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    end = j
                    break
            ranges_to_remove.append((i, end))

        for start, end in sorted(ranges_to_remove, reverse=True):
            del lines[start:end]

        insert_idx = len(lines)
        for i, line in enumerate(lines):
            if line.strip() in {"## Worth Trying Next", "## 最优方程演化"}:
                insert_idx = i
                break

        block = self._build_candidate_analysis_block(summary)
        prefix = lines[:insert_idx]
        suffix = lines[insert_idx:]
        if prefix and prefix[-1].strip():
            prefix.append("")
        return prefix + block + [""] + suffix

    def _build_candidate_analysis_block(self, summary: dict[str, Any]) -> list[str]:
        round_num = summary.get("round")
        equation = self._compact_text(str(summary.get("equation", "") or ""), limit=180)
        features = self._extract_equation_features(equation)
        rmsle = self._format_metric(summary.get("rmsle"))
        exact = self._format_metric(summary.get("exact"))
        structure_note = self._build_physical_insight_for_round(summary)

        if features["has_denominator_sensitive"]:
            sensitivity_1 = "分母里存在减法敏感项，靠近零时会放大误差，常数微调可能显著改变整体曲线。"
        elif features["has_additive_bias"]:
            sensitivity_1 = "式子包含常数偏置，低值区的拟合会强依赖这个 baseline，需警惕它只是补偿项。"
        else:
            sensitivity_1 = "当前系数主要承担尺度伸缩作用，应优先检查它们是否只是把不同量纲硬拼到一起。"

        if features["has_mixed_scales"]:
            sensitivity_2 = "系数跨多个数量级，说明表达式可能存在补丁式配平；轻微改动就可能破坏稳定性。"
        else:
            sensitivity_2 = "系数数量级相对集中，下一步应尝试减少自由常数并检查误差是否显著上升。"

        ablations: list[str] = []
        if features["has_additive_bias"]:
            ablations.append("去掉常数偏置项，检查它是否只是吸收了 proxy 的平均值。")
        if features["has_sqrt"]:
            ablations.append("去掉 `sqrt` 项或把它改写为幂律 `omega**a`，判断根号是否必要。")
        if features["has_log"]:
            ablations.append("把 `log` 项替换成同主变量上的 `exp` 或纯幂律，比较误差是否显著恶化。")
        if features["has_direct_sum"]:
            ablations.append("把 `T+omega` 一类直接相加项改成无量纲组合，排查是否只是量纲不一致的插值补丁。")
        if not ablations:
            ablations.append("围绕主变量做最小化消融，删除一个主导项后重新评测，确认哪些项是真正必要的。")

        return [
            f"## 候选方程解析（Round {round_num}）",
            "### 1) 方程与物理解释",
            f"- 当前最优返回式：`{equation}`",
            f"- 评测概览：RMSLE={rmsle}，Exact={exact}。",
            f"- 结构摘要：{structure_note[0]}",
            "### 2) 参数/系数敏感性",
            f"- {sensitivity_1}",
            f"- {sensitivity_2}",
            "### 3) 物理洞察",
            f"- {structure_note[1]}",
            f"- {structure_note[2]}",
            "### 4) 消融分析",
            *[f"- {item}" for item in ablations],
            "- 只有在删去对应项后误差显著变差，才能把该项视为机制成分而非数值补丁。",
        ]

    def _dedupe_lines_before_marker(
        self,
        lines: list[str],
        section_header: str,
        marker: str,
    ) -> list[str]:
        header_idx = -1
        marker_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == section_header:
                header_idx = i
                break
        if header_idx < 0:
            return lines

        for i in range(header_idx + 1, len(lines)):
            if marker in lines[i]:
                marker_idx = i
                break
        if marker_idx < 0 or marker_idx <= header_idx:
            return lines

        body = lines[header_idx + 1 : marker_idx]
        deduped: list[str] = []
        seen: set[str] = set()
        for line in body:
            stripped = line.strip()
            if not stripped:
                continue
            # Normalize bullet style so plain line / "- line" are treated as duplicates.
            key = re.sub(r"^[-*]\s+", "", stripped)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(line)

        return lines[: header_idx + 1] + deduped + [""] + lines[marker_idx:]

    def _keep_latest_candidate_analysis_section(self, lines: list[str]) -> list[str]:
        pattern = re.compile(r"^##\s*候选方程解析[（(]\s*Round\s*(\d+)\s*[)）]\s*$")

        heads: list[tuple[int, int]] = []
        for i, line in enumerate(lines):
            m = pattern.match(line.strip())
            if m:
                try:
                    heads.append((i, int(m.group(1))))
                except Exception:
                    continue

        if len(heads) <= 1:
            return lines

        keep_start, keep_round = max(heads, key=lambda x: (x[1], x[0]))
        keep_set = {(keep_start, keep_round)}

        ranges_to_remove: list[tuple[int, int]] = []
        for idx, (start, round_num) in enumerate(heads):
            if (start, round_num) in keep_set:
                continue
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if lines[j].startswith("## "):
                    end = j
                    break
            ranges_to_remove.append((start, end))

        for start, end in sorted(ranges_to_remove, reverse=True):
            del lines[start:end]

        return lines

    def _extract_round_num_from_table_row(self, row: str) -> int | None:
        if not isinstance(row, str):
            return None
        stripped = row.strip()
        m = re.match(r"^\|\s*Round\s+(\d+)\s*\|", stripped, flags=re.I)
        if not m:
            m = re.match(r"^\|\s*(\d+)\s*\|", stripped, flags=re.I)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _extract_return_expression(self, final_law_code: str) -> str:
        if not isinstance(final_law_code, str):
            return ""
        for line in final_law_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                return stripped[len("return ") :].strip()
        return ""

    def _normalize_expr_for_compare(self, expr: str) -> str:
        if not isinstance(expr, str):
            return ""
        # Keep comparison lightweight and robust to formatting-only differences.
        return re.sub(r"\s+", "", expr)

    def _compact_text(self, text: str, limit: int = 120) -> str:
        if not isinstance(text, str):
            return ""
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)] + "..."

    def _append_system_round_to_plan(self, signal: dict[str, Any]) -> None:
        if not self.run_dir:
            return
        plan_file = self.run_dir / "plan.md"
        if not plan_file.exists():
            return

        marker = f"<!-- HAM_SYS_BACKFILL_PLAN_ROUND_{self.round_num} -->"
        existing = plan_file.read_text(encoding="utf-8")
        if marker in existing:
            return

        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            protocol = {}
        violations = protocol.get("violations", [])
        if not isinstance(violations, list):
            violations = []

        lines = [
            "",
            marker,
            f"## 系统回填 Round {self.round_num}",
            f"- task_completed: {signal.get('task_completed')}",
            f"- satisfied: {signal.get('satisfied')}",
        ]

        if violations:
            lines.append(f"- 本轮协议问题: {', '.join(str(v) for v in violations)}")

        task_completed = str(signal.get("task_completed") or "").strip().lower()
        if task_completed != "true":
            lines.append(
                "- 下一轮要求: 请在本文件补全新的实验参数（至少 1 组 --inputs-json），"
                "并写明与上一轮不同的假设。"
            )
        else:
            lines.append("- 本轮已完成: 可将最终方程与常数估计过程整理进 findings.md 的关键洞察。")

        with plan_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        self.logger.info(
            "[Round %s] Auto-backfilled plan.md in workspace: %s",
            self.round_num,
            plan_file,
        )

    def _sync_current_best_from_protocol(self, signal: dict[str, Any], finish_message: str) -> None:
        """System-side CURRENT_BEST maintenance using marker-bounded replacement."""
        if not self.run_dir:
            return
        plan_file = self.run_dir / "plan.md"
        if not plan_file.exists():
            return

        protocol = signal.get("protocol", {})
        if not isinstance(protocol, dict):
            return

        eval_success_calls = int(protocol.get("evaluate_submission_success_calls", 0) or 0)
        if eval_success_calls < 1:
            return

        last_eval = protocol.get("last_evaluation", {})
        if not isinstance(last_eval, dict):
            return

        symbolic_equivalent = last_eval.get("symbolic_equivalent")
        if not isinstance(symbolic_equivalent, bool):
            symbolic_equivalent = None
        exact_accuracy = self._safe_float(last_eval.get("exact_accuracy"))
        rmsle = self._safe_float(last_eval.get("rmsle"))
        if symbolic_equivalent is None and exact_accuracy is None and rmsle is None:
            return

        final_law_code = self._extract_final_law_code(finish_message)
        submitted_law = str(last_eval.get("submitted_law") or "").strip()
        canonical_law_code = submitted_law or final_law_code.strip()
        equation = self._extract_return_expression(canonical_law_code)
        if not equation:
            equation = self._extract_final_law_signature(canonical_law_code) or "N/A"
        equation = self._compact_text(equation, limit=140).replace("\n", " ").replace("|", "/")

        candidate = {
            "round": self.round_num,
            "equation": equation,
            "symbolic_equivalent": symbolic_equivalent,
            "exact_accuracy": exact_accuracy,
            "rmsle": rmsle,
            "law_code": canonical_law_code if self._is_valid_discovered_law_code(canonical_law_code) else None,
        }

        text = plan_file.read_text(encoding="utf-8")
        begin_marker = "<!-- EVO_CURRENT_BEST_BEGIN -->"
        end_marker = "<!-- EVO_CURRENT_BEST_END -->"
        begin_idx = text.find(begin_marker)
        end_idx = text.find(end_marker)
        if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
            return

        inner_start = begin_idx + len(begin_marker)
        current_block = text[inner_start:end_idx]
        existing = self._parse_current_best_block(current_block)
        if existing and not self._is_candidate_better(candidate, existing):
            return

        meta = {
            "round": candidate["round"],
            "symbolic_equivalent": candidate["symbolic_equivalent"],
            "exact_accuracy": candidate["exact_accuracy"],
            "rmsle": candidate["rmsle"],
            "equation": candidate["equation"],
            "law_code": candidate["law_code"],
        }
        metrics_note = (
            "symbolic_equivalent="
            f"{self._format_metric(candidate['symbolic_equivalent'])}, "
            "exact_accuracy="
            f"{self._format_metric(candidate['exact_accuracy'])}, "
            "rmsle="
            f"{self._format_metric(candidate['rmsle'])}"
        )
        block_lines = [
            "",
            "## 当前最优",
            f"- 轮次：{candidate['round']}",
            f"- 方程：{candidate['equation']}",
            "- MSE：未知",
            f"- 更新时间：系统自动维护（{metrics_note}）",
            f"<!-- EVO_CURRENT_BEST_META: {json.dumps(meta, ensure_ascii=False)} -->",
        ]
        new_inner = "\n".join(block_lines) + "\n"
        new_text = text[:inner_start] + new_inner + text[end_idx:]
        plan_file.write_text(new_text, encoding="utf-8")

        self.logger.info(
            "[Round %s] Updated CURRENT_BEST in plan.md via protocol metrics",
            self.round_num,
        )

    def _parse_current_best_block(self, block_text: str) -> dict[str, Any] | None:
        if not isinstance(block_text, str) or not block_text.strip():
            return None

        meta_match = re.search(
            r"<!--\s*EVO_CURRENT_BEST_META:\s*(\{.*?\})\s*-->",
            block_text,
            flags=re.S,
        )
        if meta_match:
            payload = self._safe_json_loads(meta_match.group(1))
            if isinstance(payload, dict):
                return self._normalize_best_candidate(payload)

        round_match = re.search(r"轮次[:：]\s*(\d+)", block_text)
        equation_match = re.search(r"方程[:：]\s*(.+)", block_text)
        metric_match = re.search(
            r"symbolic_equivalent\s*=\s*([^,\s]+).*?"
            r"exact_accuracy\s*=\s*([^,\s]+).*?"
            r"rmsle\s*=\s*([^,\s]+)",
            block_text,
            flags=re.S,
        )

        if not round_match and not equation_match and not metric_match:
            return None

        parsed: dict[str, Any] = {}
        if round_match:
            try:
                parsed["round"] = int(round_match.group(1))
            except Exception:
                pass
        if equation_match:
            parsed["equation"] = equation_match.group(1).strip()
        if metric_match:
            symbolic_raw = re.sub(r"[)\]）】。，；;]+$", "", metric_match.group(1).strip().lower())
            if symbolic_raw in {"true", "false"}:
                parsed["symbolic_equivalent"] = symbolic_raw == "true"
            exact_raw = re.sub(r"[)\]）】。，；;]+$", "", metric_match.group(2).strip())
            rmsle_raw = re.sub(r"[)\]）】。，；;]+$", "", metric_match.group(3).strip())
            parsed["exact_accuracy"] = self._safe_float(exact_raw)
            parsed["rmsle"] = self._safe_float(rmsle_raw)

        return self._normalize_best_candidate(parsed)

    def _normalize_best_candidate(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        round_val: int | None = None
        try:
            if payload.get("round") is not None:
                round_val = int(payload.get("round"))
        except Exception:
            round_val = None

        symbolic = payload.get("symbolic_equivalent")
        if not isinstance(symbolic, bool):
            symbolic = None

        exact_accuracy = self._safe_float(payload.get("exact_accuracy"))
        rmsle = self._safe_float(payload.get("rmsle"))
        equation = str(payload.get("equation", "") or "").strip()
        law_code = str(payload.get("law_code", "") or "").strip()
        if law_code and not self._is_valid_discovered_law_code(law_code):
            law_code = ""

        if (
            round_val is None
            and not equation
            and symbolic is None
            and exact_accuracy is None
            and rmsle is None
            and not law_code
        ):
            return None

        return {
            "round": round_val,
            "equation": equation or "无",
            "symbolic_equivalent": symbolic,
            "exact_accuracy": exact_accuracy,
            "rmsle": rmsle,
            "law_code": law_code or None,
        }

    def _best_quality_key(self, candidate: dict[str, Any]) -> tuple[float, float, float]:
        symbolic = candidate.get("symbolic_equivalent")
        if symbolic is True:
            symbolic_score = 2.0
        elif symbolic is False:
            symbolic_score = 1.0
        else:
            symbolic_score = 0.0

        exact_accuracy = candidate.get("exact_accuracy")
        if isinstance(exact_accuracy, float) and math.isfinite(exact_accuracy):
            exact_score = exact_accuracy
        else:
            exact_score = -1.0

        rmsle = candidate.get("rmsle")
        if isinstance(rmsle, float) and math.isfinite(rmsle):
            rmsle_score = -rmsle
        else:
            rmsle_score = -1e12

        return (symbolic_score, exact_score, rmsle_score)

    def _is_candidate_better(self, candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
        c_key = self._best_quality_key(candidate)
        e_key = self._best_quality_key(existing)
        if c_key > e_key:
            return True
        if c_key < e_key:
            return False

        c_round = candidate.get("round")
        e_round = existing.get("round")
        if isinstance(c_round, int) and isinstance(e_round, int):
            return c_round > e_round
        return False

    def _extract_final_law_code(self, finish_message: str) -> str:
        blocks = self._extract_all_final_law_blocks(finish_message)
        if not blocks:
            return ""
        # Prefer the last valid block to avoid prose containing "<final_law>".
        for code in reversed(blocks):
            if self._is_valid_discovered_law_code(code):
                return code
        return blocks[-1]

    def _extract_all_final_law_blocks(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        blocks: list[str] = []
        matches = list(FINAL_LAW_STANDALONE_BLOCK_RE.finditer(text))
        if not matches:
            matches = list(FINAL_LAW_BLOCK_RE.finditer(text))
        for match in matches:
            code = str(match.group(1) or "").strip()
            if code:
                blocks.append(code)
        return blocks

    def _resolve_finish_message_with_eval_fallback(self, trajectory, finish_message: str) -> str:
        """Force finish.final_law to an evaluated best law; fallback to global best in plan when better."""
        source = finish_message if isinstance(finish_message, str) else ""
        script_records = self._collect_skill_script_records(trajectory)
        best_eval = self._extract_last_successful_evaluation(script_records)
        fallback_law = str(best_eval.get("submitted_law") or "").strip()

        plan_best = self._load_current_best_from_plan()
        if isinstance(plan_best, dict):
            plan_law = str(plan_best.get("law_code") or "").strip()
            if plan_law and self._is_valid_discovered_law_code(plan_law):
                current_key = self._best_quality_key(best_eval) if isinstance(best_eval, dict) else (-1e12, -1e12, -1e12)
                if self._best_quality_key(plan_best) > current_key:
                    fallback_law = plan_law

        if not fallback_law:
            return source

        current_code = self._extract_final_law_code(source)
        current_expr = self._normalize_expr_for_compare(self._extract_return_expression(current_code))
        fallback_expr = self._normalize_expr_for_compare(self._extract_return_expression(fallback_law))
        same_expr = bool(current_expr and fallback_expr and current_expr == fallback_expr)
        if self._is_valid_discovered_law_code(current_code) and same_expr:
            return source

        patched = self._replace_or_append_final_law_block(source, fallback_law)
        self.logger.warning(
            "[Round %s] final_law replaced by evaluated/global-best law for stability.",
            self.round_num,
        )
        return patched

    def _load_current_best_from_plan(self) -> dict[str, Any] | None:
        if not self.run_dir:
            return None
        plan_file = self.run_dir / "plan.md"
        if not plan_file.exists():
            return None
        text = plan_file.read_text(encoding="utf-8")
        begin_marker = "<!-- EVO_CURRENT_BEST_BEGIN -->"
        end_marker = "<!-- EVO_CURRENT_BEST_END -->"
        begin_idx = text.find(begin_marker)
        end_idx = text.find(end_marker)
        if begin_idx < 0 or end_idx < 0 or end_idx <= begin_idx:
            return None
        block = text[begin_idx + len(begin_marker) : end_idx]
        return self._parse_current_best_block(block)

    def _replace_or_append_final_law_block(self, finish_message: str, final_law_code: str) -> str:
        block = f"<final_law>\n{final_law_code.strip()}\n</final_law>"
        source = finish_message if isinstance(finish_message, str) else ""
        matches = list(FINAL_LAW_STANDALONE_BLOCK_RE.finditer(source))
        if not matches:
            matches = list(FINAL_LAW_BLOCK_RE.finditer(source))
        if matches:
            last = matches[-1]
            return source[: last.start()] + block + source[last.end() :]
        if source.strip():
            return source.rstrip() + "\n\n" + block
        return block

    def _is_valid_discovered_law_code(self, code: str) -> bool:
        if not isinstance(code, str) or not code.strip():
            return False
        if not self._canonical_discovered_law_signature(code):
            return False
        try:
            compile(code, "<final_law>", "exec")
            return True
        except Exception:
            return False

    def _format_metric(self, value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if math.isfinite(value):
                return f"{value:.12g}"
            return "None"
        return str(value)

    def _extract_agent_response(self, trajectory) -> str:
        return super()._extract_agent_response(trajectory)

    # =========================
    # Signal parsing
    # =========================

    def _parse_signal(
        self,
        agent_message: str,
        trajectory,
        task_description: str | None = None,
        finish_message_override: str | None = None,
    ) -> dict:
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
        finish_message = (
            finish_message_override
            if isinstance(finish_message_override, str)
            else self._extract_finish_message_from_trajectory(trajectory)
        )

        signal: dict[str, Any] = {
            "round": self.round_num,
            "satisfied": satisfied,
            "task_completed": task_completed,
            "notes": finish_message[:500] if finish_message else "",
        }

        if self._is_newtonbench_task(task_description):
            protocol = self._evaluate_newtonbench_protocol(
                trajectory,
                finish_message,
                task_completed=task_completed,
            )
            signal["protocol"] = protocol

            if satisfied and protocol["violations"]:
                self.logger.warning(
                    f"[Round {self.round_num}] NewtonBench protocol guard blocked completion: "
                    f"{protocol['violations']}"
                )
                signal["satisfied"] = False
                # Keep task_completed consistent with guard decision so downstream
                # summaries do not report "completed=true" when quality gates fail.
                signal["task_completed"] = "false"
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

    def _evaluate_newtonbench_protocol(
        self,
        trajectory,
        finish_message: str,
        task_completed: str | None = None,
    ) -> dict[str, Any]:
        """Validate completion protocol for NewtonBench tasks."""
        script_records = self._collect_skill_script_records(trajectory)
        script_stats = self._collect_skill_script_stats(script_records)
        run_stats = script_stats.get("run_experiment.py", {"calls": 0, "success_calls": 0})
        eval_stats = script_stats.get("evaluate_submission.py", {"calls": 0, "success_calls": 0})
        fit_stats = script_stats.get("fit_pysr_candidates.py", {"calls": 0, "success_calls": 0})

        has_final_law_block = bool(FINAL_LAW_BLOCK_RE.search(finish_message or ""))
        has_discovered_law_signature = bool(DISCOVERED_LAW_SIGNATURE_RE.search(finish_message or ""))
        expected_signature = self._extract_expected_function_signature(script_records)
        final_law_signature = self._extract_final_law_signature(finish_message)
        final_law_code = self._extract_final_law_code(finish_message)
        final_return_expr = self._extract_return_expression(final_law_code)
        signature_match = (
            expected_signature == final_law_signature
            if expected_signature and final_law_signature
            else None
        )
        last_eval_metrics = self._extract_last_successful_evaluation(script_records)

        protocol_cfg = self._get_newtonbench_protocol_cfg()
        search_mode = self._get_search_mode()
        require_signature_match = bool(protocol_cfg.get("require_signature_match", True))
        require_finite_rmsle = bool(protocol_cfg.get("require_finite_rmsle", True))
        require_symbolic_or_exact = bool(protocol_cfg.get("require_symbolic_or_exact", False))
        require_pysr_assisted_call = bool(
            protocol_cfg.get(
                "require_pysr_assisted_call",
                search_mode == "pysr_assisted",
            )
        )
        min_exact_accuracy_raw = protocol_cfg.get("min_exact_accuracy", 1.0)
        min_exact_accuracy = self._safe_float(min_exact_accuracy_raw)
        if min_exact_accuracy is None:
            min_exact_accuracy = 1.0
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
        if require_pysr_assisted_call and int(fit_stats.get("success_calls", 0)) < 1:
            violations.append("missing_successful_fit_pysr_candidates")
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
            eval_exact = last_eval_metrics.get("exact_accuracy")
            symbolic_equivalent = last_eval_metrics.get("symbolic_equivalent")
            evaluated_return_expr = str(last_eval_metrics.get("submitted_return_expr") or "").strip()
            if require_finite_rmsle and eval_rmsle is None:
                violations.append("non_finite_rmsle")
            if max_rmsle_value is not None and (
                eval_rmsle is None or eval_rmsle > max_rmsle_value
            ):
                violations.append("rmsle_above_threshold")
            # Enforce "final_law matches last evaluated" only when the agent
            # claims completion. During ongoing rounds (task_completed=false),
            # rollback to previous best may be valid.
            require_final_law_last_eval = str(task_completed or "").strip().lower() == "true"
            if require_final_law_last_eval and final_return_expr and evaluated_return_expr:
                final_expr_norm = self._normalize_expr_for_compare(final_return_expr)
                evaluated_expr_norm = self._normalize_expr_for_compare(evaluated_return_expr)
                if final_expr_norm and evaluated_expr_norm and final_expr_norm != evaluated_expr_norm:
                    violations.append("final_law_not_last_evaluated")
            if require_symbolic_or_exact:
                symbolic_ok = symbolic_equivalent is True
                exact_ok = (
                    isinstance(eval_exact, float)
                    and math.isfinite(eval_exact)
                    and eval_exact >= min_exact_accuracy
                )
                if not (symbolic_ok or exact_ok):
                    violations.append("not_symbolic_and_exact_below_threshold")

        return {
            "search_mode": search_mode,
            "run_experiment_calls": int(run_stats.get("calls", 0)),
            "run_experiment_success_calls": int(run_stats.get("success_calls", 0)),
            "evaluate_submission_calls": int(eval_stats.get("calls", 0)),
            "evaluate_submission_success_calls": int(eval_stats.get("success_calls", 0)),
            "fit_pysr_candidates_calls": int(fit_stats.get("calls", 0)),
            "fit_pysr_candidates_success_calls": int(fit_stats.get("success_calls", 0)),
            "has_final_law_block": has_final_law_block,
            "has_discovered_law_signature": has_discovered_law_signature,
            "expected_function_signature": expected_signature,
            "final_law_signature": final_law_signature,
            "signature_match": signature_match,
            "last_evaluation": last_eval_metrics,
            "quality_guard": {
                "require_signature_match": require_signature_match,
                "require_finite_rmsle": require_finite_rmsle,
                "require_symbolic_or_exact": require_symbolic_or_exact,
                "require_pysr_assisted_call": require_pysr_assisted_call,
                "min_exact_accuracy": min_exact_accuracy,
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
                output_text = output if isinstance(output, str) else str(output)

                success = self._is_tool_call_success(step, tool_call_id)
                if success and script_name == "evaluate_submission.py":
                    success = self._is_successful_evaluation_output(output_text)

                records.append(
                    {
                        "script_name": script_name,
                        "args": args,
                        "tool_call_id": tool_call_id,
                        "success": success,
                        "output": output_text,
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
        code = self._extract_final_law_code(finish_message or "")
        return self._canonical_discovered_law_signature(code)

    def _extract_last_successful_evaluation(
        self,
        script_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        best_metrics: dict[str, Any] | None = None
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
            submitted_law = payload.get("submitted_law")
            if isinstance(submitted_law, str) and submitted_law.strip():
                parsed["submitted_law"] = submitted_law.strip()
                parsed["submitted_return_expr"] = self._extract_return_expression(submitted_law)

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

            if best_metrics is None:
                best_metrics = parsed
                continue
            if self._best_quality_key(parsed) > self._best_quality_key(best_metrics):
                best_metrics = parsed
        return best_metrics or {}

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

    def _get_search_mode(self) -> str:
        """Read experiment.search_mode with safe fallback."""
        try:
            experiment_cfg = getattr(self.config, "experiment", {})
            if isinstance(experiment_cfg, dict):
                raw = str(experiment_cfg.get("search_mode", "llm_direct") or "llm_direct").strip().lower()
                if raw in {"llm_direct", "pysr_assisted"}:
                    return raw
        except Exception:
            pass
        return "llm_direct"

    def _get_newtonbench_protocol_cfg(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "require_signature_match": True,
            "require_finite_rmsle": True,
            "require_symbolic_or_exact": False,
            "require_pysr_assisted_call": False,
            "min_exact_accuracy": 1.0,
            # Quality gate default: avoid "task completed" with extremely poor fit.
            "max_rmsle": 1.0,
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

    def _is_successful_evaluation_output(self, output_text: str) -> bool:
        """Judge evaluate_submission success by payload, not only exit_code."""
        payload = self._extract_last_json_object(output_text if isinstance(output_text, str) else "")
        if not isinstance(payload, dict):
            return False
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            return False
        eval_error = evaluation.get("error")
        if isinstance(eval_error, str) and eval_error.strip():
            return False
        return True
