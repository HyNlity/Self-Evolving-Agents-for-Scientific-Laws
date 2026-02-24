"""Hamilton Round Exp - 单轮执行

负责单轮的数据分析和PySR调用。

每轮流程：
- HamiltonAgent: 读取 analysis.md, 写代码分析 + PySR, 维护 analysis.md
- Eureka Agent: 读取 analysis.md + PySR结果, 写入 insight.md

文件规范：
- analysis.md: HamiltonAgent 写入，格式 ## Round N: ...
- experiment.json: 系统自动记录 PySR 参数和结果
- insight.md: Eureka Agent 追加每轮结论；顶部 Current Best 区块由系统自动维护
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from evomaster.core.exp import BaseExp
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance


class RoundExp(BaseExp):
    """单轮实验

    负责：
    1. HamiltonAgent 执行分析（变量分析、筛选、PySR）
    2. Eureka Agent 生成靠谱发现
    """

    def __init__(self, hamilton_agent, eureka_agent, config, round_num):
        """初始化 RoundExp

        Args:
            hamilton_agent: Hamilton Agent 实例（主分析）
            eureka_agent: Eureka Agent 实例（结果分析）
            config: 配置
            round_num: 轮次编号
        """
        super().__init__(hamilton_agent, config)
        self.hamilton_agent = hamilton_agent
        self.eureka_agent = eureka_agent
        self.round_num = round_num
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def exp_name(self) -> str:
        return f"Round_{self.round_num}"

    def run(self, task_description: str, task_id: str = "exp_001") -> dict:
        """执行单轮实验

        Args:
            task_description: 任务描述
            task_id: 任务ID

        Returns:
            执行结果
        """
        self.logger.info(f"Starting Round {self.round_num}")

        # 设置实验信息
        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=self.round_num)

        # 设置环境变量，让 PySRTool 知道当前轮次
        os.environ["HAMILTON_ROUND"] = str(self.round_num)

        # 初始化本轮的文件结构
        self._init_round_files()

        # 确保本轮目录存在
        self._ensure_round_dirs()

        # ========== 步骤1: HamiltonAgent 执行分析 ==========
        self.logger.info(f"[Round {self.round_num}] Running HamiltonAgent...")
        hamilton_task = TaskInstance(
            task_id=f"{task_id}_round{self.round_num}_hamilton",
            task_type="hamilton",
            description=task_description,
            input_data={"round": self.round_num},
        )

        hamilton_trajectory = self.hamilton_agent.run(hamilton_task)
        hamilton_result = self._extract_agent_response(hamilton_trajectory)

        self.logger.info(f"[Round {self.round_num}] HamiltonAgent completed")

        # ========== 步骤2: Eureka Agent 生成发现 ==========
        self.logger.info(f"[Round {self.round_num}] Running Eureka Agent...")
        eureka_task = TaskInstance(
            task_id=f"{task_id}_round{self.round_num}_eureka",
            task_type="eureka",
            description=task_description,
            input_data={
                "round": self.round_num,
                "hamilton_result": hamilton_result,
            },
        )

        eureka_trajectory = self.eureka_agent.run(eureka_task)
        eureka_result = self._extract_agent_response(eureka_trajectory)

        self.logger.info(f"[Round {self.round_num}] Eureka Agent completed")

        # 解析 Eureka 的结构化信号，并基于其决策更新 insight.md 顶部 Current Best 区块
        eureka_signal = self._parse_eureka_signal(eureka_result, eureka_trajectory)
        self._maybe_update_current_best(eureka_signal)

        # 提取 insight.md 内容
        insight_content = self._read_insight()

        self.logger.info(f"Round {self.round_num} completed")

        return {
            "round": self.round_num,
            "hamilton_result": hamilton_result,
            "eureka_result": eureka_result,
            "eureka_signal": eureka_signal,
            "insight": insight_content,
            "hamilton_trajectory": hamilton_trajectory,
            "eureka_trajectory": eureka_trajectory,
        }

    def _ensure_round_dirs(self):
        """确保本轮目录存在"""
        if not self.run_dir:
            return

        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        scripts_dir = round_dir / "scripts"
        results_dir = round_dir / "results"

        scripts_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    def _init_round_files(self):
        """初始化本轮的文件结构

        - 在 analysis.md 中添加 ## Round N 头部
        - 初始化 experiment.json（如果不存在）
        """
        if not self.run_dir:
            return

        # 1. analysis.md - 添加本轮头部
        analysis_file = self.run_dir / "analysis.md"

        round_header = f"\n---\n\n## Round {self.round_num}\n"

        should_append_header = True
        if analysis_file.exists():
            # 检查是否已有本轮记录
            existing_content = analysis_file.read_text(encoding="utf-8")
            if f"## Round {self.round_num}" in existing_content:
                self.logger.info(f"Round {self.round_num} already initialized in analysis.md")
                should_append_header = False

        # 追加本轮头部
        if should_append_header:
            with open(analysis_file, "a", encoding="utf-8") as f:
                f.write(round_header)

        if should_append_header:
            self.logger.info(f"Initialized Round {self.round_num} in analysis.md")

        # 2. experiment.json - 初始化或确保结构存在
        experiment_file = self.run_dir / "experiment.json"

        experiment_data = None
        if experiment_file.exists():
            try:
                experiment_data = json.loads(experiment_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                experiment_data = None

        if not isinstance(experiment_data, dict):
            experiment_data = {"task": "", "rounds": {}}

        if "rounds" not in experiment_data or not isinstance(experiment_data.get("rounds"), dict):
            experiment_data["rounds"] = {}

        experiment_file.write_text(json.dumps(experiment_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_insight(self) -> str:
        """读取本轮的 insight 内容"""
        if not self.run_dir:
            return ""

        insight_file = self.run_dir / "insight.md"
        if insight_file.exists():
            with open(insight_file, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _extract_agent_response(self, trajectory) -> str:
        """从轨迹中提取最终 assistant 文本（复用 BaseExp 逻辑）"""
        return super()._extract_agent_response(trajectory)

    # =========================
    # Eureka signal + Current Best
    # =========================

    _EUREKA_SIGNAL_BEGIN = "===EVO_EUREKA_SIGNAL_BEGIN==="
    _EUREKA_SIGNAL_END = "===EVO_EUREKA_SIGNAL_END==="
    _CURRENT_BEST_BEGIN = "<!-- EVO_CURRENT_BEST_BEGIN -->"
    _CURRENT_BEST_END = "<!-- EVO_CURRENT_BEST_END -->"

    def _parse_eureka_signal(self, eureka_message: str, trajectory) -> dict:
        """Parse machine-readable Eureka signal from finish.message.

        Preferred format is a JSON block:
          ===EVO_EUREKA_SIGNAL_BEGIN===
          {...}
          ===EVO_EUREKA_SIGNAL_END===
        """
        text = eureka_message or ""
        signal = self._extract_signal_from_text(text)
        if not signal:
            # Fallback: try to recover finish.message from trajectory (in case extraction changed)
            recovered = self._extract_finish_message_from_trajectory(trajectory)
            if recovered and recovered != text:
                signal = self._extract_signal_from_text(recovered)

        if not isinstance(signal, dict):
            signal = {}

        # Lenient fallback: accept a single-line SATISFIED flag even without JSON block.
        if "satisfied" not in signal:
            satisfied = self._parse_satisfied_flag(text) or self._parse_satisfied_flag(self._extract_finish_message_from_trajectory(trajectory))
            if satisfied is not None:
                signal["satisfied"] = satisfied

        return self._normalize_eureka_signal(signal)

    def _extract_signal_from_text(self, text: str) -> dict:
        if not isinstance(text, str) or not text:
            return {}
        begin = self._EUREKA_SIGNAL_BEGIN
        end = self._EUREKA_SIGNAL_END
        if begin not in text or end not in text:
            return {}
        try:
            payload = text.split(begin, 1)[1].split(end, 1)[0].strip()
            data = json.loads(payload) if payload else {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _parse_satisfied_flag(self, text: str) -> bool | None:
        if not isinstance(text, str) or not text:
            return None
        m = re.search(r"\\bSATISFIED\\s*:\\s*(yes|no|true|false)\\b", text, flags=re.IGNORECASE)
        if not m:
            return None
        v = m.group(1).lower()
        return v in {"yes", "true"}

    def _extract_finish_message_from_trajectory(self, trajectory) -> str:
        """Extract finish.message directly from a trajectory (robust)."""
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

    def _normalize_eureka_signal(self, raw: dict) -> dict:
        def _as_bool(v) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "y"}
            if isinstance(v, (int, float)):
                return v != 0
            return False

        def _as_float(v) -> float | None:
            try:
                x = float(v)
            except Exception:
                return None
            return x

        satisfied = _as_bool(raw.get("satisfied", False))
        update_best = _as_bool(raw.get("update_best", raw.get("update_current_best", False)))

        best_equation = (
            raw.get("best_equation")
            or raw.get("current_best_equation")
            or raw.get("best_eq")
            or raw.get("equation")
        )
        if not isinstance(best_equation, str):
            best_equation = ""
        best_equation = best_equation.strip()

        best_mse = raw.get("best_mse", raw.get("mse"))
        best_mse_v = _as_float(best_mse)

        mse_source = raw.get("mse_source", raw.get("best_mse_source", "unknown"))
        if not isinstance(mse_source, str) or not mse_source.strip():
            mse_source = "unknown"

        notes = raw.get("notes", raw.get("reason", ""))
        if not isinstance(notes, str):
            notes = ""

        next_round_plan = raw.get("next_round_plan", raw.get("next_steps", []))
        if not isinstance(next_round_plan, list):
            next_round_plan = []
        next_round_plan = [x for x in next_round_plan if isinstance(x, str) and x.strip()]

        best_round = raw.get("best_round", raw.get("round", self.round_num))
        try:
            best_round_v = int(best_round)
        except Exception:
            best_round_v = self.round_num

        return {
            "round": self.round_num,
            "satisfied": satisfied,
            "update_best": update_best,
            "best_round": best_round_v,
            "best_equation": best_equation,
            "best_mse": best_mse_v,
            "mse_source": mse_source.strip(),
            "notes": notes.strip(),
            "next_round_plan": next_round_plan,
        }

    def _maybe_update_current_best(self, signal: dict) -> None:
        """Update the Current Best block in insight.md (system-managed)."""
        if not self.run_dir:
            return
        insight_file = self.run_dir / "insight.md"
        self._ensure_current_best_block(insight_file)

        if not isinstance(signal, dict) or not signal.get("update_best", False):
            return

        equation = signal.get("best_equation")
        if not isinstance(equation, str) or not equation.strip() or equation.strip().lower() == "none":
            return

        block = self._format_current_best_block(
            round_n=int(signal.get("best_round", self.round_num) or self.round_num),
            equation=equation.strip(),
            mse=signal.get("best_mse"),
            mse_source=str(signal.get("mse_source", "unknown") or "unknown"),
            notes=str(signal.get("notes", "") or ""),
        )

        try:
            text = insight_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            text = "# Insights\n\n"

        new_text = self._replace_block(text, self._CURRENT_BEST_BEGIN, self._CURRENT_BEST_END, block)
        insight_file.write_text(new_text, encoding="utf-8")

    def _ensure_current_best_block(self, insight_file: Path) -> None:
        """Ensure insight.md has a Current Best block delimited by markers."""
        placeholder = self._format_current_best_block(
            round_n=0,
            equation="none",
            mse=None,
            mse_source="unknown",
            notes="",
        )

        if not insight_file.exists():
            insight_file.write_text("# Insights\n\n" + placeholder + "\n\n", encoding="utf-8")
            return

        text = insight_file.read_text(encoding="utf-8")
        if self._CURRENT_BEST_BEGIN in text and self._CURRENT_BEST_END in text:
            return

        inserted = self._insert_after_title(text, placeholder)
        insight_file.write_text(inserted, encoding="utf-8")

    def _insert_after_title(self, text: str, block: str) -> str:
        if not isinstance(text, str):
            return block + "\n\n"
        # Prefer inserting after the first markdown H1 line.
        m = re.search(r"^#\\s+.*$", text, flags=re.MULTILINE)
        if not m:
            return block + "\n\n" + text
        line_end = text.find("\n", m.end())
        if line_end == -1:
            return text + "\n\n" + block + "\n\n"
        pos = line_end + 1
        # Skip following blank lines
        while pos < len(text) and text[pos] == "\n":
            pos += 1
        return text[:pos] + block + "\n\n" + text[pos:]

    def _replace_block(self, text: str, begin: str, end: str, new_block: str) -> str:
        if begin not in text or end not in text:
            return self._insert_after_title(text, new_block)
        start = text.find(begin)
        end_pos = text.find(end, start)
        if end_pos == -1:
            return self._insert_after_title(text, new_block)
        end_pos = end_pos + len(end)
        return text[:start] + new_block + text[end_pos:]

    def _format_current_best_block(
        self,
        round_n: int,
        equation: str,
        mse,
        mse_source: str,
        notes: str,
    ) -> str:
        mse_str = "unknown"
        try:
            if mse is not None:
                mse_str = str(float(mse))
        except Exception:
            mse_str = "unknown"

        lines = [
            self._CURRENT_BEST_BEGIN,
            "## Current Best (auto-updated)",
            f"- Round: {round_n}",
            "- Equation:",
            "```text",
            (equation or "none").strip(),
            "```",
            f"- MSE: {mse_str}",
            f"- MSESource: {(mse_source or 'unknown').strip()}",
            f"- UpdatedAt: {datetime.now().isoformat()}",
        ]
        if isinstance(notes, str) and notes.strip():
            lines.append(f"- Notes: {notes.strip()}")
        lines.append(self._CURRENT_BEST_END)
        return "\n".join(lines)
