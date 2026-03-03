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
from pathlib import Path

from evomaster.core.exp import BaseExp
from evomaster.agent import BaseAgent
from evomaster.utils.types import TaskInstance


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
        signal = self._parse_signal(agent_result, trajectory)

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

    def _parse_signal(self, agent_message: str, trajectory) -> dict:
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

        # Extract finish message for logging
        finish_message = self._extract_finish_message_from_trajectory(trajectory)

        return {
            "round": self.round_num,
            "satisfied": satisfied,
            "task_completed": task_completed,
            "notes": finish_message[:500] if finish_message else "",
        }

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
