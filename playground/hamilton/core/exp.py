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
from typing import Any

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
        self._init_round_files(task_description=task_description)

        # 确保本轮目录存在
        self._ensure_round_dirs()

        # ========== 步骤1: HamiltonAgent 执行分析 ==========
        self.logger.info(f"[Round {self.round_num}] Running HamiltonAgent...")
        previous_handoff = self._load_previous_handoff()
        hamilton_input_data = self._build_hamilton_input_data(previous_handoff)
        hamilton_task = TaskInstance(
            task_id=f"{task_id}_round{self.round_num}_hamilton",
            task_type="hamilton",
            description=task_description,
            input_data=hamilton_input_data,
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
        eureka_validation = self._validate_eureka_round_output(eureka_signal)
        eureka_artifacts = self._persist_eureka_artifacts(
            eureka_signal=eureka_signal,
            eureka_message=eureka_result,
            validation=eureka_validation,
        )

        # 提取 insight.md 内容
        insight_content = self._read_insight()

        self.logger.info(f"Round {self.round_num} completed")

        return {
            "round": self.round_num,
            "hamilton_result": hamilton_result,
            "eureka_result": eureka_result,
            "eureka_signal": eureka_signal,
            "eureka_validation": eureka_validation,
            "eureka_artifacts": eureka_artifacts,
            "insight": insight_content,
            "hamilton_trajectory": hamilton_trajectory,
            "eureka_trajectory": eureka_trajectory,
        }

    # =========================
    # Handoff (Eureka -> Hamilton)
    # =========================

    def _build_hamilton_input_data(self, previous_handoff: dict[str, Any]) -> dict[str, Any]:
        """构建 Hamilton 输入上下文（含上一轮 Eureka handoff）。"""
        handoff = previous_handoff if isinstance(previous_handoff, dict) else {}

        exists = bool(handoff.get("exists", False))
        source_round = handoff.get("source_round", max(0, self.round_num - 1))
        try:
            source_round_v = int(source_round)
        except Exception:
            source_round_v = max(0, self.round_num - 1)

        best_eq = handoff.get("best_equation", "none")
        if not isinstance(best_eq, str) or not best_eq.strip():
            best_eq = "none"

        best_mse = handoff.get("best_mse")
        best_mse_str = "unknown"
        try:
            if best_mse is not None:
                best_mse_str = str(float(best_mse))
        except Exception:
            best_mse_str = "unknown"

        mse_source = handoff.get("mse_source", "unknown")
        if not isinstance(mse_source, str) or not mse_source.strip():
            mse_source = "unknown"

        notes = handoff.get("notes", "")
        if not isinstance(notes, str) or not notes.strip():
            notes = "none"
        else:
            notes = notes.strip()

        next_round_plan = handoff.get("next_round_plan", [])
        if not isinstance(next_round_plan, list):
            next_round_plan = []
        next_round_plan = [x.strip() for x in next_round_plan if isinstance(x, str) and x.strip()]
        next_round_plan_text = "none"
        if next_round_plan:
            next_round_plan_text = " ; ".join(next_round_plan)

        return {
            "round": self.round_num,
            "previous_eureka_handoff_exists": "yes" if exists else "no",
            "previous_eureka_source_round": source_round_v,
            "previous_eureka_handoff_path": str(handoff.get("path", "none") or "none"),
            "previous_eureka_best_equation": best_eq.strip(),
            "previous_eureka_best_mse": best_mse_str,
            "previous_eureka_mse_source": mse_source.strip(),
            "previous_eureka_next_round_plan": next_round_plan_text,
            "previous_eureka_notes": notes,
            # 保留原始结构，供后续 prompt/tool 按需使用
            "previous_eureka_handoff": handoff,
        }

    def _load_previous_handoff(self) -> dict[str, Any]:
        """读取上一轮 Eureka handoff（缺失时返回默认空结构）。"""
        default = {
            "exists": False,
            "source_round": max(0, self.round_num - 1),
            "path": "none",
            "best_equation": "none",
            "best_mse": None,
            "mse_source": "unknown",
            "notes": "",
            "next_round_plan": [],
            "update_best": False,
            "satisfied": False,
        }

        if not self.run_dir or self.round_num <= 1:
            return default

        prev_round = self.round_num - 1
        handoff_path = self.run_dir / "history" / f"round{prev_round}" / "results" / self._EUREKA_HANDOFF_FILENAME
        default["path"] = self._to_workspace_rel(handoff_path)

        if not handoff_path.exists():
            return default

        try:
            loaded = json.loads(handoff_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.warning("Failed to load previous Eureka handoff: %s", e)
            return default

        if not isinstance(loaded, dict):
            return default

        merged = dict(default)
        merged.update(
            {
                "exists": True,
                "source_round": loaded.get("round", prev_round),
                "best_equation": loaded.get("best_equation", default["best_equation"]),
                "best_mse": loaded.get("best_mse", default["best_mse"]),
                "mse_source": loaded.get("mse_source", default["mse_source"]),
                "notes": loaded.get("notes", default["notes"]),
                "next_round_plan": loaded.get("next_round_plan", default["next_round_plan"]),
                "update_best": loaded.get("update_best", default["update_best"]),
                "satisfied": loaded.get("satisfied", default["satisfied"]),
                "path": default["path"],
            }
        )
        return merged

    # =========================
    # Eureka outputs: validate + persist
    # =========================

    def _validate_eureka_round_output(self, signal: dict[str, Any]) -> dict[str, Any]:
        """软校验 Eureka 本轮输出，不抛异常阻断主流程。"""
        checks: dict[str, bool] = {}
        warnings: list[str] = []

        if not self.run_dir:
            return {
                "round": self.round_num,
                "validated_at": datetime.now().isoformat(),
                "is_valid": False,
                "checks": {"run_dir_available": False},
                "warnings": ["run_dir is not set; skip Eureka artifact validation."],
                "script_files": [],
            }

        insight_file = self.run_dir / "insight.md"
        scripts_dir = self.run_dir / "history" / f"round{self.round_num}" / "scripts"
        results_dir = self.run_dir / "history" / f"round{self.round_num}" / "results"

        checks["run_dir_available"] = True
        checks["insight_file_exists"] = insight_file.exists()
        checks["scripts_dir_exists"] = scripts_dir.exists()
        checks["results_dir_exists"] = results_dir.exists()
        checks["signal_is_dict"] = isinstance(signal, dict)
        checks["signal_has_satisfied"] = isinstance(signal, dict) and "satisfied" in signal
        checks["signal_has_next_round_plan"] = isinstance(signal, dict) and isinstance(signal.get("next_round_plan", []), list)

        insight_has_round_section = False
        if insight_file.exists():
            try:
                text = insight_file.read_text(encoding="utf-8")
                insight_has_round_section = f"## Round {self.round_num}" in text
            except Exception:
                insight_has_round_section = False
        checks["insight_has_round_section"] = insight_has_round_section

        script_files: list[str] = []
        has_python_script = False
        if scripts_dir.exists():
            try:
                script_files = sorted([p.name for p in scripts_dir.iterdir() if p.is_file()])
                has_python_script = any(name.endswith(".py") for name in script_files)
            except Exception:
                script_files = []
                has_python_script = False
        checks["has_python_script"] = has_python_script

        if not checks["insight_file_exists"]:
            warnings.append("insight.md is missing.")
        if not checks["insight_has_round_section"]:
            warnings.append(f"insight.md does not contain section '## Round {self.round_num}'.")
        if not checks["has_python_script"]:
            warnings.append(f"No Python script found in history/round{self.round_num}/scripts.")
        if not checks["signal_has_next_round_plan"]:
            warnings.append("Eureka signal missing next_round_plan list.")

        is_valid = checks["run_dir_available"] and checks["results_dir_exists"] and checks["signal_is_dict"]

        return {
            "round": self.round_num,
            "validated_at": datetime.now().isoformat(),
            "is_valid": is_valid,
            "checks": checks,
            "warnings": warnings,
            "script_files": script_files,
        }

    def _persist_eureka_artifacts(
        self,
        eureka_signal: dict[str, Any],
        eureka_message: str,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        """持久化 Eureka 结构化产物与 handoff 文件。"""
        info: dict[str, Any] = {
            "round": self.round_num,
            "persisted": False,
        }
        if not self.run_dir:
            return info

        results_dir = self.run_dir / "history" / f"round{self.round_num}" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        signal_path = results_dir / f"eureka_round{self.round_num}_signal.json"
        validation_path = results_dir / f"eureka_round{self.round_num}_validation.json"
        report_index_path = results_dir / f"eureka_round{self.round_num}_report_index.json"
        handoff_path = results_dir / self._EUREKA_HANDOFF_FILENAME

        signal_payload = {
            "round": self.round_num,
            "saved_at": datetime.now().isoformat(),
            "signal": eureka_signal if isinstance(eureka_signal, dict) else {},
            "finish_message_excerpt": (eureka_message or "")[:4000],
        }
        handoff_payload = self._build_eureka_handoff_payload(eureka_signal, validation)
        report_index = {
            "round": self.round_num,
            "saved_at": datetime.now().isoformat(),
            "insight_file": "insight.md",
            "scripts_dir": self._to_workspace_rel(self.run_dir / "history" / f"round{self.round_num}" / "scripts"),
            "results_dir": self._to_workspace_rel(results_dir),
            "signal_file": self._to_workspace_rel(signal_path),
            "validation_file": self._to_workspace_rel(validation_path),
            "handoff_file": self._to_workspace_rel(handoff_path),
            "script_files": validation.get("script_files", []) if isinstance(validation, dict) else [],
        }

        try:
            signal_path.write_text(json.dumps(signal_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
            handoff_path.write_text(json.dumps(handoff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            report_index_path.write_text(json.dumps(report_index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            self.logger.warning("Failed to persist Eureka artifacts for round %s: %s", self.round_num, e)
            info["error"] = str(e)
            return info

        info.update(
            {
                "persisted": True,
                "signal_file": self._to_workspace_rel(signal_path),
                "validation_file": self._to_workspace_rel(validation_path),
                "handoff_file": self._to_workspace_rel(handoff_path),
                "report_index_file": self._to_workspace_rel(report_index_path),
            }
        )
        return info

    def _build_eureka_handoff_payload(self, signal: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        """构建供下一轮 Hamilton 使用的 handoff 结构。"""
        raw = signal if isinstance(signal, dict) else {}

        best_equation = raw.get("best_equation", "")
        if not isinstance(best_equation, str) or not best_equation.strip():
            best_equation = "none"

        best_mse = raw.get("best_mse")
        try:
            best_mse = float(best_mse) if best_mse is not None else None
        except Exception:
            best_mse = None

        next_round_plan = raw.get("next_round_plan", [])
        if not isinstance(next_round_plan, list):
            next_round_plan = []
        next_round_plan = [x.strip() for x in next_round_plan if isinstance(x, str) and x.strip()]

        notes = raw.get("notes", "")
        if not isinstance(notes, str):
            notes = ""

        mse_source = raw.get("mse_source", "unknown")
        if not isinstance(mse_source, str) or not mse_source.strip():
            mse_source = "unknown"

        return {
            "round": self.round_num,
            "generated_at": datetime.now().isoformat(),
            "best_equation": best_equation.strip(),
            "best_mse": best_mse,
            "mse_source": mse_source.strip(),
            "notes": notes.strip(),
            "next_round_plan": next_round_plan,
            "update_best": bool(raw.get("update_best", False)),
            "satisfied": bool(raw.get("satisfied", False)),
            "validation": {
                "is_valid": bool(validation.get("is_valid", False)) if isinstance(validation, dict) else False,
                "warnings": validation.get("warnings", []) if isinstance(validation, dict) else [],
            },
        }

    def _to_workspace_rel(self, path: Path) -> str:
        """将路径转为相对 workspace 的可读路径。"""
        if not self.run_dir:
            return str(path)
        try:
            return str(path.relative_to(self.run_dir))
        except Exception:
            return str(path)

    def _ensure_round_dirs(self):
        """确保本轮目录存在"""
        if not self.run_dir:
            return

        round_dir = self.run_dir / "history" / f"round{self.round_num}"
        scripts_dir = round_dir / "scripts"
        results_dir = round_dir / "results"

        scripts_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    def _init_round_files(self, task_description: str = ""):
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
            experiment_data = {"task": task_description or "", "rounds": {}}

        existing_task = experiment_data.get("task")
        if (not isinstance(existing_task, str) or not existing_task.strip()) and task_description:
            experiment_data["task"] = task_description

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
    _EUREKA_HANDOFF_FILENAME = "eureka_handoff.json"

    def _parse_eureka_signal(self, eureka_message: str, trajectory) -> dict:
        """Parse machine-readable Eureka signal from finish.message.

        Preferred format is a JSON block:
          ===EVO_EUREKA_SIGNAL_BEGIN===
          {...}
          ===EVO_EUREKA_SIGNAL_END===
        """
        signal: dict[str, Any] = {}
        candidates: list[str] = []

        def _append_candidate(text: str) -> None:
            if isinstance(text, str) and text and text not in candidates:
                candidates.append(text)

        # 候选1：当前抽取到的文本
        _append_candidate(eureka_message or "")
        # 候选2：直接从 trajectory 提取 finish.message（兼容不同提取路径）
        _append_candidate(self._extract_finish_message_from_trajectory(trajectory))

        # 候选扩展：当文本本身是 {"message": "..."} 包装时，提取 message 再解析
        expanded_candidates: list[str] = []
        for candidate in candidates:
            _append = expanded_candidates.append
            if candidate not in expanded_candidates:
                _append(candidate)
            wrapped_message = self._extract_message_from_json_wrapper(candidate)
            if wrapped_message and wrapped_message not in expanded_candidates:
                _append(wrapped_message)

        for candidate in expanded_candidates:
            signal = self._extract_signal_from_text(candidate)
            if isinstance(signal, dict) and signal:
                break

        if not isinstance(signal, dict):
            signal = {}

        # Lenient fallback: accept a single-line SATISFIED flag even without JSON block.
        if "satisfied" not in signal:
            for candidate in expanded_candidates:
                satisfied = self._parse_satisfied_flag(candidate)
                if satisfied is not None:
                    signal["satisfied"] = satisfied
                    break

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
            return self._json_loads_dict_lenient(payload)
        except Exception:
            return {}

    def _extract_message_from_json_wrapper(self, text: str) -> str:
        """解析类似 {"message": "..."} 的包装文本，提取 message 字段。"""
        parsed = self._json_loads_lenient(text)
        if isinstance(parsed, dict):
            for key in ("message", "final", "answer", "output"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return ""
        if isinstance(parsed, str):
            return parsed
        return ""

    def _json_loads_lenient(self, text: str):
        if not isinstance(text, str):
            return None
        raw = text.strip()
        if not raw:
            return None

        candidates = [raw]
        # 兼容出现 \" 和 \n 的双重转义块
        if "\\\"" in raw or "\\n" in raw or "\\t" in raw:
            # 先做最小替换，避免 unicode_escape 破坏非 ASCII 文本
            replaced = raw.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"').strip()
            if replaced and replaced not in candidates:
                candidates.append(replaced)

            try:
                decoded = bytes(raw, "utf-8").decode("unicode_escape").strip()
                if decoded and decoded not in candidates:
                    candidates.append(decoded)
            except Exception:
                pass

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return None

    def _json_loads_dict_lenient(self, text: str) -> dict:
        parsed = self._json_loads_lenient(text)
        if isinstance(parsed, dict):
            return parsed
        # 少数情况下第一次解析得到字符串，第二次可得到 dict
        if isinstance(parsed, str):
            nested = self._json_loads_lenient(parsed)
            if isinstance(nested, dict):
                return nested
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
        # 兜底：若没有 finish tool-call，尝试提取最后 assistant 文本
        try:
            fallback = super()._extract_agent_response(trajectory)
            return fallback if isinstance(fallback, str) else ""
        except Exception:
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

        # 硬门槛：只有当轮存在有效 PySR 结果时，才允许更新 Current Best。
        round_ok, round_reason = self._has_effective_round_results(self.round_num)
        if not round_ok:
            self.logger.warning(
                "Skip Current Best update at round %s: %s",
                self.round_num,
                round_reason,
            )
            signal["update_best"] = False
            gate_note = (
                f"update_best rejected: no validated PySR results for round {self.round_num} "
                f"({round_reason})."
            )
            existing_notes = signal.get("notes", "")
            if isinstance(existing_notes, str) and existing_notes.strip():
                signal["notes"] = f"{existing_notes.strip()} | {gate_note}"
            else:
                signal["notes"] = gate_note
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

    def _has_effective_round_results(self, round_num: int) -> tuple[bool, str]:
        """检查给定轮次在 experiment.json 中是否存在有效 PySR 结果。"""
        if not self.run_dir:
            return False, "run_dir_not_set"

        experiment_file = self.run_dir / "experiment.json"
        if not experiment_file.exists():
            return False, "experiment_json_missing"

        try:
            payload = json.loads(experiment_file.read_text(encoding="utf-8"))
        except Exception:
            return False, "experiment_json_invalid"

        rounds = payload.get("rounds", {})
        if not isinstance(rounds, dict):
            return False, "rounds_not_dict"

        round_data = rounds.get(str(round_num))
        if not isinstance(round_data, dict):
            return False, "round_record_missing"

        try:
            raw_exit_code = round_data.get("exit_code", 1)
            if raw_exit_code is None:
                exit_code = 1
            else:
                exit_code = int(raw_exit_code)
        except Exception:
            exit_code = 1
        if exit_code != 0:
            return False, f"round_exit_code_{exit_code}"

        results = round_data.get("results", [])
        if not isinstance(results, list) or not results:
            return False, "round_results_empty"

        has_equation = any(
            isinstance(item, dict)
            and isinstance(item.get("equation"), str)
            and item.get("equation", "").strip()
            for item in results
        )
        if not has_equation:
            return False, "round_results_no_equation"

        return True, "ok"

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
