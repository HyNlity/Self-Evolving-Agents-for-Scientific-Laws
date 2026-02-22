"""Hamilton Round Exp - 单轮执行

负责单轮的数据分析和PySR调用。

每轮流程：
- HamiltonAgent: 读取 analysis.md, 写代码分析 + PySR, 维护 analysis.md
- Eureka Agent: 读取 analysis.md + PySR结果, 写入 insight.md

文件规范：
- analysis.md: HamiltonAgent 写入，格式 ## Round N: ...
- experiment.json: 系统自动记录 PySR 参数和结果
- insight.md: Eureka Agent 写入，格式 ## Round N: ...
"""

import json
import logging
import os
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

        # 提取 insight.md 内容
        insight_content = self._read_insight()

        self.logger.info(f"Round {self.round_num} completed")

        return {
            "round": self.round_num,
            "hamilton_result": hamilton_result,
            "eureka_result": eureka_result,
            "insight": insight_content,
            # Backward-compatible alias (older code may only expect one trajectory)
            "trajectory": hamilton_trajectory,
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
