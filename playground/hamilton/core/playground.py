"""Hamilton Playground 实现

符号回归Agent - 过完备变量下的方程发现

模式：
- RoundExp: 单轮执行单元
- Playground: 循环编排，多次调用RoundExp
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 确保可以导入evomaster模块
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.agent.tools import ToolRegistry
from evomaster.agent.tools.base import BaseTool
from evomaster.agent.tools.builtin import BashTool, EditorTool, ThinkTool, FinishTool
from evomaster.skills import SkillRegistry
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from .exp import RoundExp


def create_hamilton_registry(
    skill_registry: SkillRegistry | None = None,
    exclude_pysr_tool: bool = False,
) -> ToolRegistry:
    """创建Hamilton专用的工具注册表"""
    registry = ToolRegistry()

    # 内置工具
    tools = [
        BashTool(),
        EditorTool(),
        ThinkTool(),
        FinishTool(),
    ]

    # SkillTool - Operator Skills（可选）
    if skill_registry is not None:
        from evomaster.agent.tools.skill import SkillTool
        tools.append(SkillTool(skill_registry))

    # PySRTool - Hamilton专用
    try:
        from ..tools.pysr_tool import PySRTool
        if not exclude_pysr_tool:
            tools.append(PySRTool())
    except ImportError as e:
        logging.warning(f"PySRTool not loaded: {e}")

    # 注册所有工具
    for tool in tools:
        registry.register(tool)

    return registry


@register_playground("hamilton")
class HamiltonPlayground(BasePlayground):
    """Hamilton Playground - 符号回归Agent

    编排多轮迭代：
    1. 创建 HamiltonAgent 和 Eureka Agent
    2. 循环调用 RoundExp (每轮)
    3. 记录实验结果

    每轮流程：
    - HamiltonAgent: 读取 analysis.md, 写代码分析 + PySR, 维护 analysis.md
    - Eureka Agent: 读取 analysis.md + PySR结果, 写入 insight.md

    使用方式：
        python run.py --agent hamilton --task "发现数据中的方程"
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """初始化 Hamilton Playground"""
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "hamilton"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Agents
        self.hamilton_agent = None
        self.eureka_agent = None
        self.skill_registry: SkillRegistry | None = None
        self.workspace_dir: Path | None = None

        # 实验记录
        self.experiment_record = {
            "task": "",
            "rounds": [],
            "start_time": datetime.now().isoformat(),
        }

        self.mcp_manager = None

    def set_run_dir(self, run_dir: str | Path, task_id: str | None = None) -> None:
        """设置 run 目录并为 Hamilton 初始化 workspace 模板。

        Hamilton 的 prompts 依赖 workspace 内存在一些“模板文件/目录”（如 tools/），同时数据集通常以
        `data.csv` 的形式放在 workspace 根目录（可选还有 `data_ood.csv`）。`run.py` 会为每次任务创建全新 workspace，因此这里需要：
        - 将 `playground/hamilton/workspace/` 中的模板内容拷贝到新 workspace（不覆盖已存在文件）
        - 若模板 workspace 中存在 `data.csv` / `data_ood.csv`，则一并拷贝（便于用户在模板目录放置固定数据集）
        """
        super().set_run_dir(run_dir, task_id=task_id)

        run_dir_path = Path(run_dir)
        workspace_path = run_dir_path / "workspace"
        if task_id:
            workspace_path = run_dir_path / "workspaces" / task_id

        self._seed_workspace(workspace_path)

    def _seed_workspace(self, workspace_path: Path) -> None:
        """将 Hamilton workspace 模板内容拷贝到目标 workspace（只在目标缺失时拷贝）。"""
        try:
            template_dir = project_root / "playground" / "hamilton" / "workspace"
            if not template_dir.exists():
                return

            workspace_path.mkdir(parents=True, exist_ok=True)

            # 1) tools/ 目录（Eureka 提示词中的可复用函数库入口）
            src_tools = template_dir / "tools"
            dst_tools = workspace_path / "tools"
            if src_tools.exists() and src_tools.is_dir() and not dst_tools.exists():
                shutil.copytree(src_tools, dst_tools)
                self.logger.info(f"Seeded tools/ into workspace: {dst_tools}")

            # 2) 数据文件（可选：允许用户把固定数据集放在模板目录以便自动带入每次 run）
            src_data = template_dir / "data.csv"
            dst_data = workspace_path / "data.csv"
            if src_data.exists() and src_data.is_file() and not dst_data.exists():
                shutil.copy2(src_data, dst_data)
                self.logger.info(f"Seeded data.csv into workspace: {dst_data}")

            src_data_ood = template_dir / "data_ood.csv"
            dst_data_ood = workspace_path / "data_ood.csv"
            if src_data_ood.exists() and src_data_ood.is_file() and not dst_data_ood.exists():
                shutil.copy2(src_data_ood, dst_data_ood)
                self.logger.info(f"Seeded data_ood.csv into workspace: {dst_data_ood}")

        except Exception as e:
            # seed 失败不应阻断运行（但会导致后续缺文件时显式报错）
            self.logger.warning(f"Failed to seed Hamilton workspace template: {e}", exc_info=True)

    def setup(self) -> None:
        """初始化组件"""
        self.logger.info("Setting up Hamilton playground...")

        # 1. LLM配置
        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict

        # 2. Session
        self._setup_session()
        self.workspace_dir = Path(self.session.config.workspace_path)

        # 3. Skills（可选）
        self.skill_registry = None
        config_dict = self.config.model_dump()
        skills_config = config_dict.get("skills", {})
        if skills_config.get("enabled", False):
            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
            self.skill_registry = SkillRegistry(skills_root)
            self.logger.info(f"Loaded {len(self.skill_registry.get_all_skills())} skills")

        # 4. Tools - 为不同 Agent 准备专用工具注册表
        self.hamilton_tools = create_hamilton_registry(skill_registry=self.skill_registry)
        self.eureka_tools = create_hamilton_registry(skill_registry=self.skill_registry, exclude_pysr_tool=True)
        self.tools = self.hamilton_tools
        self.logger.info(f"Loaded Hamilton tools: {self.hamilton_tools.get_tool_names()}")
        self.logger.info(f"Loaded Eureka tools: {self.eureka_tools.get_tool_names()}")

        # 5. 创建 HamiltonAgent 和 Eureka Agent
        agents_config = getattr(self.config, 'agents', {})
        if not agents_config:
            raise ValueError("No agents configuration found")

        def _get_agent_llm_config(agent_cfg: dict) -> dict:
            """为单个 agent 选择 LLM 配置（支持 per-agent 覆盖）。"""
            llm_name = agent_cfg.get("llm")
            if llm_name:
                return self.config_manager.get_llm_config(llm_name)
            return llm_config_dict

        # HamiltonAgent - 主分析Agent
        if 'hamilton' not in agents_config:
            raise ValueError("No 'hamilton' agent configuration found")

        hamilton_config = agents_config['hamilton']
        hamilton_llm_config = _get_agent_llm_config(hamilton_config)
        self.hamilton_agent = self._create_agent(
            name="hamilton",
            agent_config=hamilton_config,
            enable_tools=hamilton_config.get('enable_tools', True),
            llm_config_dict=hamilton_llm_config,
            tools=self.hamilton_tools,
            skill_registry=self.skill_registry,
        )
        self.logger.info("Hamilton Agent created")

        # Eureka Agent - 结果分析Agent
        if 'eureka' not in agents_config:
            raise ValueError("No 'eureka' agent configuration found")

        eureka_config = agents_config['eureka']
        eureka_llm_config = _get_agent_llm_config(eureka_config)
        self.eureka_agent = self._create_agent(
            name="eureka",
            agent_config=eureka_config,
            enable_tools=eureka_config.get('enable_tools', True),
            llm_config_dict=eureka_llm_config,
            tools=self.eureka_tools,
            skill_registry=self.skill_registry,
        )
        self.logger.info("Eureka Agent created")

        self.logger.info("Hamilton playground setup complete")

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """运行多轮实验

        Args:
            task_description: 任务描述
            output_file: 结果保存文件

        Returns:
            运行结果
        """
        try:
            self.setup()

            # 设置轨迹文件
            self._setup_trajectory_file(output_file)

            # 更新实验记录
            self.experiment_record["task"] = task_description

            # 获取最大轮数
            experiment_cfg = getattr(self.config, 'experiment', {})
            if not isinstance(experiment_cfg, dict):
                experiment_cfg = {}
            max_rounds = int(experiment_cfg.get('max_rounds', 5) or 5)

            self.logger.info(f"Starting Hamilton experiment with {max_rounds} max rounds")
            self.logger.info(f"Task: {task_description}")

            # 初始化workspace文件
            self._init_workspace_files(task_description)

            # 循环执行多轮
            for round_num in range(1, max_rounds + 1):
                self.logger.info("=" * 60)
                self.logger.info(f"Round {round_num}/{max_rounds}")
                self.logger.info("=" * 60)

                # 创建单轮exp
                exp = RoundExp(
                    hamilton_agent=self.hamilton_agent,
                    eureka_agent=self.eureka_agent,
                    config=self.config,
                    round_num=round_num,
                )
                if self.workspace_dir:
                    exp.set_run_dir(self.workspace_dir)

                # 执行单轮
                result = exp.run(task_description)

                # 记录结果（确保可 JSON 序列化；完整轨迹已由 trajectories/trajectory.json 持久化）
                round_record = {
                    "round": result.get("round", round_num),
                    "hamilton_result": result.get("hamilton_result", ""),
                    "eureka_result": result.get("eureka_result", ""),
                    "insight": result.get("insight", ""),
                    "hamilton": self._summarize_trajectory(result.get("hamilton_trajectory")),
                    "eureka": self._summarize_trajectory(result.get("eureka_trajectory")),
                }
                self.experiment_record["rounds"].append(round_record)

                # 检查是否完成
                satisfied_signal = "\n".join([
                    str(round_record.get("eureka_result", "")),
                    str(round_record.get("insight", "")),
                ])
                if self._is_satisfied(satisfied_signal):
                    self.logger.info("Found satisfactory result!")
                    break

            # 保存实验记录
            self._save_experiment_record()

            return {
                "status": "completed",
                "total_rounds": len(self.experiment_record["rounds"]),
                "experiment_record": self.experiment_record,
            }

        except Exception as e:
            self.logger.error(f"Hamilton experiment failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
            }

        finally:
            self.cleanup()

    def _init_workspace_files(self, task_description: str):
        """初始化workspace文件

        创建:
        - analysis.md: 分析历史（初始包含任务描述）
        - insight.md: 靠谱发现（初始为空）
        """
        workspace = self.workspace_dir
        if not workspace:
            return

        # data.csv - 必需输入
        data_file = workspace / "data.csv"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Missing required input dataset: {data_file}\n"
                "Hamilton expects a CSV named 'data.csv' in the workspace root.\n"
                "Tip: put your data.csv in 'playground/hamilton/workspace/data.csv' so it will be auto-seeded into each new run workspace."
            )

        # data_ood.csv - 可选输入
        data_ood_file = workspace / "data_ood.csv"
        if not data_ood_file.exists():
            self.logger.warning(
                "Optional OOD dataset not found: %s. "
                "PySR default tool won't break, but OOD validation scripts should guard for its absence.",
                data_ood_file
            )

        # analysis.md - 分析历史（初始写入任务描述）
        analysis_file = workspace / "analysis.md"
        if not analysis_file.exists():
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"# Data Analysis History\n\n## Task\n{task_description}\n\n")
            self.logger.info(f"Created {analysis_file}")

        # insight.md - 靠谱发现
        insight_file = workspace / "insight.md"
        if not insight_file.exists():
            with open(insight_file, 'w', encoding='utf-8') as f:
                f.write("# Insights\n\n")
            self.logger.info(f"Created {insight_file}")

    def _summarize_trajectory(self, trajectory) -> dict:
        """提取轨迹的轻量摘要（避免 experiment_record 保存巨大对象）。"""
        try:
            if trajectory is None:
                return {}
            status = getattr(trajectory, "status", None)
            steps = getattr(trajectory, "steps", None)
            steps_n = len(steps) if isinstance(steps, list) else None
            return {"status": status, "steps": steps_n}
        except Exception:
            return {}

    def _is_satisfied(self, result: str) -> bool:
        """判断是否找到满意结果"""
        keywords = ["完成", "成功", "satisfied", "success", "satisfactory", "completed"]
        return any(kw.lower() in result.lower() for kw in keywords)

    def _save_experiment_record(self):
        """保存实验记录"""
        try:
            experiment_cfg = getattr(self.config, 'experiment', {})
            if not isinstance(experiment_cfg, dict):
                experiment_cfg = {}
            record_dir = Path(experiment_cfg.get('record_dir', './playground/hamilton/records'))
            record_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_file = record_dir / f"experiment_{timestamp}.json"

            self.experiment_record["end_time"] = datetime.now().isoformat()

            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_record, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Experiment record saved to {record_file}")

        except Exception as e:
            self.logger.error(f"Failed to save experiment record: {e}")
