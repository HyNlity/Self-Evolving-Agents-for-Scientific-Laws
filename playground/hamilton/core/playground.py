"""Hamilton Playground 实现

符号回归Agent - 过完备变量下的方程发现

模式：
- RoundExp: 单轮执行单元（单 Agent 完成发现→验证→提炼闭环）
- Playground: 循环编排，多次调用RoundExp

HCC 分层记忆：
- L1 (execution_trace.md): 每轮重置的工作记忆
- L2 (plan.md, findings.md): 只增不减的知识积累
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 确保可以导入evomaster模块
_module_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_module_root) not in sys.path:
    sys.path.insert(0, str(_module_root))

from evomaster.core import BasePlayground, register_playground
from .constants import CURRENT_BEST_BEGIN, CURRENT_BEST_END, STRATEGY_QUEUE_BEGIN, STRATEGY_QUEUE_END

from .exp import RoundExp


@register_playground("hamilton")
class HamiltonPlayground(BasePlayground):
    """Hamilton Playground - 符号回归Agent

    编排多轮迭代：
    1. 创建单个 Agent（发现 + 验证 + 提炼）
    2. 循环调用 RoundExp (每轮)
    3. 记录实验结果

    每轮流程（HCC）：
    - 系统: 重置 execution_trace.md (L1)
    - Agent: 读 L2 → 发现方程 → 验证 → 提炼到 L2 → finish(satisfied)
    - 系统: 解析 signal，决定继续/停止

    使用方式：
        python run.py --agent hamilton --task "发现数据中的方程"
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """初始化 Hamilton Playground"""
        self._project_root = Path(__file__).resolve().parent.parent.parent.parent

        if config_path is None and config_dir is None:
            config_dir = self._project_root / "configs" / "hamilton"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Agents
        self.workspace_dir: Path | None = None

        # 实验记录
        self.experiment_record = {
            "task": "",
            "rounds": [],
            "start_time": datetime.now().isoformat(),
        }

    def set_run_dir(self, run_dir: str | Path, task_id: str | None = None) -> None:
        """设置 run 目录。

        Workspace seeding and file initialization are deferred to _init_workspace()
        which is called at run() time when the task description is available.
        """
        super().set_run_dir(run_dir, task_id=task_id)

    def _init_workspace(self, task_description: str) -> None:
        """Unified workspace initialization: seed template files + create runtime files.

        Steps:
        1. Copy tools/ template dir (if missing)
        2. Copy data.csv / data_ood.csv from template (if missing)
        3. Ensure skills/__init__.py for symlink compatibility
        4. Create execution_trace.md / findings.md / plan.md (if missing)
        5. Validate data.csv exists
        """
        workspace = self.workspace_dir
        if not workspace:
            return

        workspace.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: Seed from template ---
        template_dir = self._project_root / "playground" / "hamilton" / "workspace"
        if template_dir.exists():
            try:
                # task.md
                src_task = template_dir / "task.md"
                dst_task = workspace / "task.md"
                if src_task.exists() and not dst_task.exists():
                    shutil.copy2(src_task, dst_task)
                    self.logger.info("Seeded task.md into workspace")

                # data files → input/ subdirectory
                input_dir = workspace / "input"
                input_dir.mkdir(parents=True, exist_ok=True)
                src_input = template_dir / "input"
                csv_source = src_input if src_input.exists() else template_dir
                for src in csv_source.glob("*.csv"):
                    dst = input_dir / src.name
                    if not dst.exists():
                        if src.is_symlink():
                            link_target = os.readlink(src)
                            os.symlink(link_target, dst)
                        else:
                            shutil.copy2(src, dst)
                        self.logger.info(f"Seeded input/{src.name}")
            except Exception as e:
                self.logger.warning(f"Failed to seed Hamilton workspace template: {e}", exc_info=True)

        # --- Phase 3: Create runtime files ---
        # input/ must have at least one CSV
        input_dir = workspace / "input"
        csv_files = list(input_dir.glob("*.csv")) if input_dir.exists() else []
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV data files found in: {input_dir}\n"
                "Hamilton expects data CSVs in 'input/' subdirectory.\n"
                "Tip: put your CSVs in 'playground/hamilton/workspace/input/' so they will be auto-seeded."
            )

        # execution_trace.md (L1 — will be reset each round, create initial file)
        trace_file = workspace / "execution_trace.md"
        if not trace_file.exists():
            trace_file.write_text(
                "# 执行日志\n\n（每轮开始时自动重置）\n",
                encoding="utf-8",
            )
            self.logger.info(f"Created {trace_file}")

        # findings.md (L2 — knowledge accumulation, append-only)
        findings_file = workspace / "findings.md"
        if not findings_file.exists():
            findings_file.write_text(
                "# 研究发现\n\n"
                "## 关键洞察\n"
                "（经验证的数据观察和物理关系）\n\n"
                "## 实验结果\n"
                "| 轮次 | 方法 | 方程 | MSE (训练) | MSE (OOD) | 结论 |\n"
                "|------|------|------|-----------|-----------|------|\n\n"
                "## 最优方程演化\n"
                "（记录最优方程在各轮中的变化过程）\n",
                encoding="utf-8",
            )
            self.logger.info(f"Created {findings_file}")

        # lib/ (L2 — reusable scripts, persists across rounds)
        lib_dir = workspace / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        lib_readme = lib_dir / "README.md"
        if not lib_readme.exists():
            lib_readme.write_text("# lib/ 可复用脚本索引\n\n（每次新增脚本时更新）\n", encoding="utf-8")

        # plan.md (L2 — strategic plan with Current Best markers)
        plan_file = workspace / "plan.md"
        if not plan_file.exists():
            self._create_plan_file(plan_file, task_description)
            self.logger.info(f"Created {plan_file}")

    def setup(self) -> None:
        """初始化组件（复用 BasePlayground.setup）"""
        self.logger.info("Setting up Hamilton playground...")
        super().setup()

        if self.session is not None:
            try:
                self.workspace_dir = Path(self.session.config.workspace_path)
            except Exception:
                self.workspace_dir = None

        if self.agent is None:
            raise ValueError("Hamilton requires 'agents.hamilton' section in config.yaml")

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

            # 初始化workspace
            self._init_workspace(task_description)

            # 循环执行多轮
            for round_num in range(1, max_rounds + 1):
                self.logger.info("=" * 60)
                self.logger.info(f"Round {round_num}/{max_rounds}")
                self.logger.info("=" * 60)

                # 创建单轮exp
                exp = RoundExp(
                    agent=self.agent,
                    config=self.config,
                    round_num=round_num,
                )
                if self.workspace_dir:
                    exp.set_run_dir(self.workspace_dir)

                # 执行单轮
                result = exp.run(task_description)
                signal = result.get("signal") or {}

                # 记录结果（确保可 JSON 序列化；完整轨迹已由 trajectories/trajectory.json 持久化）
                round_record = {
                    "round": result.get("round", round_num),
                    "agent_result": result.get("agent_result", ""),
                    "findings": result.get("findings", ""),
                    "signal": signal,
                    "trajectory": self._summarize_trajectory(result.get("trajectory")),
                }
                self.experiment_record["rounds"].append(round_record)

                # 检查是否完成
                if self._is_satisfied(signal):
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

    def _create_plan_file(self, plan_file: Path, task_description: str):
        """创建 plan.md 研究计划文件

        优先使用 evo-protocol skill 的模板；若不可用则使用内置模板。
        """
        # Try to load template from evo-protocol skill
        template_path = self._project_root / "evomaster" / "skills" / "evo-protocol" / "references" / "plan_template.md"
        if template_path.exists():
            try:
                template = template_path.read_text(encoding="utf-8")
                plan_content = template.replace("{task_description}", task_description)
                plan_file.write_text(plan_content, encoding="utf-8")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load plan template from evo-protocol skill: {e}")

        # Fallback: inline template (includes Current Best markers)
        plan_content = f"""# 研究计划

## 任务
{task_description}

{CURRENT_BEST_BEGIN}
## 当前最优
- 轮次：0
- 方程：无
- MSE：未知
- 更新时间：{datetime.now().isoformat()}
{CURRENT_BEST_END}

## 数据概览
（首轮 EDA 后填写：变量列表、基本统计、初步观察）

## 当前假设
1. 待定

## 已确认知识
- 相关变量：待定
- 排除变量：待定
- 已发现的关键关系：无

## 策略队列
{STRATEGY_QUEUE_BEGIN}
（Agent 自行制定）
{STRATEGY_QUEUE_END}

## 失败方法
| 轮次 | 策略 | 变量 | 模板/参数 | MSE | 失败原因 |
|------|------|------|-----------|-----|----------|
"""
        plan_file.write_text(plan_content, encoding="utf-8")

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

    def _is_satisfied(self, signal) -> bool:
        """判断是否找到满意结果（只接受结构化信号，避免关键字误触发）"""
        try:
            if isinstance(signal, dict):
                return bool(signal.get("satisfied", False))
        except Exception:
            pass
        return False

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
