"""Hamilton Playground 实现

符号回归Agent - 过完备变量下的方程发现

模式：
- RoundExp: 单轮执行单元（单 Agent 完成发现→验证→提炼闭环）
- Playground: 循环编排，多次调用RoundExp

HCC 分层记忆：
- L1 (history/round{N}/trace.md): 每轮独立的工作记忆
- L2 (plan.md, findings.md): 只增不减的知识积累
"""

import json
import logging
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime

# 确保可以导入evomaster模块
_module_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_module_root) not in sys.path:
    sys.path.insert(0, str(_module_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.core.exp import extract_finish_message
from .constants import (
    CURRENT_BEST_BEGIN, CURRENT_BEST_END,
    STRATEGY_QUEUE_BEGIN, STRATEGY_QUEUE_END,
    EXPERIENCE_POSITIVE_BEGIN, EXPERIENCE_POSITIVE_END,
    EXPERIENCE_NEGATIVE_BEGIN, EXPERIENCE_NEGATIVE_END,
    CRITIC_ROUND_INTERVAL, CRITIC_MSE_PLATEAU_THRESHOLD, CRITIC_MSE_PLATEAU_WINDOW,
    FINDINGS_APPEND,
)

from .exp import RoundExp, CriticExp


@register_playground("hamilton")
class HamiltonPlayground(BasePlayground):
    """Hamilton Playground - 符号回归Agent

    编排多轮迭代：
    1. 创建单个 Agent（发现 + 验证 + 提炼）
    2. 循环调用 RoundExp (每轮)
    3. 记录实验结果

    每轮流程（HCC）：
    - 系统: L1 trace.md 在每轮 round 目录创建
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

    def _init_workspace(self) -> None:
        """Initialize workspace with L2 persistent files and L3 experience.

        Creates: findings.md, plan.md, experience.md, lib/ (if not exist).
        Agent is responsible for creating any data directories it needs.
        """
        workspace = self.workspace_dir
        if not workspace:
            return

        workspace.mkdir(parents=True, exist_ok=True)

        # findings.md (L2 — knowledge accumulation, append-only)
        # Enhanced: round-by-round reporting with formula + physical interpretation
        findings_file = workspace / "findings.md"
        if not findings_file.exists():
            findings_file.write_text(
                "# 研究发现\n\n"
                "## 关键洞察\n"
                "（经验证的数据观察和物理关系）\n\n"
                "## 逐轮发现\n\n"
                "<!-- 每轮结束后追加一个小节，格式如下：\n"
                "### 第 N 轮\n"
                "**方程**: x'' = ...\n"
                "**各项物理解释**:\n"
                "- 项1: 物理含义（如恢复力、阻尼等）\n"
                "- 项2: 物理含义\n"
                "**评估指标**:\n"
                "| 风速 | 振幅误差 | MSE(训练) | MSE(OOD) | Z2S行为 | B2S行为 |\n"
                "|------|---------|-----------|----------|---------|--------|\n"
                "**结构一致性**: 5个风速是否共享相同函数形式\n"
                "**系数趋势**: 哪些系数随风速变化、哪些恒定\n"
                "**本轮结论**: 简要总结\n"
                "-->\n\n"
                f"{FINDINGS_APPEND}\n\n"
                "## 实验结果总表\n"
                "| 轮次 | 方程 | 平均振幅误差 | MSE(训练) | 结构一致性 | 关键改进 |\n"
                "|------|------|-------------|-----------|-----------|----------|\n\n"
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
            self._create_plan_file(plan_file)
            self.logger.info(f"Created {plan_file}")

        # experience.md (L3 — cross-task persistent experience)
        self._init_experience(workspace)

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
            self._init_workspace()

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

    def _create_plan_file(self, plan_file: Path):
        """创建 plan.md 研究计划文件"""
        plan_content = f"""# 研究计划

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

    def _init_experience(self, workspace: Path) -> None:
        """Initialize L3 experience.md — cross-task persistent experience.

        L3 lives in the project-level directory (not per-run workspace),
        so knowledge accumulates across different tasks and runs.
        Both Solver and Critic write here with [POSITIVE]/[NEGATIVE] tags.
        """
        # L3 lives at project level, not per-run
        l3_dir = self._project_root / "playground" / "hamilton" / "experience"
        l3_dir.mkdir(parents=True, exist_ok=True)

        experience_file = l3_dir / "experience.md"
        if not experience_file.exists():
            experience_file.write_text(
                "# L3 跨任务经验库\n\n"
                "此文件记录跨任务的长期经验，供后续任务参考。\n"
                "使用 [POSITIVE] 和 [NEGATIVE] 标签区分正面和负面经验。\n\n"
                f"{EXPERIENCE_POSITIVE_BEGIN}\n"
                "## 正面经验（有效策略）\n"
                "<!-- 格式：\n"
                "### [POSITIVE] 经验标题\n"
                "- **任务类型**: (如 VIV-ODE, 过完备变量回归, ...)\n"
                "- **策略**: 具体做法\n"
                "- **效果**: 带来的改进\n"
                "- **适用条件**: 什么时候这个策略有效\n"
                "-->\n"
                f"{EXPERIENCE_POSITIVE_END}\n\n"
                f"{EXPERIENCE_NEGATIVE_BEGIN}\n"
                "## 负面经验（应避免的做法）\n"
                "<!-- 格式：\n"
                "### [NEGATIVE] 经验标题\n"
                "- **任务类型**: (如 VIV-ODE, 过完备变量回归, ...)\n"
                "- **错误做法**: 具体描述\n"
                "- **后果**: 导致了什么问题\n"
                "- **正确做法**: 应该怎么做\n"
                "-->\n"
                f"{EXPERIENCE_NEGATIVE_END}\n",
                encoding="utf-8",
            )
            self.logger.info(f"Created L3 experience file: {experience_file}")

        # Symlink L3 into workspace so agent can access it
        workspace_link = workspace / "experience.md"
        if not workspace_link.exists():
            try:
                workspace_link.symlink_to(experience_file.resolve())
                self.logger.info(f"Linked L3 experience into workspace")
            except OSError:
                # Fallback: copy if symlink fails (e.g. cross-filesystem)
                shutil.copy2(experience_file, workspace_link)
                self.logger.info(f"Copied L3 experience into workspace (symlink failed)")

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
        """保存实验记录到 run_dir"""
        try:
            if self.run_dir:
                record_dir = Path(self.run_dir) / "records"
            else:
                record_dir = Path("./runs") / "hamilton" / "records"
            record_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_file = record_dir / f"experiment_{timestamp}.json"

            self.experiment_record["end_time"] = datetime.now().isoformat()

            with open(record_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_record, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Experiment record saved to {record_file}")

        except Exception as e:
            self.logger.error(f"Failed to save experiment record: {e}")


@register_playground("hamilton-gan")
class GANHamiltonPlayground(HamiltonPlayground):
    """GAN-inspired adversarial Hamilton Playground.

    Architecture: Solver (Generator) + Critic (Discriminator)
    - Solver: discovers equations (same as original Hamilton agent)
    - Critic: adversarial agent that actively designs intervention experiments
      to debunk pseudo-patterns (not just passive evaluation)

    Critic trigger conditions (lightweight execution):
    1. Every N rounds (configurable, default 5)
    2. MSE plateau detected (< threshold improvement over window)
    3. Solver claims task_completed="true" (mandatory final review)

    Both agents share HCC with positive/negative experience tagging in L3.
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.solver_agent = None
        self.critic_agent = None
        self._round_mse_history: list[float | None] = []
        self._last_critic_round = 0

    def setup(self) -> None:
        """Initialize solver and critic agents."""
        self.logger.info("Setting up GAN Hamilton playground...")
        super().setup()

        # Resolve agents from config
        solver = self.agents.get("hamilton_agent")
        critic = self.agents.get("critic_agent")

        if solver is None:
            raise ValueError("GAN Hamilton requires 'agents.hamilton' in config.yaml")
        if critic is None:
            raise ValueError("GAN Hamilton requires 'agents.critic' in config.yaml")

        self.solver_agent = solver
        self.critic_agent = critic
        # Default self.agent to solver for compatibility
        self.agent = self.solver_agent

        self.logger.info("GAN Hamilton playground setup complete (Solver + Critic)")

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """Run GAN-adversarial multi-round experiment.

        Flow per round:
        1. Solver runs (same as original Hamilton)
        2. Check critic trigger conditions
        3. If triggered: Critic runs intervention experiments
        4. Critic feedback is injected into next solver round
        """
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            self.experiment_record["task"] = task_description
            self.experiment_record["mode"] = "gan-adversarial"

            experiment_cfg = getattr(self.config, 'experiment', {})
            if not isinstance(experiment_cfg, dict):
                experiment_cfg = {}
            max_rounds = int(experiment_cfg.get('max_rounds', 10) or 10)

            # Critic trigger config
            critic_cfg = experiment_cfg.get('critic', {}) or {}
            critic_interval = int(critic_cfg.get('round_interval', CRITIC_ROUND_INTERVAL))
            critic_plateau_threshold = float(critic_cfg.get('mse_plateau_threshold', CRITIC_MSE_PLATEAU_THRESHOLD))
            critic_plateau_window = int(critic_cfg.get('mse_plateau_window', CRITIC_MSE_PLATEAU_WINDOW))

            self.logger.info(f"Starting GAN Hamilton: max_rounds={max_rounds}, "
                             f"critic_interval={critic_interval}")
            self.logger.info(f"Task: {task_description}")

            self._init_workspace()

            prev_critic_feedback = None

            for round_num in range(1, max_rounds + 1):
                self.logger.info("=" * 60)
                self.logger.info(f"Round {round_num}/{max_rounds} (Solver)")
                self.logger.info("=" * 60)

                # === Phase 1: Solver round ===
                solver_exp = RoundExp(
                    agent=self.solver_agent,
                    config=self.config,
                    round_num=round_num,
                )
                if self.workspace_dir:
                    solver_exp.set_run_dir(self.workspace_dir)

                # Inject critic feedback into task description
                task_with_feedback = task_description
                if prev_critic_feedback:
                    task_with_feedback += (
                        f"\n\n---\n## Critic 反馈（第 {self._last_critic_round} 轮审查）\n"
                        f"**请认真对待以下问题，在本轮解决或证伪：**\n\n"
                        f"{prev_critic_feedback}\n---\n"
                    )

                result = solver_exp.run(task_with_feedback)
                signal = result.get("signal") or {}

                # Track MSE for plateau detection
                mse = self._extract_mse_from_signal(signal)
                self._round_mse_history.append(mse)

                # Record solver round
                round_record = {
                    "round": round_num,
                    "phase": "solver",
                    "agent_result": result.get("agent_result", ""),
                    "findings": result.get("findings", ""),
                    "signal": signal,
                    "trajectory": self._summarize_trajectory(result.get("trajectory")),
                }
                self.experiment_record["rounds"].append(round_record)

                # === Phase 2: Check critic trigger ===
                solver_claims_done = signal.get("satisfied", False)
                should_trigger_critic = self._should_trigger_critic(
                    round_num=round_num,
                    solver_claims_done=solver_claims_done,
                    critic_interval=critic_interval,
                    plateau_threshold=critic_plateau_threshold,
                    plateau_window=critic_plateau_window,
                )

                if should_trigger_critic:
                    self.logger.info("=" * 60)
                    self.logger.info(f"Round {round_num} — CRITIC TRIGGERED")
                    self.logger.info("=" * 60)

                    # Extract solver's current best for critic input
                    solver_finish_msg = self._extract_finish_message(result)

                    critic_exp = CriticExp(
                        agent=self.critic_agent,
                        config=self.config,
                        round_num=round_num,
                    )
                    if self.workspace_dir:
                        critic_exp.set_run_dir(self.workspace_dir)

                    critic_result = critic_exp.run(
                        task_description=task_description,
                        solver_summary=solver_finish_msg,
                        round_num=round_num,
                    )

                    critic_verdict = critic_result.get("verdict", {})
                    self._last_critic_round = round_num

                    # Record critic round
                    critic_record = {
                        "round": round_num,
                        "phase": "critic",
                        "verdict": critic_verdict,
                        "trajectory": self._summarize_trajectory(critic_result.get("trajectory")),
                    }
                    self.experiment_record["rounds"].append(critic_record)

                    # If critic approves AND solver claims done → stop
                    if critic_verdict.get("approved", False) and solver_claims_done:
                        self.logger.info(f"Critic APPROVED in round {round_num}. Stopping.")
                        break

                    # If critic rejects → feed critique back to solver
                    if not critic_verdict.get("approved", False):
                        prev_critic_feedback = critic_verdict.get("critique", "")
                        self.logger.info(f"Critic REJECTED — feedback will be injected into next solver round.")
                    else:
                        # Critic approved but solver didn't claim done → continue without feedback
                        prev_critic_feedback = None
                        self.logger.info(f"Critic approved current progress but solver has more to explore.")

            self._save_experiment_record()
            return {
                "status": "completed",
                "total_rounds": len(self.experiment_record["rounds"]),
                "experiment_record": self.experiment_record,
            }

        except Exception as e:
            self.logger.error(f"GAN Hamilton experiment failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

        finally:
            self.cleanup()

    def _should_trigger_critic(
        self,
        round_num: int,
        solver_claims_done: bool,
        critic_interval: int,
        plateau_threshold: float,
        plateau_window: int,
    ) -> bool:
        """Determine if critic should be activated this round.

        Trigger conditions (OR logic):
        1. Solver claims task_completed="true" → mandatory final review
        2. Every critic_interval rounds (e.g. every 5 rounds)
        3. MSE plateau detected (< threshold improvement over window consecutive rounds)
        """
        # Condition 1: solver claims done → mandatory critic review
        if solver_claims_done:
            self.logger.info("Critic trigger: solver claims task_completed=true")
            return True

        # Condition 2: periodic interval
        rounds_since_critic = round_num - self._last_critic_round
        if rounds_since_critic >= critic_interval:
            self.logger.info(f"Critic trigger: periodic ({rounds_since_critic} rounds since last critic)")
            return True

        # Condition 3: MSE plateau
        if self._detect_mse_plateau(plateau_threshold, plateau_window):
            self.logger.info("Critic trigger: MSE plateau detected")
            return True

        return False

    def _detect_mse_plateau(self, threshold: float, window: int) -> bool:
        """Detect if MSE has plateaued (< threshold relative improvement over window rounds)."""
        history = [m for m in self._round_mse_history if m is not None]
        if len(history) < window + 1:
            return False

        recent = history[-window:]
        baseline = history[-(window + 1)]
        if baseline <= 0:
            return False

        # Check if all recent values show < threshold improvement relative to baseline
        for val in recent:
            relative_improvement = (baseline - val) / baseline
            if relative_improvement >= threshold:
                return False  # at least one round showed significant improvement

        return True

    def _extract_mse_from_signal(self, signal: dict) -> float | None:
        """Try to extract MSE from signal notes or findings."""
        try:
            notes = signal.get("notes", "")
            if not notes:
                return None
            # Try to parse MSE from finish message
            mse_match = re.search(r'MSE[:\s]*([0-9.eE+-]+)', notes, re.IGNORECASE)
            if mse_match:
                return float(mse_match.group(1))
        except (ValueError, TypeError):
            pass
        return None

    def _extract_finish_message(self, result: dict) -> str:
        """Extract the solver's finish message for critic input."""
        trajectory = result.get("trajectory")
        if trajectory is None:
            return result.get("agent_result", "")

        msg = extract_finish_message(trajectory)
        return msg if msg else result.get("agent_result", "")
