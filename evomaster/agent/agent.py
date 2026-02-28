"""EvoMaster Agent 基础实现

提供 Agent 的基础抽象，支持工具调用、对话管理、轨迹记录等功能。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .context import ContextConfig, ContextManager
from evomaster.utils.types import (
    AssistantMessage,
    Dialog,
    StepRecord,
    SystemMessage,
    TaskInstance,
    ToolMessage,
    UserMessage,
)

if TYPE_CHECKING:
    from evomaster.utils import BaseLLM
    from .session import BaseSession
    from .tools import ToolRegistry
    from evomaster.skills import SkillRegistry


class AgentConfig(BaseModel):
    """Agent 配置"""
    max_turns: int = Field(default=100, description="最大执行轮数")
    max_consecutive_no_tool_calls: int = Field(
        default=0,
        description="启用工具时，连续无工具调用的最大步数；超过后提前失败（<=0 表示禁用）",
    )
    context_config: ContextConfig = Field(
        default_factory=ContextConfig,
        description="上下文管理配置"
    )


class BaseAgent(ABC):
    """Agent 基类

    提供 Agent 的基础功能：
    - 对话管理（Dialog）
    - 轨迹记录（Trajectory）
    - 工具调用执行
    - 上下文管理

    子类需要实现：
    - _get_system_prompt(): 获取系统提示词
    - _get_user_prompt(task): 获取用户提示词
    """

    VERSION: str = "1.0"
    
    # 类级别的轨迹文件路径和锁（所有agent实例共享）
    _trajectory_file_path: Path | None = None
    _trajectory_file_lock = threading.Lock()

    # Current exp info is stored thread-locally so parallel workloads do not overwrite each other.
    _exp_info_local = threading.local()

    def __init__(
        self,
        llm: BaseLLM,
        session: BaseSession,
        tools: ToolRegistry,
        config: AgentConfig | None = None,
        skill_registry: SkillRegistry | None = None,
        output_config: dict[str, Any] | None = None,
        config_dir: Path | str | None = None,
        enable_tools: bool = True,
    ):
        """初始化 Agent

        Args:
            llm: LLM 实例
            session: 环境会话，用于执行工具
            tools: 工具注册中心（始终注册，但只有在 enable_tools=True 时才会在提示词中包含工具信息）
            config: Agent 配置
            skill_registry: Skills 注册中心（可选）
            output_config: 输出显示配置
            config_dir: 配置目录路径，用于加载提示词文件
            enable_tools: 是否在提示词中包含工具信息（默认 True）。如果为 False，工具仍然注册但不会出现在提示词中
        """
        self.llm = llm
        self.session = session
        self.tools = tools
        self.config = config or AgentConfig()
        self.skill_registry = skill_registry
        self.enable_tools = enable_tools

        # 输出配置
        self.output_config = output_config or {}
        self.show_in_console = self.output_config.get("show_in_console", False)
        self.log_to_file = self.output_config.get("log_to_file", False)

        # 配置目录（用于加载提示词文件）
        self.config_dir = Path(config_dir) if config_dir else None

        # 上下文管理器
        self.context_manager = ContextManager(self.config.context_config)

        # 当前对话
        self.current_dialog: Dialog | None = None

        # 执行轨迹
        self.trajectory = None

        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)

        # 当前步骤计数
        self._step_count = 0
        self._consecutive_no_tool_calls = 0

        # 存储初始系统提示词和用户提示词（用于重置）
        self._initial_system_prompt: str | None = None
        self._initial_user_prompt: str | None = None
        
        # Agent名称（用于标识不同的agent）
        self._agent_name: str | None = None

    def run(self, task: TaskInstance):
        """执行任务

        Args:
            task: 任务实例

        Returns:
            执行轨迹
        """
        from evomaster.utils.types import Trajectory

        self.logger.info(f"Starting task: {task.task_id}")

        # 初始化
        self._initialize(task)

        try:
            # 执行循环
            for turn in range(self.config.max_turns):
                # 清晰显示当前步骤
                self.logger.info("=" * 80)
                self.logger.info(f"📍 Step [{turn + 1}/{self.config.max_turns}]")
                self.logger.info("=" * 80)

                should_finish = self._step()
                if should_finish:
                    self.logger.info("=" * 80)
                    self.logger.info("✅ Agent finished task")
                    self.logger.info("=" * 80)
                    self.trajectory.finish("completed")
                    break
            else:
                self.logger.warning("=" * 80)
                self.logger.warning("⚠️  Reached max turns limit")
                self.logger.warning("=" * 80)
                self.trajectory.finish("failed", {"reason": "max_turns_exceeded"})

        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"❌ Agent execution failed: {e}")
            self.logger.error("=" * 80)
            self.trajectory.finish("failed", {"reason": str(e)})
            raise

        return self.trajectory

    def _initialize(self, task: TaskInstance) -> None:
        """初始化执行环境

        Args:
            task: 任务实例
        """
        from evomaster.utils.types import Trajectory

        # 创建轨迹
        self.trajectory = Trajectory(
            task_id=task.task_id,
            meta={
                "agent_version": self.VERSION,
                "task_type": task.task_type,
            }
        )

        # 获取初始提示词
        system_prompt = self._get_system_prompt()
        user_prompt = self._get_user_prompt(task)

        # 保存初始提示词（用于重置）
        self._initial_system_prompt = system_prompt
        self._initial_user_prompt = user_prompt

        # 创建对话
        self.current_dialog = Dialog(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            tools=self._get_tool_specs(),
        )

        self.trajectory.dialogs.append(self.current_dialog)
        self._step_count = 0
        self._consecutive_no_tool_calls = 0

    def _step(self) -> bool:
        """执行一步

        Returns:
            是否应该结束（True 表示结束）
        """
        self._step_count += 1

        # 准备对话（可能需要截断）
        dialog_for_query = self.context_manager.prepare_for_query(self.current_dialog)

        # 查询模型（使用 LLM）
        assistant_message = self.llm.query(dialog_for_query)

        self.current_dialog.add_message(assistant_message)

        # 创建步骤记录
        step_record = StepRecord(
            step_id=self._step_count,
            assistant_message=assistant_message,
        )

        # 如果没有工具调用
        if not assistant_message.tool_calls:
            # 检查Agent是否启用了工具调用
            # 如果没有启用工具（enable_tools=False），则直接结束
            # 因为这种Agent只需要给出回答，不需要工具调用
            if hasattr(self, 'enable_tools') and not self.enable_tools:
                self.trajectory.add_step(step_record)
                # 追加保存本次step到轨迹文件（包含tool_responses）
                self._append_trajectory_entry(dialog_for_query, step_record)
                return True  # 直接结束

            self._consecutive_no_tool_calls += 1
            if self._should_fail_fast_no_tool_call():
                preview = (assistant_message.content or "").replace("\n", "\\n")
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                self.logger.warning(
                    "⚠️  Consecutive no-tool-call limit reached (%s). "
                    "Fail fast to avoid max_turns loop. assistant_preview=%s",
                    self._consecutive_no_tool_calls,
                    preview,
                )
                self.trajectory.add_step(step_record)
                self._append_trajectory_entry(dialog_for_query, step_record)
                raise RuntimeError(
                    f"consecutive_no_tool_calls_exceeded:{self._consecutive_no_tool_calls}"
                )

            # 如果启用了工具但没有工具调用，提示继续
            self._handle_no_tool_call()
            self.trajectory.add_step(step_record)
            # 追加保存本次step到轨迹文件（包含tool_responses）
            self._append_trajectory_entry(dialog_for_query, step_record)
            return False

        self._consecutive_no_tool_calls = 0

        # 处理工具调用
        should_finish = False
        for tool_call in assistant_message.tool_calls:
            self.logger.debug(f"Processing tool call: {tool_call.function.name}")

            # 检查是否是 finish 工具
            if tool_call.function.name == "finish":
                # 打印 finish 工具的参数（最终答案）
                try:
                    import json
                    finish_args = json.loads(tool_call.function.arguments)
                    self.logger.info("=" * 80)
                    self.logger.info("📝 Finish Tool Arguments:")
                    for key, value in finish_args.items():
                        # 截断过长的值用于显示
                        value_str = str(value)
                        if len(value_str) > 2000:
                            value_str = value_str[:1000] + "\n... [truncated] ...\n" + value_str[-1000:]
                        self.logger.info(f"  {key}: {value_str}")
                    self.logger.info("=" * 80)
                except Exception as e:
                    self.logger.info(f"📝 Finish Tool Raw Args: {tool_call.function.arguments}")
                should_finish = True
                break

            # 执行工具
            observation, info = self._execute_tool(tool_call)

            # 截断过长的工具输出，防止 context 溢出
            MAX_TOOL_OUTPUT = 30000
            if len(observation) > MAX_TOOL_OUTPUT:
                observation = (
                    observation[:MAX_TOOL_OUTPUT // 2]
                    + "\n\n... [output truncated due to length] ...\n\n"
                    + observation[-MAX_TOOL_OUTPUT // 2:]
                )

            # 创建工具响应消息
            tool_message = ToolMessage(
                content=observation,
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                meta={"info": info}
            )

            self.current_dialog.add_message(tool_message)
            step_record.tool_responses.append(tool_message)

        self.trajectory.add_step(step_record)
        # 追加保存本次step到轨迹文件（包含tool_responses）
        self._append_trajectory_entry(dialog_for_query, step_record)
        return should_finish

    def _execute_tool(self, tool_call) -> tuple[str, dict[str, Any]]:
        """执行工具调用

        Args:
            tool_call: 工具调用

        Returns:
            (observation, info) 元组
        """
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # 记录工具调用开始
        self._log_tool_start(tool_name, tool_args)

        # 获取工具并执行
        tool = self.tools.get_tool(tool_name)
        if tool is None:
            error_msg = f"Unknown tool: {tool_name}"
            self._log_tool_end(tool_name, error_msg, {"error": "tool_not_found"})
            return error_msg, {"error": "tool_not_found"}

        try:
            # 执行工具
            observation, info = tool.execute(self.session, tool_args)
            
            # 记录工具调用结束
            self._log_tool_end(tool_name, observation, info)
            
            return observation, info
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            self.logger.error(f"Tool execution failed: {e}", exc_info=True)
            self._log_tool_end(tool_name, error_msg, {"error": str(e)})
            return error_msg, {"error": str(e)}

    def _log_tool_start(self, tool_name: str, tool_args: str) -> None:
        """记录工具调用开始"""
        if self.log_to_file:
            self.logger.info("=" * 80)
            self.logger.info(f"Tool Call Start: {tool_name}")
            self.logger.info(f"Arguments: {tool_args}")
            self.logger.info("=" * 80)
        
        if self.show_in_console:
            print(f"\n[Tool Call] {tool_name}")
            if tool_args:
                # 尝试格式化JSON参数
                try:
                    import json
                    args_dict = json.loads(tool_args)
                    print(f"  Arguments: {json.dumps(args_dict, indent=2, ensure_ascii=False)}")
                except:
                    print(f"  Arguments: {tool_args}")
            print("-" * 60)

    def _log_tool_end(self, tool_name: str, observation: str, info: dict[str, Any]) -> None:
        """记录工具调用结束"""
        # 截断过长的输出：超过5000字符时，保留前2500和最后2500
        obs_display = observation
        if len(obs_display) > 5000:
            obs_display = obs_display[:2500] + "\n... [truncated] ...\n" + obs_display[-2500:]
        
        if self.log_to_file:
            self.logger.info("=" * 80)
            self.logger.info(f"Tool Call End: {tool_name}")
            self.logger.info(f"Output: {obs_display}")
            if info:
                self.logger.info(f"Info: {info}")
            self.logger.info("=" * 80)
        
        if self.show_in_console:
            print(f"\n[Tool Output] {tool_name}")
            print("-" * 60)
            print(obs_display)
            print("-" * 60)

    def _handle_no_tool_call(self) -> None:
        """处理没有工具调用的情况"""
        # 添加用户消息提示继续
        prompt = (
            "Please continue working on the task.\n"
            "When you have completed the task, use the finish tool.\n"
            "IMPORTANT: You should not ask for human help."
        )
        self.current_dialog.add_message(UserMessage(content=prompt))

    def _should_fail_fast_no_tool_call(self) -> bool:
        """是否触发“连续无工具调用”提前失败保护。"""
        if not self.enable_tools:
            return False
        try:
            limit = int(getattr(self.config, "max_consecutive_no_tool_calls", 0) or 0)
        except Exception:
            limit = 0
        if limit <= 0:
            return False
        return self._consecutive_no_tool_calls >= limit


    def _get_tool_specs(self) -> list:
        """获取工具规格列表
        
        只有在 enable_tools=True 时才返回工具规格列表。
        如果 enable_tools=False，返回空列表（工具仍然注册，但不会出现在提示词中）。
        """
        if not self.enable_tools:
            return []
        if self.tools is None:
            return []
        return self.tools.get_tool_specs()

    def load_prompt_from_file(
        self,
        prompt_file: str | Path,
        format_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """从文件加载提示词

        支持相对路径（相对于config_dir）和绝对路径。
        支持使用format_kwargs进行字符串格式化（{}占位符）。

        Args:
            prompt_file: 提示词文件路径（相对或绝对）
            format_kwargs: 用于格式化提示词的参数字典（可选）

        Returns:
            提示词内容（已格式化）

        Examples:
            >>> agent.load_prompt_from_file("prompts/system_prompt.txt")
            >>> agent.load_prompt_from_file("prompts/user_prompt.txt", {"task": "完成代码任务"})
        """
        # 解析文件路径
        prompt_path = Path(prompt_file)
        if not prompt_path.is_absolute():
            if self.config_dir is None:
                raise ValueError(
                    "config_dir not set. Cannot resolve relative path. "
                    "Please provide config_dir in __init__ or use absolute path."
                )
            prompt_path = self.config_dir / prompt_file

        # 读取文件内容
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Please create the file or check the path."
            )

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()

            # 如果提供了format_kwargs，进行格式化
            if format_kwargs:
                try:
                    prompt_content = prompt_content.format(**format_kwargs)
                except KeyError as e:
                    self.logger.warning(
                        f"Format key {e} not found in format_kwargs. "
                        f"Available keys: {list(format_kwargs.keys())}"
                    )
                    raise

            self.logger.debug(f"Loaded prompt from: {prompt_path}")
            return prompt_content
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt from {prompt_path}: {e}")

    def reset_context(self) -> None:
        """重置Agent的上下文到初始状态

        将对话重置为只包含初始的系统提示词和用户提示词。
        需要先调用initialize或手动设置_initial_system_prompt和_initial_user_prompt。
        """
        if self._initial_system_prompt is None:
            raise ValueError(
                "Cannot reset context: initial prompts not set. "
                "Please initialize the agent first or set _initial_system_prompt manually."
            )

        # 重新创建对话
        messages = [SystemMessage(content=self._initial_system_prompt)]
        if self._initial_user_prompt:
            messages.append(UserMessage(content=self._initial_user_prompt))

        self.current_dialog = Dialog(
            messages=messages,
            tools=self._get_tool_specs(),
        )

        # 重置步骤计数
        self._step_count = 0

        self.logger.info("Context reset to initial state")

    def add_user_message(self, content: str) -> None:
        """添加用户消息到当前对话

        Args:
            content: 用户消息内容
        """
        if self.current_dialog is None:
            raise ValueError(
                "No active dialog. Please initialize the agent first."
            )

        user_message = UserMessage(content=content)
        self.current_dialog.add_message(user_message)
        self.logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(self, content: str, tool_calls: list | None = None) -> None:
        """添加助手消息到当前对话

        Args:
            content: 助手消息内容
            tool_calls: 工具调用列表（可选）
        """
        if self.current_dialog is None:
            raise ValueError(
                "No active dialog. Please initialize the agent first."
            )

        assistant_message = AssistantMessage(content=content, tool_calls=tool_calls or [])
        self.current_dialog.add_message(assistant_message)
        content_preview = content[:50] if content else "(empty)"
        self.logger.debug(f"Added assistant message: {content_preview}...")

    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        name: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """添加工具消息到当前对话

        Args:
            content: 工具执行结果
            tool_call_id: 工具调用ID
            name: 工具名称
            meta: 元数据（可选）
        """
        if self.current_dialog is None:
            raise ValueError(
                "No active dialog. Please initialize the agent first."
            )

        tool_message = ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            meta=meta or {},
        )
        self.current_dialog.add_message(tool_message)
        self.logger.debug(f"Added tool message: {name}")

    def set_next_user_request(self, content: str) -> None:
        """设置下一次对话的用户请求

        这会添加一条用户消息到当前对话。

        Args:
            content: 用户请求内容
        """
        self.add_user_message(content)

    def get_current_dialog(self) -> Dialog | None:
        """获取当前对话

        Returns:
            当前对话对象，如果未初始化则返回None
        """
        return self.current_dialog

    def get_conversation_history(self) -> list:
        """获取对话历史

        Returns:
            消息列表
        """
        if self.current_dialog is None:
            return []
        return self.current_dialog.messages.copy()
    
    @classmethod
    def set_trajectory_file_path(cls, trajectory_file_path: str | Path) -> None:
        """设置轨迹文件路径（类级别，所有agent实例共享）

        Args:
            trajectory_file_path: 轨迹文件路径
        """
        cls._trajectory_file_path = Path(trajectory_file_path)
        # 确保目录存在
        cls._trajectory_file_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def set_exp_info(cls, exp_name: str, exp_index: int) -> None:
        """设置当前exp信息（线程本地）

        在exp运行时调用，用于记录当前step属于哪个exp阶段和迭代。

        Args:
            exp_name: exp阶段名称（如 "Solver", "Critic", "Rewriter", "Selector"）
            exp_index: 迭代序号（如 0, 1, 2, 3, 4）
        """
        setattr(cls._exp_info_local, "name", str(exp_name))
        try:
            setattr(cls._exp_info_local, "index", int(exp_index))
        except Exception:
            setattr(cls._exp_info_local, "index", None)
    
    def set_agent_name(self, name: str) -> None:
        """设置Agent名称（用于标识不同的agent）
        
        Args:
            name: Agent名称
        """
        self._agent_name = name
    
    def _append_trajectory_entry(self, dialog_for_query: Dialog, step_record: "StepRecord") -> None:
        """追加轨迹条目到轨迹文件

        每次step完成后，将prompt、response和tool_responses追加保存到轨迹文件。
        使用文件锁确保多个agent写入同一文件时的线程安全。

        保存格式与现有轨迹格式保持一致：
        [
            {
                "task_id": "...",
                "status": "...",
                "steps": ...,
                "trajectory": {...}
            }
        ]

        每次step会追加一个新的条目，包含本次调用的prompt、response和tool_responses。

        Args:
            dialog_for_query: 发送给LLM的对话（prompt）
            step_record: 步骤记录（包含assistant_message和tool_responses）
        """
        if self._trajectory_file_path is None:
            return

        try:
            # 构建新的轨迹条目
            # 格式与现有轨迹格式保持一致，但保存的是每次LLM调用的信息
            task_id = self.trajectory.task_id if self.trajectory else "unknown"
            status = self.trajectory.status if self.trajectory else "running"

            # 将dialog_for_query转换为字典格式
            prompt_dict = dialog_for_query.model_dump() if hasattr(dialog_for_query, 'model_dump') else {
                "messages": [
                    {
                        "role": msg.role.value if hasattr(msg.role, 'value') else str(msg.role),
                        "content": msg.content if hasattr(msg, 'content') else str(msg)
                    }
                    for msg in dialog_for_query.messages
                ],
                "tools": dialog_for_query.tools if hasattr(dialog_for_query, 'tools') else []
            }

            # 从step_record中获取assistant_message
            assistant_message = step_record.assistant_message

            # 将assistant_message转换为字典格式
            response_dict = assistant_message.model_dump() if hasattr(assistant_message, 'model_dump') else {
                "role": assistant_message.role.value if hasattr(assistant_message.role, 'value') else str(assistant_message.role),
                "content": assistant_message.content if hasattr(assistant_message, 'content') else "",
                "tool_calls": [
                    {
                        "id": tc.id if hasattr(tc, 'id') else "",
                        "function": {
                            "name": tc.function.name if hasattr(tc.function, 'name') else "",
                            "arguments": tc.function.arguments if hasattr(tc.function, 'arguments') else ""
                        }
                    }
                    for tc in (assistant_message.tool_calls or [])
                ] if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls else []
            }

            # 将tool_responses转换为字典格式
            tool_responses_list = []
            for tr in step_record.tool_responses:
                tr_dict = tr.model_dump() if hasattr(tr, 'model_dump') else {
                    "role": "tool",
                    "content": tr.content if hasattr(tr, 'content') else "",
                    "tool_call_id": tr.tool_call_id if hasattr(tr, 'tool_call_id') else "",
                    "name": tr.name if hasattr(tr, 'name') else ""
                }
                tool_responses_list.append(tr_dict)

            exp_name = getattr(self._exp_info_local, "name", None)
            exp_index = getattr(self._exp_info_local, "index", None)

            # 构建轨迹条目，格式与现有轨迹格式保持一致
            entry = {
                "task_id": f"{task_id}_{self._agent_name or 'agent'}_step_{self._step_count}",
                "exp_name": exp_name,      # exp阶段名称
                "exp_index": exp_index,    # exp迭代序号
                "status": status,
                "steps": self._step_count,
                "trajectory": {
                    "task_id": task_id,
                    "agent_name": self._agent_name or "unknown",
                    "step": self._step_count,
                    "dialogs": [prompt_dict],  # 保存本次调用的prompt
                    "steps": [
                        {
                            "step_id": self._step_count,
                            "assistant_message": response_dict,  # 保存本次调用的response
                            "tool_responses": tool_responses_list,  # 保存工具响应
                            "meta": {}
                        }
                    ],
                    "start_time": None,
                    "end_time": None,
                    "status": status,
                    "result": {
                        "prompt": prompt_dict,
                        "response": response_dict
                    },
                    "meta": {
                        "agent_version": self.VERSION,
                        "agent_name": self._agent_name or "unknown",
                        "step": self._step_count
                    }
                }
            }

            entry_json = json.dumps(entry, ensure_ascii=False, indent=2, default=str)
            entry_text = "\n".join("  " + line for line in entry_json.splitlines())
            entry_bytes = entry_text.encode("utf-8")

            path = self._trajectory_file_path
            with self._trajectory_file_lock:
                # Fast path: append into an existing JSON array without loading the full file.
                try:
                    if not path.exists() or path.stat().st_size == 0:
                        with open(path, "wb") as f:
                            f.write(b"[\n")
                            f.write(entry_bytes)
                            f.write(b"\n]\n")
                        return

                    with open(path, "rb+") as f:
                        f.seek(0, os.SEEK_END)
                        file_size = int(f.tell() or 0)
                        tail_size = min(file_size, 16 * 1024)
                        f.seek(max(0, file_size - tail_size))
                        tail = f.read(tail_size)

                        idx = tail.rfind(b"]")
                        if idx < 0:
                            raise ValueError("trajectory file missing closing bracket")

                        # Determine if the array already has elements by scanning back for previous non-ws byte.
                        j = idx - 1
                        while j >= 0 and tail[j] in b" \t\r\n":
                            j -= 1
                        prev = tail[j] if j >= 0 else None
                        non_empty = (prev is not None) and (prev != ord("["))

                        insert_pos = (file_size - tail_size) + idx

                        f.seek(insert_pos)
                        suffix = f.read()
                        f.seek(insert_pos)
                        f.write((b",\n" if non_empty else b"\n") + entry_bytes + b"\n")
                        f.write(suffix)
                        return
                except Exception:
                    # Fallback: load & rewrite (keeps robustness when trajectory.json is corrupted).
                    existing_data: list[Any] = []
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                        if isinstance(loaded, list):
                            existing_data = loaded
                    except Exception:
                        existing_data = []

                    existing_data.append(entry)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(existing_data, f, indent=2, default=str, ensure_ascii=False)

        except Exception as e:
            # 如果保存失败，只记录日志，不中断执行
            self.logger.warning(f"Failed to append trajectory entry: {e}", exc_info=True)

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """获取系统提示词

        子类必须实现此方法。
        """
        pass

    @abstractmethod
    def _get_user_prompt(self, task: TaskInstance) -> str:
        """获取用户提示词

        子类必须实现此方法。

        Args:
            task: 任务实例
        """
        pass


class Agent(BaseAgent):
    """标准 Agent 实现

    使用可配置的提示词模板。
    支持从配置文件加载提示词。
    """

    def __init__(
        self,
        llm: BaseLLM,
        session: BaseSession,
        tools: ToolRegistry,
        system_prompt_file: str | Path | None = None,
        user_prompt_file: str | Path | None = None,
        prompt_format_kwargs: dict[str, Any] | None = None,
        config: AgentConfig | None = None,
        skill_registry: SkillRegistry | None = None,
        output_config: dict[str, Any] | None = None,
        config_dir: Path | str | None = None,
        enable_tools: bool = True,
    ):
        """初始化 Agent

        Args:
            llm: LLM 实例
            session: 环境会话
            tools: 工具注册中心
            system_prompt_file: 系统提示词文件路径（相对于config_dir或绝对路径）
            user_prompt_file: 用户提示词文件路径（相对于config_dir或绝对路径）
            prompt_format_kwargs: 用于格式化提示词的参数字典（{}占位符）
            config: Agent 配置
            skill_registry: Skills 注册中心（可选）
            output_config: 输出显示配置
            config_dir: 配置目录路径，用于加载提示词文件
            enable_tools: 是否在提示词中包含工具信息（默认 True）。如果为 False，工具仍然注册但不会出现在提示词中，Agent 将不会调用工具
        """
        super().__init__(llm, session, tools, config, skill_registry, output_config, config_dir=config_dir, enable_tools=enable_tools)

        # 存储提示词
        self._system_prompt: str | None = None
        self._user_prompt: str | None = None
        self._prompt_format_kwargs = prompt_format_kwargs or {}
        
        # 加载系统提示词（优先级：system_prompt_file > 默认）
        if system_prompt_file:
            self._system_prompt = self.load_prompt_from_file(
                system_prompt_file,
                format_kwargs=self._prompt_format_kwargs
            )
        else:
            self._system_prompt = self._default_system_prompt()
        
        # 加载用户提示词（可选）
        if user_prompt_file:
            self._user_prompt = self.load_prompt_from_file(
                user_prompt_file,
                format_kwargs=self._prompt_format_kwargs
            )

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        prompt = """You are a helpful AI assistant that can execute tasks using tools.

You have access to the following tools:
- execute_bash: Execute bash commands in a terminal
- str_replace_editor: View, create, and edit files
- think: Think about the problem (does not affect the environment)
- finish: Signal that you have completed the task
"""

        # 如果有 skill_registry，添加 skills 信息
        if self.skill_registry is not None:
            skills_info = self.skill_registry.get_meta_info_context()
            if skills_info:
                prompt += f"\n{skills_info}\n"
                prompt += """
You can use the 'use_skill' tool to:
1. Get detailed information about a skill: action='get_info'
2. Get reference documentation: action='get_reference'
3. Run scripts from Operator skills: action='run_script'
"""

        prompt += """
When you need to complete a task:
1. First understand what needs to be done
2. Check if any available skills can help you
3. Use the available tools to accomplish the task
4. When finished, use the finish tool to signal completion

Always be careful with file operations and bash commands.
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """获取系统提示词，动态添加工作目录信息；若有 skill_registry 则自动注入 skills 信息"""
        working_dir = self.session.config.workspace_path
        # 将相对路径转换为绝对路径
        working_dir_abs = str(Path(working_dir).absolute())
        working_dir_info = f"\n\n重要提示：当前工作目录是 {working_dir_abs}。你必须在这个目录下进行所有操作，不能切换工作目录。所有文件操作、命令执行都必须在工作目录 {working_dir_abs} 下进行。"
        prompt = self._system_prompt + working_dir_info
        # 若有 skill_registry，自动注入 skills 信息（与 _default_system_prompt 一致）
        if self.skill_registry is not None:
            skills_info = self.skill_registry.get_meta_info_context()
            if skills_info:
                prompt += f"\n{skills_info}\n"
                prompt += """
You can use the 'use_skill' tool to:
1. Get detailed information about a skill: action='get_info'
2. Get reference documentation: action='get_reference'
3. Run scripts from Operator skills: action='run_script'
"""
        return prompt

    def _get_user_prompt(self, task: TaskInstance) -> str:
        """获取用户提示词"""
        # 如果设置了用户提示词，使用它（可以包含{}占位符）
        if self._user_prompt:
            try:
                return self._user_prompt.format(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    description=task.description,
                    input_data=task.input_data,
                    **self._prompt_format_kwargs
                )
            except KeyError:
                # 如果格式化失败，直接返回（可能没有占位符）
                return self._user_prompt
        
        # 默认用户提示词
        return f"""Please complete the following task:

Task ID: {task.task_id}
Task Type: {task.task_type}
Description: {task.description}

Additional Information:
{task.input_data}
"""

    def _get_tool_specs(self) -> list:
        """获取工具规格列表
        
        覆盖基类方法，但逻辑与基类相同（已移至基类）。
        保留此方法以保持向后兼容性。
        """
        return super()._get_tool_specs()
