"""EvoMaster Bash 工具

提供在环境中执行 Bash 命令的能力。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams, ToolError

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class BashToolParams(BaseToolParams):
    """Execute a bash command in the terminal within a persistent shell session.

    ### Command Execution
    * One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
    * Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
    * Default timeout: Commands have a default timeout of 1 hour. For commands that may run longer, set the "timeout" parameter.

    ### Long-running Commands
    * For commands that may run indefinitely (e.g. servers), run them in the background: `python3 app.py > server.log 2>&1 &`.
    * For scientific computing tasks (e.g. PySR, numerical integration), just run them directly — the default 1 hour timeout is sufficient for most cases.
    * If a bash command returns exit code `-1`, this means the process is still running. By setting `is_input` to `true`, you can:
      - Send empty `command` to retrieve additional logs
      - Send text (set `command` to the text) to STDIN of the running process
      - Send control commands like `C-c` (Ctrl+C), `C-z` (Ctrl+Z) to interrupt the process

    ### Best Practices
    * Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
    * Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

    ### Output Handling
    * Output truncation: If the output exceeds a maximum length, it will be truncated.
    """
    
    name: ClassVar[str] = "execute_bash"

    command: str = Field(
        description="The bash command to execute. Can be empty string to view additional logs when previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently running process."
    )
    is_input: Literal["true", "false"] = Field(
        default="false",
        description="If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.",
    )
    timeout: float = Field(
        default=-1,
        description="Optional timeout in seconds. Default is 1 hour. Only set this if you need a longer timeout.",
    )


class BashTool(BaseTool):
    """Bash 命令执行工具"""
    
    name: ClassVar[str] = "execute_bash"
    params_class: ClassVar[type[BaseToolParams]] = BashToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """执行 Bash 命令"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, BashToolParams)
        
        # 执行命令
        timeout = int(params.timeout) if params.timeout > 0 else None
        is_input = params.is_input == "true"
        
        result = session.exec_bash(
            command=params.command,
            timeout=timeout,
            is_input=is_input,
        )
        
        # 构建输出
        # 优先使用合并输出（stdout+stderr），保证报错可见
        output = result.get("output", "") or result.get("stdout", "")
        exit_code = result.get("exit_code", -1)
        working_dir = result.get("working_dir", "")
        
        # 将相对路径转换为绝对路径
        working_dir_abs = str(Path(working_dir).absolute()) if working_dir else ""
        
        # 添加状态信息
        obs = output
        if working_dir_abs:
            obs += f"\n[Current working directory: {working_dir_abs}]"
        if exit_code != -1:
            obs += f"\n[Command finished with exit code {exit_code}]"
        
        info = {
            "exit_code": exit_code,
            "working_dir": working_dir_abs,
        }
        
        return obs, info

