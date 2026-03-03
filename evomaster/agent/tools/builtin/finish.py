"""EvoMaster Finish 工具

用于标记任务完成。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from ..base import BaseTool, BaseToolParams

if TYPE_CHECKING:
    from evomaster.agent.session import BaseSession


class FinishToolParams(BaseToolParams):
    """Signals the completion of the current round of work.

    You MUST call this tool when you have finished all phases of the current round.
    Do NOT output your summary as plain text — always use this tool.

    Call this tool regardless of whether the overall task is fully solved:
    - task_completed="true": current round's objectives are met
    - task_completed="false": current round is done but overall task needs more iterations
    - task_completed="partial": current round partially completed

    The message should include a summary of actions taken, results, and any signal blocks required by the protocol.
    """

    name: ClassVar[str] = "finish"

    message: str = Field(description="Final message summarizing this round's work, including any required signal blocks")
    task_completed: Literal["true", "false", "partial"] = Field(
        description="Whether the current round's work is complete. Use 'false' if the overall task needs more iterations."
    )


class FinishTool(BaseTool):
    """完成工具"""
    
    name: ClassVar[str] = "finish"
    params_class: ClassVar[type[BaseToolParams]] = FinishToolParams

    def execute(self, session: BaseSession, args_json: str) -> tuple[str, dict[str, Any]]:
        """标记任务完成"""
        try:
            params = self.parse_params(args_json)
        except Exception as e:
            return f"Parameter validation error: {str(e)}", {"error": str(e)}
        
        assert isinstance(params, FinishToolParams)
        
        # 记录完成信息
        self.logger.info(f"Task finished. Completed: {params.task_completed}")
        self.logger.info(f"Final message: {params.message[:200]}...")
        
        return f"Task marked as {params.task_completed}. Message: {params.message}", {
            "task_completed": params.task_completed,
            "message": params.message,
        }

