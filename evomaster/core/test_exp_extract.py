import json
import unittest

from evomaster.core.exp import BaseExp
from evomaster.utils.types import (
    AssistantMessage,
    Dialog,
    FunctionCall,
    StepRecord,
    SystemMessage,
    ToolCall,
    Trajectory,
    UserMessage,
)


class TestBaseExpExtractAgentResponse(unittest.TestCase):
    def test_extracts_finish_message_from_tool_call_arguments(self):
        exp = BaseExp(agent=None, config=None)

        trajectory = Trajectory(task_id="t1")
        trajectory.dialogs.append(
            Dialog(
                messages=[
                    SystemMessage(content="sys"),
                    UserMessage(content="user"),
                ]
            )
        )

        finish_args = {"message": "FINAL ANSWER", "task_completed": "true"}
        assistant = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_finish",
                    function=FunctionCall(name="finish", arguments=json.dumps(finish_args)),
                )
            ],
        )
        trajectory.steps.append(StepRecord(step_id=1, assistant_message=assistant))

        self.assertEqual(exp._extract_agent_response(trajectory), "FINAL ANSWER")

    def test_falls_back_to_last_assistant_content(self):
        exp = BaseExp(agent=None, config=None)

        trajectory = Trajectory(task_id="t2")
        trajectory.dialogs.append(
            Dialog(
                messages=[
                    SystemMessage(content="sys"),
                    UserMessage(content="user"),
                    AssistantMessage(content="hello"),
                ]
            )
        )

        self.assertEqual(exp._extract_agent_response(trajectory), "hello")
