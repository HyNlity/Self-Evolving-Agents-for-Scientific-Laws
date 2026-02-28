import json
import unittest

from evomaster.utils.llm import BaseLLM, LLMConfig, LLMResponse
from evomaster.utils.types import (
    Dialog,
    FunctionSpec,
    SystemMessage,
    ToolSpec,
    UserMessage,
)


class _DummyLLM(BaseLLM):
    def __init__(self, response: LLMResponse):
        self._response = response
        config = LLMConfig(
            provider="openai",
            model="dummy",
            api_key="dummy-key",
        )
        super().__init__(config=config, output_config={"log_to_file": False, "show_in_console": False})

    def _setup(self) -> None:
        pass

    def _call(self, messages, tools=None, **kwargs):
        return self._response


def _build_dialog() -> Dialog:
    return Dialog(
        messages=[
            SystemMessage(content="sys"),
            UserMessage(content="user"),
        ],
        tools=[
            ToolSpec(
                function=FunctionSpec(
                    name="execute_bash",
                    description="Execute bash command",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                )
            ),
            ToolSpec(
                function=FunctionSpec(
                    name="finish",
                    description="Finish task",
                    parameters={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "task_completed": {"type": "string"},
                        },
                        "required": ["message", "task_completed"],
                    },
                )
            ),
        ],
    )


class TestLLMToolRecovery(unittest.TestCase):
    def test_recovers_json_command_when_tool_calls_missing(self):
        response = LLMResponse(
            content=(
                '{"command":"execute_bash","params":{"command":"echo hello"}}\n'
                '{"command":"execute_bash","params":{"command":"echo hello"}}'
            ),
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())

        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "execute_bash")
        args = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(args["command"], "echo hello")

    def test_ignores_unknown_command(self):
        response = LLMResponse(
            content='{"command":"not_a_tool","params":{"x":1}}',
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())
        self.assertFalse(msg.tool_calls)

    def test_recovers_legacy_shell_command_as_execute_bash(self):
        response = LLMResponse(
            content='{"command":"cat -n experiment.json"}',
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())

        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "execute_bash")
        args = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(args["command"], "cat -n experiment.json")

    def test_recovers_name_arguments_finish_format(self):
        response = LLMResponse(
            content='{"name":"finish","arguments":{"message":"ok","task_completed":"true"}}',
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())

        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "finish")
        args = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(args["message"], "ok")
        self.assertEqual(args["task_completed"], "true")

    def test_recovers_direct_finish_payload(self):
        response = LLMResponse(
            content='{"message":"done","task_completed":"true"}',
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())

        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "finish")
        args = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(args["message"], "done")
        self.assertEqual(args["task_completed"], "true")

    def test_recovers_direct_finish_payload_without_completed_flag(self):
        response = LLMResponse(
            content='{"message":"done without status"}',
            tool_calls=None,
            finish_reason="stop",
        )
        llm = _DummyLLM(response)
        msg = llm.query(_build_dialog())

        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].function.name, "finish")
        args = json.loads(msg.tool_calls[0].function.arguments)
        self.assertEqual(args["message"], "done without status")
        self.assertEqual(args["task_completed"], "partial")


if __name__ == "__main__":
    unittest.main()
