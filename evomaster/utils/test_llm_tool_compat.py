import json
import unittest

from evomaster.utils.llm import parse_compatible_tool_calls


class TestCompatibleToolCallParsing(unittest.TestCase):
    def test_recovers_single_finish_call(self):
        content = """```json
{"name":"finish","arguments":{"message":"done","task_completed":"false"}}
```"""

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "execute_bash"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "finish")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"message": "done", "task_completed": "false"},
        )

    def test_recovers_multiple_prefixed_calls(self):
        content = (
            '{"name":"functions.str_replace_editor","arguments":{"command":"create","path":"a.txt","file_text":"hello"}},'
            '{"name":"functions.finish","arguments":{"message":"done","task_completed":"false"}}'
        )

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"str_replace_editor", "finish"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual([tc.function.name for tc in tool_calls], ["str_replace_editor", "finish"])
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"command": "create", "path": "a.txt", "file_text": "hello"},
        )
        self.assertEqual(
            json.loads(tool_calls[1].function.arguments),
            {"message": "done", "task_completed": "false"},
        )

    def test_recovers_slash_prefixed_finish_call(self):
        content = '{"name":"functions/finish","arguments":{"message":"done","task_completed":"false"}}'

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "execute_bash"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "finish")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"message": "done", "task_completed": "false"},
        )

    def test_ignores_unknown_tool_names(self):
        content = '{"name":"functions.unknown_tool","arguments":{"x":1}}'

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "str_replace_editor"},
        )

        self.assertIsNone(tool_calls)
        self.assertFalse(fully_consumed)

    def test_recovers_bare_editor_arguments_without_name_wrapper(self):
        content = (
            '{"path":"/tmp/report.json","command":"create","file_text":"{}",'
            '"old_str":"","new_str":""},'
            '{"message":"critic done","task_completed":"false"}'
        )

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"str_replace_editor", "finish"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual([tc.function.name for tc in tool_calls], ["str_replace_editor", "finish"])
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {
                "path": "/tmp/report.json",
                "command": "create",
                "file_text": "{}",
                "old_str": "",
                "new_str": "",
            },
        )

    def test_recovers_mixed_think_and_multiline_bash_fragments(self):
        content = """{"name":"functions.think","arguments":{"thought":"attack this claim"}}
{"command":"mkdir -p history/round3/critic_attacks/results"}
{"command":"python3 - << 'PY'
print(1)
PY"}"""

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"think", "execute_bash", "finish"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(
            [tc.function.name for tc in tool_calls],
            ["think", "execute_bash", "execute_bash"],
        )
        self.assertIn("mkdir -p", json.loads(tool_calls[1].function.arguments)["command"])
        self.assertIn("python3 - << 'PY'", json.loads(tool_calls[2].function.arguments)["command"])

    def test_recovers_nested_finish_wrapper(self):
        content = '{"finish":{"message":"wrapped done","task_completed":"false"}}'

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "str_replace_editor"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "finish")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"message": "wrapped done", "task_completed": "false"},
        )

    def test_unwraps_finish_nested_inside_execute_bash_wrapper(self):
        content = (
            '{"name":"execute_bash","arguments":'
            '{"command":"finish","params":{"message":"wrapped via bash","task_completed":"false"}}}'
        )

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "execute_bash"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "finish")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"message": "wrapped via bash", "task_completed": "false"},
        )

    def test_flattens_execute_bash_nested_inside_execute_bash_wrapper(self):
        content = (
            '{"name":"execute_bash","arguments":'
            '{"command":"execute_bash","params":{"command":"python lib/fit_viv_analytic.py --input-dir input"}}}'
        )

        tool_calls, fully_consumed = parse_compatible_tool_calls(
            content,
            {"finish", "execute_bash"},
        )

        self.assertTrue(fully_consumed)
        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "execute_bash")
        self.assertEqual(
            json.loads(tool_calls[0].function.arguments),
            {"command": "python lib/fit_viv_analytic.py --input-dir input"},
        )


if __name__ == "__main__":
    unittest.main()
