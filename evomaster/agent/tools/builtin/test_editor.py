"""Tests for EditorTool path normalization and edits."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evomaster.agent.session import LocalSession, LocalSessionConfig
from evomaster.agent.tools.builtin.editor import EditorTool


class TestEditorTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="editor_tool_test_"))
        self.session = LocalSession(LocalSessionConfig(workspace_path=str(self.tmpdir)))
        self.session.open()
        self.tool = EditorTool()

    def tearDown(self) -> None:
        try:
            self.session.close()
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_tool(self, payload: dict) -> tuple[str, dict]:
        return self.tool.execute(self.session, json.dumps(payload, ensure_ascii=False))

    def test_create_maps_workspace_alias_to_real_workspace(self) -> None:
        message, meta = self._run_tool(
            {
                "command": "create",
                "path": "/workspace/note.txt",
                "file_text": "hello from normalized path\n",
            }
        )
        created = self.tmpdir / "note.txt"
        self.assertEqual(meta, {})
        self.assertTrue(created.exists())
        self.assertEqual(created.read_text(encoding="utf-8"), "hello from normalized path\n")
        self.assertIn(str(created), message)

    def test_str_replace_maps_workspace_alias_to_real_workspace(self) -> None:
        target = self.tmpdir / "plan.md"
        target.write_text("# Plan\n\n- old line\n", encoding="utf-8")

        message, meta = self._run_tool(
            {
                "command": "str_replace",
                "path": "/workspace/plan.md",
                "old_str": "- old line",
                "new_str": "- new line",
            }
        )
        self.assertEqual(meta, {})
        self.assertEqual(target.read_text(encoding="utf-8"), "# Plan\n\n- new line\n")
        self.assertIn(str(target), message)

    def test_str_replace_resolves_relative_workspace_path(self) -> None:
        target = self.tmpdir / "trace.md"
        target.write_text("# Trace\n\n- todo\n", encoding="utf-8")

        message, meta = self._run_tool(
            {
                "command": "str_replace",
                "path": "trace.md",
                "old_str": "- todo",
                "new_str": "- done",
            }
        )

        self.assertEqual(meta, {})
        self.assertEqual(target.read_text(encoding="utf-8"), "# Trace\n\n- done\n")
        self.assertIn(str(target), message)

    def test_create_repairs_duplicated_workspace_prefix_in_absolute_path(self) -> None:
        target = self.tmpdir / "history" / "round1" / "trace.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        doubled = Path("/tmp") / "tmp" / self.tmpdir.name / "history" / "round1" / "trace.md"

        message, meta = self._run_tool(
            {
                "command": "create",
                "path": str(doubled),
                "file_text": "trace body\n",
            }
        )

        self.assertEqual(meta, {})
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), "trace body\n")
        self.assertFalse(doubled.exists())
        self.assertIn(str(target), message)

    def test_str_replace_accepts_cat_n_prefixed_old_and_new_strings(self) -> None:
        target = self.tmpdir / "plan.md"
        target.write_text("## 当前最优\n- 轮次：0\n- 方程：无\n", encoding="utf-8")

        message, meta = self._run_tool(
            {
                "command": "str_replace",
                "path": str(target),
                "old_str": "     1\t## 当前最优\n     2\t- 轮次：0\n     3\t- 方程：无",
                "new_str": "     1\t## 当前最优\n     2\t- 轮次：3\n     3\t- 方程：x'' = -k x - b v",
            }
        )

        self.assertEqual(meta, {})
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "## 当前最优\n- 轮次：3\n- 方程：x'' = -k x - b v\n",
        )
        self.assertIn(str(target), message)

    def test_create_can_overwrite_marked_scaffold_file(self) -> None:
        target = self.tmpdir / "trace.md"
        target.write_text("<!-- EVO_SCAFFOLD_OVERWRITABLE -->\n# scaffold\n", encoding="utf-8")

        message, meta = self._run_tool(
            {
                "command": "create",
                "path": str(target),
                "file_text": "# rewritten trace\n",
            }
        )

        self.assertEqual(meta, {})
        self.assertEqual(target.read_text(encoding="utf-8"), "# rewritten trace\n")
        self.assertIn("overwritten successfully", message)

    def test_create_recovers_prefixed_editor_command(self) -> None:
        message, meta = self._run_tool(
            {
                "command": "str_replace_editor create",
                "path": str(self.tmpdir / "note.txt"),
                "file_text": "prefixed command\n",
            }
        )

        self.assertEqual(meta, {})
        self.assertEqual((self.tmpdir / "note.txt").read_text(encoding="utf-8"), "prefixed command\n")
        self.assertIn("File created successfully", message)

    def test_create_can_overwrite_placeholder_json_file(self) -> None:
        target = self.tmpdir / "critic_report.json"
        target.write_text('{"__placeholder__": true, "summary": ""}\n', encoding="utf-8")

        message, meta = self._run_tool(
            {
                "command": "create",
                "path": str(target),
                "file_text": '{"approved": false}\n',
            }
        )

        self.assertEqual(meta, {})
        self.assertEqual(target.read_text(encoding="utf-8"), '{"approved": false}\n')
        self.assertIn("overwritten successfully", message)
