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

