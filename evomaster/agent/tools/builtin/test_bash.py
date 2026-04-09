"""Tests for BashTool environment-aware python command normalization."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path

from evomaster.agent.session import LocalSession, LocalSessionConfig
from evomaster.agent.tools.builtin.bash import BashTool


class TestBashTool(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="bash_tool_test_"))
        self.session = LocalSession(LocalSessionConfig(workspace_path=str(self.tmpdir)))
        self.session.open()
        self.tool = BashTool()

    def tearDown(self) -> None:
        try:
            self.session.close()
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run_tool(self, payload: dict) -> tuple[str, dict]:
        return self.tool.execute(self.session, json.dumps(payload, ensure_ascii=False))

    def test_execute_bash_uses_environment_capabilities_python(self) -> None:
        fake_python = self.tmpdir / "fake-python.sh"
        fake_python.write_text("#!/bin/sh\necho FAKE_PYTHON:$@\n", encoding="utf-8")
        fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

        capabilities = {
            "python_executable": str(fake_python),
            "available_packages": [],
            "missing_packages": [],
        }
        (self.tmpdir / "environment_capabilities.json").write_text(
            json.dumps(capabilities, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        output, meta = self._run_tool({"command": "python3 -V"})

        self.assertEqual(meta["exit_code"], 0)
        self.assertIn("FAKE_PYTHON:-V", output)

    def test_execute_bash_recovers_nested_gateway_wrapper(self) -> None:
        output, meta = self._run_tool(
            {
                "command": "execute_bash",
                "params": {
                    "command": "echo wrapped-command",
                },
            }
        )

        self.assertEqual(meta["exit_code"], 0)
        self.assertIn("wrapped-command", output)
