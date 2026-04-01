"""Tests for Hamilton workspace asset materialization."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path

from playground.hamilton.core.playground import HamiltonPlayground


class TestHamiltonWorkspaceAssets(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="hamilton_workspace_assets_"))
        self.project_root = self.tmpdir / "project"
        self.workspace_root = self.project_root / "playground" / "hamilton" / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        input_dir = self.workspace_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "U248_train.csv").write_text("t,x,v,a\n0,0,0,0\n", encoding="utf-8")
        nested = input_dir / "subdir"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "extra.csv").write_text("id,value\n1,2\n", encoding="utf-8")

        self.runtime_workspace = self.tmpdir / "runtime_workspace"
        self.runtime_workspace.mkdir(parents=True, exist_ok=True)

        self.playground = HamiltonPlayground.__new__(HamiltonPlayground)
        self.playground._project_root = self.project_root
        self.playground.logger = logging.getLogger("TestHamiltonWorkspaceAssets")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_materialize_workspace_assets_copies_input_tree(self) -> None:
        summary = self.playground._materialize_workspace_assets(self.runtime_workspace)

        copied_train = self.runtime_workspace / "input" / "U248_train.csv"
        copied_nested = self.runtime_workspace / "input" / "subdir" / "extra.csv"
        self.assertTrue(copied_train.exists())
        self.assertTrue(copied_nested.exists())
        self.assertEqual(copied_train.read_text(encoding="utf-8"), "t,x,v,a\n0,0,0,0\n")
        self.assertEqual(summary["input"]["files_copied"], 2)

    def test_materialize_workspace_assets_overwrites_stale_file(self) -> None:
        stale_target = self.runtime_workspace / "input" / "U248_train.csv"
        stale_target.parent.mkdir(parents=True, exist_ok=True)
        stale_target.write_text("stale\n", encoding="utf-8")

        self.playground._materialize_workspace_assets(self.runtime_workspace)

        self.assertEqual(stale_target.read_text(encoding="utf-8"), "t,x,v,a\n0,0,0,0\n")

    def test_materialize_environment_capabilities_writes_runtime_probe(self) -> None:
        self.playground._detect_environment_capabilities = lambda: {
            "python_executable": "/tmp/fake-python",
            "available_packages": ["sympy"],
            "missing_packages": ["numpy", "pandas"],
        }

        capabilities_path = self.playground._materialize_environment_capabilities(self.runtime_workspace)

        self.assertTrue(capabilities_path.exists())
        payload = json.loads(capabilities_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["python_executable"], "/tmp/fake-python")
        self.assertEqual(payload["available_packages"], ["sympy"])
        self.assertEqual(payload["missing_packages"], ["numpy", "pandas"])
