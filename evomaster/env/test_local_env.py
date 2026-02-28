import os
import stat
import tempfile
import unittest
from pathlib import Path

from evomaster.agent.session.local import LocalSessionConfig
from evomaster.env.local import LocalEnv, LocalEnvConfig


class TestLocalEnvPythonToolchainEnv(unittest.TestCase):
    def test_injects_repo_venv_bin_into_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs" / "hamilton"
            config_dir.mkdir(parents=True, exist_ok=True)

            venv_bin = root / ".venv" / "bin"
            venv_bin.mkdir(parents=True, exist_ok=True)
            py = venv_bin / "python"
            py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            py.chmod(py.stat().st_mode | stat.S_IXUSR)

            session_cfg = LocalSessionConfig(
                workspace_path=str(root / "workspace"),
                config_dir=str(config_dir),
            )
            env = LocalEnv(LocalEnvConfig(session_config=session_cfg))

            injected = env._inject_python_toolchain_env({"PATH": "/usr/bin"})
            self.assertTrue(injected["PATH"].startswith(str(venv_bin)))
            self.assertEqual(injected["VIRTUAL_ENV"], str(venv_bin.parent))
            self.assertEqual(injected["HAMILTON_PYTHON_EXECUTABLE"], str(py))
            self.assertEqual(injected["EVOMASTER_PYTHON_EXECUTABLE"], str(py))
            self.assertEqual(injected["PYTHON_EXECUTABLE"], str(py))


if __name__ == "__main__":
    unittest.main()
