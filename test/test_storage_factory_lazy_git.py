from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


class StorageFactoryLazyGitTests(unittest.TestCase):
    def test_sqlite_backend_does_not_require_git_executable(self) -> None:
        script = textwrap.dedent(
            """
            import os
            import tempfile
            from pathlib import Path

            os.environ["STORAGE_BACKEND"] = "sqlite"
            os.environ["DATABASE_URL"] = ""

            from services.storage.factory import create_storage_backend

            with tempfile.TemporaryDirectory() as temp_dir:
                backend = create_storage_backend(Path(temp_dir))
                try:
                    assert backend.health_check()
                finally:
                    backend.engine.dispose()
            """
        )
        env = dict(os.environ)
        env["PATH"] = ""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
