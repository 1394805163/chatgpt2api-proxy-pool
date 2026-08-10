from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

from services.storage.database_storage import DatabaseStorageBackend


class SQLiteWalTests(unittest.TestCase):
    def test_sqlite_uses_wal_normal_sync_and_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "accounts.db"
            backend = DatabaseStorageBackend(f"sqlite:///{db_path.as_posix()}")
            try:
                with backend.engine.connect() as connection:
                    journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
                    synchronous = connection.execute(text("PRAGMA synchronous")).scalar_one()
                    busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

                self.assertEqual(str(journal_mode).lower(), "wal")
                self.assertEqual(int(synchronous), 1)
                self.assertEqual(int(busy_timeout), 30000)
            finally:
                backend.engine.dispose()

    def test_non_sqlite_engine_is_not_given_sqlite_connect_args(self) -> None:
        self.assertFalse(DatabaseStorageBackend._is_sqlite_url("postgresql://user:pass@db/app"))
        self.assertTrue(DatabaseStorageBackend._is_sqlite_url("sqlite:////app/data/accounts.db"))


if __name__ == "__main__":
    unittest.main()
