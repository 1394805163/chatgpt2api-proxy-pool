import json
import tempfile
import unittest
from pathlib import Path

from services.config import ConfigStore
from services.storage.database_storage import DatabaseStorageBackend


class SettingsPersistenceTests(unittest.TestCase):
    def test_database_settings_survive_backend_recreation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            database_url = f"sqlite:///{Path(tmp_dir) / 'settings.db'}"
            backends = [DatabaseStorageBackend(database_url)]
            try:
                self.assertEqual(backends[0].load_settings(), {})
                self.assertTrue(backends[0].save_settings({"image_task_timeout_secs": 180}))

                backends.append(DatabaseStorageBackend(database_url))
                self.assertEqual(backends[1].load_settings(), {"image_task_timeout_secs": 180})
            finally:
                for backend in backends:
                    backend.engine.dispose()

    def test_config_store_restores_and_updates_database_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({"auth-key": "test-auth", "image_poll_timeout_secs": 70}),
                encoding="utf-8",
            )
            database_url = f"sqlite:///{root / 'settings.db'}"
            backends = [DatabaseStorageBackend(database_url)]
            try:
                backend = backends[0]
                backend.save_settings({"image_task_timeout_secs": 180, "image_poll_timeout_secs": 180})

                store = ConfigStore(config_path)
                store._storage_backend = backend
                store._restore_persisted_settings(backend)

                self.assertEqual(store.image_task_timeout_secs, 180)
                updated = store.update({"image_task_timeout_secs": 240})
                self.assertEqual(updated["image_task_timeout_secs"], 240)
                self.assertEqual(updated["image_poll_timeout_secs"], 240)
                backends.append(DatabaseStorageBackend(database_url))
                self.assertEqual(backends[1].load_settings()["image_task_timeout_secs"], 240)
                self.assertNotIn("auth-key", backend.load_settings())
            finally:
                for backend in backends:
                    backend.engine.dispose()


if __name__ == "__main__":
    unittest.main()
