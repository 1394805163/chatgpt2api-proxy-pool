from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services.auth_service import AuthService
from services.log_service import LoggedCall
from services.storage.database_storage import DatabaseStorageBackend
from services.storage.json_storage import JSONStorageBackend


def make_auth_service(root: Path, *, daily_limit: int = 3) -> tuple[AuthService, dict[str, object]]:
    service = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
    _, raw_key = service.create_key(role="user", name="quota-user", daily_request_limit=daily_limit)
    identity = service.authenticate(raw_key)
    assert identity is not None
    return service, identity


class AuthQuotaPersistenceTests(unittest.TestCase):
    def test_reset_save_failure_rolls_back_state_and_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service, identity = make_auth_service(Path(tmp_dir), daily_limit=1)
            key_id = str(identity["id"])
            with service._lock:
                found = service._find_item_locked(key_id)
                self.assertIsNotNone(found)
                index, item = found
                previous = dict(item)
                previous["daily_request_date"] = "2000-01-01"
                previous["daily_request_used"] = 1
                service._items[index] = previous

            with mock.patch.object(service, "_save_item_locked", side_effect=OSError("storage unavailable")):
                with self.assertRaisesRegex(OSError, "storage unavailable"):
                    service.reserve_daily_request(identity, "reset-failure")

            with service._lock:
                current = service._find_item_locked(key_id)
                self.assertIsNotNone(current)
                self.assertEqual(current[1]["daily_request_date"], "2000-01-01")
                self.assertEqual(current[1]["daily_request_used"], 1)
                self.assertNotIn(key_id, service._daily_reservations)

    def test_success_save_failure_keeps_reservation_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service, identity = make_auth_service(Path(tmp_dir), daily_limit=1)
            service.reserve_daily_request(identity, "persist-retry")
            original_save = service._save_item_locked

            with mock.patch.object(service, "_save_item_locked", side_effect=OSError("storage unavailable")):
                with self.assertRaisesRegex(OSError, "storage unavailable"):
                    service.finish_daily_request(identity, "persist-retry", success=True)

            key_id = str(identity["id"])
            with service._lock:
                self.assertIn("persist-retry", service._daily_reservations[key_id])
                current = service._find_item_locked(key_id)
                self.assertIsNotNone(current)
                self.assertEqual(current[1]["daily_request_used"], 0)

            service._save_item_locked = original_save
            self.assertTrue(service.finish_daily_request(identity, "persist-retry", success=True))
            self.assertEqual(service.list_keys(role="user")[0]["daily_request_used"], 1)

    def test_database_success_counter_updates_only_one_auth_key_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            storage = DatabaseStorageBackend(f"sqlite:///{root / 'quota.db'}")
            try:
                service = AuthService(storage)
                _, first_key = service.create_key(role="user", name="first", daily_request_limit=2)
                service.create_key(role="user", name="second", daily_request_limit=2)
                identity = service.authenticate(first_key)
                self.assertIsNotNone(identity)

                with mock.patch.object(storage, "save_auth_keys", side_effect=AssertionError("full rewrite used")):
                    service.reserve_daily_request(identity, "single-row")
                    self.assertTrue(service.finish_daily_request(identity, "single-row", success=True))

                items = {str(item["name"]): item for item in service.list_keys(role="user")}
                self.assertEqual(items["first"]["daily_request_used"], 1)
                self.assertEqual(items["second"]["daily_request_used"], 0)
            finally:
                storage.engine.dispose()


class LoggedCallQuotaTests(unittest.TestCase):
    def test_non_stream_success_counts_and_failure_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service, identity = make_auth_service(Path(tmp_dir), daily_limit=2)
            with mock.patch("services.auth_service.auth_service", service):
                success = LoggedCall(identity, "/v1/chat/completions", "auto", "text")
                with mock.patch.object(success, "log"):
                    result = asyncio.run(success.run(lambda: {"ok": True}))
                self.assertEqual(result, {"ok": True})

                failed = LoggedCall(identity, "/v1/chat/completions", "auto", "text")
                with mock.patch.object(failed, "log"):
                    response = asyncio.run(failed.run(lambda: (_ for _ in ()).throw(RuntimeError("upstream failed"))))
                self.assertEqual(response.status_code, 502)

            item = service.list_keys(role="user")[0]
            self.assertEqual(item["daily_request_used"], 1)
            self.assertEqual(item["daily_request_remaining"], 1)

    def test_stream_counts_only_after_full_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service, identity = make_auth_service(Path(tmp_dir), daily_limit=2)
            with mock.patch("services.auth_service.auth_service", service):
                completed = LoggedCall(identity, "/v1/chat/completions", "auto", "text")
                completed._reserve_quota()
                with mock.patch.object(completed, "log"):
                    self.assertEqual(list(completed.stream(iter([{"value": 1}, {"value": 2}]))), [{"value": 1}, {"value": 2}])

                disconnected = LoggedCall(identity, "/v1/chat/completions", "auto", "text")
                disconnected._reserve_quota()
                with mock.patch.object(disconnected, "log"):
                    stream = disconnected.stream(iter([{"value": 1}, {"value": 2}]))
                    self.assertEqual(next(stream), {"value": 1})
                    stream.close()

            item = service.list_keys(role="user")[0]
            self.assertEqual(item["daily_request_used"], 1)
            self.assertEqual(item["daily_request_remaining"], 1)


if __name__ == "__main__":
    unittest.main()
