from __future__ import annotations

import copy
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("CHATGPT2API_AUTH_KEY", "test-auth")

from services.account_service import AccountService
from services.auth_service import AuthService, DailyRequestQuotaExceeded, ImageRequestLimitExceeded
from services.config import config
from services.openai_backend_api import InvalidAccessTokenError, OpenAIBackendAPI
from services.storage.json_storage import JSONStorageBackend
from utils.helper import anonymize_token, split_image_model


class AccountCapabilityTests(unittest.TestCase):
    def test_unknown_quota_accounts_are_available_only_when_not_throttled(self) -> None:
        self.assertFalse(
            AccountService._is_image_account_available(
                {"status": "限流", "image_quota_unknown": True, "quota": 0}
            )
        )
        self.assertTrue(
            AccountService._is_image_account_available(
                {"status": "正常", "image_quota_unknown": True, "quota": 0}
            )
        )

    def test_prolite_variants_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertEqual(service._normalize_account_type("prolite"), "ProLite")
            self.assertEqual(service._normalize_account_type("pro_lite"), "ProLite")

    def test_search_account_type_ignores_unrelated_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            self.assertIsNone(
                service._search_account_type(
                    {
                        "amr": ["pwd", "otp", "mfa"],
                        "chatgpt_compute_residency": "no_constraint",
                        "chatgpt_data_residency": "no_constraint",
                        "user_id": "user-I52GFfLGFM0dokFk2dBiKEBn",
                    }
                )
            )

    def test_mark_image_result_does_not_consume_unknown_quota(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])
            service.update_account(
                "token-1",
                {
                    "status": "正常",
                    "quota": 0,
                    "image_quota_unknown": True,
                },
            )

            updated = service.mark_image_result("token-1", success=True)

            self.assertIsNotNone(updated)
            self.assertEqual(updated["quota"], 0)
            self.assertEqual(updated["status"], "正常")
            self.assertTrue(updated["image_quota_unknown"])

    def test_mark_image_result_tracks_consecutive_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items([{"access_token": "token-1", "status": "正常", "quota": 3}])

            first_failure = service.mark_image_result("token-1", success=False)
            second_failure = service.mark_image_result("token-1", success=False)
            success = service.mark_image_result("token-1", success=True)

            self.assertEqual(first_failure["consecutive_image_failures"], 1)
            self.assertEqual(second_failure["consecutive_image_failures"], 2)
            self.assertEqual(success["consecutive_image_failures"], 0)

    def test_split_image_model_supports_plan_type_prefix(self) -> None:
        self.assertEqual(split_image_model("gpt-image-2"), (None, "gpt-image-2"))
        self.assertEqual(split_image_model("plus-codex-gpt-image-2"), ("plus", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("team-codex-gpt-image-2"), ("team", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("pro-codex-gpt-image-2"), ("pro", "codex-gpt-image-2"))
        self.assertEqual(split_image_model("plus-gpt-image-2"), (None, None))
        self.assertEqual(split_image_model("unknown-image-model"), (None, None))

    def test_get_available_access_token_filters_by_plan_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_account_items(
                [
                    {"access_token": "token-plus", "type": "Plus", "status": "正常", "quota": 3},
                    {"access_token": "token-pro", "type": "Pro", "status": "正常", "quota": 3},
                ]
            )

            service.fetch_remote_info = lambda access_token, event="fetch_remote_info": service.get_account(access_token)

            plus_token = service.get_available_access_token(plan_type="plus")
            pro_token = service.get_available_access_token(plan_type="pro")
            service.release_image_slot(plus_token)
            service.release_image_slot(pro_token)

            self.assertEqual(plus_token, "token-plus")
            self.assertEqual(pro_token, "token-pro")

    def test_refresh_accounts_can_remove_invalid_token_without_confirmation_delay(self) -> None:
        original_value = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "invalid-token", "status": "正常"}])

                with patch(
                    "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                    side_effect=InvalidAccessTokenError("token invalidated (/backend-api/me)"),
                ):
                    result = service.refresh_accounts(["invalid-token"], defer_invalid_removal=False)

                self.assertEqual(result["refreshed"], 0)
                self.assertEqual(len(result["errors"]), 1)
                self.assertEqual(result["items"], [])
                self.assertIsNone(service.get_account("invalid-token"))
        finally:
            if original_value is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_value

    def test_refresh_accounts_defers_invalid_token_removal_by_default(self) -> None:
        original_value = config.data.get("auto_remove_invalid_accounts")
        config.data["auto_remove_invalid_accounts"] = True
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "invalid-token", "status": "正常"}])

                with patch(
                    "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                    side_effect=InvalidAccessTokenError("token invalidated (/backend-api/me)"),
                ):
                    result = service.refresh_accounts(["invalid-token"])

                account = service.get_account("invalid-token")
                self.assertEqual(result["refreshed"], 0)
                self.assertEqual(len(result["errors"]), 1)
                self.assertIsNotNone(account)
                self.assertEqual(account["invalid_count"], 1)
        finally:
            if original_value is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_value

    def test_refresh_accounts_limits_concurrency_and_persists_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = JSONStorageBackend(Path(tmp_dir) / "accounts.json")
            service = AccountService(storage)
            tokens = [f"token-{index}" for index in range(6)]
            service.add_accounts(tokens)

            active = 0
            peak_active = 0
            active_lock = threading.Lock()

            def fetch_remote_info(token: str, *args, **kwargs):
                nonlocal active, peak_active
                with active_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    time.sleep(0.02)
                    return service.update_account(token, {"quota": 5, "status": "姝ｅ父"})
                finally:
                    with active_lock:
                        active -= 1

            service.fetch_remote_info = fetch_remote_info
            original_save = storage.save_accounts
            storage.save_accounts = MagicMock(wraps=original_save)

            result = service.refresh_accounts(tokens)

            self.assertEqual(result["refreshed"], len(tokens))
            self.assertLessEqual(peak_active, service._MAX_REFRESH_WORKERS)
            self.assertEqual(storage.save_accounts.call_count, 1)

    def test_overlapping_refresh_batches_persist_when_each_batch_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = JSONStorageBackend(Path(tmp_dir) / "accounts.json")
            service = AccountService(storage)
            service.add_accounts(["token-1"])
            original_save = storage.save_accounts
            storage.save_accounts = MagicMock(wraps=original_save)

            service._begin_account_save_batch()
            service._begin_account_save_batch()
            service.update_account("token-1", {"quota": 3})
            service._end_account_save_batch()

            self.assertEqual(storage.save_accounts.call_count, 1)
            self.assertEqual(service._account_save_batch_depth, 1)

            service.update_account("token-1", {"quota": 2})
            service._end_account_save_batch()

            self.assertEqual(storage.save_accounts.call_count, 2)
            self.assertEqual(service._account_save_batch_depth, 0)

    def test_refresh_accounts_batch_timeout_finishes_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            tokens = [f"token-{index}" for index in range(4)]
            service.add_accounts(tokens)
            service._MAX_REFRESH_BATCH_SECONDS = 0.05
            release = threading.Event()

            def blocked_refresh(token: str, *args, **kwargs):
                release.wait(timeout=2)
                return service.get_account(token)

            service.fetch_remote_info = blocked_refresh
            started = time.monotonic()
            with patch("services.account_service.log_service.add"):
                result = service.refresh_accounts(tokens, progress_id="batch-timeout")
            elapsed = time.monotonic() - started
            release.set()

            progress = service.get_refresh_progress("batch-timeout")
            self.assertLess(elapsed, 1.0)
            self.assertEqual(len(result["errors"]), len(tokens))
            self.assertIsNotNone(progress)
            self.assertTrue(progress["done"])
            self.assertEqual(progress["processed"], len(tokens))

    def test_scheduled_free_pool_check_uses_bounded_shared_refresh(self) -> None:
        original_settings = copy.deepcopy(config.data.get("free_account_cleanup"))
        config.data["free_account_cleanup"] = {
            "enabled": True,
            "action": "mark_abnormal",
            "failure_threshold": 2,
        }
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([
                    {"access_token": "free-token", "type": "free", "status": "正常", "quota": 1}
                ])
                expected = {
                    "refreshed": 1,
                    "errors": [],
                    "items": service.list_accounts(),
                    "relogined": 0,
                }

                with patch.object(service, "refresh_accounts", return_value=expected) as refresh:
                    result = service.refresh_normal_free_accounts(event="scheduled-free-check")

                refresh.assert_called_once_with(
                    ["free-token"],
                    defer_invalid_removal=False,
                    cleanup_action="mark_abnormal",
                    event="scheduled-free-check",
                )
                self.assertEqual(result["checked"], 1)
        finally:
            if original_settings is None:
                config.data.pop("free_account_cleanup", None)
            else:
                config.data["free_account_cleanup"] = original_settings

    def test_fetch_remote_info_closes_temporary_backend_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
            service.add_accounts(["token-1"])

            with patch("services.openai_backend_api.OpenAIBackendAPI") as backend_class:
                backend = backend_class.return_value
                backend.get_user_info.return_value = {"status": "姝ｅ父", "quota": 5}

                result = service.fetch_remote_info("token-1")

            self.assertEqual(result["quota"], 5)
            backend.get_user_info.assert_called_once_with(request_workers=1, timeout_secs=20.0)
            backend.close.assert_called_once_with()

    def test_get_user_info_uses_one_shared_deadline(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "token"
        backend._get_me = MagicMock(return_value={"email": "user@example.com", "id": "user-1"})
        backend._get_conversation_init = MagicMock(return_value={"limits_progress": []})
        backend._get_default_account = MagicMock(return_value={"plan_type": "free"})

        with patch("services.openai_backend_api.time.monotonic", side_effect=[0.0, 0.0, 30.0, 44.0]):
            result = backend.get_user_info(request_workers=1, timeout_secs=45.0)

        self.assertEqual(result["email"], "user@example.com")
        backend._get_me.assert_called_once_with(20.0)
        backend._get_conversation_init.assert_called_once_with(15.0)
        backend._get_default_account.assert_called_once_with(1.0)

    def test_free_cleanup_verifies_after_failure_threshold_and_marks_invalid_account(self) -> None:
        original_settings = copy.deepcopy(config.data.get("free_account_cleanup"))
        original_auto_remove = config.data.get("auto_remove_invalid_accounts")
        config.data["free_account_cleanup"] = {
            "enabled": True,
            "failure_threshold": 2,
            "action": "mark_abnormal",
        }
        config.data["auto_remove_invalid_accounts"] = False
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "invalid-token", "type": "free", "status": "正常", "quota": 5}])

                service.mark_image_result("invalid-token", success=False)
                below_threshold = service.verify_free_account_after_image_failure(
                    "invalid-token",
                    "image_generation_error",
                    "first failure",
                )

                self.assertFalse(below_threshold["checked"])
                self.assertEqual(service.get_account("invalid-token")["status"], "正常")

                service.mark_image_result("invalid-token", success=False)
                with patch(
                    "services.openai_backend_api.OpenAIBackendAPI.get_user_info",
                    side_effect=InvalidAccessTokenError("token invalidated (/backend-api/me)"),
                ):
                    result = service.verify_free_account_after_image_failure(
                        "invalid-token",
                        "image_generation_error",
                        "second failure",
                    )

                account = service.get_account("invalid-token")
                self.assertTrue(result["checked"])
                self.assertIsNotNone(account)
                self.assertEqual(account["status"], "异常")
                self.assertEqual(account["quota"], 0)
        finally:
            if original_settings is None:
                config.data.pop("free_account_cleanup", None)
            else:
                config.data["free_account_cleanup"] = original_settings
            if original_auto_remove is None:
                config.data.pop("auto_remove_invalid_accounts", None)
            else:
                config.data["auto_remove_invalid_accounts"] = original_auto_remove

    def test_free_cleanup_does_not_touch_paid_accounts(self) -> None:
        original_settings = copy.deepcopy(config.data.get("free_account_cleanup"))
        config.data["free_account_cleanup"] = {
            "enabled": True,
            "failure_threshold": 1,
            "action": "delete",
        }
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                service = AccountService(JSONStorageBackend(Path(tmp_dir) / "accounts.json"))
                service.add_account_items([{"access_token": "plus-token", "type": "Plus", "status": "正常", "quota": 5}])
                service.mark_image_result("plus-token", success=False)

                result = service.verify_free_account_after_image_failure(
                    "plus-token",
                    "image_generation_error",
                    "failure",
                    force=True,
                )

                self.assertFalse(result["checked"])
                self.assertIsNotNone(service.get_account("plus-token"))
        finally:
            if original_settings is None:
                config.data.pop("free_account_cleanup", None)
            else:
                config.data["free_account_cleanup"] = original_settings


class TokenLogTests(unittest.TestCase):
    def test_anonymize_token_hides_raw_value(self) -> None:
        token = "super-secret-token"
        token_ref = anonymize_token(token)

        self.assertTrue(token_ref.startswith("token:"))
        self.assertNotIn(token, token_ref)


class AuthServiceTests(unittest.TestCase):
    def test_daily_request_quota_counts_only_successful_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice", daily_request_limit=2)
            identity = service.authenticate(raw_key)
            self.assertIsNotNone(identity)

            service.reserve_daily_request(identity, "failed-request")
            service.finish_daily_request(identity, "failed-request", success=False)
            service.reserve_daily_request(identity, "success-1")
            service.finish_daily_request(identity, "success-1", success=True)
            service.reserve_daily_request(identity, "success-2")
            service.finish_daily_request(identity, "success-2", success=True)

            current = service.list_keys(role="user")[0]
            self.assertEqual(current["daily_request_used"], 2)
            self.assertEqual(current["daily_request_remaining"], 0)
            with self.assertRaises(DailyRequestQuotaExceeded):
                service.reserve_daily_request(identity, "blocked-request")
            self.assertFalse(service.finish_daily_request(identity, "success-2", success=True))

    def test_daily_request_quota_resets_on_new_configured_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice", daily_request_limit=1)
            identity = service.authenticate(raw_key)
            self.assertIsNotNone(identity)
            service.reserve_daily_request(identity, "success")
            service.finish_daily_request(identity, "success", success=True)
            with service._lock:
                index, stored = service._find_item_locked(str(item["id"]))
                previous = dict(stored)
                previous["daily_request_date"] = "2000-01-01"
                service._items[index] = previous

            service.reserve_daily_request(identity, "next-day")
            service.finish_daily_request(identity, "next-day", success=True)

            current = service.list_keys(role="user")[0]
            self.assertEqual(current["daily_request_used"], 1)
            self.assertEqual(current["daily_request_remaining"], 0)

    def test_image_request_limit_is_per_user_key_and_admin_is_unlimited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            _, raw_key = service.create_key(role="user", name="Alice", image_request_limit=3)
            identity = service.authenticate(raw_key)
            self.assertIsNotNone(identity)

            service.validate_image_request(identity, 3)
            with self.assertRaises(ImageRequestLimitExceeded):
                service.validate_image_request(identity, 4)
            service.validate_image_request({"id": "admin", "role": "admin"}, 100)

    def test_create_authenticate_disable_and_delete_user_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))

            item, raw_key = service.create_key(role="user", name="Alice")

            self.assertEqual(item["role"], "user")
            self.assertEqual(item["name"], "Alice")
            self.assertTrue(item["enabled"])
            self.assertTrue(raw_key.startswith("sk-"))

            authed = service.authenticate(raw_key)
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertEqual(authed["role"], "user")
            self.assertIsNotNone(authed["last_used_at"])

            updated = service.update_key(item["id"], {"enabled": False}, role="user")
            self.assertIsNotNone(updated)
            self.assertFalse(updated["enabled"])
            self.assertIsNone(service.authenticate(raw_key))

            self.assertTrue(service.delete_key(item["id"], role="user"))
            self.assertFalse(service.delete_key(item["id"], role="user"))
            self.assertEqual(service.list_keys(role="user"), [])

    def test_authenticate_ignores_last_used_save_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            def fail_save() -> None:
                raise OSError("disk unavailable")

            service._save = fail_save

            authed = service.authenticate(raw_key)

            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])
            self.assertIsNotNone(authed["last_used_at"])

    def test_update_user_key_replaces_raw_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            item, raw_key = service.create_key(role="user", name="Alice")

            updated = service.update_key(item["id"], {"key": "sk-user-custom-key"}, role="user")

            self.assertIsNotNone(updated)
            self.assertIsNone(service.authenticate(raw_key))

            authed = service.authenticate("sk-user-custom-key")
            self.assertIsNotNone(authed)
            self.assertEqual(authed["id"], item["id"])

    def test_user_key_name_must_be_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = AuthService(JSONStorageBackend(Path(tmp_dir) / "accounts.json", Path(tmp_dir) / "auth_keys.json"))
            first, _ = service.create_key(role="user", name="Alice")
            second, _ = service.create_key(role="user", name="Bob")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.create_key(role="user", name="Alice")

            with self.assertRaisesRegex(ValueError, "这个名称已经在使用中了"):
                service.update_key(second["id"], {"name": "Alice"}, role="user")

            updated = service.update_key(first["id"], {"name": "Alice"}, role="user")
            self.assertIsNotNone(updated)
            self.assertEqual(updated["name"], "Alice")


if __name__ == "__main__":
    unittest.main()
