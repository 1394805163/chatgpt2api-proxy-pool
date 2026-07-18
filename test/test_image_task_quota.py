from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from services.auth_service import AuthService, DailyRequestQuotaExceeded, ImageRequestLimitExceeded
from services.image_task_service import ImageTaskService
from services.storage.json_storage import JSONStorageBackend


def wait_for_status(
    service: ImageTaskService,
    identity: dict[str, object],
    task_id: str,
    status: str,
    timeout: float = 2.0,
) -> dict[str, object]:
    deadline = time.time() + timeout
    last: dict[str, object] | None = None
    while time.time() < deadline:
        items = service.list_tasks(identity, [task_id])["items"]
        last = items[0] if items else None
        if last and last.get("status") == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {status}: {last}")


def make_identity(
    root: Path,
    *,
    daily_limit: int = 5,
    image_limit: int = 5,
) -> tuple[AuthService, dict[str, object]]:
    auth = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
    _, raw_key = auth.create_key(
        role="user",
        name="image-user",
        daily_request_limit=daily_limit,
        image_request_limit=image_limit,
    )
    identity = auth.authenticate(raw_key)
    assert identity is not None
    return auth, identity


def make_service(path: Path, handler) -> ImageTaskService:
    return ImageTaskService(
        path,
        generation_handler=handler,
        edit_handler=handler,
        retention_days_getter=lambda: 30,
        stale_task_timeout_getter=lambda: 5,
        max_task_duration_getter=lambda: 0.05,
    )


class ImageTaskQuotaTests(unittest.TestCase):
    def test_success_counts_once_and_failure_does_not_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auth, identity = make_identity(root, daily_limit=3)
            calls = 0

            def handler(_payload):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("upstream failed")
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = make_service(root / "tasks.json", handler)
            with mock.patch("services.image_task_service.auth_service", auth):
                service.submit_generation(identity, client_task_id="success", prompt="cat", model="gpt-image-2", size=None)
                wait_for_status(service, identity, "success", "success")
                service.submit_generation(identity, client_task_id="failure", prompt="cat", model="gpt-image-2", size=None)
                wait_for_status(service, identity, "failure", "error")

            item = auth.list_keys(role="user")[0]
            self.assertEqual(item["daily_request_used"], 1)
            self.assertEqual(item["daily_request_remaining"], 2)

    def test_user_tasks_use_180_second_limit_and_admin_uses_global_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _, identity = make_identity(root)
            service = make_service(root / "tasks.json", lambda _payload: {"data": [{"url": "ok"}]})

            self.assertEqual(service._max_task_duration(identity), 180.0)
            self.assertEqual(service._max_task_duration({"id": "admin", "role": "admin"}), 0.05)

    def test_selected_account_email_is_kept_as_internal_task_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            def handler(payload):
                payload["progress_callback"]("account_email:pool-user@example.com")
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = make_service(root / "tasks.json", handler)
            service.submit_generation(
                {"id": "admin", "role": "admin"},
                client_task_id="account-metadata",
                prompt="cat",
                model="gpt-image-2",
                size=None,
            )
            wait_for_status(service, {"id": "admin", "role": "admin"}, "account-metadata", "success")

            reloaded = make_service(root / "tasks.json", handler)
            with reloaded._lock:
                task = reloaded._tasks["admin:account-metadata"]
                self.assertEqual(task["account_email"], "pool-user@example.com")
            public = reloaded.list_tasks({"id": "admin", "role": "admin"}, ["account-metadata"])["items"][0]
            self.assertNotIn("account_email", public)

    def test_active_image_task_limit_cannot_be_bypassed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auth, identity = make_identity(root, image_limit=1)
            release = threading.Event()

            def handler(_payload):
                release.wait(1)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = make_service(root / "tasks.json", handler)
            with mock.patch("services.image_task_service.auth_service", auth):
                service.submit_generation(identity, client_task_id="first", prompt="cat", model="gpt-image-2", size=None)
                wait_for_status(service, identity, "first", "running")
                with self.assertRaises(ImageRequestLimitExceeded):
                    service.submit_generation(identity, client_task_id="second", prompt="cat", model="gpt-image-2", size=None)
                release.set()
                wait_for_status(service, identity, "first", "success")

            self.assertEqual(auth.list_keys(role="user")[0]["daily_request_used"], 1)

    def test_resume_poll_reserves_and_counts_a_new_successful_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auth, identity = make_identity(root, daily_limit=1)
            service = make_service(root / "tasks.json", lambda _payload: {"data": []})
            key = f"{identity['id']}:resume-me"
            now = time.time()
            with service._lock:
                service._tasks[key] = {
                    "id": "resume-me",
                    "owner_id": identity["id"],
                    "status": "error",
                    "mode": "generate",
                    "model": "gpt-image-2",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "created_ts": now,
                    "updated_ts": now,
                    "conversation_id": "conversation-1",
                    "base_url": "http://local.test",
                    "error": "图片任务超时",
                }
                service._save_locked()

            backend = mock.MagicMock()
            backend._poll_image_results.return_value = (["file-1"], [])
            backend.resolve_conversation_image_urls.return_value = ["https://example.test/source.png"]
            backend.download_image_bytes.return_value = [b"png"]
            with (
                mock.patch("services.image_task_service.auth_service", auth),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", return_value=backend),
                mock.patch(
                    "services.protocol.conversation.format_image_result",
                    return_value={"data": [{"url": "http://example.test/resumed.png"}]},
                ) as format_result,
            ):
                service.resume_poll(identity, "resume-me", 30)
                task = wait_for_status(service, identity, "resume-me", "success")

            self.assertEqual(task["data"][0]["url"], "http://example.test/resumed.png")
            self.assertEqual(format_result.call_args.args[2], "url")
            self.assertEqual(format_result.call_args.args[3], "http://local.test")
            self.assertEqual(auth.list_keys(role="user")[0]["daily_request_used"], 1)

    def test_resume_poll_failure_releases_quota_and_exhausted_quota_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auth, identity = make_identity(root, daily_limit=1)
            service = make_service(root / "tasks.json", lambda _payload: {"data": []})

            def add_timeout_task(task_id: str) -> None:
                now = time.time()
                with service._lock:
                    service._tasks[f"{identity['id']}:{task_id}"] = {
                        "id": task_id,
                        "owner_id": identity["id"],
                        "status": "error",
                        "mode": "generate",
                        "model": "gpt-image-2",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "created_ts": now,
                        "updated_ts": now,
                        "conversation_id": "conversation-1",
                        "error": "图片任务超时",
                    }
                    service._save_locked()

            add_timeout_task("resume-fails")
            backend = mock.MagicMock()
            backend._poll_image_results.side_effect = RuntimeError("still unavailable")
            with (
                mock.patch("services.image_task_service.auth_service", auth),
                mock.patch("services.openai_backend_api.OpenAIBackendAPI", return_value=backend),
            ):
                service.resume_poll(identity, "resume-fails", 30)
                wait_for_status(service, identity, "resume-fails", "error")
            self.assertEqual(auth.list_keys(role="user")[0]["daily_request_used"], 0)

            auth.reserve_daily_request(identity, "consume-limit")
            auth.finish_daily_request(identity, "consume-limit", success=True)
            add_timeout_task("resume-blocked")
            with mock.patch("services.image_task_service.auth_service", auth):
                with self.assertRaises(DailyRequestQuotaExceeded):
                    service.resume_poll(identity, "resume-blocked", 30)


if __name__ == "__main__":
    unittest.main()
