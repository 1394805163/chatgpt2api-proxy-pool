from __future__ import annotations

import json
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from services.config import config
from services.image_task_service import ImageTaskService


OWNER = {"id": "owner-1", "name": "Owner", "role": "admin"}
OTHER_OWNER = {"id": "owner-2", "name": "Other", "role": "user"}


def wait_for_task(service: ImageTaskService, identity: dict[str, object], task_id: str, status: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        result = service.list_tasks(identity, [task_id])
        last = (result.get("items") or [None])[0]
        if last and last.get("status") == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not reach {status}, last={last}")


class ImageTaskServiceTests(unittest.TestCase):
    def make_service(self, path: Path, handler=None) -> ImageTaskService:
        return ImageTaskService(
            path,
            generation_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/image.png"}]}),
            edit_handler=handler or (lambda _payload: {"data": [{"url": "http://example.test/edit.png"}]}),
            retention_days_getter=lambda: 30,
        )

    def test_list_tasks_keeps_live_running_task_until_handler_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            block = threading.Event()

            def handler(_payload):
                block.wait(1)
                return {"data": [{"url": "http://example.test/late.png"}]}

            service = ImageTaskService(
                path,
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                stale_task_timeout_getter=lambda: 0.05,
            )
            service.submit_generation(
                OWNER,
                client_task_id="stale-running-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "stale-running-task", "running")
            time.sleep(0.08)
            result = service.list_tasks(OWNER, ["stale-running-task"])

            self.assertEqual(result["items"][0]["status"], "running")
            self.assertNotIn("error", result["items"][0])

            block.set()
            task = wait_for_task(service, OWNER, "stale-running-task", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/late.png")

    def test_live_task_stops_at_total_duration_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            cancel_seen = threading.Event()

            def handler(payload):
                cancel_event = payload["cancel_event"]
                if cancel_event.wait(1):
                    cancel_seen.set()
                release.wait(1)
                return {"data": [{"url": "http://example.test/too-late.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                stale_task_timeout_getter=lambda: 5,
                max_task_duration_getter=lambda: 0.05,
            )
            service.submit_generation(
                OWNER,
                client_task_id="deadline-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            task = wait_for_task(service, OWNER, "deadline-task", "error")
            self.assertIn("总时限", task["error"])
            self.assertTrue(cancel_seen.wait(0.5))

            release.set()
            time.sleep(0.05)
            task = service.list_tasks(OWNER, ["deadline-task"])["items"][0]
            self.assertEqual(task["status"], "error")
            self.assertEqual(task["data"], [])

    def test_list_tasks_marks_orphaned_running_task_as_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                retention_days_getter=lambda: 30,
                stale_task_timeout_getter=lambda: 0.05,
            )
            now = time.time()
            with service._lock:
                service._tasks["owner-1:orphaned-running-task"] = {
                    "id": "orphaned-running-task",
                    "owner_id": "owner-1",
                    "status": "running",
                    "mode": "generate",
                    "model": "gpt-image-2",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "created_ts": now - 1,
                    "updated_ts": now - 1,
                    "started_ts": now - 1,
                }

            result = service.list_tasks(OWNER, ["orphaned-running-task"])

            self.assertEqual(result["items"][0]["status"], "error")
            self.assertIn("超时", result["items"][0]["error"])

    def test_default_stale_timeout_tracks_total_task_timeout_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            now = time.time()
            with service._lock:
                service._tasks["owner-1:runtime-stale-task"] = {
                    "id": "runtime-stale-task",
                    "owner_id": "owner-1",
                    "status": "running",
                    "mode": "generate",
                    "model": "gpt-image-2",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:30Z",
                    "created_ts": now - 240,
                    "updated_ts": now - 230,
                    "started_ts": now - 230,
                }

            with mock.patch.dict(
                config.data,
                {
                    "image_poll_timeout_secs": 70,
                    "image_task_timeout_secs": 150,
                    "user_image_task_timeout_secs": 180,
                    "image_poll_initial_wait_secs": 10,
                    "image_poll_interval_secs": 10,
                },
            ):
                result = service.list_tasks(OWNER, ["runtime-stale-task"])

            self.assertEqual(result["items"][0]["status"], "error")
            self.assertIn("超时", result["items"][0]["error"])

    def test_duplicate_submit_uses_existing_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            calls = 0

            def handler(_payload):
                nonlocal calls
                calls += 1
                time.sleep(0.05)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = self.make_service(Path(tmp_dir) / "image_tasks.json", handler)
            first = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            second = service.submit_generation(
                OWNER,
                client_task_id="task-1",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            self.assertEqual(first["id"], "task-1")
            self.assertEqual(second["id"], "task-1")
            task = wait_for_task(service, OWNER, "task-1", "success")
            self.assertEqual(task["data"][0]["url"], "http://example.test/image.png")
            self.assertEqual(calls, 1)

    def test_different_owner_cannot_query_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = self.make_service(Path(tmp_dir) / "image_tasks.json")
            service.submit_generation(
                OWNER,
                client_task_id="private-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )

            wait_for_task(service, OWNER, "private-task", "success")
            result = service.list_tasks(OTHER_OWNER, ["private-task"])

            self.assertEqual(result["items"], [])
            self.assertEqual(result["missing_ids"], ["private-task"])

    def test_success_task_persists_to_new_service_instance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            service = self.make_service(path)
            service.submit_generation(
                OWNER,
                client_task_id="persisted-task",
                prompt="cat",
                model="gpt-image-2",
                size=None,
                base_url="http://local.test",
            )
            wait_for_task(service, OWNER, "persisted-task", "success")

            reloaded = self.make_service(path)
            result = reloaded.list_tasks(OWNER, ["persisted-task"])

            self.assertEqual(result["missing_ids"], [])
            self.assertEqual(result["items"][0]["status"], "success")
            self.assertEqual(result["items"][0]["data"][0]["url"], "http://example.test/image.png")

    def test_startup_marks_unfinished_tasks_as_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "image_tasks.json"
            path.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "id": "queued-task",
                                "owner_id": "owner-1",
                                "status": "queued",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                            {
                                "id": "running-task",
                                "owner_id": "owner-1",
                                "status": "running",
                                "mode": "generate",
                                "model": "gpt-image-2",
                                "created_at": "2099-01-01 00:00:00",
                                "updated_at": "2099-01-01 00:00:00",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            service = self.make_service(path)
            result = service.list_tasks(OWNER, ["queued-task", "running-task"])

            self.assertEqual([item["status"] for item in result["items"]], ["error", "error"])
            self.assertTrue(all("已中断" in item.get("error", "") for item in result["items"]))


if __name__ == "__main__":
    unittest.main()
