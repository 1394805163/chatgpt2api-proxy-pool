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

    def test_timed_out_unresponsive_handler_does_not_permanently_block_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release_stuck = threading.Event()
            second_started = threading.Event()

            def handler(payload):
                if payload["client_task_id"] == "stuck":
                    release_stuck.wait(2)
                else:
                    second_started.set()
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                stale_task_timeout_getter=lambda: 5,
                max_task_duration_getter=lambda: 0.05,
                global_concurrency_getter=lambda: 1,
                per_owner_concurrency_getter=lambda: 1,
            )
            service.submit_generation(OWNER, client_task_id="stuck", prompt="cat", model="gpt-image-2", size=None)
            wait_for_task(service, OWNER, "stuck", "error")

            service.submit_generation(OWNER, client_task_id="after-stuck", prompt="cat", model="gpt-image-2", size=None)
            self.assertTrue(second_started.wait(0.5))
            wait_for_task(service, OWNER, "after-stuck", "success")

            with service._lock:
                stuck_thread = service._threads.get("owner-1:stuck")
                self.assertIsNotNone(stuck_thread)
                self.assertTrue(stuck_thread.is_alive())
            release_stuck.set()

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

    def test_scheduler_limits_global_and_per_owner_concurrency_fairly(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            state_lock = threading.Lock()
            running = 0
            max_running = 0
            running_by_owner: dict[str, int] = {}
            max_running_by_owner: dict[str, int] = {}
            first_wave: list[str] = []

            def handler(payload):
                nonlocal running, max_running
                owner_id = str(payload["client_task_id"]).rsplit("-", 1)[0]
                with state_lock:
                    running += 1
                    running_by_owner[owner_id] = running_by_owner.get(owner_id, 0) + 1
                    max_running = max(max_running, running)
                    max_running_by_owner[owner_id] = max(
                        max_running_by_owner.get(owner_id, 0),
                        running_by_owner[owner_id],
                    )
                    if len(first_wave) < 3:
                        first_wave.append(owner_id)
                release.wait(2)
                with state_lock:
                    running -= 1
                    running_by_owner[owner_id] -= 1
                return {"data": [{"url": f"http://example.test/{owner_id}.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                global_concurrency_getter=lambda: 3,
                per_owner_concurrency_getter=lambda: 1,
            )
            owners = [
                {"id": f"owner-{index}", "name": f"Owner {index}", "role": "user"}
                for index in range(1, 4)
            ]
            quota = mock.Mock()
            quota.reserve_daily_request.return_value = False
            quota_patcher = mock.patch("services.image_task_service.auth_service", quota)
            quota_patcher.start()
            self.addCleanup(quota_patcher.stop)
            for round_index in range(2):
                for owner in owners:
                    task_id = f"{owner['id']}-{round_index}"
                    service.submit_generation(
                        owner,
                        client_task_id=task_id,
                        prompt="cat",
                        model="gpt-image-2",
                        size=None,
                        base_url="http://local.test",
                    )

            deadline = time.time() + 1
            while time.time() < deadline:
                with state_lock:
                    if running == 3:
                        break
                time.sleep(0.01)

            with state_lock:
                self.assertEqual(running, 3)
                self.assertEqual(set(first_wave), {"owner-1", "owner-2", "owner-3"})
            queued = sum(
                1
                for owner in owners
                for item in service.list_tasks(owner, [f"{owner['id']}-1"])["items"]
                if item["status"] == "queued"
            )
            self.assertEqual(queued, 3)

            release.set()
            for owner in owners:
                for round_index in range(2):
                    wait_for_task(service, owner, f"{owner['id']}-{round_index}", "success")

            self.assertEqual(max_running, 3)
            self.assertTrue(all(value == 1 for value in max_running_by_owner.values()))

    def test_admin_can_fill_global_concurrency_for_ten_image_batch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            state_lock = threading.Lock()
            running = 0
            max_running = 0
            started_task_ids: list[str] = []

            def handler(payload):
                nonlocal running, max_running
                with state_lock:
                    running += 1
                    max_running = max(max_running, running)
                    started_task_ids.append(str(payload["client_task_id"]))
                release.wait(10)
                with state_lock:
                    running -= 1
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                global_concurrency_getter=lambda: 10,
                per_owner_concurrency_getter=lambda: 2,
            )
            for index in range(10):
                service.submit_generation(
                    OWNER,
                    client_task_id=f"admin-batch-{index}",
                    prompt="same prompt",
                    model="gpt-image-2",
                    size=None,
                    base_url="http://local.test",
                )

            deadline = time.time() + 2
            while time.time() < deadline:
                with state_lock:
                    if running == 10:
                        break
                time.sleep(0.01)

            with state_lock:
                observed_running = running
                observed_started = len(set(started_task_ids))
            release.set()
            for index in range(10):
                wait_for_task(service, OWNER, f"admin-batch-{index}", "success")

            self.assertEqual(observed_running, 10)
            self.assertEqual(max_running, 10)
            self.assertEqual(observed_started, 10)

    def test_queued_time_does_not_consume_generation_deadline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release_first = threading.Event()

            def handler(payload):
                if payload["client_task_id"] == "first":
                    release_first.wait(1)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                max_task_duration_getter=lambda: 0.2,
                global_concurrency_getter=lambda: 1,
                per_owner_concurrency_getter=lambda: 1,
            )
            service.submit_generation(
                OWNER,
                client_task_id="first",
                prompt="cat",
                model="gpt-image-2",
                size=None,
            )
            service.submit_generation(
                OWNER,
                client_task_id="second",
                prompt="cat",
                model="gpt-image-2",
                size=None,
            )

            wait_for_task(service, OWNER, "first", "running")
            time.sleep(0.08)
            queued = service.list_tasks(OWNER, ["second"])["items"][0]
            self.assertEqual(queued["status"], "queued")

            release_first.set()
            wait_for_task(service, OWNER, "first", "success")
            second = wait_for_task(service, OWNER, "second", "success")
            self.assertGreaterEqual(second["queue_duration_ms"], 50)

    def test_queued_task_expires_without_starting_the_handler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            started: list[str] = []

            def handler(payload):
                task_id = str(payload["client_task_id"])
                started.append(task_id)
                if task_id == "blocking":
                    release.wait(1)
                return {"data": [{"url": "http://example.test/image.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                max_task_duration_getter=lambda: 1,
                global_concurrency_getter=lambda: 1,
                per_owner_concurrency_getter=lambda: 1,
                queue_timeout_getter=lambda: 0.05,
            )
            for task_id in ("blocking", "expires"):
                service.submit_generation(
                    OWNER,
                    client_task_id=task_id,
                    prompt="cat",
                    model="gpt-image-2",
                    size=None,
                )

            wait_for_task(service, OWNER, "blocking", "running")
            deadline = time.time() + 0.5
            expired_status = "queued"
            while time.time() < deadline:
                with service._lock:
                    expired_status = str(service._tasks["owner-1:expires"]["status"])
                if expired_status == "error":
                    break
                time.sleep(0.01)
            expired = service.list_tasks(OWNER, ["expires"])["items"][0]

            self.assertEqual(expired_status, "error")
            self.assertEqual(expired["status"], "error")
            self.assertIn("排队超过", expired["error"])
            self.assertNotIn("expires", started)
            release.set()
            wait_for_task(service, OWNER, "blocking", "success")

    def test_five_users_can_submit_ten_staggered_tasks_each(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_lock = threading.Lock()
            running = 0
            max_running = 0
            running_by_owner: dict[str, int] = {}
            max_running_by_owner: dict[str, int] = {}

            def handler(payload):
                nonlocal running, max_running
                owner_id = str(payload["client_task_id"]).split(":", 1)[0]
                with state_lock:
                    running += 1
                    running_by_owner[owner_id] = running_by_owner.get(owner_id, 0) + 1
                    max_running = max(max_running, running)
                    max_running_by_owner[owner_id] = max(
                        max_running_by_owner.get(owner_id, 0),
                        running_by_owner[owner_id],
                    )
                time.sleep(0.03)
                with state_lock:
                    running -= 1
                    running_by_owner[owner_id] -= 1
                return {"data": [{"url": f"http://example.test/{owner_id}.png"}]}

            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=handler,
                edit_handler=handler,
                retention_days_getter=lambda: 30,
                max_task_duration_getter=lambda: 2,
                global_concurrency_getter=lambda: 10,
                per_owner_concurrency_getter=lambda: 2,
                queue_timeout_getter=lambda: 10,
            )
            owners = [
                {
                    "id": f"user-{index}",
                    "name": f"User {index}",
                    "role": "user",
                    "image_request_limit": 10,
                }
                for index in range(5)
            ]
            start_gate = threading.Barrier(len(owners) + 1)

            def submit_owner(owner: dict[str, object], owner_index: int) -> None:
                start_gate.wait()
                for task_index in range(10):
                    if task_index:
                        time.sleep(((owner_index + task_index) % 4) * 0.001)
                    service.submit_generation(
                        owner,
                        client_task_id=f"{owner['id']}:{task_index}",
                        prompt="cat",
                        model="gpt-image-2",
                        size=None,
                    )

            submitters = [
                threading.Thread(target=submit_owner, args=(owner, index), daemon=True)
                for index, owner in enumerate(owners)
            ]
            quota = mock.Mock()
            quota.reserve_daily_request.return_value = False
            with mock.patch("services.image_task_service.auth_service", quota):
                for thread in submitters:
                    thread.start()
                start_gate.wait()
                for thread in submitters:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                completed = [
                    wait_for_task(service, owner, f"{owner['id']}:{task_index}", "success", timeout=5)
                    for owner in owners
                    for task_index in range(10)
                ]

            self.assertEqual(len(completed), 50)
            self.assertEqual(max_running, 10)
            self.assertEqual(set(max_running_by_owner), {str(owner["id"]) for owner in owners})
            self.assertTrue(all(value == 2 for value in max_running_by_owner.values()))
            self.assertTrue(any(int(task.get("queue_duration_ms") or 0) > 0 for task in completed))

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

    def test_resume_poll_waits_for_the_same_global_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            resume_started = threading.Event()
            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=lambda _payload: (release.wait(1), {"data": [{"url": "ok"}]})[1],
                retention_days_getter=lambda: 30,
                global_concurrency_getter=lambda: 1,
                per_owner_concurrency_getter=lambda: 1,
                max_task_duration_getter=lambda: 2,
            )
            service.submit_generation(OWNER, client_task_id="blocking", prompt="cat", model="gpt-image-2", size=None)
            wait_for_task(service, OWNER, "blocking", "running")

            resume_owner = {"id": "resume-owner", "name": "Resume", "role": "admin"}
            now = time.time()
            resume_key = "resume-owner:resume-task"
            with service._lock:
                service._tasks[resume_key] = {
                    "id": "resume-task",
                    "owner_id": "resume-owner",
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

            def fake_resume(*_args):
                resume_started.set()
                service._update_task(resume_key, status="success", data=[{"url": "resumed"}], error="")

            service._run_resume_poll = fake_resume
            submitted = service.resume_poll(resume_owner, "resume-task", 30)

            self.assertEqual(submitted["status"], "queued")
            self.assertFalse(resume_started.wait(0.1))
            release.set()
            wait_for_task(service, OWNER, "blocking", "success")
            self.assertTrue(resume_started.wait(0.5))
            wait_for_task(service, resume_owner, "resume-task", "success")

    def test_resume_poll_queue_timeout_starts_from_resume_attempt(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            release = threading.Event()
            resume_started = threading.Event()
            service = ImageTaskService(
                Path(tmp_dir) / "image_tasks.json",
                generation_handler=lambda _payload: (release.wait(1), {"data": [{"url": "ok"}]})[1],
                retention_days_getter=lambda: 30,
                global_concurrency_getter=lambda: 1,
                per_owner_concurrency_getter=lambda: 1,
                queue_timeout_getter=lambda: 0.2,
                max_task_duration_getter=lambda: 2,
            )
            service.submit_generation(OWNER, client_task_id="blocking", prompt="cat", model="gpt-image-2", size=None)
            wait_for_task(service, OWNER, "blocking", "running")

            resume_owner = {"id": "resume-owner", "name": "Resume", "role": "admin"}
            now = time.time()
            resume_key = "resume-owner:old-resume-task"
            with service._lock:
                service._tasks[resume_key] = {
                    "id": "old-resume-task",
                    "owner_id": "resume-owner",
                    "status": "error",
                    "mode": "generate",
                    "model": "gpt-image-2",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "created_ts": now - 3600,
                    "updated_ts": now - 3600,
                    "conversation_id": "conversation-1",
                    "error": "图片任务超时",
                }
                service._save_locked()

            def fake_resume(*_args):
                resume_started.set()
                service._update_task(resume_key, status="success", data=[{"url": "resumed"}], error="")

            service._run_resume_poll = fake_resume
            submitted = service.resume_poll(resume_owner, "old-resume-task", 30)

            self.assertEqual(submitted["status"], "queued")
            time.sleep(0.05)
            queued = service.list_tasks(resume_owner, ["old-resume-task"])["items"][0]
            self.assertEqual(queued["status"], "queued")
            release.set()
            wait_for_task(service, OWNER, "blocking", "success")
            self.assertTrue(resume_started.wait(0.5))
            wait_for_task(service, resume_owner, "old-resume-task", "success")


if __name__ == "__main__":
    unittest.main()
