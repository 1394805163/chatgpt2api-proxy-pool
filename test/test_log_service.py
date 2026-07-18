from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.log_service import LOG_TYPE_CALL, LogService
from services.storage.database_storage import DatabaseStorageBackend, LogModel


def append_log(service: LogService, *, log_id: str, time: str, summary: str, detail: dict):
    item = {
        "id": log_id,
        "time": time,
        "type": LOG_TYPE_CALL,
        "summary": summary,
        "detail": detail,
    }
    with service.path.open("a", encoding="utf-8") as file:
        file.write(service._serialize_item(item) + "\n")


class LogServiceTests(unittest.TestCase):
    def test_local_log_retention_applies_time_and_count_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            for index, timestamp in enumerate(
                [
                    "2026-07-15T00:00:00Z",
                    "2099-07-17T00:00:00Z",
                    "2099-07-18T00:00:00Z",
                    "2099-07-19T00:00:00Z",
                ]
            ):
                append_log(
                    service,
                    log_id=f"log-{index}",
                    time=timestamp,
                    summary=f"log {index}",
                    detail={},
                )

            removed = service._prune_file(retention_days=3, max_items=2)

            self.assertEqual(removed, 2)
            self.assertEqual([item["id"] for item in service.list(limit=None)], ["log-3", "log-2"])

    def test_database_log_retention_applies_time_and_count_limits(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            storage = DatabaseStorageBackend(f"sqlite:///{root / 'logs.db'}")
            for index, timestamp in enumerate(
                [
                    "2026-07-15T00:00:00Z",
                    "2099-07-17T00:00:00Z",
                    "2099-07-18T00:00:00Z",
                    "2099-07-19T00:00:00Z",
                ]
            ):
                storage.save_log({
                    "id": f"log-{index}",
                    "time": timestamp,
                    "type": LOG_TYPE_CALL,
                    "summary": f"log {index}",
                    "detail": {},
                })

            removed = storage.prune_logs(retention_days=3, max_items=2)
            session = storage.Session()
            try:
                remaining = [row.id for row in session.query(LogModel).order_by(LogModel.time.desc()).all()]
            finally:
                session.close()
                storage.engine.dispose()

            self.assertEqual(removed, 2)
            self.assertEqual(remaining, ["log-3", "log-2"])

    def test_database_and_file_logs_are_merged_in_time_order(self):
        class FakeStorage:
            def load_logs(self, limit=None, type=""):
                return [{
                    "id": "database-old",
                    "time": "2026-07-18T00:00:00Z",
                    "type": LOG_TYPE_CALL,
                    "summary": "database old",
                    "detail": {},
                }]

        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl", storage_backend=FakeStorage())
            append_log(
                service,
                log_id="file-new",
                time="2026-07-19T00:00:00Z",
                summary="file new",
                detail={},
            )

            self.assertEqual([item["id"] for item in service.list(limit=None)], ["file-new", "database-old"])

    def test_sqlite_backed_logs_survive_service_restart(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            storage = DatabaseStorageBackend(f"sqlite:///{root / 'logs.db'}")
            service = LogService(root / "logs.jsonl", storage_backend=storage)
            service.add(LOG_TYPE_CALL, "persisted", {"status": "success"})
            (root / "logs.jsonl").unlink()

            restarted = LogService(root / "new-logs.jsonl", storage_backend=storage)
            items = restarted.list(limit=None)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["summary"], "persisted")
            storage.engine.dispose()

    def test_database_backed_logs_survive_missing_local_file(self):
        class FakeStorage:
            def __init__(self):
                self.items: list[dict] = []

            def save_log(self, item):
                self.items.append(dict(item))
                return True

            def load_logs(self, limit=None, type=""):
                items = [item for item in reversed(self.items) if not type or item.get("type") == type]
                return items if limit is None else items[:limit]

            def delete_logs(self, ids):
                target = set(ids)
                before = len(self.items)
                self.items = [item for item in self.items if item["id"] not in target]
                return before - len(self.items)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "logs.jsonl"
            storage = FakeStorage()
            service = LogService(path, storage_backend=storage)
            service.add(LOG_TYPE_CALL, "persisted", {"status": "success"})
            path.unlink()

            items = service.list(limit=None)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["summary"], "persisted")

    def test_date_range_can_return_all_matching_logs_without_default_cap(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            for index in range(250):
                append_log(
                    service,
                    log_id=f"call-{index}",
                    time=f"2026-07-05T00:{index % 60:02d}:00Z",
                    summary="call done",
                    detail={"endpoint": "/v1/chat/completions", "status": "success"},
                )

            items = service.list(type=LOG_TYPE_CALL, start_date="2026-07-05", end_date="2026-07-05", limit=None)

            self.assertEqual(len(items), 250)
            self.assertEqual(items[0]["id"], "call-249")
            self.assertEqual(items[-1]["id"], "call-0")

    def test_date_range_matches_display_timezone_day(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            append_log(
                service,
                log_id="local-july-6",
                time="2026-07-05T16:49:50Z",
                summary="image generation failed",
                detail={
                    "endpoint": "/v1/images/generations",
                    "status": "failed",
                    "request_text": "same prompt",
                },
            )

            items = service.list(
                type=LOG_TYPE_CALL,
                start_date="2026-07-06",
                end_date="2026-07-06",
                limit=None,
                display_timezone="Asia/Shanghai",
            )

            self.assertEqual([item["id"] for item in items], ["local-july-6"])

    def test_same_prompt_failed_image_calls_are_collapsed_with_failure_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            for index in range(3):
                append_log(
                    service,
                    log_id=f"failed-image-{index}",
                    time=f"2026-07-05T01:00:0{index}Z",
                    summary="image generation failed",
                    detail={
                        "endpoint": "/v1/images/generations",
                        "status": "failed",
                        "request_text": "same prompt",
                        "error": f"failed {index}",
                    },
                )
            append_log(
                service,
                log_id="failed-other-prompt",
                time="2026-07-05T01:00:05Z",
                summary="image generation failed",
                detail={
                    "endpoint": "/v1/images/generations",
                    "status": "failed",
                    "request_text": "other prompt",
                    "error": "failed other",
                },
            )
            append_log(
                service,
                log_id="successful-image",
                time="2026-07-05T01:00:06Z",
                summary="image generation done",
                detail={
                    "endpoint": "/v1/images/generations",
                    "status": "success",
                    "request_text": "same prompt",
                },
            )

            items = service.list(
                type=LOG_TYPE_CALL,
                start_date="2026-07-05",
                end_date="2026-07-05",
                limit=None,
                collapse_image_failures=True,
            )

            grouped = [item for item in items if item["id"].startswith("group:")]
            self.assertEqual(len(grouped), 1)
            self.assertEqual(grouped[0]["detail"]["failure_count"], 3)
            self.assertEqual(
                grouped[0]["detail"]["grouped_log_ids"],
                ["failed-image-2", "failed-image-1", "failed-image-0"],
            )
            self.assertEqual([item["id"] for item in items], ["successful-image", "failed-other-prompt", grouped[0]["id"]])

    def test_failed_image_calls_without_prompt_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            for index in range(2):
                append_log(
                    service,
                    log_id=f"failed-no-prompt-{index}",
                    time=f"2026-07-05T01:00:0{index}Z",
                    summary="image generation failed",
                    detail={
                        "endpoint": "/v1/images/generations",
                        "status": "failed",
                        "conversation_id": "same-conversation",
                        "error": f"failed {index}",
                    },
                )

            items = service.list(
                type=LOG_TYPE_CALL,
                start_date="2026-07-05",
                end_date="2026-07-05",
                limit=None,
                collapse_image_failures=True,
            )

            self.assertEqual([item["id"] for item in items], ["failed-no-prompt-1", "failed-no-prompt-0"])

    def test_delete_group_id_removes_all_collapsed_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = LogService(Path(tmp_dir) / "logs.jsonl")
            for index in range(3):
                append_log(
                    service,
                    log_id=f"failed-image-{index}",
                    time=f"2026-07-05T01:00:0{index}Z",
                    summary="image generation failed",
                    detail={
                        "endpoint": "/v1/images/generations",
                        "status": "failed",
                        "request_text": "same prompt",
                        "error": f"failed {index}",
                    },
                )
            append_log(
                service,
                log_id="failed-other-prompt",
                time="2026-07-05T01:00:05Z",
                summary="image generation failed",
                detail={
                    "endpoint": "/v1/images/generations",
                    "status": "failed",
                    "request_text": "other prompt",
                    "error": "failed other",
                },
            )

            [grouped] = [
                item
                for item in service.list(type=LOG_TYPE_CALL, limit=None, collapse_image_failures=True)
                if item["id"].startswith("group:")
            ]
            result = service.delete([grouped["id"]])
            remaining = service.list(type=LOG_TYPE_CALL, limit=None)

            self.assertEqual(result, {"removed": 3})
            self.assertEqual([item["id"] for item in remaining], ["failed-other-prompt"])


if __name__ == "__main__":
    unittest.main()
