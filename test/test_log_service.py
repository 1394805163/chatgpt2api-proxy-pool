from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.log_service import LOG_TYPE_CALL, LogService


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


if __name__ == "__main__":
    unittest.main()
