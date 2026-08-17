from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.system as system_module


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}


class FakeLogService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return []


class LogsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_log_service = FakeLogService()
        self.patchers = [
            mock.patch.object(system_module, "require_admin", lambda _authorization: {"role": "admin"}),
            mock.patch.object(system_module, "log_service", self.fake_log_service),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(system_module.create_router("9.9.9-test"))
        self.client = TestClient(app)

    def test_date_filtered_call_logs_request_all_items_and_collapse_failures(self) -> None:
        response = self.client.get(
            "/api/logs?type=call&start_date=2026-07-05&end_date=2026-07-05",
            headers=AUTH_HEADERS,
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.fake_log_service.calls,
            [
                {
                    "type": "call",
                    "start_date": "2026-07-05",
                    "end_date": "2026-07-05",
                    "limit": None,
                    "collapse_image_failures": True,
                    "display_timezone": "Asia/Shanghai",
                }
            ],
        )

    def test_unfiltered_call_logs_keep_default_limit_and_collapse_failures(self) -> None:
        response = self.client.get("/api/logs?type=call", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.fake_log_service.calls,
            [
                {
                    "type": "call",
                    "start_date": "",
                    "end_date": "",
                    "limit": 200,
                    "collapse_image_failures": True,
                    "display_timezone": "Asia/Shanghai",
                }
            ],
        )

    def test_unfiltered_non_call_logs_keep_default_limit_without_collapse(self) -> None:
        response = self.client.get("/api/logs?type=account", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            self.fake_log_service.calls,
            [
                {
                    "type": "account",
                    "start_date": "",
                    "end_date": "",
                    "limit": 200,
                    "collapse_image_failures": False,
                    "display_timezone": "Asia/Shanghai",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
