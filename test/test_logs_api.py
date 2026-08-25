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

    def test_v183_date_filtered_call_logs_are_bounded_and_collapse_failures(self) -> None:
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
                    "limit": 100,
                    "cursor": "",
                    "collapse_image_failures": True,
                    "display_timezone": "Asia/Shanghai",
                }
            ],
        )

    def test_v183_log_limit_cannot_exceed_page_bound(self) -> None:
        response = self.client.get("/api/logs?limit=201", headers=AUTH_HEADERS)
        self.assertEqual(response.status_code, 422, response.text)

    def test_log_and_image_limits_are_bounded_independently(self) -> None:
        self.assertEqual(self.client.get("/api/logs?limit=101", headers=AUTH_HEADERS).status_code, 422)
        self.assertEqual(self.client.get("/api/images?limit=51", headers=AUTH_HEADERS).status_code, 422)

    def test_v183_image_index_read_is_offloaded_from_the_event_loop(self) -> None:
        calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def fake_list_images(_base_url: str, *, start_date: str = "", end_date: str = "", limit: int = 50, cursor: str = ""):
            return {"items": [], "groups": [], "range": [start_date, end_date], "limit": limit, "cursor": cursor}

        async def fake_run_in_threadpool(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        with (
            mock.patch.object(system_module, "list_images", side_effect=fake_list_images) as image_list,
            mock.patch.object(system_module, "run_in_threadpool", side_effect=fake_run_in_threadpool),
        ):
            response = self.client.get(
                "/api/images?start_date=2026-08-01&end_date=2026-08-02",
                headers=AUTH_HEADERS,
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["range"], ["2026-08-01", "2026-08-02"])
        self.assertEqual(response.json()["limit"], 50)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], image_list)
        self.assertEqual(calls[0][2], {"start_date": "2026-08-01", "end_date": "2026-08-02", "limit": 50, "cursor": ""})

    def test_log_reads_are_offloaded_from_the_event_loop(self) -> None:
        calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        async def fake_run_in_threadpool(func, *args, **kwargs):
            calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        with mock.patch.object(system_module, "run_in_threadpool", side_effect=fake_run_in_threadpool):
            response = self.client.get("/api/logs?type=account", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0][0], "__name__", ""), "list")
        self.assertIs(getattr(calls[0][0], "__self__", None), self.fake_log_service)

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
                    "limit": 100,
                    "cursor": "",
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
                    "limit": 100,
                    "cursor": "",
                    "collapse_image_failures": False,
                    "display_timezone": "Asia/Shanghai",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
