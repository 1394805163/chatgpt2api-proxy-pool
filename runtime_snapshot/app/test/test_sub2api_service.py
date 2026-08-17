from __future__ import annotations

import unittest
from unittest import mock

from services import sub2api_service


SERVER = {
    "id": "server-1",
    "base_url": "https://sub2api.example.test/",
    "api_key": "admin-key",
}


class FakeResponse:
    def __init__(self, payload: object, *, ok: bool = True, status_code: int = 200) -> None:
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = "response"

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.get_calls: list[tuple[str, dict]] = []
        self.closed = False

    def get(self, url: str, **kwargs) -> FakeResponse:
        self.get_calls.append((url, kwargs))
        return self.response

    def close(self) -> None:
        self.closed = True


class MemoryImportConfig:
    def __init__(self, job: dict) -> None:
        self.job = dict(job)

    def get_import_job(self, _server_id: str) -> dict:
        return dict(self.job)

    def set_import_job(self, _server_id: str, job: dict) -> dict:
        self.job = dict(job)
        return {"import_job": dict(self.job)}


class Sub2APIServiceTests(unittest.TestCase):
    def test_list_remote_accounts_keeps_rows_without_embedded_access_token(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "items": [
                            {
                                "id": 17,
                                "name": "user@example.test",
                                "status": "active",
                                "credentials": {"email": "user@example.test"},
                            }
                        ],
                        "total": 1,
                    },
                }
            )
        )

        with mock.patch.object(sub2api_service, "Session", return_value=session):
            result = sub2api_service.list_remote_accounts(SERVER)

        self.assertEqual([item["id"] for item in result], ["17"])
        self.assertTrue(session.closed)

    def test_batch_export_uses_one_request_and_reports_missing_tokens(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "accounts": [
                            {"id": "1", "credentials": {"access_token": "token-1"}},
                            {"id": "2", "credentials": {}},
                        ]
                    },
                }
            )
        )

        with mock.patch.object(sub2api_service, "Session", return_value=session):
            tokens, errors = sub2api_service._fetch_access_tokens_for_accounts(SERVER, ["1", "2"])

        self.assertEqual(tokens, ["token-1"])
        self.assertEqual(errors, [{"name": "2", "error": "missing access_token"}])
        self.assertEqual(len(session.get_calls), 1)
        url, kwargs = session.get_calls[0]
        self.assertEqual(url, "https://sub2api.example.test/api/v1/admin/accounts/data")
        self.assertEqual(kwargs["params"]["ids"], "1,2")
        self.assertTrue(session.closed)

    def test_partial_batch_import_preserves_job_totals_and_errors(self) -> None:
        config = MemoryImportConfig(
            {
                "job_id": "job-1",
                "status": "pending",
                "created_at": "2026-07-22T00:00:00+00:00",
                "updated_at": "2026-07-22T00:00:00+00:00",
                "total": 2,
                "completed": 0,
                "added": 0,
                "skipped": 0,
                "refreshed": 0,
                "failed": 0,
                "errors": [],
            }
        )
        service = sub2api_service.Sub2APIImportService(config)

        with (
            mock.patch.object(
                sub2api_service,
                "_fetch_access_tokens_for_accounts",
                return_value=(["token-1"], [{"name": "2", "error": "missing access_token"}]),
            ),
            mock.patch.object(
                sub2api_service.account_service,
                "add_accounts",
                return_value={"added": 1, "skipped": 0},
            ) as add_accounts,
            mock.patch.object(
                sub2api_service.account_service,
                "refresh_accounts",
                return_value={"refreshed": 1},
            ),
        ):
            service._run_import("server-1", SERVER, ["1", "2"])

        self.assertEqual(config.job["status"], "completed")
        self.assertEqual(config.job["completed"], 2)
        self.assertEqual(config.job["failed"], 1)
        self.assertEqual(config.job["errors"], [{"name": "2", "error": "missing access_token"}])
        add_accounts.assert_called_once_with(["token-1"], source_type="codex")


if __name__ == "__main__":
    unittest.main()
