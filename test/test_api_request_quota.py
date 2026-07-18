from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai_api
import api.support as api_support
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend


class ApiRequestQuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.auth = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
        _, raw_key = self.auth.create_key(
            role="user",
            name="api-user",
            daily_request_limit=2,
            image_request_limit=2,
        )
        self.identity = self.auth.authenticate(raw_key)
        assert self.identity is not None

        self.patches = [
            mock.patch.object(ai_api, "require_identity", return_value=self.identity),
            mock.patch.object(ai_api, "filter_or_log", mock.AsyncMock()),
            mock.patch("services.auth_service.auth_service", self.auth),
            mock.patch.object(api_support, "auth_service", self.auth),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temp_dir.cleanup)

        app = FastAPI()
        app.include_router(ai_api.create_router())
        self.client = TestClient(app)

    def test_v1_successes_count_failures_do_not_and_models_are_free(self) -> None:
        handler = mock.MagicMock(
            side_effect=[
                {"id": "first", "choices": []},
                RuntimeError("upstream failed"),
                {"id": "second", "choices": []},
            ]
        )
        with (
            mock.patch.object(ai_api.openai_v1_chat_complete, "handle", handler),
            mock.patch.object(ai_api.openai_v1_models, "list_models", return_value={"object": "list", "data": []}),
        ):
            models = self.client.get("/v1/models", headers={"Authorization": "Bearer user"})
            first = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer user"},
                json={"model": "auto", "messages": [{"role": "user", "content": "one"}]},
            )
            failed = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer user"},
                json={"model": "auto", "messages": [{"role": "user", "content": "fail"}]},
            )
            second = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer user"},
                json={"model": "auto", "messages": [{"role": "user", "content": "two"}]},
            )
            blocked = self.client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer user"},
                json={"model": "auto", "messages": [{"role": "user", "content": "three"}]},
            )

        self.assertEqual(models.status_code, 200, models.text)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(failed.status_code, 502, failed.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(blocked.status_code, 429, blocked.text)
        self.assertEqual(handler.call_count, 3)
        item = self.auth.list_keys(role="user")[0]
        self.assertEqual(item["daily_request_used"], 2)
        self.assertEqual(item["daily_request_remaining"], 0)

    def test_direct_v1_image_request_cannot_bypass_key_image_limit(self) -> None:
        with mock.patch.object(ai_api.openai_v1_image_generations, "handle") as handler:
            response = self.client.post(
                "/v1/images/generations",
                headers={"Authorization": "Bearer user"},
                json={"model": "gpt-image-2", "prompt": "three cats", "n": 3},
            )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()["detail"]["limit"], 2)
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
