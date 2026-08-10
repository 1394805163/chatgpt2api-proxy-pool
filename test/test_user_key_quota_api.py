from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.accounts as accounts_api
from services.auth_service import AuthService
from services.storage.json_storage import JSONStorageBackend


class UserKeyQuotaApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.auth = AuthService(JSONStorageBackend(root / "accounts.json", root / "auth_keys.json"))
        self.auth_patch = mock.patch.object(accounts_api, "auth_service", self.auth)
        self.admin_patch = mock.patch.object(
            accounts_api,
            "require_admin",
            return_value={"id": "admin", "role": "admin"},
        )
        self.auth_patch.start()
        self.admin_patch.start()
        self.addCleanup(self.auth_patch.stop)
        self.addCleanup(self.admin_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)
        app = FastAPI()
        app.include_router(accounts_api.create_router())
        self.client = TestClient(app)

    def test_create_and_update_user_key_quota_fields(self) -> None:
        created = self.client.post(
            "/api/auth/users",
            headers={"Authorization": "Bearer admin"},
            json={
                "name": "limited-user",
                "daily_request_limit": 12,
                "image_request_limit": 4,
            },
        )

        self.assertEqual(created.status_code, 200, created.text)
        item = created.json()["item"]
        self.assertEqual(item["daily_request_limit"], 12)
        self.assertEqual(item["daily_request_used"], 0)
        self.assertEqual(item["daily_request_remaining"], 12)
        self.assertEqual(item["image_request_limit"], 4)

        updated = self.client.post(
            f"/api/auth/users/{item['id']}",
            headers={"Authorization": "Bearer admin"},
            json={"daily_request_limit": 20, "image_request_limit": 7},
        )

        self.assertEqual(updated.status_code, 200, updated.text)
        updated_item = updated.json()["item"]
        self.assertEqual(updated_item["daily_request_limit"], 20)
        self.assertEqual(updated_item["image_request_limit"], 7)

        listed = self.client.get("/api/auth/users", headers={"Authorization": "Bearer admin"})
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual(listed.json()["items"][0]["daily_request_remaining"], 20)


if __name__ == "__main__":
    unittest.main()
