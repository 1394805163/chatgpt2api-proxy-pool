from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.accounts import create_router


class AccountImportAPITests(unittest.TestCase):
    def test_import_returns_before_background_refresh_finishes(self) -> None:
        refresh_started = threading.Event()
        release_refresh = threading.Event()
        service = MagicMock()
        service.add_accounts.return_value = {
            "added": 2,
            "skipped": 0,
            "items": [
                {"access_token": "token-1"},
                {"access_token": "token-2"},
            ],
        }

        def refresh_accounts(tokens, progress_id, defer_invalid_removal):
            refresh_started.set()
            release_refresh.wait(timeout=2)
            return {"refreshed": len(tokens), "errors": [], "items": service.add_accounts.return_value["items"]}

        service.refresh_accounts.side_effect = refresh_accounts

        app = FastAPI()
        with patch("api.accounts.account_service", service), patch("api.accounts.require_admin"):
            app.include_router(create_router())
            with TestClient(app) as client:
                response = client.post(
                    "/api/accounts",
                    headers={"Authorization": "Bearer test"},
                    json={"tokens": ["token-1", "token-2"]},
                )

                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["added"], 2)
                self.assertEqual(payload["refreshing"], 2)
                self.assertTrue(payload["refresh_progress_id"])
                self.assertTrue(refresh_started.wait(timeout=1))
                release_refresh.set()


if __name__ == "__main__":
    unittest.main()
