from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.app import create_app


class CorsTests(unittest.TestCase):
    def test_preflight_allows_authorization_and_is_cached_for_one_day(self) -> None:
        response = TestClient(create_app()).options(
            "/api/image-tasks?ids=probe",
            headers={
                "Origin": "https://ximage.xtools.fun",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "*")
        self.assertIn("authorization", response.headers.get("access-control-allow-headers", "").lower())
        self.assertEqual(response.headers.get("access-control-max-age"), "86400")


if __name__ == "__main__":
    unittest.main()
