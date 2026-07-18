from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import Response
from fastapi.testclient import TestClient

import api.app as app_module
import api.system as system_module


class HeadRouteTests(unittest.TestCase):
    def test_frontend_routes_support_head_prefetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            index = Path(tmp_dir) / "index.html"
            index.write_text("<html></html>", encoding="utf-8")
            with mock.patch.object(app_module, "resolve_web_asset", return_value=index):
                response = TestClient(app_module.create_app()).head("/image/")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, b"")

    def test_generated_image_head_keeps_cache_headers(self) -> None:
        image_response = Response(
            content=b"image",
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=7200, immutable"},
        )
        with mock.patch.object(system_module, "get_image_response", return_value=image_response):
            response = TestClient(app_module.create_app()).head("/images/generated.png")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["cache-control"], "public, max-age=7200, immutable")
        self.assertEqual(response.content, b"")


if __name__ == "__main__":
    unittest.main()
