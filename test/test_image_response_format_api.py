from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai_module


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}
PNG_BYTES = b"fake-png"


class ImageResponseFormatApiTests(unittest.TestCase):
    def setUp(self):
        self.generation_calls: list[dict[str, object]] = []
        self.edit_calls: list[dict[str, object]] = []

        def fake_generation(payload):
            self.generation_calls.append(dict(payload))
            return {"created": 1, "data": [{"url": "/images/generated.png"}]}

        def fake_edit(payload):
            self.edit_calls.append(dict(payload))
            return {"created": 1, "data": [{"url": "/images/edited.png"}]}

        self.generation_patcher = mock.patch.object(ai_module.openai_v1_image_generations, "handle", fake_generation)
        self.edit_patcher = mock.patch.object(ai_module.openai_v1_image_edit, "handle", fake_edit)
        self.filter_patcher = mock.patch.object(ai_module, "filter_or_log", mock.AsyncMock())
        self.identity_patcher = mock.patch.object(
            ai_module,
            "require_identity",
            return_value={"id": "admin", "role": "admin"},
        )
        self.generation_patcher.start()
        self.edit_patcher.start()
        self.filter_patcher.start()
        self.identity_patcher.start()
        self.addCleanup(self.generation_patcher.stop)
        self.addCleanup(self.edit_patcher.stop)
        self.addCleanup(self.filter_patcher.stop)
        self.addCleanup(self.identity_patcher.stop)

        app = FastAPI()
        app.include_router(ai_module.create_router())
        self.client = TestClient(app)

    def test_generation_defaults_to_url(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "默认 URL"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.generation_calls[0]["response_format"], "url")
        self.assertEqual(response.json()["data"][0]["url"], "/images/generated.png")

    def test_generation_accepts_explicit_url(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "显式 URL", "response_format": "url"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.generation_calls[0]["response_format"], "url")

    def test_generation_rejects_base64_response(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "拒绝 Base64", "response_format": "b64_json"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("请求格式错误", response.text)
        self.assertEqual(self.generation_calls, [])

    def test_generation_rejects_unknown_response_format(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "拒绝未知格式", "response_format": "binary"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("请求格式错误", response.text)
        self.assertEqual(self.generation_calls, [])

    def test_edit_defaults_to_url(self):
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            files={"image": ("source.png", PNG_BYTES, "image/png")},
            data={"prompt": "编辑默认 URL"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.edit_calls[0]["response_format"], "url")
        self.assertEqual(response.json()["data"][0]["url"], "/images/edited.png")

    def test_edit_rejects_base64_response(self):
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            files={"image": ("source.png", PNG_BYTES, "image/png")},
            data={"prompt": "编辑拒绝 Base64", "response_format": "b64_json"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("请求格式错误", response.text)
        self.assertEqual(self.edit_calls, [])


if __name__ == "__main__":
    unittest.main()
