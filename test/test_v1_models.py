from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import requests

from services.protocol import openai_v1_models


AUTH_KEY = os.getenv("CHATGPT2API_TEST_AUTH_KEY", "chatgpt2api")
BASE_URL = os.getenv("CHATGPT2API_TEST_BASE_URL", "http://localhost:8000")
RUN_HTTP_MODEL_TEST = bool(os.getenv("CHATGPT2API_TEST_BASE_URL")) or os.getenv("CHATGPT2API_RUN_HTTP_MODEL_TEST") == "1"
RUN_LIVE_MODEL_TEST = os.getenv("CHATGPT2API_RUN_LIVE_MODEL_TEST") == "1"


class ModelListTests(unittest.TestCase):
    def setUp(self):
        openai_v1_models.clear_model_cache()

    def tearDown(self):
        openai_v1_models.clear_model_cache()

    def test_list_models_only_returns_image_models_backed_by_account_types(self):
        with (
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-free", "type": "free"},
                    {"access_token": "token-web-team", "type": "Team", "source_type": "web"},
                    {"access_token": "token-codex-team", "type": "Team", "source_type": "codex"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertIn("codex-gpt-image-2", ids)
        self.assertIn("team-codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)
        self.assertNotIn("pro-codex-gpt-image-2", ids)

    def test_list_models_does_not_return_codex_models_for_web_plus_accounts(self):
        with (
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI,
                "list_models",
                return_value={"object": "list", "data": []},
            ),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-web-plus", "type": "Plus", "source_type": "web"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertNotIn("codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)

    def test_list_models_reuses_recent_backend_response(self):
        backend_result = {"object": "list", "data": [{"id": "auto", "object": "model"}]}
        with (
            mock.patch.object(
                openai_v1_models.OpenAIBackendAPI,
                "list_models",
                return_value=backend_result,
            ) as list_models,
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[],
            ),
        ):
            first = openai_v1_models.list_models()
            second = openai_v1_models.list_models()

        self.assertEqual(first["data"], [{"id": "auto", "object": "model"}])
        self.assertEqual(second["data"], [{"id": "auto", "object": "model"}])
        list_models.assert_called_once()

    @unittest.skipUnless(RUN_LIVE_MODEL_TEST, "live upstream model-list test disabled")
    def test_list_models_function(self):
        """测试直接调用服务层获取模型列表。"""
        result = openai_v1_models.list_models()
        print("function result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    @unittest.skipUnless(RUN_HTTP_MODEL_TEST, "HTTP model-list test disabled")
    def test_list_models_http(self):
        """测试通过 HTTP 接口获取模型列表。"""
        response = requests.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {AUTH_KEY}"},
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)
        print("http status:")
        print(response.status_code)
        print("http result:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
