from __future__ import annotations

import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI


class UpstreamModelSettingsTests(unittest.TestCase):
    def test_image_model_uses_configured_upstream_model_and_effort(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        with mock.patch.object(
            type(config),
            "default_upstream_model_name",
            new_callable=mock.PropertyMock,
            return_value="gpt-5-5",
        ), mock.patch.object(
            type(config),
            "default_thinking_effort",
            new_callable=mock.PropertyMock,
            return_value="extended",
        ):
            self.assertEqual(backend._image_model_settings("gpt-image-2"), ("gpt-5-5", "extended"))

    def test_model_suffix_overrides_default_effort(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        with mock.patch.object(
            type(config),
            "default_upstream_model_name",
            new_callable=mock.PropertyMock,
            return_value="gpt-5-5-standard",
        ), mock.patch.object(
            type(config),
            "default_thinking_effort",
            new_callable=mock.PropertyMock,
            return_value="max",
        ):
            self.assertEqual(backend._image_model_settings("gpt-image-2"), ("gpt-5-5", "standard"))

    def test_auto_effort_is_not_added_to_text_payload(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        with mock.patch.object(
            type(config),
            "default_thinking_effort",
            new_callable=mock.PropertyMock,
            return_value="auto",
        ):
            payload = backend._conversation_payload([], "auto", "Asia/Shanghai")
        self.assertNotIn("thinking_effort", payload)


if __name__ == "__main__":
    unittest.main()
