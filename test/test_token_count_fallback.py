from __future__ import annotations

import unittest
from unittest import mock

from services.protocol import conversation


class TokenCountFallbackTests(unittest.TestCase):
    def test_text_token_count_falls_back_when_tiktoken_cache_download_fails(self) -> None:
        with mock.patch.object(
            conversation,
            "encoding_for_model",
            side_effect=OSError("tiktoken cache download failed"),
        ):
            count = conversation.count_text_tokens("abc测试", "gpt-image-2")

        self.assertEqual(count, 3)

    def test_message_token_count_falls_back_without_losing_chat_overhead(self) -> None:
        messages = [{"role": "user", "content": "hello世界"}]
        with mock.patch.object(
            conversation,
            "encoding_for_model",
            side_effect=OSError("tiktoken cache download failed"),
        ):
            count = conversation.count_message_text_tokens(messages, "gpt-image-2")

        self.assertEqual(count, 11)


if __name__ == "__main__":
    unittest.main()
