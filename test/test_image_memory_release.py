from __future__ import annotations

import asyncio
from unittest import TestCase, mock

from services.log_service import LoggedCall


class ImageMemoryReleaseTests(TestCase):
    def test_sync_image_result_releases_process_memory_after_response(self) -> None:
        identity = {"id": "key-1", "name": "test", "role": "user"}
        call = LoggedCall(
            identity,
            "/v1/images/generations",
            "gpt-image-2",
            "image",
            quota_is_image=True,
        )
        with (
            mock.patch.object(call, "_reserve_quota"),
            mock.patch.object(call, "_decorate_image_payload"),
            mock.patch.object(call, "_finish_quota"),
            mock.patch.object(call, "log"),
            mock.patch("services.log_service.release_process_memory") as release,
        ):
            result = asyncio.run(call.run(lambda: {"data": [{"url": "http://local/image.png"}]}))

        self.assertEqual(result["data"][0]["url"], "http://local/image.png")
        release.assert_called_once_with()

    def test_text_result_does_not_trigger_image_memory_cleanup(self) -> None:
        identity = {"id": "key-1", "name": "test", "role": "user"}
        call = LoggedCall(identity, "/v1/chat/completions", "auto", "text")
        with (
            mock.patch.object(call, "_reserve_quota"),
            mock.patch.object(call, "_decorate_image_payload"),
            mock.patch.object(call, "_finish_quota"),
            mock.patch.object(call, "log"),
            mock.patch("services.log_service.release_process_memory") as release,
        ):
            result = asyncio.run(call.run(lambda: {"ok": True}))

        self.assertEqual(result, {"ok": True})
        release.assert_not_called()
