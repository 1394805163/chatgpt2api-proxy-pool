from __future__ import annotations

import unittest
from unittest import mock

from services.config import config
from services.openai_backend_api import OpenAIBackendAPI
from services.protocol import conversation


class FakeResponse:
    status_code = 200

    def json(self) -> dict[str, bool]:
        return {"success": True}


class ImageConversationCleanupTests(unittest.TestCase):
    def test_cleanup_setting_defaults_to_disabled(self) -> None:
        with mock.patch.dict(config.data, {}, clear=True):
            self.assertFalse(config.image_remove_conversation_after_result)

    def test_delete_conversation_hides_upstream_record(self) -> None:
        backend = object.__new__(OpenAIBackendAPI)
        backend.base_url = "https://chatgpt.com"
        backend.session = mock.Mock()
        backend.session.headers = {"Authorization": "Bearer token-1"}
        backend.session.patch.return_value = FakeResponse()

        result = backend.delete_conversation("conv-1")

        self.assertEqual(result, {"success": True})
        backend.session.patch.assert_called_once()
        _, kwargs = backend.session.patch.call_args
        self.assertEqual(kwargs["json"], {"is_visible": False})
        self.assertEqual(kwargs["timeout"], 60)
        self.assertEqual(
            backend.session.patch.call_args.args[0],
            "https://chatgpt.com/backend-api/conversation/conv-1",
        )

    def test_disabled_cleanup_does_not_start_worker(self) -> None:
        with (
            mock.patch.dict(config.data, {"image_remove_conversation_after_result": False}),
            mock.patch.object(conversation.threading, "Thread") as thread_factory,
        ):
            conversation._remove_image_conversation_later("token-1", "conv-1")

        thread_factory.assert_not_called()

    def test_cleanup_worker_uses_independent_backend_and_always_closes(self) -> None:
        cleanup_backend = mock.Mock()
        cleanup_backend.delete_conversation.side_effect = RuntimeError("upstream failed")

        with mock.patch.object(conversation, "OpenAIBackendAPI", return_value=cleanup_backend) as backend_factory:
            conversation._remove_image_conversation("token-1", "conv-1")

        backend_factory.assert_called_once_with(access_token="token-1")
        cleanup_backend.delete_conversation.assert_called_once_with("conv-1")
        cleanup_backend.close.assert_called_once_with()

    def test_successful_image_result_schedules_cleanup(self) -> None:
        image_backend = mock.Mock()
        output = conversation.ImageOutput(
            kind="result",
            model="gpt-image-2",
            index=1,
            total=1,
            data=[{"url": "https://example.test/image.png"}],
            conversation_id="conv-1",
        )

        with (
            mock.patch.object(conversation.account_service, "get_available_access_token", return_value="token-1"),
            mock.patch.object(conversation.account_service, "get_account", return_value={"email": "a@example.test"}),
            mock.patch.object(conversation.account_service, "mark_image_result"),
            mock.patch.object(conversation, "OpenAIBackendAPI", return_value=image_backend),
            mock.patch.object(conversation, "stream_image_outputs", return_value=iter([output])),
            mock.patch.object(conversation, "_remove_image_conversation_later") as schedule_cleanup,
        ):
            outputs = conversation._generate_single_image(
                conversation.ConversationRequest(model="gpt-image-2", prompt="draw a cat"),
                1,
                1,
            )

        self.assertEqual(outputs, [output])
        schedule_cleanup.assert_called_once_with("token-1", "conv-1")
        image_backend.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
