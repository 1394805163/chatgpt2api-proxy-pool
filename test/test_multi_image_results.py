from __future__ import annotations

import base64
import unittest
from unittest import mock

from services.account_service import account_service
from services.config import config
from services.openai_backend_api import ImagePollTimeoutError, ImageTaskDeadlineError, OpenAIBackendAPI
from services.protocol.conversation import (
    ConversationRequest,
    ImageGenerationError,
    ImageOutput,
    _generate_single_image,
    extract_conversation_ids,
    stream_image_outputs,
)
from services.protocol.openai_v1_response import stream_image_response


def _conversation(file_ids: list[str], sediment_ids: list[str] | None = None) -> dict:
    parts: list[object] = [
        {"content_type": "image_asset_pointer", "asset_pointer": f"file-service://{file_id}"}
        for file_id in file_ids
    ]
    parts.extend(f"sediment://{sediment_id}" for sediment_id in (sediment_ids or []))
    return {
        "mapping": {
            "tool": {
                "message": {
                    "author": {"role": "tool"},
                    "create_time": 1,
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {"content_type": "multimodal_text", "parts": parts},
                }
            }
        }
    }


class FakeBackend(OpenAIBackendAPI):
    def __init__(self, conversations: list[dict] | None = None) -> None:
        self.conversations = conversations or []
        self.calls = 0
        self.file_urls: dict[str, str] = {}
        self.sediment_urls: dict[str, str] = {}

    def _get_conversation(self, conversation_id: str) -> dict:
        self.calls += 1
        index = min(self.calls - 1, len(self.conversations) - 1)
        return self.conversations[index]

    def _get_file_download_url(self, file_id: str) -> str:
        return self.file_urls.get(file_id, "")

    def _get_attachment_download_url(self, conversation_id: str, attachment_id: str) -> str:
        return self.sediment_urls.get(attachment_id, "")


class FakeStreamingResponse:
    def __init__(self, close_error: Exception | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    def iter_lines(self):
        yield b"data: [DONE]\n"

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class MultiImageResultTests(unittest.TestCase):
    def _picture_stream_backend(self, response: FakeStreamingResponse) -> OpenAIBackendAPI:
        backend = object.__new__(OpenAIBackendAPI)
        backend.access_token = "token-1"
        backend.progress_callback = None
        backend._ensure_image_task_active = mock.Mock()
        backend._upload_image = mock.Mock(return_value={})
        backend._bootstrap = mock.Mock()
        backend._get_chat_requirements = mock.Mock(return_value=mock.Mock())
        backend._prepare_image_conversation = mock.Mock(return_value="conduit-token")
        backend._start_image_generation = mock.Mock(return_value=response)
        return backend

    def test_picture_stream_suppresses_curl_write_error_caused_by_close(self) -> None:
        response = FakeStreamingResponse(RuntimeError(
            "Failed to perform, curl: (23) client returned ERROR on write of 45 bytes."
        ))
        backend = self._picture_stream_backend(response)

        payloads = list(backend._stream_picture_conversation("cat", "gpt-image-2", []))

        self.assertEqual(payloads, ["[DONE]"])
        self.assertEqual(response.close_calls, 1)

    def test_picture_stream_keeps_unexpected_close_errors(self) -> None:
        response = FakeStreamingResponse(RuntimeError("unexpected close failure"))
        backend = self._picture_stream_backend(response)

        with self.assertRaisesRegex(RuntimeError, "unexpected close failure"):
            list(backend._stream_picture_conversation("cat", "gpt-image-2", []))

    def test_task_deadline_releases_account_slot_without_marking_failure(self) -> None:
        with (
            mock.patch.object(account_service, "get_available_access_token", return_value="token-1"),
            mock.patch.object(account_service, "get_account", return_value={"email": "test@example.com"}),
            mock.patch.object(account_service, "release_image_slot") as release_slot,
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=mock.Mock()),
            mock.patch(
                "services.protocol.conversation.stream_image_outputs",
                side_effect=ImageTaskDeadlineError("deadline reached"),
            ),
        ):
            with self.assertRaises(ImageGenerationError) as raised:
                _generate_single_image(ConversationRequest(model="gpt-image-2", prompt="cat"), 1, 1)

        self.assertEqual(raised.exception.code, "image_task_timeout")
        release_slot.assert_called_once_with("token-1")

    def test_stream_id_extractor_keeps_full_file_ids(self) -> None:
        payload = (
            '{"conversation_id":"conv-1"} '
            'file-service://file-first_123-extra sediment://sed-second_456-extra'
        )

        conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)

        self.assertEqual(conversation_id, "conv-1")
        self.assertEqual(file_ids, ["file-first_123-extra"])
        self.assertEqual(sediment_ids, ["sed-second_456-extra"])

    def test_conversation_record_extractor_finds_all_generated_assets(self) -> None:
        backend = FakeBackend()
        conversation = {
            "mapping": {
                "user": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["file-service://file-user-input"]},
                    }
                },
                "tool": {
                    "message": {
                        "author": {"role": "tool"},
                        "create_time": 1,
                        "metadata": {
                            "async_task_type": "image_gen",
                            "nested": {"asset": "file-service://file-second"},
                        },
                        "content": {
                            "content_type": "text",
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-first"},
                                "sediment://sed-first",
                            ],
                        },
                    }
                },
                "assistant": {
                    "message": {
                        "author": {"role": "assistant"},
                        "create_time": 2,
                        "metadata": {},
                        "content": {
                            "parts": [
                                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-third"}
                            ]
                        },
                    }
                },
            }
        }

        records = backend._extract_image_tool_records(conversation)
        file_ids = [file_id for record in records for file_id in record["file_ids"]]
        sediment_ids = [sediment_id for record in records for sediment_id in record["sediment_ids"]]

        self.assertEqual(file_ids, ["file-first", "file-second", "file-third"])
        self.assertEqual(sediment_ids, ["sed-first"])

    def test_poll_waits_for_generated_asset_ids_to_settle(self) -> None:
        backend = FakeBackend([
            _conversation(["file-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
            _conversation(["file-one", "file-two"], ["sed-one"]),
        ])

        with (
            mock.patch.dict(config.data, {"image_poll_initial_wait_secs": 0, "image_poll_interval_secs": 0.5}),
            mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None),
        ):
            file_ids, sediment_ids = backend._poll_image_results("conv-1", timeout_secs=10)

        self.assertEqual(file_ids, ["file-one", "file-two"])
        self.assertEqual(sediment_ids, ["sed-one"])
        self.assertEqual(backend.calls, 3)

    def test_resolver_uses_file_and_sediment_urls(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend.sediment_urls = {
            "sed-one": "https://attachments.test/one.png",
            "sed-two": "https://attachments.test/two.png",
        }

        urls = backend._resolve_image_urls("conv-1", ["file-one"], ["sed-one", "sed-two"])

        self.assertEqual(urls, [
            "https://files.test/one.png",
            "https://attachments.test/one.png",
            "https://attachments.test/two.png",
        ])

    def test_resolver_keeps_stream_ids_when_poll_extension_fails(self) -> None:
        backend = FakeBackend()
        backend.file_urls = {"file-one": "https://files.test/one.png"}
        backend._get_conversation = mock.Mock(side_effect=RuntimeError("poll failed"))

        with mock.patch("services.openai_backend_api.time.sleep", lambda _seconds: None):
            urls = backend.resolve_conversation_image_urls("conv-1", ["file-one"], [], poll=True)

        self.assertEqual(urls, ["https://files.test/one.png"])

    def test_text_reply_poll_uses_total_task_timeout(self) -> None:
        backend = FakeBackend()
        backend.resolve_conversation_image_urls = mock.Mock(return_value=["https://files.test/one.png"])
        backend.download_image_bytes = mock.Mock(return_value=[b"image-bytes"])
        events = [
            {
                "type": "conversation.completed",
                "conversation_id": "conv-1",
                "file_ids": [],
                "sediment_ids": [],
                "text": '{"size":"1024x1024","n":1}',
                "turn_use_case": "image gen",
            }
        ]

        with (
            mock.patch.dict(config.data, {"image_poll_timeout_secs": 70, "image_task_timeout_secs": 150}),
            mock.patch("services.protocol.conversation.conversation_events", return_value=iter(events)),
            mock.patch("services.protocol.conversation._get_detailed_error_from_tasks", return_value=""),
            mock.patch("services.protocol.conversation.save_image_bytes", return_value="http://local.test/one.png"),
        ):
            outputs = list(stream_image_outputs(
                backend,
                ConversationRequest(model="gpt-image-2", prompt="draw a cat", response_format="b64_json"),
            ))

        backend.resolve_conversation_image_urls.assert_called_once()
        self.assertEqual(backend.resolve_conversation_image_urls.call_args.kwargs["poll_timeout_secs"], 150)
        self.assertTrue(any(output.kind == "result" for output in outputs))

    def test_progress_event_does_not_block_poll_timeout_retry(self) -> None:
        attempts = 0

        def stream_outputs(_backend, _request, index, total):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                yield ImageOutput(kind="progress", model="gpt-image-2", index=index, total=total)
                raise ImagePollTimeoutError("poll timed out")
            yield ImageOutput(
                kind="result",
                model="gpt-image-2",
                index=index,
                total=total,
                data=[{"url": "http://example.test/image.png"}],
            )

        with (
            mock.patch.object(account_service, "get_available_access_token", side_effect=["token-1", "token-2"]),
            mock.patch.object(account_service, "get_account", return_value={"email": "test@example.com"}),
            mock.patch.object(account_service, "mark_image_result"),
            mock.patch("services.protocol.conversation.record_image_failure"),
            mock.patch("services.protocol.conversation.OpenAIBackendAPI", return_value=mock.Mock()),
            mock.patch("services.protocol.conversation.stream_image_outputs", side_effect=stream_outputs),
        ):
            outputs = _generate_single_image(ConversationRequest(model="gpt-image-2", prompt="cat"), 1, 1)

        self.assertEqual(attempts, 2)
        self.assertTrue(any(output.kind == "result" for output in outputs))

    def test_responses_stream_emits_all_image_output_items(self) -> None:
        first = base64.b64encode(b"first").decode("ascii")
        second = base64.b64encode(b"second").decode("ascii")
        events = list(stream_image_response(
            [ImageOutput(
                kind="result",
                model="gpt-image-2",
                index=1,
                total=1,
                data=[{"b64_json": first}, {"b64_json": second}],
            )],
            "draw two options",
            "gpt-image-2",
        ))

        done_events = [event for event in events if event.get("type") == "response.output_item.done"]
        completed = next(event["response"] for event in events if event.get("type") == "response.completed")

        self.assertEqual([event["output_index"] for event in done_events], [0, 1])
        self.assertEqual([item["result"] for item in completed["output"]], [first, second])


if __name__ == "__main__":
    unittest.main()
