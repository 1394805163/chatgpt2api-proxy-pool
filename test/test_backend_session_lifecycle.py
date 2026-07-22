from __future__ import annotations

import unittest
from unittest import mock

from services import openai_backend_api
from services.protocol import conversation, openai_search, openai_v1_models, web_search_tool


class RecordingSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class BackendSessionLifecycleTests(unittest.TestCase):
    def test_backend_close_is_idempotent_and_context_managed(self) -> None:
        session = RecordingSession()

        with mock.patch.object(openai_backend_api.requests, "Session", return_value=session):
            backend = openai_backend_api.OpenAIBackendAPI()
            with backend as entered:
                self.assertIs(entered, backend)
            backend.close()

        self.assertEqual(session.close_calls, 1)

    def test_stream_text_deltas_uses_and_closes_supplied_backend(self) -> None:
        backend = mock.Mock(access_token="token-1")
        request = conversation.ConversationRequest(prompt="hello")

        with (
            mock.patch.object(
                conversation,
                "conversation_events",
                return_value=iter([{"type": "conversation.delta", "delta": "ok"}]),
            ),
            mock.patch.object(conversation, "OpenAIBackendAPI") as backend_factory,
            mock.patch.object(conversation.account_service, "mark_text_used"),
        ):
            self.assertEqual(list(conversation.stream_text_deltas(backend, request)), ["ok"])

        backend_factory.assert_not_called()
        backend.close.assert_called_once_with()

    def test_stream_text_deltas_closes_backend_when_consumer_stops_early(self) -> None:
        backend = mock.Mock(access_token="token-1")
        request = conversation.ConversationRequest(prompt="hello")

        def events(*_args, **_kwargs):
            yield {"type": "conversation.delta", "delta": "first"}
            yield {"type": "conversation.delta", "delta": "second"}

        with mock.patch.object(conversation, "conversation_events", side_effect=events):
            stream = conversation.stream_text_deltas(backend, request)
            self.assertEqual(next(stream), "first")
            stream.close()

        backend.close.assert_called_once_with()

    def test_search_handler_closes_backend_after_failure(self) -> None:
        backend = mock.Mock()
        backend.search.side_effect = RuntimeError("upstream failed")

        with (
            mock.patch.object(openai_search, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(openai_search.account_service, "get_text_access_token", return_value="token-1"),
            mock.patch.object(openai_search.account_service, "get_account", return_value={}),
        ):
            with self.assertRaisesRegex(RuntimeError, "upstream failed"):
                openai_search.handle({"prompt": "query"})

        backend.close.assert_called_once_with()

    def test_web_search_tool_closes_backend_after_success(self) -> None:
        backend = mock.Mock()
        backend.search.return_value = {"answer": "ok"}

        with (
            mock.patch.object(web_search_tool, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(web_search_tool.account_service, "get_text_access_token", return_value="token-1"),
            mock.patch.object(web_search_tool.account_service, "mark_text_used"),
        ):
            result = web_search_tool.run_web_search("query")

        self.assertEqual(result, {"answer": "ok"})
        backend.close.assert_called_once_with()

    def test_model_fetch_closes_backend_before_caching_result(self) -> None:
        backend = mock.Mock()
        backend.list_models.return_value = {"object": "list", "data": []}
        openai_v1_models.clear_model_cache()

        try:
            with mock.patch.object(openai_v1_models, "OpenAIBackendAPI", return_value=backend):
                result = openai_v1_models._backend_models()
        finally:
            openai_v1_models.clear_model_cache()

        self.assertEqual(result, {"object": "list", "data": []})
        backend.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
