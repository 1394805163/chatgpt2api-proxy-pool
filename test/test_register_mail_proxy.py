from __future__ import annotations

import unittest
import copy
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from services.register import mail_provider, openai_register


class FakeSession:
    def __init__(self, responses=None, **kwargs):
        self.kwargs = kwargs
        self.responses = list(responses or [])
        self.closed = False
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def json(self):
        return self._payload


def cloudmail_entry() -> dict:
    return {
        "enable": True,
        "type": "cloudmail_gen",
        "api_base": "https://mail.example",
        "admin_email": "admin@example.com",
        "admin_password": "secret",
        "domain": ["example.com"],
    }


class RegisterMailProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = {
            "mail": copy.deepcopy(openai_register.config.get("mail")),
            "proxy": openai_register.config.get("proxy"),
        }
        mail_provider.cloudmail_token_cache.clear()

    def tearDown(self) -> None:
        openai_register.config["mail"] = self.original_config["mail"]
        openai_register.config["proxy"] = self.original_config["proxy"]
        mail_provider.cloudmail_token_cache.clear()

    def test_mail_api_uses_register_proxy_when_enabled(self) -> None:
        openai_register.config["proxy"] = "http://register.example:8080"
        openai_register.config["mail"] = {
            "api_use_register_proxy": True,
            "providers": [cloudmail_entry()],
        }

        self.assertEqual(openai_register._mail_config("http://worker.example:9000")["proxy"], "http://worker.example:9000")

    def test_mail_api_uses_direct_connection_when_disabled(self) -> None:
        openai_register.config["proxy"] = "http://register.example:8080"
        openai_register.config["mail"] = {
            "api_use_register_proxy": False,
            "providers": [cloudmail_entry()],
        }

        self.assertEqual(openai_register._mail_config("http://worker.example:9000")["proxy"], "")

    def test_provider_session_receives_no_proxy_when_disabled(self) -> None:
        openai_register.config["mail"] = {
            "api_use_register_proxy": False,
            "providers": [cloudmail_entry()],
        }
        created: list[FakeSession] = []

        def session_factory(**kwargs):
            session = FakeSession(
                responses=[
                    FakeResponse({"code": 200, "data": {"token": "mail-token"}}),
                    FakeResponse({"code": 200, "data": {}}),
                ],
                **kwargs,
            )
            created.append(session)
            return session

        with patch.object(mail_provider.requests, "Session", side_effect=session_factory):
            mailbox = openai_register.create_mailbox(register_proxy="http://worker.example:9000")

        self.assertEqual(mailbox["provider"], "cloudmail_gen")
        self.assertIsInstance(mailbox.get("_code_not_before"), datetime)
        self.assertEqual(len(created), 1)
        self.assertNotIn("proxy", created[0].kwargs)

    def test_cloudmail_gen_registers_new_address_before_returning_it(self) -> None:
        session = FakeSession(
            responses=[
                FakeResponse({"code": 200, "data": {"token": "mail-token"}}),
                FakeResponse({"code": 200, "data": {}}),
            ]
        )

        with patch.object(mail_provider, "_create_session", return_value=session):
            provider = mail_provider.CloudMailGenProvider(cloudmail_entry(), mail_provider._config({"proxy": ""}))
            mailbox = provider.create_mailbox("new-user")

        self.assertEqual(mailbox["address"], "new-user@example.com")
        self.assertEqual([call["url"] for call in session.calls], [
            "https://mail.example/api/public/genToken",
            "https://mail.example/api/public/addUser",
        ])
        self.assertEqual(session.calls[1]["headers"]["Authorization"], "mail-token")
        self.assertEqual(session.calls[1]["json"], {"list": [{"email": "new-user@example.com"}]})

    def test_cloudmail_gen_refreshes_cached_token_when_add_user_rejects_it(self) -> None:
        entry = cloudmail_entry()
        cache_key = f"{entry['api_base']}|{entry['admin_email']}"
        mail_provider.cloudmail_token_cache[cache_key] = ("stale-token", time.time() + 3600)
        session = FakeSession(
            responses=[
                FakeResponse({"code": 401, "message": "invalid token"}),
                FakeResponse({"code": 200, "data": {"token": "fresh-token"}}),
                FakeResponse({"code": 200, "data": {}}),
            ]
        )

        with patch.object(mail_provider, "_create_session", return_value=session):
            provider = mail_provider.CloudMailGenProvider(entry, mail_provider._config({"proxy": ""}))
            mailbox = provider.create_mailbox("new-user")

        self.assertEqual(mailbox["address"], "new-user@example.com")
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "stale-token")
        self.assertEqual(session.calls[2]["headers"]["Authorization"], "fresh-token")

    def test_base_provider_ignores_codes_received_before_mailbox_boundary(self) -> None:
        boundary = datetime.now(timezone.utc)
        messages = [
            {
                "message_id": "old",
                "text_content": "Verification code: 111111",
                "received_at": boundary - timedelta(seconds=1),
            },
            {
                "message_id": "new",
                "text_content": "Verification code: 222222",
                "received_at": boundary + timedelta(seconds=1),
            },
        ]

        class SequenceProvider(mail_provider.BaseMailProvider):
            def fetch_latest_message(self, _mailbox):
                return messages.pop(0) if messages else None

        provider = SequenceProvider({"wait_timeout": 0.5, "wait_interval": 0.001})

        self.assertEqual(provider.wait_for_code({"_code_not_before": boundary}), "222222")

    def test_outlook_provider_ignores_codes_received_before_mailbox_boundary(self) -> None:
        boundary = datetime.now(timezone.utc)
        provider = object.__new__(mail_provider.OutlookTokenProvider)
        provider.conf = {"wait_timeout": 0.1, "wait_interval": 0.001}
        provider.fetch_recent_messages = lambda _mailbox: [
            {
                "message_id": "old",
                "text_content": "Verification code: 111111",
                "received_at": boundary - timedelta(seconds=1),
            },
            {
                "message_id": "new",
                "text_content": "Verification code: 222222",
                "received_at": boundary + timedelta(seconds=1),
            },
        ]

        self.assertEqual(provider.wait_for_code({"_code_not_before": boundary}), "222222")

    def test_cloudmail_gen_parses_current_field_names_and_retries_transient_errors(self) -> None:
        responses = [
            RuntimeError("temporary tls failure"),
            RuntimeError("temporary empty response"),
            FakeResponse({"code": 200, "data": {"token": "mail-token"}}),
            FakeResponse(
                {
                    "code": 200,
                    "data": [
                        {
                            "emailId": "mail-1",
                            "toEmail": "user@example.com",
                            "sendEmail": "noreply@example.com",
                            "subject": "OpenAI verification",
                            "text": "Verification code: 123456",
                            "createTime": "2026-06-15T01:02:03Z",
                        }
                    ],
                }
            ),
        ]
        session = FakeSession(responses=responses)

        with patch.object(mail_provider, "_create_session", return_value=session), patch.object(mail_provider.time, "sleep", return_value=None):
            provider = mail_provider.CloudMailGenProvider(cloudmail_entry(), mail_provider._config({"proxy": ""}))
            message = provider.fetch_latest_message({"address": "user@example.com"})

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message["message_id"], "mail-1")
        self.assertEqual(message["sender"], "noreply@example.com")
        self.assertEqual(message["text_content"], "Verification code: 123456")
        self.assertIsInstance(message["received_at"], datetime)
        self.assertEqual(len(session.calls), 4)

    def test_cloudmail_gen_refreshes_cached_token_when_email_list_rejects_it(self) -> None:
        entry = cloudmail_entry()
        cache_key = f"{entry['api_base']}|{entry['admin_email']}"
        mail_provider.cloudmail_token_cache[cache_key] = ("stale-token", time.time() + 3600)
        responses = [
            FakeResponse({"code": 401, "message": "invalid token"}),
            FakeResponse({"code": 200, "data": {"token": "fresh-token"}}),
            FakeResponse(
                {
                    "code": 200,
                    "data": [
                        {
                            "emailId": "mail-2",
                            "toEmail": "user@example.com",
                            "subject": "OpenAI verification",
                            "text": "Verification code: 654321",
                        }
                    ],
                }
            ),
        ]
        session = FakeSession(responses=responses)

        with patch.object(mail_provider, "_create_session", return_value=session), patch.object(mail_provider.time, "sleep", return_value=None):
            provider = mail_provider.CloudMailGenProvider(entry, mail_provider._config({"proxy": ""}))
            message = provider.fetch_latest_message({"address": "user@example.com"})

        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message["message_id"], "mail-2")
        self.assertEqual(mail_provider._extract_code(message), "654321")
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(session.calls[0]["headers"]["Authorization"], "stale-token")
        self.assertEqual(session.calls[2]["headers"]["Authorization"], "fresh-token")


if __name__ == "__main__":
    unittest.main()
