from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen

from utils import sentinel_sdk
from utils.sentinel_sdk import SentinelSDKClient, SentinelSDKError


class FakeResponse:
    def __init__(self, *, status_code=200, text="", content=b"", json_data=None):
        self.status_code = status_code
        self.text = text
        self.content = content or text.encode("utf-8")
        self._json_data = json_data

    def json(self):
        return self._json_data


class FakeSession:
    SDK_URL = "https://sentinel.openai.com/sentinel/20260725abcd/sdk.js"

    def __init__(self):
        self.get_calls = []
        self.post_calls = []
        self.cookies = FakeCookies()

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        if url == SentinelSDKClient.FRAME_URL:
            return FakeResponse(
                text=f'<html><script src="{self.SDK_URL}"></script></html>'
            )
        if url == self.SDK_URL:
            return FakeResponse(content=b"var SentinelSDK={};")
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse(
            json_data={
                "token": "challenge-token",
                "turnstile": {"required": True, "dx": "turnstile-dx"},
                "so": {
                    "required": True,
                    "collector_dx": "collector-dx",
                    "snapshot_dx": "snapshot-dx",
                },
            }
        )


class FakeCookies:
    def __init__(self):
        self.values = {}

    def set(self, name, value, **kwargs):
        self.values[(name, kwargs.get("domain", ""))] = value


class SentinelSDKClientTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_client(self, *, session=None, process_runner=None, user_agent="Test Browser UA"):
        return SentinelSDKClient(
            session=session or FakeSession(),
            device_id="device-id",
            user_agent=user_agent,
            cache_dir=Path(self.temp_dir.name),
            process_runner=process_runner,
        )

    def test_uses_node_permission_flag_supported_by_runtime_major_version(self):
        self.assertEqual(
            SentinelSDKClient._permission_flag_for_version("v20.19.2"),
            "--experimental-permission",
        )
        self.assertEqual(
            SentinelSDKClient._permission_flag_for_version("v24.11.1"),
            "--permission",
        )

    @staticmethod
    def token(flow: str, *, t: str = "turnstile-token") -> str:
        return json.dumps(
            {"p": "proof", "t": t, "c": "challenge-token", "id": "device-id", "flow": flow},
            separators=(",", ":"),
        )

    @staticmethod
    def so_token(
        flow: str,
        *,
        so: str = "observer-token",
        challenge: str = "challenge-token",
    ) -> str:
        return json.dumps(
            {"so": so, "c": challenge, "id": "device-id", "flow": flow},
            separators=(",", ":"),
        )

    def test_discovers_current_sdk_relays_challenge_and_generates_token_pair(self):
        session = FakeSession()
        runner_calls = []

        def process_runner(command, *, timeout):
            runner_calls.append((command, timeout))
            challenge_url = command[command.index("--challenge-url") + 1]
            request = Request(
                challenge_url,
                data=json.dumps({"p": "request-proof", "id": "device-id", "flow": "oauth_create_account"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5) as response:
                challenge = json.loads(response.read().decode())
            self.assertEqual(challenge["token"], "challenge-token")
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "token": self.token("oauth_create_account"),
                        "soToken": self.so_token("oauth_create_account"),
                    }
                ),
                stderr="",
            )

        client = self.make_client(session=session, process_runner=process_runner)
        result = client.get_tokens("oauth_create_account", include_so=True)

        self.assertEqual(result.sdk_version, "20260725abcd")
        self.assertEqual(json.loads(result.token)["t"], "turnstile-token")
        self.assertEqual(json.loads(result.so_token)["so"], "observer-token")
        self.assertEqual(len(session.post_calls), 1)
        req_url, req_kwargs = session.post_calls[0]
        self.assertEqual(req_url, SentinelSDKClient.SENTINEL_REQ_URL)
        self.assertEqual(json.loads(req_kwargs["data"])["flow"], "oauth_create_account")
        self.assertEqual(req_kwargs["headers"]["user-agent"], "Test Browser UA")
        self.assertTrue(req_kwargs["verify"])
        self.assertTrue(all(kwargs["verify"] for _url, kwargs in session.get_calls))
        self.assertEqual(
            session.cookies.values[("oai-sc", ".openai.com")],
            "0challenge-token",
        )
        command = runner_calls[0][0]
        self.assertIn("--include-so", command)
        self.assertEqual(command[command.index("--include-so") + 1], "1")
        self.assertIn("--experimental-permission", command)
        self.assertTrue((Path(self.temp_dir.name) / "20260725abcd" / "sdk.js").is_file())

    def test_runner_fingerprint_matches_windows_chrome_user_agent(self):
        commands = []

        def process_runner(command, *, timeout):
            commands.append(command)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"token": self.token("authorize_continue"), "soToken": ""}),
                stderr="",
            )

        client = self.make_client(
            process_runner=process_runner,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.7123.45 Safari/537.36"
            ),
        )
        client.get_tokens("authorize_continue")

        command = commands[0]
        self.assertEqual(command[command.index("--navigator-platform") + 1], "Win32")
        self.assertEqual(command[command.index("--user-agent-data-platform") + 1], "Windows")
        self.assertEqual(command[command.index("--chrome-major") + 1], "145")
        self.assertEqual(command[command.index("--chrome-full-version") + 1], "145.0.7123.45")
        self.assertEqual(command[command.index("--sec-ch-ua-platform") + 1], "Windows")

    def test_default_runner_does_not_inherit_application_secrets(self):
        with patch.dict(
            os.environ,
            {"PATH": os.environ.get("PATH", ""), "DATABASE_URL": "database-secret"},
            clear=True,
        ), patch.object(sentinel_sdk.subprocess, "run") as run:
            sentinel_sdk._default_process_runner(["node", "runner.js"], timeout=1)

        child_env = run.call_args.kwargs["env"]
        self.assertNotIn("DATABASE_URL", child_env)
        self.assertEqual(child_env["PATH"], os.environ.get("PATH", ""))

    def test_reuses_cached_sdk_without_downloading_script_again(self):
        session = FakeSession()

        def process_runner(_command, *, timeout):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"token": self.token("authorize_continue"), "soToken": ""}),
                stderr="",
            )

        client = self.make_client(session=session, process_runner=process_runner)
        client.get_tokens("authorize_continue")
        client.get_tokens("authorize_continue")

        sdk_downloads = [url for url, _kwargs in session.get_calls if url == session.SDK_URL]
        self.assertEqual(sdk_downloads, [session.SDK_URL])

    def test_rejects_empty_turnstile_token(self):
        def process_runner(_command, *, timeout):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"token": self.token("authorize_continue", t=""), "soToken": ""}),
                stderr="",
            )

        client = self.make_client(process_runner=process_runner)

        with self.assertRaisesRegex(SentinelSDKError, "turnstile"):
            client.get_tokens("authorize_continue")

    def test_requires_session_observer_token_for_create_account(self):
        def process_runner(_command, *, timeout):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"token": self.token("oauth_create_account"), "soToken": ""}),
                stderr="",
            )

        client = self.make_client(process_runner=process_runner)

        with self.assertRaisesRegex(SentinelSDKError, "session observer"):
            client.get_tokens("oauth_create_account", include_so=True)

    def test_rejects_session_observer_token_from_another_challenge(self):
        def process_runner(_command, *, timeout):
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "token": self.token("oauth_create_account"),
                        "soToken": self.so_token(
                            "oauth_create_account",
                            challenge="different-challenge",
                        ),
                    }
                ),
                stderr="",
            )

        client = self.make_client(process_runner=process_runner)

        with self.assertRaisesRegex(SentinelSDKError, "challenge mismatch"):
            client.get_tokens("oauth_create_account", include_so=True)

    def test_runner_failure_does_not_expose_stderr_tokens(self):
        secret = "sentinel-secret-that-must-not-be-logged"

        def process_runner(_command, *, timeout):
            return SimpleNamespace(returncode=1, stdout="", stderr=secret)

        client = self.make_client(process_runner=process_runner)

        with self.assertRaises(SentinelSDKError) as context:
            client.get_tokens("authorize_continue")

        self.assertNotIn(secret, str(context.exception))


if __name__ == "__main__":
    unittest.main()
