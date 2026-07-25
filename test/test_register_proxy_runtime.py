import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services.proxy_service import ClearanceBundle
from services.register import openai_register


class FakeResponse:
    def __init__(self, status_code=200, text="", headers=None, url="https://auth.openai.com/test", json_data=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.url = url
        self.json_data = json_data or {}

    def json(self):
        return self.json_data


class FakeCookieJar:
    def __init__(self):
        self.items = []

    def set(self, name, value, domain=None):
        self.items.append({"name": name, "value": value, "domain": domain})


class FakeSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.headers = {}
        self.cookies = FakeCookieJar()
        self.closed = False

    def close(self):
        self.closed = True


class FakeProxyProfile:
    clearance_enabled = True


class FakeProxySettings:
    def __init__(self, bundle=None):
        self.bundle = bundle
        self.refreshed = False
        self.session_kwargs_calls = []
        self.build_headers_calls = []
        self.refresh_calls = []

    def build_session_kwargs(self, **kwargs):
        self.session_kwargs_calls.append(kwargs)
        return dict(kwargs, proxy="http://runtime.example:8118")

    def build_headers(self, headers=None, target_url="", proxy="", upstream=True, **kwargs):
        self.build_headers_calls.append({"target_url": target_url, "proxy": proxy, "upstream": upstream})
        merged = dict(headers or {})
        if self.refreshed and self.bundle and self.bundle.cookies:
            merged["Cookie"] = "; ".join(f"{key}={value}" for key, value in self.bundle.cookies.items())
        return merged

    def get_profile(self, **kwargs):
        return FakeProxyProfile()

    def refresh_clearance(self, target_url="", proxy="", force=False, upstream=True, **kwargs):
        self.refresh_calls.append({"target_url": target_url, "proxy": proxy, "force": force, "upstream": upstream})
        self.refreshed = self.bundle is not None
        return self.bundle


class FakeSentinelSDKClient:
    def __init__(self):
        self.calls = []
        self.closed = False

    def get_tokens(self, flow, *, include_so=False):
        self.calls.append((flow, include_so))
        return SimpleNamespace(
            token="sentinel-secret-value",
            so_token="so-secret-value" if include_so else "",
            sdk_version="20260219f9f6",
        )

    def close(self):
        self.closed = True


class RegisterProxyRuntimeTests(unittest.TestCase):
    def test_browser_fallback_user_agent_matches_runtime_chromium(self):
        value = openai_register.BrowserPlatformRegistrar._user_agent_for_version(
            "149.0.7827.55"
        )

        self.assertIn("Chrome/149.0.7827.55", value)
        self.assertIn("X11; Linux x86_64", value)
        self.assertNotIn("HeadlessChrome", value)

    def test_browser_fallback_only_accepts_protocol_and_sentinel_failures(self):
        retryable = [
            openai_register.SentinelSDKError("Sentinel Node runner failed"),
            RuntimeError("Cloudflare clearance retry is still blocked"),
            RuntimeError('create_account_http_400, detail={"code":"registration_disallowed"}'),
        ]
        non_retryable = [
            RuntimeError('TempMail.lol HTTP 429 {"error":"Rate limited (free)"}'),
            RuntimeError("CloudflareTempMail request failed: HTTP 500"),
            RuntimeError("waiting for registration OTP timed out"),
            RuntimeError("Failed to create account. Please try again. mailbox domain rejected"),
        ]

        self.assertTrue(all(openai_register.should_use_browser_fallback(error) for error in retryable))
        self.assertTrue(all(not openai_register.should_use_browser_fallback(error) for error in non_retryable))

    def test_worker_retries_once_in_browser_with_next_pool_proxy(self):
        class TrackingLock:
            def __init__(self):
                self.active = False

            def __enter__(self):
                self.active = True

            def __exit__(self, *_args):
                self.active = False

        browser_lock = TrackingLock()
        selections = [
            SimpleNamespace(
                proxy="http://protocol.example:8080",
                source="text",
                count=2,
                last_error="",
                last_fetch=0.0,
            ),
            SimpleNamespace(
                proxy="http://browser.example:8080",
                source="text",
                count=2,
                last_error="",
                last_fetch=0.0,
            ),
        ]
        def next_proxy(**_kwargs):
            if len(selections) == 1:
                self.assertTrue(browser_lock.active)
            return selections.pop(0)

        pool = SimpleNamespace(
            next_proxy=next_proxy,
            record_result=unittest.mock.Mock(return_value={"bucket": "other", "cooldown_seconds": 0}),
        )
        protocol_instances = []
        browser_instances = []

        class FailingProtocolRegistrar:
            def __init__(self, proxy):
                self.proxy = proxy
                self.stage_timings = {"platform_authorize_ms": 12.0}
                self.closed = False
                protocol_instances.append(self)

            def register(self, _index):
                raise openai_register.SentinelSDKError("turnstile token is empty")

            def close(self):
                self.closed = True

        class SuccessfulBrowserRegistrar:
            def __init__(self, proxy):
                self.proxy = proxy
                browser_instances.append(self)

            def register(self, _index):
                return {
                    "email": "browser@example.com",
                    "password": "Password1!",
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "id_token": "id-token",
                    "source_type": "web",
                    "created_at": "2026-07-25T00:00:00+00:00",
                }

        with patch.object(openai_register, "proxy_pool", pool), patch.object(
            openai_register,
            "PlatformRegistrar",
            FailingProtocolRegistrar,
        ), patch.object(
            openai_register,
            "BrowserPlatformRegistrar",
            SuccessfulBrowserRegistrar,
        ), patch.object(
            openai_register.account_service,
            "add_account_items",
        ), patch.object(
            openai_register.account_service,
            "refresh_accounts",
            return_value={"errors": []},
        ), patch.object(
            openai_register,
            "log",
        ), patch.object(
            openai_register,
            "step",
        ), patch.object(
            openai_register,
            "browser_fallback_lock",
            browser_lock,
        ):
            original_mode = openai_register.config.get("proxy_input_mode")
            original_stats = dict(openai_register.stats)
            try:
                openai_register.config["proxy_input_mode"] = "text"
                openai_register.stats.update(
                    {"done": 0, "success": 0, "fail": 0, "start_time": 1.0}
                )
                result = openai_register.worker(1)
            finally:
                openai_register.config["proxy_input_mode"] = original_mode
                openai_register.stats.clear()
                openai_register.stats.update(original_stats)

        self.assertTrue(result["ok"])
        self.assertTrue(protocol_instances[0].closed)
        self.assertEqual(browser_instances[0].proxy, "http://browser.example:8080")
        self.assertEqual(pool.record_result.call_count, 2)
        first_record = pool.record_result.call_args_list[0]
        second_record = pool.record_result.call_args_list[1]
        self.assertEqual(first_record.args[0], "http://protocol.example:8080")
        self.assertFalse(first_record.kwargs["success"])
        self.assertEqual(second_record.args[0], "http://browser.example:8080")
        self.assertTrue(second_record.kwargs["success"])

    def test_create_session_uses_proxy_settings_without_breaking_existing_proxy_argument(self):
        fake_proxy = FakeProxySettings()
        created = []

        def fake_session_factory(**kwargs):
            session = FakeSession(**kwargs)
            created.append(session)
            return session

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register.requests,
            "Session",
            side_effect=fake_session_factory,
        ):
            session = openai_register.create_session("http://legacy-register.example:8080")

        self.assertIs(session, created[0])
        self.assertEqual(fake_proxy.session_kwargs_calls[0]["proxy"], "http://legacy-register.example:8080")
        self.assertTrue(fake_proxy.session_kwargs_calls[0]["upstream"])
        self.assertEqual(fake_proxy.session_kwargs_calls[0]["impersonate"], "chrome")
        self.assertFalse(fake_proxy.session_kwargs_calls[0]["verify"])
        self.assertEqual(session.kwargs["proxy"], "http://runtime.example:8118")

    def test_clearance_user_agent_updates_matching_client_hints(self):
        fake_proxy = FakeProxySettings()
        clearance_user_agent = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/149.0.7827.55 Safari/537.36"
        )

        with patch.object(openai_register, "proxy_settings", fake_proxy):
            headers = openai_register._headers_with_clearance(
                dict(openai_register.common_headers),
                "https://auth.openai.com/api/accounts/create_account",
                user_agent_override=clearance_user_agent,
            )

        normalized = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(normalized["user-agent"], clearance_user_agent)
        self.assertIn('v="149"', normalized["sec-ch-ua"])
        self.assertIn('149.0.7827.55', normalized["sec-ch-ua-full-version-list"])
        self.assertEqual(normalized["sec-ch-ua-platform"], '"Linux"')

    def test_cloudflare_without_clearance_keeps_clear_register_error(self):
        fake_proxy = FakeProxySettings(bundle=None)
        cf_response = FakeResponse(
            status_code=403,
            text="<html><title>Just a moment...</title></html>",
            headers={"server": "cloudflare", "content-type": "text/html"},
            url="https://auth.openai.com/api/accounts/authorize",
        )

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", return_value=(cf_response, "")):
            registrar = openai_register.PlatformRegistrar(proxy="http://legacy-register.example:8080")
            with self.assertRaisesRegex(RuntimeError, "Cloudflare") as ctx:
                registrar._platform_authorize("user@example.com", 1)

        self.assertEqual(len(fake_proxy.refresh_calls), 1)
        self.assertIn("status=403", str(ctx.exception))
        self.assertIn("Just a moment", str(ctx.exception))

    def test_openai_html_behind_cloudflare_is_not_treated_as_challenge(self):
        response = FakeResponse(
            status_code=200,
            text="""
            <!DOCTYPE html><html lang=\"en-US\"><head>
            <title>Create a password - OpenAI</title>
            </head><body>OpenAI account page</body></html>
            """,
            headers={"server": "cloudflare", "content-type": "text/html; charset=utf-8"},
            url="https://auth.openai.com/create-account/password",
        )

        self.assertFalse(openai_register._is_cloudflare_challenge(response))

    def test_cloudflare_challenge_refreshes_clearance_and_retries_once_with_matching_headers(self):
        bundle = ClearanceBundle(
            target_host="auth.openai.com",
            proxy_url="http://runtime.example:8118",
            cookies={"cf_clearance": "flare-token"},
            user_agent="Flare UA",
        )
        fake_proxy = FakeProxySettings(bundle=bundle)
        responses = [
            FakeResponse(
                status_code=403,
                text="<html><title>Just a moment...</title></html>",
                headers={"server": "cloudflare", "content-type": "text/html"},
                url="https://auth.openai.com/api/accounts/authorize",
            ),
            FakeResponse(status_code=200, text="{}", headers={"content-type": "application/json"}),
            FakeResponse(status_code=200, text="{}", headers={"content-type": "application/json"}),
        ]
        request_calls = []

        def fake_request(session, method, url, retry_attempts=3, **kwargs):
            request_calls.append({"method": method, "url": url, "headers": dict(kwargs.get("headers") or {})})
            return responses.pop(0), ""

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", side_effect=fake_request):
            registrar = openai_register.PlatformRegistrar(proxy="http://legacy-register.example:8080")
            registrar.sentinel_sdk = FakeSentinelSDKClient()
            registrar._platform_authorize("user@example.com", 1)

        self.assertEqual(len(request_calls), 3)
        self.assertEqual(len(fake_proxy.refresh_calls), 1)
        retry_headers = {key.lower(): value for key, value in request_calls[1]["headers"].items()}
        self.assertEqual(retry_headers["user-agent"], "Flare UA")
        self.assertEqual(retry_headers["cookie"], "cf_clearance=flare-token")
        self.assertEqual(fake_proxy.refresh_calls[0]["target_url"], openai_register.auth_base)
        self.assertEqual(fake_proxy.refresh_calls[0]["proxy"], "http://legacy-register.example:8080")
        self.assertTrue(fake_proxy.refresh_calls[0]["force"])

    def test_refresh_failure_reports_cloudflare_detail_without_infinite_retry(self):
        fake_proxy = FakeProxySettings(bundle=None)
        cf_response = FakeResponse(
            status_code=403,
            text="<html><title>Just a moment...</title><body>challenge body</body></html>",
            headers={"server": "cloudflare", "content-type": "text/html"},
            url="https://auth.openai.com/api/accounts/authorize",
        )
        request_calls = []

        def fake_request(session, method, url, retry_attempts=3, **kwargs):
            request_calls.append({"method": method, "url": url})
            return cf_response, ""

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", side_effect=fake_request):
            registrar = openai_register.PlatformRegistrar(proxy="")
            with self.assertRaisesRegex(RuntimeError, "Cloudflare") as ctx:
                registrar._platform_authorize("user@example.com", 1)

        self.assertEqual(len(request_calls), 1)
        self.assertEqual(len(fake_proxy.refresh_calls), 1)
        message = str(ctx.exception)
        self.assertIn("status=403", message)
        self.assertIn("challenge body", message)

    def test_platform_authorize_submits_email_through_authorize_continue(self):
        fake_proxy = FakeProxySettings()
        sentinel = FakeSentinelSDKClient()
        responses = [
            FakeResponse(
                status_code=200,
                text="<html>signup</html>",
                headers={"content-type": "text/html"},
                url="https://auth.openai.com/create-account",
            ),
            FakeResponse(
                status_code=200,
                text='{"page":{"type":"create_account_password"}}',
                headers={"content-type": "application/json"},
                url="https://auth.openai.com/api/accounts/authorize/continue",
                json_data={"page": {"type": "create_account_password"}},
            ),
        ]
        request_calls = []

        def fake_request(session, method, url, retry_attempts=3, **kwargs):
            request_calls.append({"method": method, "url": url, **kwargs})
            return responses.pop(0), ""

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", side_effect=fake_request):
            registrar = openai_register.PlatformRegistrar(proxy="")
            registrar.sentinel_sdk = sentinel
            registrar._platform_authorize("user@example.com", 1)

        self.assertEqual(len(request_calls), 2)
        continue_call = request_calls[1]
        self.assertEqual(continue_call["method"], "post")
        self.assertEqual(continue_call["url"], "https://auth.openai.com/api/accounts/authorize/continue")
        self.assertEqual(
            continue_call["json"],
            {"username": {"kind": "email", "value": "user@example.com"}},
        )
        continue_headers = {key.lower(): value for key, value in continue_call["headers"].items()}
        self.assertEqual(continue_headers["openai-sentinel-token"], "sentinel-secret-value")
        self.assertNotIn("openai-sentinel-so-token", continue_headers)
        self.assertEqual(sentinel.calls, [("authorize_continue", False)])

    def test_platform_authorize_skips_duplicate_continue_after_password_landing(self):
        fake_proxy = FakeProxySettings()
        sentinel = FakeSentinelSDKClient()
        response = FakeResponse(
            status_code=200,
            text="<html>Create a password</html>",
            headers={"content-type": "text/html"},
            url="https://auth.openai.com/create-account/password",
        )
        request_calls = []

        def fake_request(session, method, url, retry_attempts=3, **kwargs):
            request_calls.append({"method": method, "url": url, **kwargs})
            return response, ""

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", side_effect=fake_request):
            registrar = openai_register.PlatformRegistrar(proxy="")
            registrar.sentinel_sdk = sentinel
            registrar._platform_authorize("user@example.com", 1)

        self.assertEqual(len(request_calls), 1)
        self.assertEqual(sentinel.calls, [])

    def test_create_account_uses_sdk_sentinel_and_so_tokens_without_logging_values(self):
        fake_proxy = FakeProxySettings()
        sentinel = FakeSentinelSDKClient()
        request_calls = []
        log_lines = []

        def fake_request(session, method, url, retry_attempts=3, **kwargs):
            request_calls.append({"method": method, "url": url, **kwargs})
            return FakeResponse(
                status_code=200,
                text="{}",
                headers={"content-type": "application/json"},
                url=url,
                json_data={},
            ), ""

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ), patch.object(openai_register, "request_with_local_retry", side_effect=fake_request), patch.object(
            openai_register,
            "build_sentinel_token",
            side_effect=AssertionError("legacy sentinel builder must not be used"),
        ), patch.object(openai_register, "step", side_effect=lambda _index, text, _color="": log_lines.append(text)):
            registrar = openai_register.PlatformRegistrar(proxy="")
            registrar.sentinel_sdk = sentinel
            registrar._create_account("Test User", "2000-01-01", 1)

        self.assertEqual(sentinel.calls, [("oauth_create_account", True)])
        self.assertEqual(len(request_calls), 1)
        headers = {key.lower(): value for key, value in request_calls[0]["headers"].items()}
        self.assertEqual(headers["openai-sentinel-token"], "sentinel-secret-value")
        self.assertEqual(headers["openai-sentinel-so-token"], "so-secret-value")
        logs = "\n".join(log_lines)
        self.assertIn("sdk=20260219f9f6", logs)
        self.assertIn("token_len=21", logs)
        self.assertIn("so_token=yes", logs)
        self.assertNotIn("sentinel-secret-value", logs)
        self.assertNotIn("so-secret-value", logs)

    def test_sentinel_uses_cloudflare_clearance_user_agent(self):
        fake_proxy = FakeProxySettings()
        sentinel = FakeSentinelSDKClient()

        with patch.object(openai_register, "proxy_settings", fake_proxy), patch.object(
            openai_register,
            "create_session",
            return_value=FakeSession(),
        ):
            registrar = openai_register.PlatformRegistrar(proxy="")
            registrar.sentinel_sdk = sentinel
            registrar.clearance_user_agent = "Cloudflare Browser UA"
            registrar._sentinel_headers("authorize_continue", 1)

        self.assertEqual(sentinel.user_agent, "Cloudflare Browser UA")


if __name__ == "__main__":
    unittest.main()
