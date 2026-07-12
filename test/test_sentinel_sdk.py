from __future__ import annotations

import unittest

from utils.sentinel_sdk import SentinelSDKClient


class FakePage:
    def __init__(self):
        self.goto_calls = []
        self.wait_calls = []
        self.evaluate_calls = []

    def goto(self, url, **kwargs):
        self.goto_calls.append((url, kwargs))

    def wait_for_function(self, expression, **kwargs):
        self.wait_calls.append((expression, kwargs))

    def eval_on_selector_all(self, selector, expression):
        return ["https://sentinel.openai.com/sentinel/20260219f9f6/sdk.js"]

    def evaluate(self, expression, payload):
        self.evaluate_calls.append((expression, payload))
        return {
            "token": "sentinel-token",
            "soToken": "sentinel-so-token",
        }


class FakeContext:
    def __init__(self, page):
        self.page = page
        self.cookies = []

    def add_cookies(self, cookies):
        self.cookies.extend(cookies)

    def new_page(self):
        return self.page


class FakeBrowser:
    def __init__(self, page):
        self.page = page
        self.context_kwargs = None
        self.context = FakeContext(page)
        self.closed = False

    def new_context(self, **kwargs):
        self.context_kwargs = kwargs
        return self.context

    def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, browser):
        self.browser = browser
        self.launch_kwargs = None

    def launch(self, **kwargs):
        self.launch_kwargs = kwargs
        return self.browser


class FakePlaywright:
    def __init__(self):
        self.page = FakePage()
        self.browser = FakeBrowser(self.page)
        self.chromium = FakeChromium(self.browser)
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakePlaywrightStarter:
    def __init__(self, playwright):
        self.playwright = playwright

    def start(self):
        return self.playwright


class SentinelSDKClientTests(unittest.TestCase):
    def test_generates_token_pair_with_current_sdk_and_5000ms_observer_wait(self):
        playwright = FakePlaywright()
        client = SentinelSDKClient(
            device_id="device-id",
            user_agent="Test Browser UA",
            proxy="http://user:pass@proxy.example:8080",
            playwright_factory=lambda: FakePlaywrightStarter(playwright),
        )

        result = client.get_tokens("oauth_create_account", include_so=True)
        client.close()

        self.assertEqual(result.token, "sentinel-token")
        self.assertEqual(result.so_token, "sentinel-so-token")
        self.assertEqual(result.sdk_version, "20260219f9f6")
        self.assertEqual(playwright.page.goto_calls[0][0], SentinelSDKClient.FRAME_URL)
        _, payload = playwright.page.evaluate_calls[0]
        self.assertEqual(payload["flow"], "oauth_create_account")
        self.assertTrue(payload["includeSo"])
        self.assertEqual(payload["observerWaitMs"], 5000)
        self.assertEqual(
            playwright.chromium.launch_kwargs["proxy"],
            {"server": "http://proxy.example:8080", "username": "user", "password": "pass"},
        )
        self.assertEqual(playwright.browser.context_kwargs["user_agent"], "Test Browser UA")
        self.assertTrue(playwright.browser.closed)
        self.assertTrue(playwright.stopped)


if __name__ == "__main__":
    unittest.main()
