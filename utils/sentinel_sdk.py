from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import unquote, urlparse


@dataclass(frozen=True)
class SentinelSDKTokens:
    token: str
    so_token: str
    sdk_version: str


class SentinelSDKClient:
    """Run the current official Sentinel SDK in Chromium for one register worker."""

    FRAME_URL = "https://sentinel.openai.com/backend-api/sentinel/frame.html"
    OBSERVER_WAIT_MS = 5000
    PAGE_TIMEOUT_MS = 60000

    _TOKEN_SCRIPT = """
    async ({ flow, includeSo, observerWaitMs }) => {
        const sdk = window.SentinelSDK;
        if (!sdk || typeof sdk.init !== "function" || typeof sdk.token !== "function") {
            throw new Error("SentinelSDK is unavailable");
        }
        const withTimeout = (value, timeoutMs, label) => Promise.race([
            Promise.resolve(value),
            new Promise((_, reject) => setTimeout(
                () => reject(new Error(`${label} timed out after ${timeoutMs}ms`)),
                timeoutMs,
            )),
        ]);

        // init() performs the official Sentinel req for this exact flow.
        await withTimeout(sdk.init(flow), 15000, "Sentinel init");
        const token = await withTimeout(sdk.token(flow), 45000, "Sentinel token");
        let soToken = "";
        if (includeSo) {
            if (typeof sdk.sessionObserverToken !== "function") {
                throw new Error("SentinelSDK.sessionObserverToken is unavailable");
            }
            await new Promise((resolve) => setTimeout(resolve, observerWaitMs));
            soToken = await withTimeout(
                sdk.sessionObserverToken(flow),
                45000,
                "Sentinel session observer token",
            );
        }
        return { token: token || "", soToken: soToken || "" };
    }
    """

    def __init__(
        self,
        *,
        device_id: str,
        user_agent: str,
        proxy: str = "",
        playwright_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.device_id = str(device_id or "").strip()
        self.user_agent = str(user_agent or "").strip()
        self.proxy = str(proxy or "").strip()
        self._playwright_factory = playwright_factory
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self.sdk_url = ""
        self.sdk_version = "unknown"

    @staticmethod
    def _proxy_config(proxy: str) -> dict[str, str] | None:
        value = str(proxy or "").strip()
        if not value:
            return None
        if "://" not in value:
            value = f"http://{value}"
        parsed = urlparse(value)
        if not parsed.hostname:
            raise ValueError("Sentinel browser proxy is invalid")
        scheme = "socks5" if parsed.scheme.lower() == "socks5h" else parsed.scheme.lower()
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        server = f"{scheme}://{host}"
        if parsed.port:
            server += f":{parsed.port}"
        result = {"server": server}
        if parsed.username:
            result["username"] = unquote(parsed.username)
        if parsed.password:
            result["password"] = unquote(parsed.password)
        return result

    @staticmethod
    def _sdk_version(script_urls: list[str]) -> tuple[str, str]:
        for url in script_urls:
            match = re.search(r"/sentinel/([^/]+)/sdk\.js(?:$|\?)", str(url or ""))
            if match:
                return str(url), match.group(1)
        return "", "unknown"

    def _start(self) -> None:
        if self._page is not None:
            return
        if not self.device_id:
            raise RuntimeError("Sentinel SDK device_id is required")
        if not self.user_agent:
            raise RuntimeError("Sentinel SDK user_agent is required")

        if self._playwright_factory is None:
            try:
                from playwright.sync_api import sync_playwright
            except Exception as exc:
                raise RuntimeError(
                    "Playwright is required for official Sentinel SDK tokens"
                ) from exc
            self._playwright_factory = sync_playwright

        starter = self._playwright_factory()
        self._playwright = starter.start()
        launch_kwargs: dict[str, Any] = {
            "headless": True,
            "args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        }
        proxy_config = self._proxy_config(self.proxy)
        if proxy_config:
            launch_kwargs["proxy"] = proxy_config
        self._browser = self._playwright.chromium.launch(**launch_kwargs)
        self._context = self._browser.new_context(
            user_agent=self.user_agent,
            locale="en-US",
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True,
        )
        self._context.add_cookies(
            [
                {
                    "name": "oai-did",
                    "value": self.device_id,
                    "domain": ".openai.com",
                    "path": "/",
                    "secure": True,
                    "sameSite": "Lax",
                }
            ]
        )
        self._page = self._context.new_page()
        self._page.goto(
            self.FRAME_URL,
            wait_until="load",
            timeout=self.PAGE_TIMEOUT_MS,
        )
        self._page.wait_for_function(
            "() => !!window.SentinelSDK && typeof window.SentinelSDK.token === 'function'",
            timeout=30000,
        )
        script_urls = self._page.eval_on_selector_all(
            "script[src]",
            "(scripts) => scripts.map((script) => script.src).filter(Boolean)",
        )
        self.sdk_url, self.sdk_version = self._sdk_version(
            [str(item) for item in (script_urls or [])]
        )

    def get_tokens(self, flow: str, *, include_so: bool = False) -> SentinelSDKTokens:
        flow_name = str(flow or "").strip()
        if not flow_name:
            raise ValueError("Sentinel flow is required")
        self._start()
        result = self._page.evaluate(
            self._TOKEN_SCRIPT,
            {
                "flow": flow_name,
                "includeSo": bool(include_so),
                "observerWaitMs": self.OBSERVER_WAIT_MS,
            },
        )
        result = result if isinstance(result, dict) else {}
        token = str(result.get("token") or "").strip()
        so_token = str(result.get("soToken") or "").strip()
        if not token:
            raise RuntimeError(f"Sentinel SDK returned an empty token for {flow_name}")
        if include_so and not so_token:
            raise RuntimeError(
                f"Sentinel SDK returned an empty session observer token for {flow_name}"
            )
        return SentinelSDKTokens(
            token=token,
            so_token=so_token,
            sdk_version=self.sdk_version,
        )

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._browser = None
            self._context = None
            self._page = None
            self._playwright = None
