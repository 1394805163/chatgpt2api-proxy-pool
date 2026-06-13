from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

from curl_cffi import requests

from services.proxy_service import normalize_proxy_url


PROXY_INPUT_MODES = {"single", "url", "text"}


@dataclass(frozen=True)
class ProxyPoolSelection:
    proxy: str
    source: str
    count: int
    last_error: str
    last_fetch: float


@dataclass(frozen=True)
class ProxyPoolState:
    mode: str
    source: str
    count: int
    current_proxy: str
    last_fetch: float
    last_error: str


def normalize_proxy_input_mode(value: object) -> str:
    mode = str(value or "single").strip().lower()
    return mode if mode in PROXY_INPUT_MODES else "single"


def normalize_proxy_refresh_interval(value: object) -> int:
    try:
        return max(10, int(value or 120))
    except (OverflowError, TypeError, ValueError):
        return 120


def parse_proxy_lines(text: str) -> list[str]:
    proxies: list[str] = []
    seen: set[str] = set()
    for raw_line in str(text or "").replace(",", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        proxy = normalize_proxy_url(line)
        if not _is_supported_proxy_url(proxy) or proxy in seen:
            continue
        seen.add(proxy)
        proxies.append(proxy)
    return proxies


def _is_supported_proxy_url(proxy: str) -> bool:
    parsed = urlparse(proxy)
    return parsed.scheme in {"http", "https", "socks5", "socks5h"} and bool(parsed.netloc)


def _is_supported_source_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class RegisterProxyPool:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode = "single"
        self._single_proxy = ""
        self._proxy_url = ""
        self._proxy_list_text = ""
        self._refresh_interval = 120
        self._proxies: list[str] = []
        self._index = 0
        self._current_proxy = ""
        self._last_fetch = 0.0
        self._last_error = ""

    def configure(
        self,
        *,
        mode: str,
        single_proxy: str,
        proxy_url: str,
        proxy_list_text: str,
        refresh_interval: int,
        fetch_now: bool = False,
    ) -> ProxyPoolState:
        normalized_mode = normalize_proxy_input_mode(mode)
        normalized_single = normalize_proxy_url(single_proxy)
        normalized_url = str(proxy_url or "").strip()
        normalized_text = str(proxy_list_text or "")
        normalized_interval = normalize_proxy_refresh_interval(refresh_interval)

        with self._lock:
            changed_source = (
                normalized_mode != self._mode
                or normalized_single != self._single_proxy
                or normalized_url != self._proxy_url
                or normalized_text != self._proxy_list_text
            )
            self._mode = normalized_mode
            self._single_proxy = normalized_single
            self._proxy_url = normalized_url
            self._proxy_list_text = normalized_text
            self._refresh_interval = normalized_interval
            self._last_error = ""
            if changed_source:
                self._index = 0
                self._current_proxy = ""
            if normalized_mode == "single":
                self._proxies = [normalized_single] if normalized_single else []
                self._last_fetch = 0.0
            elif normalized_mode == "text":
                self._proxies = parse_proxy_lines(normalized_text)
                self._last_fetch = 0.0
            elif changed_source:
                self._proxies = []
                self._last_fetch = 0.0

        if normalized_mode == "url" and fetch_now:
            self.refresh_url(force=fetch_now)
        return self.state()

    def prepare(self) -> ProxyPoolState:
        with self._lock:
            mode = self._mode
        if mode == "url":
            self.refresh_url(force=True)
        state = self.state()
        if state.mode in {"url", "text"} and state.count == 0:
            message = state.last_error or f"no proxies available for {state.mode} proxy source"
            raise RuntimeError(message)
        return state

    def next_proxy(self) -> ProxyPoolSelection:
        with self._lock:
            mode = self._mode
        if mode == "url" and self._should_refresh():
            self.refresh_url(force=False)

        with self._lock:
            if self._mode == "single":
                return ProxyPoolSelection(
                    proxy=self._single_proxy,
                    source="single" if self._single_proxy else "default",
                    count=len(self._proxies),
                    last_error=self._last_error,
                    last_fetch=self._last_fetch,
                )
            if not self._proxies:
                return ProxyPoolSelection(proxy="", source=self._mode, count=0, last_error=self._last_error, last_fetch=self._last_fetch)
            proxy = self._proxies[self._index % len(self._proxies)]
            self._index += 1
            self._current_proxy = proxy
            return ProxyPoolSelection(
                proxy=proxy,
                source=self._mode,
                count=len(self._proxies),
                last_error=self._last_error,
                last_fetch=self._last_fetch,
            )

    def refresh_url(self, *, force: bool) -> ProxyPoolState:
        with self._lock:
            url = self._proxy_url
            if self._mode != "url":
                return self.state()
            if not force and not self._should_refresh_locked():
                return self.state()

        if not _is_supported_source_url(url):
            self._record_error("proxy_url must be a valid http or https URL")
            return self.state()

        try:
            response = requests.get(url, timeout=15, verify=False)
            response.raise_for_status()
            proxies = parse_proxy_lines(response.text)
            with self._lock:
                self._proxies = proxies
                self._index = 0
                self._last_fetch = time.time()
                self._last_error = "" if proxies else "proxy URL returned no valid proxies"
        except Exception as error:
            self._record_error(f"failed to fetch proxy URL: {error}")
        return self.state()

    def state(self) -> ProxyPoolState:
        with self._lock:
            return ProxyPoolState(
                mode=self._mode,
                source=self._mode,
                count=len(self._proxies),
                current_proxy=self._current_proxy,
                last_fetch=self._last_fetch,
                last_error=self._last_error,
            )

    def _should_refresh(self) -> bool:
        with self._lock:
            return self._should_refresh_locked()

    def _should_refresh_locked(self) -> bool:
        if self._mode != "url":
            return False
        if not self._last_fetch:
            return True
        return time.time() - self._last_fetch >= self._refresh_interval

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._last_error = str(message or "").strip()
