from __future__ import annotations

import json
import hashlib
import os
import re
import secrets
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urljoin, urlparse


class SentinelSDKError(RuntimeError):
    """Raised when the official Sentinel SDK cannot produce valid tokens."""


@dataclass(frozen=True)
class SentinelSDKTokens:
    token: str
    so_token: str
    sdk_version: str


def _default_process_runner(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    allowed_environment = {
        key: value
        for key in (
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "USERPROFILE",
            "WINDIR",
        )
        if (value := os.environ.get(key))
    }
    allowed_environment["NODE_NO_WARNINGS"] = "1"
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=allowed_environment,
    )


class SentinelSDKClient:
    """Execute the current official Sentinel SDK in a lightweight Node VM."""

    FRAME_URL = "https://sentinel.openai.com/backend-api/sentinel/frame.html"
    SENTINEL_REQ_URL = "https://sentinel.openai.com/backend-api/sentinel/req"
    PAGE_URL = "https://auth.openai.com/create-account"
    REQUEST_TIMEOUT_SECS = 20
    PROCESS_TIMEOUT_SECS = 75

    _SDK_PATTERN = re.compile(
        r"/sentinel/(?P<version>[A-Za-z0-9._-]+)/sdk\.js(?:$|[?#])"
    )
    _SCRIPT_PATTERN = re.compile(
        r"<script\b[^>]*\bsrc\s*=\s*['\"](?P<src>[^'\"]+)['\"]",
        re.IGNORECASE,
    )
    _CHROME_VERSION_PATTERN = re.compile(
        r"(?:Chrome|Chromium)/(?P<version>\d+(?:\.\d+){0,3})",
        re.IGNORECASE,
    )
    _NODE_PERMISSION_FLAGS: dict[str, str] = {}
    _NODE_PERMISSION_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        session: Any,
        device_id: str,
        user_agent: str,
        cache_dir: str | Path | None = None,
        process_runner: Callable[..., Any] | None = None,
        node_binary: str = "",
        runner_path: str | Path | None = None,
    ) -> None:
        self.session = session
        self.device_id = str(device_id or "").strip()
        self.user_agent = str(user_agent or "").strip()
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("SENTINEL_SDK_CACHE_DIR")
            or Path(__file__).resolve().parents[1] / "data" / "sentinel_sdk"
        )
        self.process_runner = process_runner or _default_process_runner
        self.node_binary = str(
            node_binary or os.environ.get("SENTINEL_NODE_BINARY") or "node"
        ).strip()
        self.runner_path = Path(
            runner_path or Path(__file__).with_name("sentinel_node_runner.js")
        )
        self.sdk_url = ""
        self.sdk_version = "unknown"
        self._sdk_path: Path | None = None

    @classmethod
    def _parse_sdk_url(cls, frame_html: str) -> tuple[str, str]:
        for match in cls._SCRIPT_PATTERN.finditer(str(frame_html or "")):
            sdk_url = urljoin(cls.FRAME_URL, match.group("src"))
            parsed = urlparse(sdk_url)
            version_match = cls._SDK_PATTERN.search(sdk_url)
            if (
                parsed.scheme == "https"
                and parsed.hostname == "sentinel.openai.com"
                and version_match
            ):
                return sdk_url, version_match.group("version")
        raise SentinelSDKError("current Sentinel SDK URL was not found in frame.html")

    @staticmethod
    def _response_json(response: Any) -> dict[str, Any]:
        try:
            value = response.json()
        except Exception:
            try:
                value = json.loads(str(getattr(response, "text", "") or "{}"))
            except Exception:
                value = {}
        return value if isinstance(value, dict) else {}

    def _request_headers(self) -> dict[str, str]:
        return {
            "accept": "*/*",
            "content-type": "text/plain;charset=UTF-8",
            "origin": "https://sentinel.openai.com",
            "referer": self.FRAME_URL,
            "user-agent": self.user_agent,
            "oai-device-id": self.device_id,
        }

    @classmethod
    def client_hints_for_user_agent(cls, user_agent: str) -> dict[str, str]:
        value = str(user_agent or "")
        version_match = cls._CHROME_VERSION_PATTERN.search(value)
        full_version = version_match.group("version") if version_match else "145.0.0.0"
        full_version = ".".join((full_version.split(".") + ["0", "0", "0"])[0:4])
        major = full_version.split(".", 1)[0]
        lowered = value.lower()
        if "windows" in lowered:
            platform = "Windows"
            navigator_platform = "Win32"
            platform_version = "10.0.0"
            architecture = "x86_64"
        elif "macintosh" in lowered or "mac os" in lowered:
            platform = "macOS"
            navigator_platform = "MacIntel"
            platform_version = "10.15.7"
            architecture = "arm" if "arm" in lowered else "x86_64"
        else:
            platform = "Linux"
            navigator_platform = "Linux x86_64"
            platform_version = ""
            architecture = "x86_64"
        return {
            "chrome_major": major,
            "chrome_full_version": full_version,
            "navigator_platform": navigator_platform,
            "platform": platform,
            "platform_version": platform_version,
            "architecture": architecture,
            "sec_ch_ua": (
                f'"Google Chrome";v="{major}", "Not?A_Brand";v="8", '
                f'"Chromium";v="{major}"'
            ),
            "sec_ch_ua_full_version_list": (
                f'"Chromium";v="{full_version}", "Not:A-Brand";v="99.0.0.0", '
                f'"Google Chrome";v="{full_version}"'
            ),
        }

    @staticmethod
    def _permission_flag_for_version(version: str) -> str:
        match = re.search(r"(?:^|\D)(\d+)", str(version or ""))
        major = int(match.group(1)) if match else 20
        return "--permission" if major >= 23 else "--experimental-permission"

    def _node_permission_flag(self) -> str:
        with self._NODE_PERMISSION_LOCK:
            cached = self._NODE_PERMISSION_FLAGS.get(self.node_binary)
            if cached:
                return cached
            try:
                result = _default_process_runner(
                    [self.node_binary, "--version"],
                    timeout=5,
                )
                version = (
                    str(getattr(result, "stdout", "") or "")
                    if int(getattr(result, "returncode", 1) or 0) == 0
                    else ""
                )
            except Exception:
                version = ""
            flag = self._permission_flag_for_version(version)
            self._NODE_PERMISSION_FLAGS[self.node_binary] = flag
            return flag

    @staticmethod
    def _cache_matches(path: Path, metadata_path: Path, sdk_url: str) -> bool:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            content = path.read_bytes()
        except Exception:
            return False
        return bool(content) and metadata == {
            "sdk_url": sdk_url,
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _ensure_sdk(self) -> Path:
        if self._sdk_path is not None and self._sdk_path.is_file():
            return self._sdk_path
        if not self.device_id:
            raise SentinelSDKError("Sentinel SDK device_id is required")
        if not self.user_agent:
            raise SentinelSDKError("Sentinel SDK user_agent is required")

        try:
            frame_response = self.session.get(
                self.FRAME_URL,
                headers={"user-agent": self.user_agent},
                timeout=self.REQUEST_TIMEOUT_SECS,
                verify=True,
            )
        except Exception as exc:
            raise SentinelSDKError(f"failed to load Sentinel frame: {exc}") from exc
        if int(getattr(frame_response, "status_code", 0) or 0) != 200:
            raise SentinelSDKError(
                f"Sentinel frame returned HTTP {getattr(frame_response, 'status_code', 'unknown')}"
            )

        self.sdk_url, self.sdk_version = self._parse_sdk_url(
            str(getattr(frame_response, "text", "") or "")
        )
        sdk_path = self.cache_dir / self.sdk_version / "sdk.js"
        metadata_path = sdk_path.with_suffix(".verified.json")
        if not self._cache_matches(sdk_path, metadata_path, self.sdk_url):
            try:
                sdk_response = self.session.get(
                    self.sdk_url,
                    headers={
                        "referer": self.FRAME_URL,
                        "user-agent": self.user_agent,
                    },
                    timeout=self.REQUEST_TIMEOUT_SECS,
                    verify=True,
                )
            except Exception as exc:
                raise SentinelSDKError(f"failed to download Sentinel SDK: {exc}") from exc
            if int(getattr(sdk_response, "status_code", 0) or 0) != 200:
                raise SentinelSDKError(
                    f"Sentinel SDK returned HTTP {getattr(sdk_response, 'status_code', 'unknown')}"
                )
            content = bytes(getattr(sdk_response, "content", b"") or b"")
            if not content:
                content = str(getattr(sdk_response, "text", "") or "").encode("utf-8")
            if not content:
                raise SentinelSDKError("Sentinel SDK response was empty")
            sdk_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = sdk_path.with_suffix(f".{secrets.token_hex(4)}.tmp")
            temporary_path.write_bytes(content)
            temporary_path.replace(sdk_path)
            metadata = {
                "sdk_url": self.sdk_url,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            metadata_path.write_text(
                json.dumps(metadata, separators=(",", ":")),
                encoding="utf-8",
            )
        self._sdk_path = sdk_path
        return sdk_path

    def _store_oai_sc_cookie(self, challenge: dict[str, Any]) -> None:
        challenge_token = str(challenge.get("token") or "").strip()
        cookies = getattr(self.session, "cookies", None)
        if not challenge_token or cookies is None or not hasattr(cookies, "set"):
            return
        cookies.set("oai-sc", f"0{challenge_token}", domain=".openai.com")

    @contextmanager
    def _challenge_relay(self) -> Iterator[str]:
        client = self
        relay_path = f"/sentinel/{secrets.token_urlsafe(18)}"

        class RelayHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path != relay_path:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("content-length") or 0)
                    if length <= 0 or length > 1_000_000:
                        raise ValueError("invalid challenge request length")
                    body = self.rfile.read(length)
                    request_data = json.loads(body.decode("utf-8"))
                    if not isinstance(request_data, dict):
                        raise ValueError("challenge request must be an object")
                    if str(request_data.get("id") or "") != client.device_id:
                        raise ValueError("challenge device id mismatch")
                    if not str(request_data.get("flow") or "").strip():
                        raise ValueError("challenge flow is missing")
                    response = client.session.post(
                        client.SENTINEL_REQ_URL,
                        data=body.decode("utf-8"),
                        headers=client._request_headers(),
                        timeout=client.REQUEST_TIMEOUT_SECS,
                        verify=True,
                    )
                    status_code = int(getattr(response, "status_code", 502) or 502)
                    challenge = client._response_json(response)
                    if 200 <= status_code < 300:
                        client._store_oai_sc_cookie(challenge)
                    response_text = str(getattr(response, "text", "") or "")
                    payload = response_text.encode("utf-8") if response_text else json.dumps(
                        challenge, separators=(",", ":")
                    ).encode("utf-8")
                except Exception as exc:
                    status_code = 502
                    payload = json.dumps(
                        {"error": f"Sentinel challenge relay failed: {exc}"},
                        separators=(",", ":"),
                    ).encode("utf-8")
                self.send_response(status_code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            yield f"http://{host}:{port}{relay_path}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    @staticmethod
    def _parse_runner_output(stdout: Any) -> dict[str, Any]:
        text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else str(stdout or "")
        for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise SentinelSDKError("Sentinel Node runner returned no JSON result")

    def _validate_token(self, raw_token: str, flow: str) -> str:
        try:
            token = json.loads(raw_token)
        except Exception as exc:
            raise SentinelSDKError(f"Sentinel token is invalid JSON: {exc}") from exc
        if not isinstance(token, dict):
            raise SentinelSDKError("Sentinel token must be an object")
        for field in ("p", "c", "id", "flow"):
            if not str(token.get(field) or "").strip():
                raise SentinelSDKError(f"Sentinel token field {field} is empty")
        if not str(token.get("t") or "").strip():
            raise SentinelSDKError("Sentinel turnstile token is empty")
        if str(token.get("id")) != self.device_id or str(token.get("flow")) != flow:
            raise SentinelSDKError("Sentinel token context mismatch")
        return raw_token

    def _validate_so_token(
        self,
        raw_token: str,
        flow: str,
        challenge_token: str,
    ) -> str:
        try:
            token = json.loads(raw_token)
        except Exception as exc:
            raise SentinelSDKError(f"Sentinel session observer token is invalid JSON: {exc}") from exc
        if not isinstance(token, dict) or not str(token.get("so") or "").strip():
            raise SentinelSDKError("Sentinel session observer token is empty")
        if not str(token.get("c") or "").strip():
            raise SentinelSDKError("Sentinel session observer challenge is empty")
        if str(token.get("id")) != self.device_id or str(token.get("flow")) != flow:
            raise SentinelSDKError("Sentinel session observer token context mismatch")
        if str(token.get("c")) != challenge_token:
            raise SentinelSDKError("Sentinel session observer challenge mismatch")
        return raw_token

    def get_tokens(self, flow: str, *, include_so: bool = False) -> SentinelSDKTokens:
        flow_name = str(flow or "").strip()
        if not flow_name:
            raise ValueError("Sentinel flow is required")
        sdk_path = self._ensure_sdk()
        if not self.runner_path.is_file():
            raise SentinelSDKError(f"Sentinel Node runner is missing: {self.runner_path}")

        with self._challenge_relay() as challenge_url:
            fingerprint = self.client_hints_for_user_agent(self.user_agent)
            command = [
                self.node_binary,
                self._node_permission_flag(),
                f"--allow-fs-read={self.runner_path}",
                f"--allow-fs-read={sdk_path}",
                str(self.runner_path),
                "--no-config",
                "1",
                "--sdk",
                str(sdk_path),
                "--flow",
                flow_name,
                "--device-id",
                self.device_id,
                "--user-agent",
                self.user_agent,
                "--navigator-platform",
                fingerprint["navigator_platform"],
                "--user-agent-data-platform",
                fingerprint["platform"],
                "--language",
                "en-US",
                "--languages",
                "en-US,en",
                "--time-zone",
                "UTC",
                "--timezone-name",
                "Coordinated Universal Time",
                "--timezone-offset-minutes",
                "0",
                "--chrome-major",
                fingerprint["chrome_major"],
                "--chrome-full-version",
                fingerprint["chrome_full_version"],
                "--sec-ch-ua",
                fingerprint["sec_ch_ua"],
                "--sec-ch-ua-platform",
                fingerprint["platform"],
                "--sec-ch-ua-full-version-list",
                fingerprint["sec_ch_ua_full_version_list"],
                "--sec-ch-ua-platform-version",
                fingerprint["platform_version"],
                "--sec-ch-ua-arch",
                fingerprint["architecture"],
                "--sec-ch-ua-bitness",
                "64",
                "--device-pixel-ratio",
                "1",
                "--width",
                "1920",
                "--height",
                "1080",
                "--page-url",
                self.PAGE_URL,
                "--script-src",
                self.sdk_url,
                "--challenge-url",
                challenge_url,
                "--include-so",
                "1" if include_so else "0",
                "--no-cookie",
                "1",
            ]
            try:
                result = self.process_runner(command, timeout=self.PROCESS_TIMEOUT_SECS)
            except subprocess.TimeoutExpired as exc:
                raise SentinelSDKError("Sentinel Node runner timed out") from exc
            except Exception as exc:
                raise SentinelSDKError(f"Sentinel Node runner failed to start: {exc}") from exc

        if int(getattr(result, "returncode", 1) or 0) != 0:
            raise SentinelSDKError(
                f"Sentinel Node runner failed with exit code {result.returncode}"
            )
        output = self._parse_runner_output(getattr(result, "stdout", ""))
        token = self._validate_token(str(output.get("token") or "").strip(), flow_name)
        so_token = str(output.get("soToken") or "").strip()
        if include_so:
            challenge_token = str(json.loads(token).get("c") or "")
            so_token = self._validate_so_token(
                so_token,
                flow_name,
                challenge_token,
            )
        return SentinelSDKTokens(
            token=token,
            so_token=so_token,
            sdk_version=self.sdk_version,
        )

    def close(self) -> None:
        return
