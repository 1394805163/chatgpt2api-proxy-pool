from __future__ import annotations

import base64
import hashlib
import json
import random
import secrets
import string
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse

from curl_cffi import requests

from services.account_service import account_service
from services.proxy_service import ClearanceBundle, proxy_settings
from services.register import mail_provider
from services.register.proxy_pool import RegisterProxyPool
from utils.sentinel_sdk import SentinelSDKClient, SentinelSDKError, SentinelSDKTokens

base_dir = Path(__file__).resolve().parent
config = {
    "mail": {
        "request_timeout": 30,
        "wait_timeout": 30,
        "wait_interval": 2,
        "api_use_register_proxy": True,
        "providers": [],
    },
    "proxy": "",
    "proxy_input_mode": "single",
    "proxy_url": "",
    "proxy_list_text": "",
    "proxy_refresh_interval": 120,
    "total": 10,
    "threads": 3,
}
register_config_file = base_dir.parents[1] / "data" / "register.json"
try:
    saved_config = json.loads(register_config_file.read_text(encoding="utf-8"))
    config.update(
        {
            key: saved_config[key]
            for key in (
                "mail",
                "proxy",
                "proxy_input_mode",
                "proxy_url",
                "proxy_list_text",
                "proxy_refresh_interval",
                "total",
                "threads",
            )
            if key in saved_config
        }
    )
except Exception:
    pass

auth_base = "https://auth.openai.com"
platform_base = "https://platform.openai.com"
platform_oauth_client_id = "app_2SKx67EdpoN0G6j64rFvigXD"
platform_oauth_redirect_uri = f"{platform_base}/auth/callback"
platform_oauth_audience = "https://api.openai.com/v1"
platform_auth0_client = "eyJuYW1lIjoiYXV0aDAtc3BhLWpzIiwidmVyc2lvbiI6IjEuMjEuMCJ9"
user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)
sec_ch_ua = '"Google Chrome";v="145", "Not?A_Brand";v="8", "Chromium";v="145"'
sec_ch_ua_full_version_list = '"Chromium";v="145.0.0.0", "Not:A-Brand";v="99.0.0.0", "Google Chrome";v="145.0.0.0"'
default_timeout = 30
print_lock = threading.Lock()
stats_lock = threading.Lock()
browser_fallback_lock = threading.Lock()
stats = {
    "done": 0,
    "success": 0,
    "fail": 0,
    "start_time": 0.0,
    "current_proxy": "",
    "proxy_pool_count": 0,
    "proxy_source": "single",
    "proxy_pool_last_error": "",
    "proxy_pool_last_fetch": 0,
}
register_log_sink = None
proxy_pool = RegisterProxyPool()

common_headers = {
    "accept": "application/json",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "content-type": "application/json",
    "dnt": "1",
    "origin": auth_base,
    "priority": "u=1, i",
    "sec-gpc": "1",
    "sec-ch-ua": sec_ch_ua,
    "sec-ch-ua-arch": '"x86_64"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version-list": sec_ch_ua_full_version_list,
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-platform-version": '"10.0.0"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": user_agent,
}

navigate_headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "connection": "keep-alive",
    "dnt": "1",
    "sec-gpc": "1",
    "sec-ch-ua": sec_ch_ua,
    "sec-ch-ua-arch": '"x86_64"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version-list": sec_ch_ua_full_version_list,
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": '"Windows"',
    "sec-ch-ua-platform-version": '"10.0.0"',
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": user_agent,
}


def log(text: str, color: str = "") -> None:
    colors = {"red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m"}
    if register_log_sink:
        try:
            register_log_sink(text, color)
        except Exception:
            pass
    with print_lock:
        prefix = colors.get(color, "")
        suffix = "\033[0m" if prefix else ""
        print(f"{prefix}{datetime.now().strftime('%H:%M:%S')} {text}{suffix}")


def step(index: int, text: str, color: str = "") -> None:
    log(f"[任务{index}] {text}", color)


def _make_trace_headers() -> dict[str, str]:
    trace_id = str(random.getrandbits(64))
    parent_id = str(random.getrandbits(64))
    return {
        "traceparent": f"00-{uuid.uuid4().hex}-{format(int(parent_id), '016x')}-01",
        "tracestate": "dd=s:1;o:rum",
        "x-datadog-origin": "rum",
        "x-datadog-parent-id": parent_id,
        "x-datadog-sampling-priority": "1",
        "x-datadog-trace-id": trace_id,
    }


from utils.pkce import generate_pkce as _generate_pkce  # noqa: F401


def _random_password(length: int = 16) -> str:
    chars = string.ascii_letters + string.digits + "!@#$%"
    value = list(
        secrets.choice(string.ascii_uppercase)
        + secrets.choice(string.ascii_lowercase)
        + secrets.choice(string.digits)
        + secrets.choice("!@#$%")
        + "".join(secrets.choice(chars) for _ in range(max(0, length - 4)))
    )
    random.shuffle(value)
    return "".join(value)


def _random_name() -> tuple[str, str]:
    return random.choice(["James", "Robert", "John", "Michael", "David", "Mary", "Emma", "Olivia"]), random.choice(
        ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
    )


def _random_birthdate() -> str:
    return f"{random.randint(1996, 2006):04d}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"


def _platform_authorize_url(email: str, device_id: str) -> tuple[str, str]:
    code_verifier, code_challenge = _generate_pkce()
    params = {
        "issuer": auth_base,
        "client_id": platform_oauth_client_id,
        "audience": platform_oauth_audience,
        "redirect_uri": platform_oauth_redirect_uri,
        "device_id": device_id,
        "screen_hint": "signup",
        "max_age": "0",
        "login_hint": email,
        "scope": "openid profile email offline_access",
        "response_type": "code",
        "response_mode": "query",
        "state": secrets.token_urlsafe(32),
        "nonce": secrets.token_urlsafe(32),
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "auth0Client": platform_auth0_client,
    }
    return f"{auth_base}/api/accounts/authorize?{urlencode(params)}", code_verifier


def _response_json(resp) -> dict:
    try:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _response_debug_detail(resp, limit: int = 800) -> str:
    if resp is None:
        return ""
    data = _response_json(resp)
    parts = [
        f"url={str(getattr(resp, 'url', '') or '')[:300]}",
        f"content_type={str(getattr(resp, 'headers', {}).get('content-type') or '')}",
    ]
    for key in ("cf-ray", "x-request-id", "openai-processing-ms"):
        value = str(getattr(resp, "headers", {}).get(key) or "").strip()
        if value:
            parts.append(f"{key}={value}")
    if data:
        parts.append(f"json={json.dumps(data, ensure_ascii=False)[:limit]}")
    else:
        parts.append(f"body={str(getattr(resp, 'text', '') or '')[:limit]}")
    return ", ".join(parts)


def _is_cloudflare_challenge(resp) -> bool:
    if resp is None:
        return False
    try:
        status_code = int(getattr(resp, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status_code = 0
    if status_code not in (403, 503):
        return False
    text = str(getattr(resp, "text", "") or "").lower()
    return (
        "<title>just a moment" in text
        or "<title>attention required! | cloudflare" in text
        or "cf-chl-" in text
        or "__cf_chl_" in text
        or "cf-browser-verification" in text
    )


def _truthy(value: object, fallback: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return fallback


def _mail_config(register_proxy: str = "") -> dict:
    mail = config["mail"] if isinstance(config.get("mail"), dict) else {}
    use_register_proxy = _truthy(mail.get("api_use_register_proxy"), True)
    proxy = str(register_proxy or "").strip() if use_register_proxy else ""
    return {**mail, "api_use_register_proxy": use_register_proxy, "proxy": proxy}


def _authorize_landed_page(resp) -> str:
    """诊断用：粗判 authorize 之后落在哪个页面。返回 signup / login / "" 仅供日志。

    注意：email-verification / email_otp_verification 在注册和登录流程里都会出现，
    无法据此可靠区分，所以这里只用于打日志，绝不据此中断注册流程。
    """
    if resp is None:
        return ""
    final_url = str(getattr(resp, "url", "") or "").lower()
    data = _response_json(resp)
    page_type = ""
    page = data.get("page") if isinstance(data, dict) else None
    if isinstance(page, dict):
        page_type = str(page.get("type") or "").lower()
    if "create-account" in final_url or "signup" in final_url or "create_account" in page_type:
        return "signup"
    if "/log-in" in final_url or "/login" in final_url or page_type in {"login", "password_verification"}:
        return "login"
    return ""


def _authorize_already_at_password_step(resp) -> bool:
    if resp is None:
        return False
    final_url = str(getattr(resp, "url", "") or "").lower()
    if "/create-account/password" in final_url:
        return True
    data = _response_json(resp)
    page = data.get("page") if isinstance(data, dict) else None
    page_type = str(page.get("type") or "").lower() if isinstance(page, dict) else ""
    return page_type in {"create_account_password", "create-account-password"}


def create_mailbox(username: str | None = None, register_proxy: str = "") -> dict:
    return mail_provider.create_mailbox(_mail_config(register_proxy), username)


def wait_for_code(
    mailbox: dict,
    register_proxy: str = "",
    *,
    wait_timeout: int | None = None,
) -> str | None:
    mail_config = _mail_config(register_proxy)
    if wait_timeout is not None:
        mail_config["wait_timeout"] = max(1, int(wait_timeout))
    return mail_provider.wait_for_code(mail_config, mailbox)


from utils.sentinel import SentinelTokenGenerator, build_sentinel_token as _build_sentinel_token_tuple  # noqa: F401


def build_sentinel_token(session: requests.Session, device_id: str, flow: str) -> str:
    """请求 sentinel token，返回 sentinel header 字符串（兼容旧接口）。"""
    sentinel_val, _oai_sc_val = _build_sentinel_token_tuple(session, device_id, flow, user_agent=user_agent, sec_ch_ua=sec_ch_ua)
    return sentinel_val


def create_session(proxy: str = "") -> Any:
    kwargs = proxy_settings.build_session_kwargs(
        proxy=proxy,
        upstream=True,
        impersonate="chrome",
        verify=False,
    )
    return requests.Session(**kwargs)


def _apply_clearance_to_session(session: requests.Session, bundle: ClearanceBundle | None) -> None:
    if bundle is None:
        return
    if bundle.user_agent:
        session.headers["User-Agent"] = bundle.user_agent
        session.headers["user-agent"] = bundle.user_agent
    for name, value in bundle.cookies.items():
        try:
            session.cookies.set(name, value, domain=f".{bundle.target_host or 'openai.com'}")
            session.cookies.set(name, value, domain=bundle.target_host or "auth.openai.com")
        except Exception:
            continue


def _headers_with_clearance(
    headers: dict[str, str],
    target_url: str,
    proxy: str = "",
    user_agent_override: str = "",
) -> dict[str, str]:
    merged = proxy_settings.build_headers(
        headers=headers,
        target_url=target_url,
        proxy=proxy,
        upstream=True,
    )
    normalized = {str(key): str(value) for key, value in merged.items()}
    if user_agent_override:
        ua_key = next((key for key in normalized if key.lower() == "user-agent"), "user-agent")
        normalized[ua_key] = user_agent_override
        fingerprint = SentinelSDKClient.client_hints_for_user_agent(user_agent_override)
        hint_values = {
            "sec-ch-ua": fingerprint["sec_ch_ua"],
            "sec-ch-ua-full-version-list": fingerprint["sec_ch_ua_full_version_list"],
            "sec-ch-ua-platform": f'"{fingerprint["platform"]}"',
            "sec-ch-ua-platform-version": f'"{fingerprint["platform_version"]}"',
            "sec-ch-ua-arch": f'"{fingerprint["architecture"]}"',
            "sec-ch-ua-bitness": '"64"',
        }
        for header_name, header_value in hint_values.items():
            key = next(
                (item for item in normalized if item.lower() == header_name),
                header_name,
            )
            normalized[key] = header_value
    return normalized


def _cloudflare_block_message(resp, prefix: str = "被 Cloudflare 拦截", reason: str = "") -> str:
    status = getattr(resp, "status_code", "unknown")
    debug = _response_debug_detail(resp)
    reason = reason or "clearance 刷新失败或重试后仍失败，请更换 IP/代理重试"
    return f"{prefix}，{reason}: status={status}, {debug}"


def request_with_local_retry(session: requests.Session, method: str, url: str, retry_attempts: int = 3, **kwargs):
    last_error = ""
    for _ in range(max(1, retry_attempts)):
        try:
            return session.request(method.upper(), url, timeout=default_timeout, **kwargs), ""
        except Exception as error:
            last_error = str(error)
            time.sleep(1)
    return None, last_error


def validate_otp(session: requests.Session, device_id: str, code: str):
    headers = dict(common_headers)
    headers["referer"] = f"{auth_base}/email-verification"
    headers["oai-device-id"] = device_id
    headers.update(_make_trace_headers())
    resp, error = request_with_local_retry(session, "post", f"{auth_base}/api/accounts/email-otp/validate", json={"code": code}, headers=headers, verify=False)
    if resp is not None and resp.status_code == 200:
        return resp, ""
    headers["openai-sentinel-token"] = build_sentinel_token(session, device_id, "authorize_continue")
    resp, error = request_with_local_retry(session, "post", f"{auth_base}/api/accounts/email-otp/validate", json={"code": code}, headers=headers, verify=False)
    return resp, error


def extract_oauth_callback_params_from_url(url: str) -> dict[str, str] | None:
    if not url:
        return None
    try:
        params = parse_qs(urlparse(url).query)
    except Exception:
        return None
    code = str((params.get("code") or [""])[0]).strip()
    if not code:
        return None
    return {"code": code, "state": str((params.get("state") or [""])[0]).strip(), "scope": str((params.get("scope") or [""])[0]).strip()}


def request_platform_oauth_token(session: requests.Session, code: str, code_verifier: str) -> dict | None:
    headers = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "auth0-client": platform_auth0_client,
        "cache-control": "no-cache",
        "content-type": "application/json",
        "origin": platform_base,
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": f"{platform_base}/",
        "sec-ch-ua": sec_ch_ua,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": user_agent,
    }
    resp = session.post(
        f"{auth_base}/api/accounts/oauth/token",
        headers=headers,
        json={
            "client_id": platform_oauth_client_id,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": platform_oauth_redirect_uri,
        },
        verify=False,
        timeout=60,
    )
    if resp.status_code != 200:
        print(resp.text)
        return None
    return _response_json(resp)


class PlatformRegistrar:
    def __init__(self, proxy: str = "") -> None:
        self.proxy = str(proxy or "").strip()
        self.session = create_session(self.proxy)
        self.clearance_user_agent = ""
        self.clearance_failure_reason = ""
        self.device_id = str(uuid.uuid4())
        self.code_verifier = ""
        self.platform_auth_code = ""
        self.stage_timings: dict[str, float] = {}
        self.sentinel_sdk = SentinelSDKClient(
            session=self.session,
            device_id=self.device_id,
            user_agent=user_agent,
        )

    def close(self) -> None:
        try:
            self.sentinel_sdk.close()
        finally:
            self.session.close()

    def _sentinel_headers(
        self,
        flow: str,
        index: int,
        *,
        include_so: bool = False,
    ) -> dict[str, str]:
        self.sentinel_sdk.user_agent = self.clearance_user_agent or user_agent
        tokens: SentinelSDKTokens = self.sentinel_sdk.get_tokens(
            flow,
            include_so=include_so,
        )
        step(
            index,
            (
                f"Sentinel SDK token 完成 flow={flow} sdk={tokens.sdk_version} "
                f"token_len={len(tokens.token)} so_token={'yes' if tokens.so_token else 'no'} "
                f"so_token_len={len(tokens.so_token)}"
            ),
        )
        headers = {"OpenAI-Sentinel-Token": tokens.token}
        if include_so:
            if not tokens.so_token:
                raise RuntimeError(f"Sentinel SDK 未生成 SO token: flow={flow}")
            headers["OpenAI-Sentinel-SO-Token"] = tokens.so_token
        return headers

    def _navigate_headers(self, referer: str = "") -> dict[str, str]:
        headers = dict(navigate_headers)
        if referer:
            headers["referer"] = referer
        return headers

    def _json_headers(self, referer: str) -> dict[str, str]:
        headers = dict(common_headers)
        headers["referer"] = referer
        headers["oai-device-id"] = self.device_id
        headers.update(_make_trace_headers())
        return headers

    def _refresh_cloudflare_clearance(self, target_url: str, index: int) -> ClearanceBundle | None:
        self.clearance_failure_reason = ""
        profile = proxy_settings.get_profile(proxy=self.proxy, upstream=True)
        if not profile.clearance_enabled:
            self.clearance_failure_reason = (
                "可尝试使用 FlareSolverr 清障方式，注意需要 Docker 部署 flaresolverr、privoxy、warp-proxy 等相关容器"
            )
            step(index, f"检测到 Cloudflare 拦截，{self.clearance_failure_reason}", "yellow")
            return None
        step(index, "检测到 Cloudflare 拦截，尝试刷新 clearance", "yellow")
        bundle = proxy_settings.refresh_clearance(
            target_url=target_url,
            proxy=self.proxy,
            force=True,
            upstream=True,
        )
        if bundle is not None:
            _apply_clearance_to_session(self.session, bundle)
            self.clearance_user_agent = bundle.user_agent or self.clearance_user_agent
            step(index, "Cloudflare clearance 刷新完成，重试当前请求", "yellow")
        else:
            self.clearance_failure_reason = "clearance 刷新未返回可用 Cookie，请检查 FlareSolverr URL、代理和出口 IP"
            step(index, f"Cloudflare clearance 刷新失败：{self.clearance_failure_reason}", "yellow")
        return bundle

    def _platform_authorize(self, email: str, index: int) -> None:
        step(index, "开始 platform authorize")
        self.session.cookies.set("oai-did", self.device_id, domain=".auth.openai.com")
        self.session.cookies.set("oai-did", self.device_id, domain="auth.openai.com")
        self.code_verifier, code_challenge = _generate_pkce()
        params = {
            "issuer": auth_base,
            "client_id": platform_oauth_client_id,
            "audience": platform_oauth_audience,
            "redirect_uri": platform_oauth_redirect_uri,
            "device_id": self.device_id,
            # 注册流程显式声明 signup：throwaway 域名 OpenAI 会自动当新账号走注册，
            # 但 @outlook.com/@hotmail.com 这类真实消费邮箱会被 login_or_signup 路由到登录分支，
            # 后续 user/register 落在错误的 auth step 上报 invalid_auth_step。
            "screen_hint": "signup",
            "max_age": "0",
            "login_hint": email,
            "scope": "openid profile email offline_access",
            "response_type": "code",
            "response_mode": "query",
            "state": secrets.token_urlsafe(32),
            "nonce": secrets.token_urlsafe(32),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "auth0Client": platform_auth0_client,
        }
        target_url = f"{auth_base}/api/accounts/authorize?{urlencode(params)}"
        headers = self._navigate_headers(f"{platform_base}/")
        headers = _headers_with_clearance(headers, target_url, self.proxy, self.clearance_user_agent)
        resp, error = request_with_local_retry(self.session, "get", target_url, headers=headers, allow_redirects=True, verify=False)
        if _is_cloudflare_challenge(resp):
            bundle = self._refresh_cloudflare_clearance(auth_base, index)
            if bundle is None:
                raise RuntimeError(_cloudflare_block_message(resp, reason=self.clearance_failure_reason))
            retry_headers = _headers_with_clearance(self._navigate_headers(f"{platform_base}/"), target_url, self.proxy, self.clearance_user_agent)
            resp, error = request_with_local_retry(self.session, "get", target_url, headers=retry_headers, allow_redirects=True, verify=False)
            if _is_cloudflare_challenge(resp):
                raise RuntimeError(_cloudflare_block_message(resp, "Cloudflare clearance 重试仍被拦截"))
        if resp is None or resp.status_code != 200:
            err = _response_json(resp).get("error", {}) if resp is not None else {}
            detail = f": {err.get('code', '')} - {err.get('message', '')}".strip(" -") if err else ""
            debug = _response_debug_detail(resp)
            status = getattr(resp, "status_code", "unknown")
            raise RuntimeError(error or f"platform_authorize_http_{status}{detail}, {debug}")
        landed = _authorize_landed_page(resp)
        step(index, f"platform authorize 完成[{landed or '?'}] url={str(getattr(resp, 'url', '') or '')[:160]}")

        if _authorize_already_at_password_step(resp):
            step(index, "authorize 已进入密码步骤，跳过重复 continue")
            return

        continue_url = f"{auth_base}/api/accounts/authorize/continue"
        continue_referer = str(getattr(resp, "url", "") or f"{auth_base}/create-account")
        continue_headers = self._json_headers(continue_referer)
        continue_headers.update(self._sentinel_headers("authorize_continue", index))
        continue_headers = _headers_with_clearance(
            continue_headers,
            continue_url,
            self.proxy,
            self.clearance_user_agent,
        )
        continue_body = {"username": {"kind": "email", "value": email}}
        resp, error = request_with_local_retry(
            self.session,
            "post",
            continue_url,
            json=continue_body,
            headers=continue_headers,
            verify=False,
        )
        if _is_cloudflare_challenge(resp):
            bundle = self._refresh_cloudflare_clearance(auth_base, index)
            if bundle is None:
                raise RuntimeError(_cloudflare_block_message(resp, reason=self.clearance_failure_reason))
            continue_headers = self._json_headers(continue_referer)
            continue_headers.update(self._sentinel_headers("authorize_continue", index))
            continue_headers = _headers_with_clearance(
                continue_headers,
                continue_url,
                self.proxy,
                self.clearance_user_agent,
            )
            resp, error = request_with_local_retry(
                self.session,
                "post",
                continue_url,
                json=continue_body,
                headers=continue_headers,
                verify=False,
            )
            if _is_cloudflare_challenge(resp):
                raise RuntimeError(_cloudflare_block_message(resp, "Cloudflare clearance 重试仍被拦截"))
        if resp is None or resp.status_code not in (200, 302):
            data = _response_json(resp) if resp is not None else {}
            detail = f", detail={json.dumps(data, ensure_ascii=False)}" if data else ""
            raise RuntimeError(
                error
                or f"authorize_continue_http_{getattr(resp, 'status_code', 'unknown')}{detail}"
            )
        step(index, "邮箱 authorize/continue 完成")

    def _register_user(self, email: str, password: str, index: int) -> None:
        step(index, "开始提交注册密码")
        url = f"{auth_base}/api/accounts/user/register"
        headers = self._json_headers(f"{auth_base}/create-account/password")
        headers.update(self._sentinel_headers("username_password_create", index))
        headers = _headers_with_clearance(headers, url, self.proxy, self.clearance_user_agent)
        resp, error = request_with_local_retry(self.session, "post", url, json={"username": email, "password": password}, headers=headers, verify=False)
        if _is_cloudflare_challenge(resp):
            bundle = self._refresh_cloudflare_clearance(auth_base, index)
            if bundle is None:
                raise RuntimeError(_cloudflare_block_message(resp, reason=self.clearance_failure_reason))
            headers = self._json_headers(f"{auth_base}/create-account/password")
            headers.update(self._sentinel_headers("username_password_create", index))
            headers = _headers_with_clearance(headers, url, self.proxy, self.clearance_user_agent)
            resp, error = request_with_local_retry(self.session, "post", url, json={"username": email, "password": password}, headers=headers, verify=False)
            if _is_cloudflare_challenge(resp):
                raise RuntimeError(_cloudflare_block_message(resp, "Cloudflare clearance 重试仍被拦截"))
        if resp is None or resp.status_code != 200:
            data = _response_json(resp) if resp is not None else {}
            if data.get("message") == "Failed to create account. Please try again.":
                step(index, "注册失败提示: 邮箱域名很可能因滥用被封禁，请更换邮箱域名", "yellow")
            detail = f", detail={json.dumps(data, ensure_ascii=False)}" if data else ""
            raise RuntimeError(error or f"user_register_http_{getattr(resp, 'status_code', 'unknown')}{detail}")
        step(index, "提交注册密码完成")

    def _send_otp(self, index: int) -> None:
        step(index, "开始发送验证码")
        url = f"{auth_base}/api/accounts/email-otp/send"
        headers = _headers_with_clearance(self._navigate_headers(f"{auth_base}/create-account/password"), url, self.proxy, self.clearance_user_agent)
        resp, error = request_with_local_retry(self.session, "get", url, headers=headers, allow_redirects=True, verify=False)
        if _is_cloudflare_challenge(resp):
            bundle = self._refresh_cloudflare_clearance(auth_base, index)
            if bundle is None:
                raise RuntimeError(_cloudflare_block_message(resp, reason=self.clearance_failure_reason))
            headers = _headers_with_clearance(self._navigate_headers(f"{auth_base}/create-account/password"), url, self.proxy, self.clearance_user_agent)
            resp, error = request_with_local_retry(self.session, "get", url, headers=headers, allow_redirects=True, verify=False)
            if _is_cloudflare_challenge(resp):
                raise RuntimeError(_cloudflare_block_message(resp, "Cloudflare clearance 重试仍被拦截"))
        if resp is None or resp.status_code not in (200, 302):
            raise RuntimeError(error or f"send_otp_http_{getattr(resp, 'status_code', 'unknown')}")
        step(index, "发送验证码完成")

    def _validate_otp(self, code: str, index: int) -> None:
        step(index, f"开始校验验证码 {code}")
        resp, error = validate_otp(self.session, self.device_id, code)
        if resp is None or resp.status_code != 200:
            body = ""
            try:
                body = (resp.text or "")[:500] if resp is not None else ""
            except Exception:
                pass
            raise RuntimeError(error or f"validate_otp_http_{getattr(resp, 'status_code', 'unknown')}_body={body}")
        step(index, "验证码校验完成")

    def _create_account(self, name: str, birthdate: str, index: int) -> None:
        step(index, "开始创建账号资料")
        url = f"{auth_base}/api/accounts/create_account"
        headers = self._json_headers(f"{auth_base}/about-you")
        headers.update(self._sentinel_headers("oauth_create_account", index, include_so=True))
        headers = _headers_with_clearance(headers, url, self.proxy, self.clearance_user_agent)
        resp, error = request_with_local_retry(self.session, "post", url, json={"name": name, "birthdate": birthdate}, headers=headers, verify=False)
        if _is_cloudflare_challenge(resp):
            bundle = self._refresh_cloudflare_clearance(auth_base, index)
            if bundle is None:
                raise RuntimeError(_cloudflare_block_message(resp, reason=self.clearance_failure_reason))
            headers = self._json_headers(f"{auth_base}/about-you")
            headers.update(self._sentinel_headers("oauth_create_account", index, include_so=True))
            headers = _headers_with_clearance(headers, url, self.proxy, self.clearance_user_agent)
            resp, error = request_with_local_retry(self.session, "post", url, json={"name": name, "birthdate": birthdate}, headers=headers, verify=False)
            if _is_cloudflare_challenge(resp):
                raise RuntimeError(_cloudflare_block_message(resp, "Cloudflare clearance 重试仍被拦截"))
        if resp is None or resp.status_code not in (200, 302):
            data = _response_json(resp) if resp is not None else {}
            if data.get("message") == "Failed to create account. Please try again.":
                step(index, "创建账号失败提示: 邮箱域名很可能因滥用被封禁，请更换邮箱域名", "yellow")
            detail = f", detail={json.dumps(data, ensure_ascii=False)}" if data else ""
            raise RuntimeError(error or f"create_account_http_{getattr(resp, 'status_code', 'unknown')}{detail}")
        data = _response_json(resp)
        callback_params = extract_oauth_callback_params_from_url(str(data.get("continue_url") or "").strip())
        self.platform_auth_code = str((callback_params or {}).get("code") or "").strip()
        step(index, "创建账号资料完成")

    def _exchange_registered_tokens(self, index: int) -> dict:
        step(index, "开始换 token")
        tokens = request_platform_oauth_token(self.session, self.platform_auth_code, self.code_verifier)
        if not tokens:
            raise RuntimeError("token换取失败")
        step(index, "token 换取完成")
        return tokens

    def register(self, index: int) -> dict:
        step(index, "开始创建邮箱")
        mailbox = create_mailbox(register_proxy=self.proxy)
        email = str(mailbox.get("address") or "").strip()
        if not email:
            mail_provider.release_mailbox(mailbox)
            raise RuntimeError("邮箱服务未返回 address")
        label = str(mailbox.get("label") or "")
        step(index, f"邮箱创建完成[{label}]: {email}")
        try:
            password = _random_password()
            first_name, last_name = _random_name()
            platform_authorize_started = time.time()
            try:
                self._platform_authorize(email, index)
            finally:
                self.stage_timings["platform_authorize_ms"] = round((time.time() - platform_authorize_started) * 1000, 1)
            self._register_user(email, password, index)
            self._send_otp(index)
            step(index, "开始等待注册验证码")
            code = wait_for_code(mailbox, register_proxy=self.proxy)
            if not code:
                raise RuntimeError("等待注册验证码超时")
            step(index, f"收到注册验证码: {code}")
            self._validate_otp(code, index)
            self._create_account(f"{first_name} {last_name}", _random_birthdate(), index)
            tokens = self._exchange_registered_tokens(index)
        except Exception as error:
            mail_provider.mark_mailbox_result(mailbox, success=False, error=error)
            raise
        mail_provider.mark_mailbox_result(mailbox, success=True)
        return {
            "email": email,
            "password": password,
            "access_token": str(tokens.get("access_token") or "").strip(),
            "refresh_token": str(tokens.get("refresh_token") or "").strip(),
            "id_token": str(tokens.get("id_token") or "").strip(),
            "source_type": "web",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def should_use_browser_fallback(error: BaseException) -> bool:
    if isinstance(error, SentinelSDKError):
        return True
    text = str(error or "").lower()
    if "registration_disallowed" in text:
        return True
    non_retryable = (
        "rate limited (free)",
        "http 429",
        "otp timed out",
        "otp_timeout",
        "failed to create account. please try again",
        "mailbox domain",
        "email provider",
        "tempmail",
    )
    if any(marker in text for marker in non_retryable):
        return False
    return "sentinel" in text or "cloudflare" in text or "just a moment" in text


class BrowserPlatformRegistrar:
    EMAIL_SELECTORS = (
        'input[type="email"]',
        'input[name="email"]',
        'input[name="username"]',
        '#email-input',
    )
    PASSWORD_SELECTORS = (
        'input[type="password"]',
        'input[name="password"]',
    )
    OTP_SELECTORS = (
        'input[autocomplete="one-time-code"]',
        'input[name="code"]',
        'input[inputmode="numeric"]',
    )
    NAME_SELECTORS = (
        'input[name="name"]',
        'input[autocomplete="name"]',
        'input[placeholder*="name" i]',
    )
    BIRTHDATE_SELECTORS = (
        'input[name="birthdate"]',
        'input[name="birthday"]',
        'input[type="date"]',
        'input[placeholder*="MM" i]',
    )
    SUBMIT_SELECTORS = (
        'button[type="submit"]',
        'button:has-text("Continue")',
        'button:has-text("Create account")',
        'button:has-text("Verify")',
    )

    def __init__(self, proxy: str = "") -> None:
        self.proxy = str(proxy or "").strip()
        self.device_id = str(uuid.uuid4())
        self.response_events: list[str] = []

    @staticmethod
    def _user_agent_for_version(version: str) -> str:
        browser_version = str(version or "").strip().rsplit("/", 1)[-1]
        if not browser_version or any(
            character not in "0123456789." for character in browser_version
        ):
            browser_version = "145.0.0.0"
        return (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{browser_version} Safari/537.36"
        )

    @staticmethod
    def _proxy_config(proxy: str) -> dict[str, str] | None:
        value = str(proxy or "").strip()
        if not value:
            return None
        if "://" not in value:
            value = f"http://{value}"
        parsed = urlparse(value)
        if not parsed.hostname:
            raise RuntimeError("Browser fallback proxy is invalid")
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
    def _visible_locator(page: Any, selectors: tuple[str, ...], timeout_ms: int = 60000) -> Any:
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            for selector in selectors:
                locator = page.locator(selector).first
                try:
                    if locator.count() and locator.is_visible():
                        return locator
                except Exception:
                    continue
            page.wait_for_timeout(250)
        return None

    @classmethod
    def _fill(cls, page: Any, selectors: tuple[str, ...], value: str, label: str) -> Any:
        locator = cls._visible_locator(page, selectors)
        if locator is None:
            raise RuntimeError(f"Browser fallback could not find {label} input")
        locator.fill(value)
        return locator

    @classmethod
    def _submit(cls, page: Any, fallback_locator: Any) -> None:
        button = cls._visible_locator(page, cls.SUBMIT_SELECTORS, timeout_ms=5000)
        if button is not None:
            button.click()
        else:
            fallback_locator.press("Enter")

    @classmethod
    def _fill_otp(cls, page: Any, code: str) -> Any:
        locator = cls._visible_locator(page, cls.OTP_SELECTORS)
        if locator is None:
            raise RuntimeError("Browser fallback could not find OTP input")
        try:
            locator.fill(code)
            return locator
        except Exception:
            inputs = page.locator('input[inputmode="numeric"]:visible')
            if inputs.count() < len(code):
                raise RuntimeError("Browser fallback OTP input layout is unsupported")
            for index, digit in enumerate(code):
                inputs.nth(index).fill(digit)
            return inputs.nth(len(code) - 1)

    @classmethod
    def _fill_birthdate(cls, page: Any, birthdate: str) -> Any:
        locator = cls._visible_locator(page, cls.BIRTHDATE_SELECTORS, timeout_ms=10000)
        if locator is not None:
            locator.fill(birthdate)
            return locator
        year, month, day = birthdate.split("-")
        segment_values = (
            (('input[name="month"]', '[role="spinbutton"][aria-label*="month" i]'), month),
            (('input[name="day"]', '[role="spinbutton"][aria-label*="day" i]'), day),
            (('input[name="year"]', '[role="spinbutton"][aria-label*="year" i]'), year),
        )
        last_locator = None
        for selectors, value in segment_values:
            last_locator = cls._visible_locator(page, selectors, timeout_ms=5000)
            if last_locator is None:
                raise RuntimeError("Browser fallback could not find birthdate input")
            last_locator.fill(value)
        return last_locator

    def _record_response(self, response: Any) -> None:
        url = str(getattr(response, "url", "") or "")
        if "/api/accounts/" not in url:
            return
        try:
            status = int(getattr(response, "status", 0) or 0)
        except (TypeError, ValueError):
            status = 0
        if status < 400:
            return
        code = ""
        message = ""
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get("error") if isinstance(data.get("error"), dict) else data
                code = str(error.get("code") or error.get("type") or "")
                message = str(error.get("message") or data.get("message") or "")
        except Exception:
            pass
        path = urlparse(url).path
        self.response_events.append(
            f"{path}:HTTP {status}, code={code[:80]}, message={message[:180]}"
        )
        self.response_events = self.response_events[-5:]

    def _page_error_context(self, page: Any) -> str:
        try:
            title = str(page.title() or "")[:120]
        except Exception:
            title = ""
        try:
            body = " ".join(str(page.locator("body").inner_text(timeout=1000) or "").split())[:240]
        except Exception:
            body = ""
        events = " | ".join(self.response_events[-3:])
        return (
            f"url={str(getattr(page, 'url', '') or '')[:240]}, title={title}, "
            f"body={body}, responses={events}"
        )

    def _complete_browser_flow(
        self,
        page: Any,
        *,
        authorize_url: str,
        mailbox: dict,
        email: str,
        password: str,
        name: str,
        birthdate: str,
        index: int,
    ) -> str:
        page.goto(authorize_url, wait_until="domcontentloaded", timeout=90000)
        stage_locator = self._visible_locator(
            page,
            self.EMAIL_SELECTORS + self.PASSWORD_SELECTORS,
        )
        if stage_locator is None:
            raise RuntimeError(
                f"Browser fallback did not reach signup form: {self._page_error_context(page)}"
            )
        if str(stage_locator.get_attribute("type") or "").lower() != "password":
            stage_locator.fill(email)
            self._submit(page, stage_locator)
        password_input = self._fill(page, self.PASSWORD_SELECTORS, password, "password")
        self._submit(page, password_input)

        if self._visible_locator(page, self.OTP_SELECTORS) is None:
            raise RuntimeError(
                f"Browser fallback did not reach OTP form: {self._page_error_context(page)}"
            )
        step(index, "Browser fallback is waiting for a fresh mailbox OTP")
        code = wait_for_code(mailbox, register_proxy=self.proxy, wait_timeout=90)
        if not code:
            raise RuntimeError("Browser fallback OTP timed out")
        otp_input = self._fill_otp(page, code)
        self._submit(page, otp_input)

        deadline = time.monotonic() + 60
        name_input = None
        while time.monotonic() < deadline:
            callback = extract_oauth_callback_params_from_url(str(page.url or ""))
            if callback:
                return callback["code"]
            name_input = self._visible_locator(page, self.NAME_SELECTORS, timeout_ms=500)
            if name_input is not None:
                break
        if name_input is None:
            raise RuntimeError(
                f"Browser fallback did not reach profile form: {self._page_error_context(page)}"
            )
        name_input.fill(name)
        birthdate_input = self._fill_birthdate(page, birthdate)
        self._submit(page, birthdate_input)

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            callback = extract_oauth_callback_params_from_url(str(page.url or ""))
            if callback:
                return callback["code"]
            page.wait_for_timeout(250)
        raise RuntimeError(
            f"Browser fallback did not receive OAuth callback: {self._page_error_context(page)}"
        )

    def register(self, index: int) -> dict:
        mailbox = create_mailbox(register_proxy=self.proxy)
        email = str(mailbox.get("address") or "").strip()
        if not email:
            mail_provider.release_mailbox(mailbox)
            raise RuntimeError("Browser fallback mailbox has no address")
        password = _random_password()
        first_name, last_name = _random_name()
        name = f"{first_name} {last_name}"
        birthdate = _random_birthdate()
        authorize_url, code_verifier = _platform_authorize_url(email, self.device_id)
        step(index, f"Browser fallback created fresh mailbox [{mailbox.get('label', '')}]: {email}")
        try:
            try:
                from playwright.sync_api import sync_playwright
            except Exception as exc:
                raise RuntimeError("Playwright is required for browser registration fallback") from exc

            profile = proxy_settings.get_profile(proxy=self.proxy, upstream=True)
            browser_proxy = str(getattr(profile, "proxy_url", "") or self.proxy).strip()
            launch_kwargs: dict[str, Any] = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            }
            proxy_config = self._proxy_config(browser_proxy)
            if proxy_config:
                launch_kwargs["proxy"] = proxy_config

            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(**launch_kwargs)
                try:
                    context = browser.new_context(
                        user_agent=self._user_agent_for_version(browser.version),
                        locale="en-US",
                        viewport={"width": 1365, "height": 900},
                        ignore_https_errors=True,
                    )
                    context.add_init_script(
                        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                    )
                    context.add_cookies(
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
                    page = context.new_page()
                    page.set_default_timeout(60000)
                    page.on("response", self._record_response)
                    auth_code = self._complete_browser_flow(
                        page,
                        authorize_url=authorize_url,
                        mailbox=mailbox,
                        email=email,
                        password=password,
                        name=name,
                        birthdate=birthdate,
                        index=index,
                    )
                finally:
                    browser.close()

            token_session = create_session(self.proxy)
            try:
                tokens = request_platform_oauth_token(token_session, auth_code, code_verifier)
            finally:
                token_session.close()
            if not tokens:
                raise RuntimeError("Browser fallback OAuth token exchange failed")
        except Exception as error:
            mail_provider.mark_mailbox_result(mailbox, success=False, error=error)
            raise
        mail_provider.mark_mailbox_result(mailbox, success=True)
        return {
            "email": email,
            "password": password,
            "access_token": str(tokens.get("access_token") or "").strip(),
            "refresh_token": str(tokens.get("refresh_token") or "").strip(),
            "id_token": str(tokens.get("id_token") or "").strip(),
            "source_type": "web",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def configure_proxy_pool(fetch_now: bool = False) -> dict[str, object]:
    try:
        refresh_interval = int(config.get("proxy_refresh_interval") or 120)
    except (OverflowError, TypeError, ValueError):
        refresh_interval = 120
    state = proxy_pool.configure(
        mode=str(config.get("proxy_input_mode") or "single"),
        single_proxy=str(config.get("proxy") or ""),
        proxy_url=str(config.get("proxy_url") or ""),
        proxy_list_text=str(config.get("proxy_list_text") or ""),
        refresh_interval=refresh_interval,
        fetch_now=fetch_now,
    )
    with stats_lock:
        stats["proxy_pool_count"] = state.count
        stats["proxy_source"] = state.source
        stats["proxy_pool_last_error"] = state.last_error
        stats["proxy_pool_last_fetch"] = state.last_fetch
    return {
        "proxy_pool_count": state.count,
        "proxy_source": state.source,
        "proxy_pool_last_error": state.last_error,
        "proxy_pool_last_fetch": state.last_fetch,
    }


def prepare_proxy_pool() -> dict[str, object]:
    state = proxy_pool.prepare()
    with stats_lock:
        stats["current_proxy"] = state.current_proxy
        stats["proxy_pool_count"] = state.count
        stats["proxy_source"] = state.source
        stats["proxy_pool_last_error"] = state.last_error
        stats["proxy_pool_last_fetch"] = state.last_fetch
    return {
        "proxy_pool_count": state.count,
        "proxy_source": state.source,
        "proxy_pool_last_error": state.last_error,
        "proxy_pool_last_fetch": state.last_fetch,
    }


def reset_proxy_pool_cycle() -> dict[str, object]:
    state = proxy_pool.reset_selection_cycle()
    with stats_lock:
        stats["current_proxy"] = ""
        stats["proxy_pool_count"] = state.count
        stats["proxy_source"] = state.source
        stats["proxy_pool_last_error"] = state.last_error
        stats["proxy_pool_last_fetch"] = state.last_fetch
    return {
        "proxy_pool_count": state.count,
        "proxy_source": state.source,
        "proxy_pool_last_error": state.last_error,
        "proxy_pool_last_fetch": state.last_fetch,
    }


def worker(index: int) -> dict:
    start = time.time()
    selection = proxy_pool.next_proxy()
    with stats_lock:
        stats["current_proxy"] = selection.proxy
        stats["proxy_pool_count"] = selection.count
        stats["proxy_source"] = selection.source
        stats["proxy_pool_last_error"] = selection.last_error
        stats["proxy_pool_last_fetch"] = selection.last_fetch
    if str(config.get("proxy_input_mode") or "single") in {"url", "text"} and not selection.proxy:
        step(index, selection.last_error or "No proxy available, skipping", "yellow")
        return {"ok": False, "index": index, "error": selection.last_error or "no_proxy"}
    active_selection = selection
    attempt_started = start
    attempt_recorded = False
    registrar: PlatformRegistrar | None = PlatformRegistrar(selection.proxy)
    try:
        step(index, "任务启动")
        try:
            result = registrar.register(index)
        except Exception as protocol_error:
            if not should_use_browser_fallback(protocol_error):
                raise
            platform_authorize_ms = float(
                registrar.stage_timings.get("platform_authorize_ms") or 0.0
            )
            proxy_pool.record_result(
                active_selection.proxy,
                success=False,
                error=str(protocol_error),
                cost_seconds=time.time() - attempt_started,
                platform_authorize_ms=platform_authorize_ms,
            )
            attempt_recorded = True
            registrar.close()
            registrar = None
            with browser_fallback_lock:
                if str(config.get("proxy_input_mode") or "single") in {"url", "text"}:
                    active_selection = proxy_pool.next_proxy(
                        exclude_proxy=selection.proxy,
                    )
                    if not active_selection.proxy:
                        raise RuntimeError(
                            active_selection.last_error or "No proxy available for browser fallback"
                        )
                with stats_lock:
                    stats["current_proxy"] = active_selection.proxy
                attempt_started = time.time()
                attempt_recorded = False
                step(
                    index,
                    "Protocol registration failed in Sentinel/Cloudflare stage; "
                    "starting one serialized browser fallback with a fresh mailbox",
                    "yellow",
                )
                result = BrowserPlatformRegistrar(active_selection.proxy).register(index)
        cost = time.time() - start
        platform_authorize_ms = float(
            registrar.stage_timings.get("platform_authorize_ms") or 0.0
        ) if registrar is not None else 0.0
        proxy_pool.record_result(
            active_selection.proxy,
            success=True,
            cost_seconds=time.time() - attempt_started,
            platform_authorize_ms=platform_authorize_ms,
        )
        attempt_recorded = True
        access_token = str(result["access_token"])
        account_service.add_account_items([result])
        refresh_result = account_service.refresh_accounts([access_token])
        if refresh_result.get("errors"):
            step(index, f"账号已保存，刷新状态暂未成功，稍后可重试: {refresh_result['errors']}", "yellow")
        with stats_lock:
            stats["done"] += 1
            stats["success"] += 1
            avg = (time.time() - stats["start_time"]) / stats["success"]
        log(f'{result["email"]} 注册成功，本次耗时{cost:.1f}s，全局平均每个号注册耗时{avg:.1f}s', "green")
        return {"ok": True, "index": index, "result": result}
    except Exception as e:
        cost = time.time() - start
        platform_authorize_ms = float(
            registrar.stage_timings.get("platform_authorize_ms") or 0.0
        ) if registrar is not None else 0.0
        if not attempt_recorded:
            proxy_pool.record_result(
                active_selection.proxy,
                success=False,
                error=str(e),
                cost_seconds=time.time() - attempt_started,
                platform_authorize_ms=platform_authorize_ms,
            )
        with stats_lock:
            stats["done"] += 1
            stats["fail"] += 1
        log(f"任务{index} 注册失败，本次耗时{cost:.1f}s，原因: {e}", "red")
        return {"ok": False, "index": index, "error": str(e)}
    finally:
        if registrar is not None:
            registrar.close()
