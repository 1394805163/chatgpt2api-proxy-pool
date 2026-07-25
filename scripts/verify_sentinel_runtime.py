from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.register.openai_register import create_session, user_agent
from utils.sentinel_sdk import SentinelSDKClient, _default_process_runner


def safe_process_runner(command: list[str], *, timeout: float):
    result = _default_process_runner(command, timeout=timeout)
    if result.returncode:
        error_text = str(result.stderr or "")
        print(
            json.dumps(
                {
                    "runner_exit_code": result.returncode,
                    "access_denied": "ERR_ACCESS_DENIED" in error_text,
                    "access_restricted": "access to this api has been restricted" in error_text.lower(),
                    "permission_error": "permission" in error_text.lower(),
                    "bad_option": "bad option" in error_text.lower(),
                    "cannot_find": "cannot find" in error_text.lower(),
                    "challenge_error": "challenge" in error_text.lower(),
                    "fetch_error": "fetch" in error_text.lower(),
                    "not_defined": "not defined" in error_text.lower(),
                    "observer_unavailable": "sessionobservertoken is unavailable" in error_text.lower(),
                    "os_error": next(
                        (
                            code
                            for code in ("EACCES", "ENOENT", "EPERM")
                            if code in error_text
                        ),
                        "",
                    ),
                    "exception_types": sorted(
                        set(re.findall(r"\b(?:Type|Reference|Syntax|Range|Eval)?Error\b", error_text))
                    ),
                    "error_codes": re.findall(r"ERR_[A-Z_]+", error_text)[:5],
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
    return result


def main() -> int:
    session = create_session("")
    try:
        client = SentinelSDKClient(
            session=session,
            device_id=str(uuid.uuid4()),
            user_agent=user_agent,
            process_runner=safe_process_runner,
        )
        result = client.get_tokens("oauth_create_account", include_so=True)
        token = json.loads(result.token)
        so_token = json.loads(result.so_token)
        print(
            json.dumps(
                {
                    "sdk": result.sdk_version,
                    "token_t_nonempty": bool(token.get("t")),
                    "token_flow": token.get("flow"),
                    "so_nonempty": bool(so_token.get("so")),
                    "so_flow": so_token.get("flow"),
                    "same_challenge": token.get("c") == so_token.get("c"),
                    "token_len": len(result.token),
                    "so_len": len(result.so_token),
                },
                separators=(",", ":"),
            )
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
