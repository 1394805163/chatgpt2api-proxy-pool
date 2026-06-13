from __future__ import annotations

import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

from services.register.proxy_pool import RegisterProxyPool, parse_proxy_lines


class _ProxyListHandler(BaseHTTPRequestHandler):
    body = "127.0.0.1:8080\nsocks5://127.0.0.2:1080\nhttp://127.0.0.3:8080\n"

    def do_GET(self) -> None:
        payload = self.body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def proxy_list_server(body: str) -> Iterator[str]:
    _ProxyListHandler.body = body
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyListHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/proxies.txt"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class RegisterProxyPoolTests(unittest.TestCase):
    def test_parse_proxy_lines_normalizes_and_deduplicates(self) -> None:
        proxies = parse_proxy_lines(
            """
            # comment
            127.0.0.1:8080
            http://127.0.0.1:8080
            socks5://127.0.0.2:1080
            invalid
            """
        )

        self.assertEqual(proxies, ["http://127.0.0.1:8080", "socks5h://127.0.0.2:1080"])

    def test_text_mode_rotates_proxies(self) -> None:
        pool = RegisterProxyPool()
        pool.configure(
            mode="text",
            single_proxy="",
            proxy_url="",
            proxy_list_text="127.0.0.1:8080\n127.0.0.2:8080",
            refresh_interval=120,
        )

        self.assertEqual(pool.next_proxy().proxy, "http://127.0.0.1:8080")
        self.assertEqual(pool.next_proxy().proxy, "http://127.0.0.2:8080")
        self.assertEqual(pool.next_proxy().proxy, "http://127.0.0.1:8080")

    def test_url_mode_fetches_proxy_list(self) -> None:
        with proxy_list_server("127.0.0.1:8080\nsocks5://127.0.0.2:1080\n") as url:
            pool = RegisterProxyPool()
            state = pool.configure(
                mode="url",
                single_proxy="",
                proxy_url=url,
                proxy_list_text="",
                refresh_interval=120,
                fetch_now=True,
            )

            self.assertEqual(state.count, 2)
            self.assertEqual(pool.next_proxy().proxy, "http://127.0.0.1:8080")
            self.assertEqual(pool.next_proxy().proxy, "socks5h://127.0.0.2:1080")

    def test_url_refresh_failure_keeps_existing_pool(self) -> None:
        pool = RegisterProxyPool()
        with proxy_list_server("127.0.0.1:8080\n127.0.0.2:8080\n") as url:
            pool.configure(
                mode="url",
                single_proxy="",
                proxy_url=url,
                proxy_list_text="",
                refresh_interval=120,
                fetch_now=True,
            )
            self.assertEqual(pool.state().count, 2)

        state = pool.refresh_url(force=True)

        self.assertEqual(state.count, 2)
        self.assertIn("failed to fetch proxy URL", state.last_error)
        self.assertEqual(pool.next_proxy().proxy, "http://127.0.0.1:8080")


if __name__ == "__main__":
    unittest.main()

