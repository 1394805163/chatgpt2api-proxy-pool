from __future__ import annotations

import json
import time
import unittest

import requests

AUTH_KEY = "chatgpt2api"
BASE_URL = "http://localhost:8000"


class ImageGenerationsTests(unittest.TestCase):
    def test_image_generation_http(self):
        """测试图片生成的非流式 HTTP 调用。"""
        response = requests.post(
            f"{BASE_URL}/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_KEY}"},
            json={
                "model": "gpt-image-2",
                "prompt": "我想做一张南京城市宣传海报图。",
                "n": 1,
                "response_format": "url",
            },
            timeout=300,
        )
        payload = response.json()
        items = [item for item in payload.get("data") or [] if isinstance(item, dict)]
        urls = [str(item.get("url") or "") for item in items]
        self.assertTrue(items and all(urls) and all(not item.get("b64_json") for item in items), "非流式接口未返回纯 URL 图片结果。")
        print("images generations non-stream status:")
        print(response.status_code)
        print("images generations non-stream created:")
        print(payload.get("created"))
        print("images generations non-stream urls:")
        for url in urls:
            print(url)

    def test_image_generation_stream_http(self):
        """测试图片生成的流式 HTTP 调用。"""
        response = requests.post(
            f"{BASE_URL}/v1/images/generations",
            headers={"Authorization": f"Bearer {AUTH_KEY}"},
            json={
                "model": "gpt-image-2",
                "prompt": "我想做一张南京城市宣传海报图。",
                "n": 1,
                "response_format": "url",
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        image_items: list[dict[str, object]] = []
        started_at = time.time()
        print("images generations stream status:")
        print(response.status_code)
        print("images generations stream chunks:")
        for line in response.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8", errors="replace")
            print(f"{time.time() - started_at:6.2f}s {text}")
            if not text.startswith("data:"):
                continue
            payload = text[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            data = chunk.get("data")
            if isinstance(data, list):
                image_items.extend(item for item in data if isinstance(item, dict))

        urls = [str(item.get("url") or "") for item in image_items]
        self.assertTrue(urls and all(urls) and all(not item.get("b64_json") for item in image_items), "流式接口未返回纯 URL 图片结果。")
        print("images generations stream urls:")
        for url in urls:
            print(url)


if __name__ == "__main__":
    unittest.main()
