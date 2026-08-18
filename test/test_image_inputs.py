from __future__ import annotations

import asyncio
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

from fastapi import HTTPException
from starlette.datastructures import UploadFile

from api import image_inputs


class ImageInputTests(unittest.TestCase):
    def test_data_url_is_spooled_when_output_path_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "input.bin"
            result = image_inputs._download_image_url("data:image/png;base64,YWJj", output)
            stored = Path(result[0]).read_bytes()

        self.assertEqual(Path(result[0]).suffix, ".png")
        self.assertEqual(result[1:], ("image_url.png", "image/png"))
        self.assertEqual(stored, b"abc")

    def test_read_image_sources_rejects_too_many_sources(self) -> None:
        with self.assertRaisesRegex(HTTPException, "at most"):
            asyncio.run(
                image_inputs.read_image_sources(
                    ["data:image/png;base64,YWJj"] * (image_inputs.MAX_IMAGE_SOURCES + 1),
                    spool_to_disk=True,
                )
            )

    def test_spooled_upload_stops_before_writing_oversized_chunk(self) -> None:
        upload = UploadFile(filename="large.png", file=BytesIO(b"12345"))
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            image_inputs, "MAX_IMAGE_REFERENCE_BYTES", 4
        ):
            output = Path(tmp_dir) / "input.bin"
            with self.assertRaisesRegex(HTTPException, "image file exceeds"):
                image_inputs._copy_upload_to_path(upload, output)
            self.assertEqual(output.stat().st_size, 0)

    def test_remote_download_streams_to_disk_and_closes_response(self) -> None:
        response = mock.Mock(
            status_code=200,
            headers={"content-type": "image/png", "content-length": "6"},
        )
        response.iter_content.return_value = iter([b"abc", b"def"])
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            image_inputs.requests, "get", return_value=response
        ) as request_get:
            output = Path(tmp_dir) / "remote.png"
            result = image_inputs._download_image_url("https://example.test/a.png", output)
            stored = output.read_bytes()

        self.assertEqual(result, (str(output), "a.png", "image/png"))
        self.assertEqual(stored, b"abcdef")
        self.assertTrue(request_get.call_args.kwargs["stream"])
        response.close.assert_called_once_with()

    def test_remote_download_closes_response_when_limit_is_exceeded(self) -> None:
        response = mock.Mock(
            status_code=200,
            headers={"content-type": "image/png"},
        )
        response.iter_content.return_value = iter([b"abc", b"def"])
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            image_inputs.requests, "get", return_value=response
        ), mock.patch.object(image_inputs, "MAX_IMAGE_REFERENCE_BYTES", 4):
            output = Path(tmp_dir) / "remote.png"
            with self.assertRaisesRegex(HTTPException, "image_url exceeds"):
                image_inputs._download_image_url("https://example.test/a.png", output)

        response.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
