from pathlib import Path

import pytest
from pydantic_ai import BinaryContent, ImageUrl

from robot_agent import load_image
from robot_agent.inputs import sniff_media_type


def test_load_bytes_png(png_bytes):
    part = load_image(png_bytes)
    assert isinstance(part, BinaryContent)
    assert part.media_type == "image/png"
    assert part.data == png_bytes


def test_load_file(tmp_path, png_bytes):
    path = tmp_path / "picture.png"
    path.write_bytes(png_bytes)
    part = load_image(path)
    assert isinstance(part, BinaryContent)
    assert part.media_type == "image/png"


def test_load_file_detects_format_from_content_not_extension(tmp_path, png_bytes):
    path = tmp_path / "picture.jpg"  # wrong extension on purpose
    path.write_bytes(png_bytes)
    assert load_image(path).media_type == "image/png"


def test_load_url():
    part = load_image("https://example.com/cat.png")
    assert isinstance(part, ImageUrl)
    assert part.url == "https://example.com/cat.png"


def test_passthrough(png_bytes):
    binary = BinaryContent(data=png_bytes, media_type="image/png")
    url = ImageUrl(url="https://example.com/cat.png")
    assert load_image(binary) is binary
    assert load_image(url) is url


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_unrecognized_bytes():
    with pytest.raises(ValueError, match="recognize"):
        load_image(b"this is not an image")


def test_unsupported_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(path)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n....", "image/png"),
        (b"\xff\xd8\xff\xe0....", "image/jpeg"),
        (b"GIF89a....", "image/gif"),
        (b"GIF87a....", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBP....", "image/webp"),
        (b"plain text", None),
    ],
)
def test_sniff_media_type(data, expected):
    assert sniff_media_type(data) == expected


def test_path_like_string(tmp_path, png_bytes):
    path = tmp_path / "picture.png"
    path.write_bytes(png_bytes)
    assert isinstance(load_image(str(path)), BinaryContent)
    assert isinstance(load_image(Path(path)), BinaryContent)
