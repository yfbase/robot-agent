"""Helpers for turning image sources into pydantic-ai content parts.

Local files and raw bytes become ``BinaryContent`` (sent as base64 data URLs);
http(s) links become ``ImageUrl``. DeepSeek's Flash models accept JPEG, PNG,
GIF and WebP.
"""

from pathlib import Path

from pydantic_ai import BinaryContent, ImageUrl

ImageInput = str | Path | bytes | BinaryContent | ImageUrl
"""Anything ``load_image`` accepts: a file path, an http(s) URL, raw bytes,
or an already-built content part."""


def sniff_media_type(data: bytes) -> str | None:
    """Detect the image format from magic bytes, ignoring the file extension."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image(source: ImageInput) -> BinaryContent | ImageUrl:
    """Convert an image source into a content part ready for the model."""
    if isinstance(source, (BinaryContent, ImageUrl)):
        return source

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return ImageUrl(url=source)

    if isinstance(source, bytes):
        media_type = sniff_media_type(source)
        if media_type is None:
            raise ValueError(
                "Could not recognize the image format from bytes (expected JPEG, PNG, GIF or WebP)."
            )
        return BinaryContent(data=source, media_type=media_type)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    data = path.read_bytes()
    media_type = sniff_media_type(data)
    if media_type is None:
        raise ValueError(f"Unsupported image format: {path} (expected JPEG, PNG, GIF or WebP).")
    return BinaryContent(data=data, media_type=media_type)
