"""
图片编码工具 — 统一将各类图片输入转为 base64 data URI
"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Union

from PIL import Image

# 后缀 → MIME 类型映射
_MIME_MAP = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
}


def image_to_base64(
    image: Union[str, Path, Image.Image, bytes],
    mime: str | None = None,
) -> str:
    """
    将图片转换为 base64 data URI 字符串

    参数：
        image: 文件路径 / PIL.Image / 原始字节
        mime: 强制指定 MIME 类型（如 "png"），默认按文件后缀推断

    返回：
        "data:image/png;base64,xxxx" 格式的字符串
        （OpenAI 兼容接口 image_url 标准格式）
    """
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        mime = mime or "png"
    elif isinstance(image, bytes):
        data = image
        mime = mime or "png"
    else:
        path = Path(image)
        suffix = path.suffix.lower()
        mime = mime or _MIME_MAP.get(suffix, "png")
        data = path.read_bytes()

    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"