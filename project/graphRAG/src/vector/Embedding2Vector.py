"""
本地 Embedding 向量转换模块

基于 sentence-transformers 的多模态模型（如 CLIP），统一处理文本和图文混合场景。
所有方法共用同一个模型，文本数据也通过多模态模型的文本分支编码。

编码格式统一使用 sentence-transformers 原生多模态输入：
  - 纯文本： {"text": "..."}
  - 图文：   {"text": "...", "image": PIL.Image}

配置优先级：函数参数 > 环境变量 > 默认值
"""
from typing import Optional, Union
from pathlib import Path

from PIL import Image

from config.logger import get_app_logger
from utils.model_loader import get_model

_logger = get_app_logger("vector")


def _ensure_image(image: Union[str, Path, Image.Image]) -> Image.Image:
    """统一输入为 PIL.Image"""
    if isinstance(image, Image.Image):
        return image
    return Image.open(image).convert("RGB")


def text_to_vector(
    text: str,
    model: Optional[str] = None,
    normalize: bool = True,
) -> list[float]:
    """
    将纯文本转换为向量（使用多模态模型的文本编码分支）

    参数：
        text: 待转换的文本
        model: sentence-transformers 多模态模型名，默认 clip-ViT-B-32-multilingual-v1
        normalize: 是否 L2 归一化，默认 True

    返回：
        向量列表 list[float]
    """
    enc = get_model(model)
    _logger.debug("纯文本向量化，文本长度: %d", len(text))
    vec = enc.encode([{"text": text}], normalize_embeddings=normalize)
    _logger.debug("纯文本向量化完成，向量维度: %d", len(vec[0]))
    return vec[0].tolist()


def text_image_to_vector(
    text: str,
    image: Union[str, Path, Image.Image],
    model: Optional[str] = None,
    normalize: bool = True,
) -> list[float]:
    """
    将文本 + 图片转换为联合向量

    模型原生融合文本和图像语义，比手动平均更准确。

    参数：
        text: 文本内容
        image: 图片（文件路径 / PIL.Image）
        model: 模型名，默认使用全局多模态模型
        normalize: 是否 L2 归一化，默认 True

    返回：
        向量列表 list[float]
    """
    enc = get_model(model)
    img = _ensure_image(image)
    _logger.debug("图文联合向量化，文本长度: %d", len(text))
    vec = enc.encode([{"text": text, "image": img}], normalize_embeddings=normalize)
    _logger.debug("图文联合向量化完成，向量维度: %d", len(vec[0]))
    return vec[0].tolist()


def texts_to_vectors(
    texts: list[str],
    model: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 32,
) -> list[list[float]]:
    """
    批量将多个文本转换为向量

    参数：
        texts: 文本列表
        model: 模型名，默认使用全局多模态模型
        normalize: 是否 L2 归一化
        batch_size: 批大小

    返回：
        向量列表的列表 list[list[float]]
    """
    enc = get_model(model)
    _logger.info("批量文本向量化，共 %d 条，batch_size=%d", len(texts), batch_size)
    inputs: list[dict] = [{"text": t} for t in texts]
    vecs = enc.encode(
        inputs,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    _logger.info("批量文本向量化完成，输出形状: %s", getattr(vecs, "shape", f"len={len(vecs)}"))
    return vecs.tolist()

