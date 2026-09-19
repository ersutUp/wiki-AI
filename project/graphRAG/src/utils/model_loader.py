"""
模型加载工具 — 延迟加载 + 单例缓存

避免重复加载同一模型，所有模块共享缓存。
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from langchain_core.embeddings import Embeddings

# 默认多模态模型
_DEFAULT_MODEL = os.environ.get("MULTIMODAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")

# 模型单例缓存
_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str | None = None) -> SentenceTransformer:
    """延迟加载 sentence-transformers 模型，同名模型只加载一次"""
    name = model_name or _DEFAULT_MODEL
    if name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


class LangChainEmbeddingsAdapter(Embeddings):
    """
    将 SentenceTransformer 实例包装为 langchain 兼容的 embeddings 对象

    用于 SemanticChunker 等需要 langchain embeddings 接口的场景。
    """

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本"""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """编码单条文本"""
        return self.model.encode(text).tolist()


def get_langchain_embeddings(model_name: str | None = None) -> LangChainEmbeddingsAdapter:
    """
    获取 langchain 兼容的 embeddings 对象

    内部复用 get_model() 的缓存，避免重复加载模型。
    """
    model = get_model(model_name)
    return LangChainEmbeddingsAdapter(model)