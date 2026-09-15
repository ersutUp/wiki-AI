"""
Markdown 目录切割器模块

提供 Markdown 文档的目录级切割功能，支持：
- 按标题等级切割
- 跨页章节合并
- 语义切割（SemanticChunker）
- 图片提取与保存
- 全局有序输出
"""

from .MarkDownSplitter import (
    MarkdownDirectorySplitter,
    split_markdown_directory,
    DEFAULT_HEADERS_TO_SPLIT_ON,
)

__all__ = [
    'MarkdownDirectorySplitter',
    'split_markdown_directory',
    'DEFAULT_HEADERS_TO_SPLIT_ON',
]
