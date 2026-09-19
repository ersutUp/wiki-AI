from typing import TypedDict

from langchain_core.documents import Document

from llm import qwen3_vl_llm
from splitter import split_markdown_directory
from splitter.MarkDownSplitter import (
    KEY_DOC_TYPE,
    KEY_IMAGE_PATH,
    KEY_PRECEDING_TEXT,
    KEY_FOLLOWING_TEXT,
    DOC_TYPE_IMAGE,
    DOC_TYPE_TEXT,
)
from utils.image_encoder import image_to_base64
from vector import text_to_vector, text_image_to_vector

split_doc: list[Document] = split_markdown_directory(
    "/Users/ersut/my/code/wiki-AI/project/graphRAG/output/PDF合并",
)


for doc in split_doc:

    doc_type = doc.metadata.get(KEY_DOC_TYPE)
    if doc_type == DOC_TYPE_IMAGE:
        # 图片 Document：page_content 为空，内容在 metadata 里
        image_path = doc.metadata.get(KEY_IMAGE_PATH)
        preceding = doc.metadata.get(KEY_PRECEDING_TEXT, "")
        following = doc.metadata.get(KEY_FOLLOWING_TEXT, "")

        # 先让多模态大模型根据图片生成描述
        summary = qwen3_vl_llm.invoke([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                    根据图片以及位于图片上边内容和下边内容，以图片为主，文字为辅，生成一段小于 300 字的概括。
                    
                    上边内容：{preceding}
                    
                    下边内容: {following}
                    """},
                    {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}},
                ],
            }
        ])

        # 用「图片描述 + 前后文」编码为向量
        vector = text_image_to_vector(
            summary,
            image_path,
        )
    elif doc_type == DOC_TYPE_TEXT:
        # 文本 Document：直接编码
        vector = text_to_vector(doc.page_content)
    else:
        continue

print("okk")