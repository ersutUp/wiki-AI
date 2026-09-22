"""
OCR 文档向量化入库流水线

流程：
1. 用 MarkdownDirectorySplitter 按标题切分 OCR 产物目录（每页一个 _page_N.md）
2. 文本块：正文直接做文本向量化
3. 图片块：多模态 LLM 结合前后文生成摘要 → 摘要 + 图片联合向量化
4. 组装为 OcrDocument 写入 Milvus
"""
import requests
from langchain_core.documents import Document
from pymilvus import MilvusClient

from schema import OcrDocument, OcrFileType
from splitter import split_markdown_directory
from splitter.MarkDownSplitter import (
    KEY_DOC_TYPE,
    KEY_IMAGE_PATH,
    KEY_PRECEDING_TEXT,
    KEY_FOLLOWING_TEXT,
    DOC_TYPE_IMAGE,
    DOC_TYPE_TEXT, KEY_START_PAGE, KEY_END_PAGE,
)
from utils.env_utils import LLM_API_KEY, LLM_BASE_URL
from utils.image_encoder import image_to_base64
from vector import text_to_vector, text_image_to_vector

# Milvus 服务地址
MILVUS_URI = "http://localhost:19530"
# 入库目标集合名
COLLECTION_NAME = "ocr_documents"


def describe_image(image_path: str, preceding: str, following: str) -> str | None:
    """调用多模态 API 描述图片，失败时打印详细错误并返回 None"""
    payload = {
        "model": "qwen3-vl-plus",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "根据图片以及位于图片上边内容和下边内容，生成一段小于 300 字的概括。\n\n"
                    f"上边内容：{preceding}\n\n"
                    f"下边内容: {following}"
                )},
                {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}},
            ],
        }],
    }
    try:
        resp = requests.post(
            f"{LLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=payload,
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"❌ API HTTP {resp.status_code}: {resp.text[:500]}")
            return None
        body = resp.json()
        if body.get("code"):
            print(f"❌ API 业务错误 code={body['code']} code_msg={body.get('code_msg', '')}")
            return None
        return body["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return None


def doc_to_record(doc: Document, source: str) -> OcrDocument | None:
    """将切分后的 Document 转换为 OcrDocument（生成内容与向量），无法处理时返回 None

    参数：
        doc: 切分后的块（doc_type 区分文本块 / 图片块）
        source: 来源标识，写入记录的 source 字段
    """
    doc_type = doc.metadata.get(KEY_DOC_TYPE)
    start_page = doc.metadata.get(KEY_START_PAGE)
    end_page = doc.metadata.get(KEY_END_PAGE)

    # 基础元数据：页码范围 + 标题链（h1 > h2 > h3，缺失的层级自动跳过）
    record = OcrDocument(
        doc_type=doc_type,
        source=source,
        file_type=OcrFileType.PDF,
        pageNo=f"{start_page}-{end_page}",
        chapter=" > ".join(
            h for h in [
                doc.metadata.get("h1"),
                doc.metadata.get("h2"),
                doc.metadata.get("h3"),
            ] if h
        ) or None,
    )

    if doc_type == DOC_TYPE_IMAGE:
        # 图片块：先用多模态 LLM 结合前后文生成摘要，再用「摘要 + 图片」联合向量化
        image_path = doc.metadata.get(KEY_IMAGE_PATH)
        preceding = doc.metadata.get(KEY_PRECEDING_TEXT, "")
        following = doc.metadata.get(KEY_FOLLOWING_TEXT, "")

        summary = describe_image(image_path, preceding, following)
        if summary is None:
            return None

        record.file_path = image_path
        record.content = summary
        record.embeddings = text_image_to_vector(summary, image_path)

    elif doc_type == DOC_TYPE_TEXT:
        # 文本块：正文直接向量化
        record.content = doc.page_content
        record.embeddings = text_to_vector(doc.page_content)
    else:
        return None

    return record


def ingest_to_milvus(
    dir_path: str,
    source: str,
    uri: str = MILVUS_URI,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """切分目录、向量化并写入 Milvus

    参数：
        dir_path: OCR 产物目录（每页一个 _page_N.md）
        source: 来源标识，写入每条记录的 source 字段
        uri: Milvus 服务地址
        collection_name: 目标集合名（需已建好 schema）
    """
    client = MilvusClient(uri=uri)
    # 按标题切分，得到有序的文本块与图片块
    split_doc: list[Document] = split_markdown_directory(dir_path)
    for doc in split_doc:
        # 返回 None 表示该块无法处理（如图片描述生成失败），跳过不入库
        record = doc_to_record(doc, source)
        if record is None:
            continue
        client.insert(collection_name, [record.to_dict()])


if __name__ == "__main__":

    # dir_path 为 OCR 产物目录，source 为本次入库的文档来源标识
    ingest_to_milvus(
        "/Users/ersut/my/code/wiki-AI/project/graphRAG/output/PDF合并",
        source="文件路径",
    )
    print("okk")