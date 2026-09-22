"""
OCR 文档向量化入库流水线

流程：
1. 用 DotsOCRParser 解析 PDF / 图片文件，产出每页的 Markdown
2. 用 MarkdownDirectorySplitter 按标题切分 OCR 产物目录（每页一个 _page_N.md）
3. 文本块：正文直接做文本向量化
4. 图片块：多模态 LLM 结合前后文生成摘要 → 摘要 + 图片联合向量化
5. 组装为 OcrDocument 写入 Milvus
"""
import os

import requests
from langchain_core.documents import Document
from pymilvus import MilvusClient

from config.logger import get_app_logger
from dots_ocr.parser import DotsOCRParser
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

_logger = get_app_logger("ocr_to_milvus")


def describe_image(image_path: str, preceding: str, following: str) -> str | None:
    """调用多模态 API 描述图片，失败时打印详细错误并返回 None"""
    _logger.info("开始生成图片描述，图片路径: %s", image_path)
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
            _logger.error("图片描述 API HTTP %d: %s", resp.status_code, resp.text[:500])
            return None
        body = resp.json()
        if body.get("code"):
            _logger.error("图片描述 API 业务错误 code=%s code_msg=%s", body["code"], body.get("code_msg", ""))
            return None
        content = body["choices"][0]["message"]["content"]
        _logger.info("图片描述生成成功，内容长度: %d", len(content))
        return content
    except Exception as e:
        _logger.exception("图片描述请求异常: %s", e)
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

    _logger.info("转换 Document → OcrDocument，doc_type=%s，页码: %s-%s", doc_type, start_page, end_page)

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
        _logger.warning("未知 doc_type: %s，跳过该块", doc_type)
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
    _logger.info("开始入库，目录: %s，来源: %s，集合: %s", dir_path, source, collection_name)
    client = MilvusClient(uri=uri)
    # 按标题切分，得到有序的文本块与图片块
    split_doc: list[Document] = split_markdown_directory(dir_path)
    _logger.info("目录切分完成，共 %d 个块", len(split_doc))
    inserted = 0
    skipped = 0
    for doc in split_doc:
        # 返回 None 表示该块无法处理（如图片描述生成失败），跳过不入库
        record = doc_to_record(doc, source)
        if record is None:
            skipped += 1
            continue
        client.insert(collection_name, [record.to_dict()])
        inserted += 1
    _logger.info("入库完成，成功: %d，跳过: %d，总计: %d", inserted, skipped, len(split_doc))


def process_file_to_milvus(
    input_path: str,
    source: str = "",
    output_dir: str = "./output",
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = True,
    parser_kwargs: dict | None = None,
    uri: str = MILVUS_URI,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    完整流水线：OCR 解析文件 → 按标题切分 → 向量化 → 写入 Milvus

    参数：
        input_path: 待解析的 PDF / 图片文件路径
        source: 来源标识，默认取文件名
        output_dir: OCR 产物根目录（产物在 <output_dir>/<文件名>/ 下）
        prompt_mode: OCR 解析提示模式，默认 prompt_layout_all_en
        fitz_preprocess: 是否对图片做 fitz 上采样预处理（仅图片生效，PDF 已按 dpi 渲染）
        parser_kwargs: DotsOCRParser 额外构造参数（如 ip / port / model_name / num_thread / dpi）
        uri: Milvus 服务地址
        collection_name: 目标集合名
    """
    _logger.info("流水线启动，输入文件: %s", input_path)
    # ① OCR 解析：产物保存到 <output_dir>/<文件名>/，每页一个 <文件名>_page_N.md
    parser = DotsOCRParser(output_dir=output_dir, **(parser_kwargs or {}))
    parser.parse_file(
        input_path,
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess,
    )
    _logger.info("OCR 解析完成，输入文件: %s", input_path)

    # ② 计算 OCR 产物目录（parse_file 内部按文件名建同名子目录）
    filename = os.path.splitext(os.path.basename(input_path))[0]
    md_dir = os.path.join(os.path.abspath(output_dir), filename)

    # ③ 切分 + 向量化 + 入库
    ingest_to_milvus(md_dir, source or filename, uri=uri, collection_name=collection_name)
    _logger.info("流水线完成，输入文件: %s", input_path)


if __name__ == "__main__":
    from config.logger import configure_logging
    configure_logging()
    # 完整流水线入口：输入一个 PDF / 图片文件，OCR 识别 → 切分 → 向量化 → 入库
    # 注意：需要先启动 dots_ocr 的 vllm 服务（默认 localhost:9180）和 Milvus
    parser_kwargs = {
        "ip" : "localhost",
        "port" : 6009,
    }

    process_file_to_milvus(
        "/output/01_尚硅谷大数据技术之ClickHouse入门V1.01.pdf",
        output_dir="/Us/code/wiki-AI/project/graphRAG/output",
        parser_kwargs=parser_kwargs,
    )
    print("okk")