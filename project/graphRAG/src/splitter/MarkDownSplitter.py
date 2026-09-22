"""
Markdown 目录级流式切割器

功能：
1. 逐页读取目录下的 Markdown 文件（按页码排序）
2. 提取并保存 base64 图片为文件
3. 用 MarkdownHeaderTextSplitter 按标题等级切割
4. 跨页延续的章节合并（相同标题或页首无标题时）
5. 超过 1000 字符的章节用 SemanticChunker 语义切割
6. 生成图片 Document（含前后文本）
7. 全局有序输出（chunk_index 递增）
"""
import os
import re
import base64
from pathlib import Path
from typing import Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.logger import get_app_logger

_logger = get_app_logger("splitter")

# 可选导入
try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False


# 默认标题层级配置（h1-h3，避免 OCR 产物的 #### 垃圾空标题）
DEFAULT_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# 图片提取正则：匹配 ![alt](data:image/...;base64,...)
IMAGE_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(data:image/(png|jpeg|jpg|webp);base64,([^)]+)\)',
    re.IGNORECASE
)

# 页文件匹配正则
PAGE_FILE_PATTERN = re.compile(r'^(?P<name>.+)_page_(?P<n>\d+)\.md$')

# 图片占位标记正则
IMAGE_PLACEHOLDER = re.compile(r'<<IMAGE:([^>]+)>>')

# ── metadata 键名常量 ────────────────────────────────────
KEY_DOC_TYPE = 'doc_type'
KEY_START_PAGE = 'start_page'
KEY_END_PAGE = 'end_page'
KEY_IMAGE_PATH = 'image_path'
KEY_ALT = 'alt'
KEY_PAGE = 'page'
KEY_PRECEDING_TEXT = 'preceding_text'
KEY_FOLLOWING_TEXT = 'following_text'
KEY_CHUNK_INDEX = 'chunk_index'
KEY_PATH = 'path'

# doc_type 取值常量
DOC_TYPE_TEXT = 'text'
DOC_TYPE_IMAGE = 'image'

# 图片上下文截取长度（前文/后文各取的字符数）
IMAGE_CONTEXT_CHARS = 300

class MarkdownDirectorySplitter:
    """
    Markdown 目录级流式切割器

    参数：
        embeddings: 语义切割用的 embedding 模型，默认 OpenAIEmbeddings()
                   （通过 OPENAI_API_KEY / OPENAI_BASE_URL 环境变量配置）
        headers_to_split_on: 标题层级配置，默认 h1-h3
        chunk_threshold: 超过此字符数才触发语义切割，默认 1000
        semantic_config: SemanticChunker 额外配置（如 breakpoint_threshold_type）
        strip_headers: 是否从正文中移除标题行（默认 False，保留标题利于 embedding）
    """

    def __init__(
        self,
        embeddings=None,
        headers_to_split_on=None,
        chunk_threshold: int = 1000,
        semantic_config: Optional[dict] = None,
        strip_headers: bool = False,
    ):
        self.headers_to_split_on = headers_to_split_on or DEFAULT_HEADERS_TO_SPLIT_ON
        self.chunk_threshold = chunk_threshold
        self.strip_headers = strip_headers

        # 存储 embedding 规格，延迟加载
        self._embeddings_spec = embeddings
        self._semantic_splitter = None
        self._has_semantic = HAS_SEMANTIC
        # SemanticChunker 默认 sentence_split_regex 只匹配英文标点，不兼容中文
        # 合并用户配置，sentence_split_regex 优先中文断句
        self._semantic_config = dict(
            # ⚠️原版正则不支持中文句号，这里修复中文分句！
            sentence_split_regex=r'(?<=[。！？.!?])\s+',
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,  # 默认70，落差前30%才切分
            min_chunk_size=50
        )
        self._semantic_config.update(semantic_config or {})

        # 初始化 MarkdownHeaderTextSplitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=self.strip_headers,
        )

        # 初始化规则切割（始终可用，作为回退）
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_threshold,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def _get_semantic_splitter(self):
        """
        延迟初始化语义切割器，只加载一次

        self._embeddings_spec 支持三种类型：
        - str: SentenceTransformer 模型名，延迟加载
        - 有 encode() 方法的对象: SentenceTransformer 实例，直接用
        - 有 embed_documents() 的对象: langchain embedding，直接用
        - None: 尝试默认 OpenAIEmbeddings

        返回 None 表示语义切割不可用
        """
        if self._semantic_splitter is not None:
            return self._semantic_splitter

        if not self._has_semantic:
            return None

        spec = self._embeddings_spec

        if isinstance(spec, str):
            # 模型名 → 通过共享缓存的模型包装为 langchain 兼容对象
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                spec = HuggingFaceEmbeddings(model=spec)
            except Exception as e:
                _logger.warning("Embedding 模型加载失败: %s", e)
                return None
        elif spec is not None and hasattr(spec, 'encode'):
            pass
        elif spec is None:
            # 使用 共享缓存 的模型
            # 模型名 → 通过共享缓存获取 SentenceTransformer，再包装为 langchain 兼容对象
            try:
                from utils.model_loader import get_langchain_embeddings
                spec = get_langchain_embeddings(spec)
            except Exception as e:
                _logger.warning("Embedding 模型加载失败: %s", e)
                return None

        try:
            self._semantic_splitter = SemanticChunker(
                embeddings=spec,
                **self._semantic_config,
            )
        except Exception as e:
            _logger.warning("SemanticChunker 初始化失败: %s", e)
            return None

        return self._semantic_splitter

    # ── 公共接口 ──────────────────────────────────────────

    def split_directory(self, dir_path: str | Path) -> list[Document]:
        """
        切割整个目录下的 Markdown 文件

        参数：
            dir_path: 目录路径（如 output/demo_pdf1）

        返回：
            List[Document]，全局有序（chunk_index 递增）
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise ValueError(f"目录不存在: {dir_path}")

        page_files = self._iter_page_files(dir_path)
        if not page_files:
            _logger.warning("目录 %s 下未找到页文件（_page_N.md）", dir_path)
            return []

        _logger.info("开始切分目录 %s，共 %d 个页文件", dir_path, len(page_files))

        image_registry = {}
        results = []
        buffered = None
        active_headers = {}  # 跨页维护当前标题链，如 {'h1': '路费', 'h2': '飞机：'}

        for page_no, page_path in page_files:
            page_text = page_path.read_text(encoding='utf-8')
            page_text = self._preprocess(page_text)

            page_text, page_images = self._extract_images(page_text, dir_path, page_no)
            image_registry.update(page_images)

            sections = self.header_splitter.split_text(page_text)
            _logger.debug("第 %d 页按标题切分为 %d 个 section", page_no, len(sections))

            for i, sec in enumerate(sections):
                sec.metadata = self._inherit_parent_headers(sec.metadata, active_headers)
                self._update_active_headers(active_headers, sec.metadata)

                if self._should_merge(buffered, sec, i):
                    buffered['text'] += '\n\n' + sec.page_content
                    buffered[KEY_END_PAGE] = page_no
                else:
                    if buffered:
                        results.extend(self._flush(buffered))
                    buffered = {
                        'text': sec.page_content,
                        'headers': sec.metadata,
                        KEY_START_PAGE: page_no,
                        KEY_END_PAGE: page_no,
                    }

        if buffered:
            results.extend(self._flush(buffered))

        # 最后统一处理图片 Document
        results = self._insert_image_documents(results, image_registry)

        for idx, doc in enumerate(results):
            doc.metadata[KEY_CHUNK_INDEX] = idx

        _logger.info(
            "目录切分完成，共 %d 个结果块（文本块 + 图片块），图片注册表 %d 条",
            len(results), len(image_registry),
        )
        return results

    # ── 跨页合并判断 ──────────────────────────────────────

    def _should_merge(self, buffered: dict | None, sec: Document, idx: int) -> bool:
        """
        当前 section 是否和缓冲中的章节是同一节

        两种情况：
        ① 标题链完全一致 → 同节跨页延续
        ② 当前页首 section 无标题、缓冲中有标题 → 上页章节的正文跨页延续
        """
        if buffered is None:
            return False
        # 情况①
        if sec.metadata == buffered['headers']:
            return True
        # 情况②：页首 section，无标题，缓冲中已有标题
        if idx == 0 and not sec.metadata and buffered['headers']:
            return True
        return False

    # ── 父级标题继承 ──────────────────────────────────────

    def _inherit_parent_headers(
        self, metadata: dict, active_headers: dict
    ) -> dict:
        """
        用 active_headers 补全 metadata 中缺失的父级标题

        例：上一页的 active_headers = {'h1': '路费'}，
        当前 metadata = {'h2': '方案四'} → 补全为 {'h1': '路费', 'h2': '方案四'}
        """
        result = dict(metadata)
        for level in ('h1', 'h2', 'h3'):
            if level not in result and level in active_headers:
                result[level] = active_headers[level]
        return result

    def _update_active_headers(
        self, active_headers: dict, metadata: dict
    ) -> None:
        """
        根据当前 section 的标题更新 active_headers

        有 h1 → 覆盖 h1，清除 h2/h3
        有 h2 → 覆盖 h2，清除 h3
        有 h3 → 覆盖 h3
        无标题 → 保持不动（正文延续）
        """
        if not metadata:
            return
        has_h1 = 'h1' in metadata
        has_h2 = 'h2' in metadata
        has_h3 = 'h3' in metadata

        if has_h1:
            active_headers['h1'] = metadata['h1']
            active_headers.pop('h2', None)
            active_headers.pop('h3', None)
        if has_h2:
            active_headers['h2'] = metadata['h2']
            active_headers.pop('h3', None)
        if has_h3:
            active_headers['h3'] = metadata['h3']

    # ── 文件发现 ──────────────────────────────────────────

    def _iter_page_files(self, dir_path: Path) -> list[tuple[int, Path]]:
        """发现并排序页文件，排除 _nohf.md"""
        page_files = []
        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue
            match = PAGE_FILE_PATTERN.match(file_path.name)
            if not match:
                continue
            if file_path.name.endswith('_nohf.md'):
                continue
            page_no = int(match.group('n'))
            page_files.append((page_no, file_path))
        page_files.sort(key=lambda x: x[0])
        return page_files

    # ── 预处理 ────────────────────────────────────────────

    def _preprocess(self, text: str) -> str:
        """压缩连续空行"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    # ── 图片提取 ──────────────────────────────────────────

    def _extract_images(self, text: str, dir_path: Path, page_no: int) -> tuple[str, dict]:
        """提取 base64 图片保存为文件，返回 (替换后文本, 图片注册表)"""
        images = {}
        img_dir = dir_path / 'images'
        img_dir.mkdir(exist_ok=True)

        def replace_image(match):
            alt = match.group(1)
            ext = match.group(2).lower()
            if ext == 'jpeg':
                ext = 'jpg'
            b64_data = match.group(3)

            img_idx = len(images)
            img_id = f"page_{page_no}_img_{img_idx}"
            img_filename = f"{dir_path.name}_{img_id}.{ext}"
            img_path = img_dir / img_filename

            try:
                img_bytes = base64.b64decode(b64_data)
                img_path.write_bytes(img_bytes)
            except Exception:
                return match.group(0)

            images[img_id] = {
                KEY_PATH: str(img_path),
                KEY_ALT: alt,
                KEY_PAGE: page_no,
            }
            return f'<<IMAGE:{img_id}>>'

        new_text = IMAGE_PATTERN.sub(replace_image, text)
        if images:
            _logger.debug("第 %d 页提取 %d 张图片", page_no, len(images))
        return new_text, images

    # ── Flush ─────────────────────────────────────────────

    def _flush(self, buffered: dict) -> list[Document]:
        """输出缓冲章节，超出阈值则切割"""
        text = buffered['text']
        headers = buffered['headers']
        start_page = buffered[KEY_START_PAGE]
        end_page = buffered[KEY_END_PAGE]

        base_doc = Document(
            page_content=text,
            metadata={
                **headers,
                KEY_START_PAGE: start_page,
                KEY_END_PAGE: end_page,
                KEY_DOC_TYPE: DOC_TYPE_TEXT,
            }
        )

        if len(text) > self.chunk_threshold:
            _logger.debug("文本块超过阈值 %d（当前 %d 字符），触发语义切割", self.chunk_threshold, len(text))
            semantic = self._get_semantic_splitter()
            if semantic:
                # 语义切割，超长的块继续切，最多 3 次
                original_count = 1
                chunks = [base_doc]
                chunks = self._semantic_split(chunks, semantic)
                _logger.debug("语义切割完成，%d 个块 → %d 个块", original_count, len(chunks))
            else:
                chunks = self.fallback_splitter.split_documents([base_doc])
                _logger.debug("语义切割不可用，回退到规则切割，产出 %d 个块", len(chunks))
            for chunk in chunks:
                chunk.metadata.update({
                    KEY_START_PAGE: start_page,
                    KEY_END_PAGE: end_page,
                    KEY_DOC_TYPE: DOC_TYPE_TEXT,
                })
            return chunks
        else:
            return [base_doc]

    def _semantic_split(self, chunks, semantic, retry_num = 3 ):
        for attempt in range(retry_num):
            new_chunks = []
            for chunk in chunks:
                if len(chunk.page_content) > self.chunk_threshold:
                    new_chunks.extend(semantic.split_documents([chunk]))
                else:
                    new_chunks.append(chunk)
            if new_chunks == chunks:
                break
            chunks = new_chunks
        return chunks

    # ── 图片 Document 后处理 ───────────────────────────────

    def _insert_image_documents(
        self, results: list[Document], image_registry: dict
    ) -> list[Document]:
        """
        在所有文本块产出后，扫描并生成图片 Document

        文本块保持完整（只移除占位标记），图片 Document 追加在后面。
        原地替换，不建新列表。
        """
        if not image_registry:
            return results

        self._replace_image_placeholders(results, image_registry)
        self._fill_image_context(results)
        return results

    def _replace_image_placeholders(
        self, results: list[Document], image_registry: dict
    ) -> None:
        """遍历文本块，将图片占位替换为图片 Document 原地插入"""
        i = 0
        while i < len(results):
            doc = results[i]
            matches = list(IMAGE_PLACEHOLDER.finditer(doc.page_content))
            if not matches:
                i += 1
                continue

            # 清空占位，文本块保持完整
            clean_text = IMAGE_PLACEHOLDER.sub('', doc.page_content).strip()
            if clean_text:
                results[i] = Document(
                    page_content=clean_text,
                    metadata={**doc.metadata, KEY_DOC_TYPE: DOC_TYPE_TEXT}
                )
                i += 1
            else:
                results.pop(i)

            # 追加本块内的图片
            for match in matches:
                img_id = match.group(1)
                if img_id not in image_registry:
                    continue
                img_info = image_registry[img_id]
                img_meta = {
                    KEY_DOC_TYPE: DOC_TYPE_IMAGE,
                    KEY_IMAGE_PATH: img_info[KEY_PATH],
                    KEY_ALT: img_info[KEY_ALT],
                    KEY_PAGE: img_info[KEY_PAGE],
                    KEY_PRECEDING_TEXT: '',
                    KEY_FOLLOWING_TEXT: '',
                    KEY_START_PAGE: doc.metadata.get(KEY_START_PAGE),
                    KEY_END_PAGE: doc.metadata.get(KEY_END_PAGE),
                }
                for level in ('h1', 'h2', 'h3'):
                    if level in doc.metadata:
                        img_meta[level] = doc.metadata[level]
                results.insert(i, Document(
                    page_content="",
                    metadata=img_meta,
                ))
                i += 1

    def _fill_image_context(self, results: list[Document]) -> None:
        """
        为图片 Document 补全 preceding_text / following_text

        前文：上一个文本块末尾 IMAGE_CONTEXT_CHARS 字符
        后文：下一个文本块开头 IMAGE_CONTEXT_CHARS 字符
        """
        for i, doc in enumerate(results):
            if doc.metadata.get(KEY_DOC_TYPE) != DOC_TYPE_IMAGE:
                continue
            for j in range(i - 1, -1, -1):
                prev = results[j]
                if prev.metadata.get(KEY_DOC_TYPE) == DOC_TYPE_TEXT and prev.page_content:
                    doc.metadata[KEY_PRECEDING_TEXT] = prev.page_content[-IMAGE_CONTEXT_CHARS:]
                    break
            for j in range(i + 1, len(results)):
                nxt = results[j]
                if nxt.metadata.get(KEY_DOC_TYPE) == DOC_TYPE_TEXT and nxt.page_content:
                    doc.metadata[KEY_FOLLOWING_TEXT] = nxt.page_content[:IMAGE_CONTEXT_CHARS]
                    break


def split_markdown_directory(
    dir_path: str | Path,
    embeddings=None,
    headers_to_split_on=None,
    chunk_threshold: int = 1000,
    semantic_config: Optional[dict] = None,
    strip_headers: bool = False,
) -> list[Document]:
    """
    便捷函数：切割整个目录下的 Markdown 文件

    参数：
        dir_path: 目录路径
        embeddings: 语义切割用的 embedding 模型
        headers_to_split_on: 标题层级配置，默认 h1-h3
        chunk_threshold: 超过此字符数才触发语义切割，默认 1000
        semantic_config: SemanticChunker 额外配置
        strip_headers: 是否从正文中移除标题行

    返回：
        List[Document]，全局有序
    """
    splitter = MarkdownDirectorySplitter(
        embeddings=embeddings,
        headers_to_split_on=headers_to_split_on,
        chunk_threshold=chunk_threshold,
        semantic_config=semantic_config,
        strip_headers=strip_headers,
    )
    return splitter.split_directory(dir_path)

if __name__ == '__main__':
    from config.logger import configure_logging
    configure_logging()
    start = os.times().elapsed
    print(start)
    split_doc:list[Document] = split_markdown_directory(
        "/Users/ersut/my/code/wiki-AI/project/graphRAG/output/PDF合并",
    )
    end = os.times().elapsed
    print(end)
    print(end - start)

    print(split_doc)

    print("ok")