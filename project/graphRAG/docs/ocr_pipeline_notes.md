# OCR → 切割 → 向量入库 → 知识图谱 流水线笔记

## 整体流程

```
PDF / 图片文件
      │
      ▼ ① DotsOCRParser（多模态 vLLM）
每页 Markdown（output/<文件名>/<文件名>_page_N.md）
      │
      ▼ ② MarkdownDirectorySplitter
文本块（doc_type=text） / 图片块（doc_type=image）
      │
      ├──▶ ③ 向量化 → Milvus（语义检索）
      │
      └──▶ ④ 实体/关系抽取（仅文本块）→ Neo4j（知识图谱）
```

---

## ① OCR 识别：PDF → Markdown

### 原理

`DotsOCRParser` 把 PDF 每页按 dpi 渲染为图片，再调用远端 vLLM 服务（`dotsmorc:1.5`）识别布局和文字，最终将结构化 JSON 通过 `layoutjson2md` 转成 Markdown。图片文件走 fitz 上采样预处理后再送模型。

每页产出一个独立文件，命名规则为 `<文件名>_page_N.md`，同目录下还有 `_page_N.json`（布局结构）和 `_page_N.jpg`（标注图）。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ip / port` | localhost / 6009 | vLLM 服务地址 |
| `model_name` | dotsmorc:1.5 | OCR 模型 |
| `dpi` | 200 | PDF 渲染分辨率，越高识别越准但越慢 |
| `num_thread` | 64 | 多页并发线程数 |
| `prompt_mode` | prompt_layout_all_en | 识别模式，含布局+文字 |
| `fitz_preprocess` | True | 图片输入时做 fitz 上采样 |

### 代码

```python
from dots_ocr import DotsOCRParser

parser = DotsOCRParser(
    ip="localhost",
    port=6009,
    dpi=200,
    num_thread=64,
    output_dir="./output",
)
parser.parse_file(
    "document.pdf",
    prompt_mode="prompt_layout_all_en",
    fitz_preprocess=True,
)
# 产物目录：./output/document/
#   document_page_1.md / .json / .jpg
#   document_page_2.md / .json / .jpg ...
```

---

## ② 文档切割：Markdown → 文本块 / 图片块

### 切割策略

`MarkdownDirectorySplitter` 按以下顺序处理：

1. **提取 base64 图片**：OCR 产物中图片以 `![alt](data:image/...;base64,...)` 内嵌，先提取保存为文件，在原文位置替换为占位标记 `<<IMAGE:path>>`。
2. **按标题切割**：用 `MarkdownHeaderTextSplitter` 按 `#` / `##` / `###` 三级标题切分，每个 section 带 h1/h2/h3 元数据。
3. **跨页合并**：两种情况触发合并——标题链完全一致（同节跨页延续），或页首 section 无标题（上页章节正文跨页续写）。
4. **超长语义切割**：section 字符数超过阈值（默认 1000）时，用 `SemanticChunker` 做语义切割。`SemanticChunker` 默认不支持中文句号，已修复正则为 `r'(?<=[。！？.!?])\s+'`。语义切割不可用时回退到 `RecursiveCharacterTextSplitter`（chunk_size=1000, overlap=150）。
5. **生成图片块**：图片块记录前后各 300 字符的上下文，用于后续多模态 LLM 生成摘要。

### 每个 Document 的 metadata

```python
{
    "doc_type": "text" | "image",
    "h1": "章节标题",       # 可能为 None
    "h2": "小节标题",
    "h3": "子节标题",
    "start_page": 1,
    "end_page": 2,
    "chunk_index": 0,        # 全局有序编号
    # 图片块专有：
    "image_path": "/output/document/document_page_2_img_0.png",
    "preceding_text": "图片前 300 字符...",
    "following_text": "图片后 300 字符...",
}
```

### 代码

```python
from splitter import split_markdown_directory

docs = split_markdown_directory(
    "./output/document",
    chunk_threshold=1000,   # 超过此长度触发语义切割
)
# 返回 list[Document]，按 chunk_index 全局有序
text_docs  = [d for d in docs if d.metadata["doc_type"] == "text"]
image_docs = [d for d in docs if d.metadata["doc_type"] == "image"]
```

---

## ③ 向量化入库 Milvus

### 向量化方式

底层使用 `sentence-transformers` 多模态模型（默认 `clip-ViT-B-32-multilingual-v1`），文本和图文走同一模型的不同编码路径，输出向量经 L2 归一化。

| 块类型 | 处理流程 |
|--------|----------|
| 文本块 | 正文 → `text_to_vector()` → 向量 |
| 图片块 | 多模态 LLM（qwen3-vl-plus）结合前后文生成 ≤300 字摘要 → `text_image_to_vector(摘要, 图片)` → 向量 |

图片块走图文联合编码，比单独对摘要编码更能融合视觉语义。

### 图片摘要生成

```python
# 调用多模态 API 生成摘要
payload = {
    "model": "qwen3-vl-plus",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "根据图片以及位于图片上边内容和下边内容，生成一段小于 300 字的概括。\n\n"
                f"上边内容：{preceding}\n\n下边内容：{following}"
            )},
            {"type": "image_url", "image_url": {"url": image_to_base64(image_path)}},
        ],
    }],
}
```

### 向量化代码

```python
from vector import text_to_vector, text_image_to_vector

# 文本块
vector = text_to_vector(doc.page_content)

# 图片块：先生成摘要，再联合编码
summary = describe_image(image_path, preceding_text, following_text)
vector  = text_image_to_vector(summary, image_path)
```

### 写入 Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")
client.insert("ocr_documents", [{
    "doc_type":   record.doc_type,
    "source":     record.source,
    "pageNo":     record.pageNo,
    "chapter":    record.chapter,   # "h1 > h2 > h3"
    "content":    record.content,
    "file_path":  record.file_path, # 图片块专有
    "embeddings": record.embeddings,
}])
```

---

## ④ 实体/关系抽取入库 Neo4j

### 为什么只处理文本块

图片块的语义已在步骤③中由多模态 LLM 转化为文字摘要并入库 Milvus，将原始图片路径送给 `LLMGraphTransformer` 没有意义，故跳过。

### 实体与关系约束

为防止大模型产生噪声节点（如"步骤"、"示例"），通过白名单限制实体类型和关系搭配：

**实体类型**（8 类）
```
技术、组件、概念、操作、配置项、工具、公司、版本
```

**关系三元组约束**（部分）
```
(技术)  -[:包含]->   (组件)    # ClickHouse 包含 MergeTree
(技术)  -[:依赖]->   (技术)    # ClickHouse 依赖 ZooKeeper
(组件)  -[:实现]->   (概念)    # MergeTree 实现 列式存储
(操作)  -[:作用于]-> (组件)    # INSERT 作用于 MergeTree
(配置项)-[:属于]->   (技术)    # max_memory_usage 属于 ClickHouse
```

### 实体名标准化

写入前对所有实体名做 NFKC 归一（全角→半角）并去除空白，防止同一实体以不同字符串形式产生重复节点：

```python
import unicodedata, re

def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", "", name).strip()
```

**注意**：`LLMGraphTransformer` 构建关系时会为端点新建独立的 Node 对象，不引用 nodes 列表中已有节点，因此必须对 nodes 列表和关系端点分别做标准化。

### 幂等写入

写入前建唯一约束，保证重复执行不产生重复节点：

```python
graph.query(
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
)
# baseEntityLabel=True：所有节点加 __Entity__ 标签，便于跨类型查询
# include_source=False：不存 Document 节点，溯源由 Milvus 侧负责
graph.add_graph_documents(graph_docs, include_source=False, baseEntityLabel=True)
```

### 代码

```python
from utils.neo4j_graph import extract_and_store

text_docs = [d for d in split_docs if d.metadata["doc_type"] == "text"]
for doc in text_docs:
    doc.metadata.setdefault("source", source)

extract_and_store(
    text_docs,
    uri=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
```

---

## 一键运行完整流水线

```python
from src.ocr_to_milvus import process_file_to_milvus

process_file_to_milvus(
    input_path="document.pdf",
    source="document",            # 可选，默认取文件名
    output_dir="./output",
    prompt_mode="prompt_layout_all_en",
    fitz_preprocess=True,
    parser_kwargs={"ip": "localhost", "port": 6009},
    uri="http://localhost:19530",
    collection_name="ocr_documents",
    enable_neo4j=True,            # False 则跳过知识图谱步骤
)
```

内部执行顺序：`DotsOCRParser.parse_file` → `split_markdown_directory` → 逐块 `doc_to_record` + `client.insert` → `extract_and_store`。

---

## 依赖服务

| 服务 | 默认地址 | 用途 |
|------|----------|------|
| DotsOCR vLLM | localhost:6009 | OCR 识别 |
| qwen3-vl-plus API | LLM_BASE_URL | 图片摘要生成 |
| sentence-transformers | 本地 | 向量化 |
| Milvus | localhost:19530 | 向量存储 |
| Neo4j | NEO4J_URI | 知识图谱存储 |
