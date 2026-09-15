# MarkDownSplitter 笔记

## 整体设计

OCR 产出的多页 Markdown 存在三个问题，这个模块逐一解决：

1. **标题层级被分页截断** — 跨页合并 + 父级标题继承
2. **(图片的) base64 数据嵌在文本中** — 提取为文件，用占位符替换
3. **章节过长不利于检索** — 超阈值时触发语义切割

最终输出全局有序（`chunk_index` 递增）的 `Document` 列表，文本和图片各自独立为 Document。

---

## 主要阶段

### 阶段一：逐页切割

```python
for page_path in sorted_page_files(dir_path):           # 按页码排序
    text = preprocess(page_path.read_text())            # 压缩空行
    text = extract_base64_images(text, dir_path)        # base64 → 文件，替换为 <<IMAGE:id>> 占位符
    sections = header_splitter.split(text)              # 按 h1/h2/h3 切割
    # 进入阶段二...
```

### 阶段二：跨页合并 + 标题继承

```python
buffered = None          # {text, headers, start_page, end_page}
active_headers = {}      # 跨页维护的标题链，如 {'h1': '路费', 'h2': '飞机'}

for sec in sections:
    sec.metadata = inherit_parent_headers(sec.metadata, active_headers)  # 补全缺失的父级标题
    update_active_headers(active_headers, sec.metadata)  # 更新当前标题链

    if should_merge(buffered, sec, idx):
        buffered.text += '\n\n' + sec.text              # 拼接，更新 end_page
    else:
        if buffered: results += flush(buffered)         # 输出旧缓冲区
        buffered = {'text': sec.text, 'headers': sec.metadata, ...}  # 新建缓冲区

if buffered: results += flush(buffered)                 # 最后一页收尾
```

合并判断 (`should_merge`)：

| 情形 | 条件 | 例子 |
|---|---|---|
| 同节跨页 | `sec.metadata == buffered.headers` | 第3页 `## 方案四`，第4页也是 `## 方案四`，正文被分页截断 |
| 正文延续 | `idx == 0` 且 sec 无标题、buffered 有标题 | 第2页末尾是 `## 概述` 的正文，第3页开头无标题的段落是它的一部分 |

标题链更新 (`update_active_headers`)：

```
遇到 h1 → 覆盖 h1，清空 h2/h3
遇到 h2 → 覆盖 h2，清空 h3
遇到 h3 → 覆盖 h3
无标题  → 不动（正文延续，标题链保持不变）
```

### 阶段三：Flush — 语义切割

```python
def flush(buffered):
    doc = Document(text=buffered.text, metadata={**buffered.headers, 'doc_type': 'text'})

    if len(doc.text) <= chunk_threshold:                # 短块不切
        return [doc]

    splitter = get_semantic_splitter()
    if splitter is None:
        return fallback_splitter.split([doc])           # 回退：RecursiveCharacterTextSplitter

    chunks = [doc]
    for _ in range(3):                                  # 最多迭代 3 次
        chunks = [
            splitter.split([c]) if len(c.text) > chunk_threshold else [c]
            for c in chunks
        ]
        if all(len(c.text) <= chunk_threshold for c in chunks):
            break                                       # 收敛
    return flatten(chunks)
```

语义切割原理：把文本按标点拆成句子 → 计算相邻句子 embedding 相似度 → 在"落差"最大的位置切断。比纯规则切割更自然。

中文兼容配置：

```python
SemanticChunker(
    embeddings=embedding_model,
    # 默认正则 r'(?<=[.?!])\s+' 只认英文标点，中文句子之间无法断开
    # 加入 。！？ 后中文段落才能正确拆成句子，否则整段被当成一个"句子"
    sentence_split_regex=r'(?<=[。！？.!?])\s+',

    # 断点判定策略：percentile = 只切"相似度落差"排在尾部（最低）的位置
    # 还可用 standard_deviation / interquartile
    breakpoint_threshold_type="percentile",

    # 百分位阈值 85：相似度落差必须低于第 85 百分位（即落差足够大）才允许切断
    # 阈值越高，切割越保守（块更大）；默认 70，这里调高到 85 避免过度切割
    breakpoint_threshold_amount=85,
)
```

### 阶段四：图片后处理

```python
# 在所有文本块产出后统一处理

for doc in results:                                 # 遍历每个文本块
    for match in re.finditer(r'<<IMAGE:(.+)>>', doc.text):
        img_id = match.group(1)
        info = image_registry[img_id]

        doc.text = re.sub(r'<<IMAGE:.+?>>', '', doc.text)  # 移除占位符

        results.insert(
            after=doc,
            Document(                                       # 原地插入 image Document
                text='',
                metadata={
                    'doc_type': 'image',
                    'image_path': info.path,                # 图片文件路径
                    'alt': info.alt,                        # 图片描述
                    'preceding_text': prev_text_doc.text[-300:],   # 上文
                    'following_text': next_text_doc.text[:300],    # 下文
                }
            )
        )
```

---

## 输出结构

```python
# 文本
Document(text="机票票价目前主要分为三类...",
         metadata={'h1': '路费', 'h2': '飞机', 'doc_type': 'text', 'chunk_index': 12})

# 图片
Document(text="",
         metadata={'doc_type': 'image', 'image_path': '...img.jpg',
                   'alt': '机票价格表', 'preceding_text': '...', 'following_text': '...',
                   'chunk_index': 13})
```

## 使用

```python
from splitter import split_markdown_directory

docs = split_markdown_directory("output/某PDF", embeddings="Qwen/Qwen3-VL-Embedding-2B")
```