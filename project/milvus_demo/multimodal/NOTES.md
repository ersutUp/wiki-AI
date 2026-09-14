# 图文融合向量化笔记 — multimodal/

**核心流程：** 准备图片 → LLM 自动生成描述 → 图文融合编码（一张图+一段描述 → 一个向量）→ 插入 → 融合搜索

---

## 写入（multimodal_insert.py）

### 第一步：大模型生成图片描述

用多模态 LLM（qwen3-vl）看图写描述，填充 description 字段：

```python
from llm.my_llm import qwen3_vl_llm
from langchain_core.messages import HumanMessage
import base64

for item in items:
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "请用一句话描述这张图片的内容，不超过 30 个字。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )
    item["description"] = qwen3_vl_llm.invoke([message]).content
```

### 第二步：图文融合编码

一张图 + 一段描述 → **一个**融合向量（不是两个）：

```python
# MultimodalInput：{"text": 描述, "image": PIL图片}
multimodal_inputs = [
    {"text": item["description"], "image": Image.open(...).convert("RGB")}
    for item in items
]
embeddings = model.encode(multimodal_inputs)  # shape: (6, 2048)
```

---

## 查询（multimodal_search.py）

### 纯文本查询

```python
query_text_emb = model.encode("一只橘猫趴在窗台上，懒洋洋地吃着碗里的猫粮")
results = client.search(collection_name=COLLECTION_NAME,
                        data=[query_text_emb.tolist()], limit=5, ...)
```

### 图文融合查询（图 + 文本一起查）

```python
multimodal_query = {"text": query_text, "image": query_image}
multimodal_query_emb = model.encode(multimodal_query)   # 同样融合为一个向量

results = client.search(collection_name=COLLECTION_NAME,
                        data=[multimodal_query_emb.tolist()], limit=5, ...)
```

---

## 与 image/ 目录纯图片向量的区别

| | image/（纯图片） | multimodal/（图文融合） |
|---|---|---|
| 向量来源 | 只编码图片像素 | 图片像素 + LLM 生成的文字描述融合 |
| 语义信息 | 模型视觉理解（有限） | 视觉 + 显式语言描述（更丰富） |
| 描述来源 | 人工手写 | LLM 看图自动生成 |

## 要点

- 融合编码是 Qwen3-VL-Embedding 的特性：`{"text": ..., "image": ...}` 字典输入，模型内部把两种模态的语义融合成单个向量
- LLM 描述相当于给图片「打标签」：视觉上难表达的抽象概念（情绪、场景、动作）由文字补充进向量
- 插入侧和查询侧编码方式要**对称**：插入用图文融合向量，查询既可以纯文本也可以图文融合
- LLM 生成描述有成本（每张图一次调用），但只发生在写入时；查询时无额外 LLM 开销
- `item.pop("path")` 在组装输入时移除路径字段，避免把 path 插进 Milvus（也可以留在动态字段里）
