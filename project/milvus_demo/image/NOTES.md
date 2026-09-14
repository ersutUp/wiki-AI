# 图片向量化搜索笔记 — image/

**核心流程：** 批量加载图片 → 批量编码 embedding → 插入 Milvus → 以图搜图 / 文本搜图

---

## 写入（image_insert.py）

### 批量向量化插入

```python
from PIL import Image

# 批量加载图片 → 批量生成 embedding（比逐张编码快很多）
pil_images = [Image.open(os.path.join(PROJECT_ROOT, img["path"])).convert("RGB") for img in images]
embeddings = model.encode(pil_images)  # ndarray, shape: (N, 2048)

# 组装插入数据：ndarray 每行转 list
data = [
    {**img, VECTOR_FIELD: emb.tolist()} for img, emb in zip(images, embeddings)
]
client.insert(collection_name=COLLECTION_NAME, data=data)
```

---

## 查询（image_search.py）

### 以图搜图（同模态）

```python
query_image = Image.open(os.path.join(PROJECT_ROOT, query_name)).convert("RGB")
query_emb = model.encode(query_image)   # 单张编码，返回一维向量

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_emb.tolist()],
    limit=3,
    output_fields=["id", "title", "description"],
)
```

### 文本搜图（跨模态）

```python
query_text_emb = model.encode("一只猫")   # 直接编码文本

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_text_emb.tolist()],
    limit=3,
    output_fields=["id", "title", "description"],
)
```

---

## 要点

- 模型：`Qwen/Qwen3-VL-Embedding-2B`（多模态 embedding，图文共用同一向量空间，维度 2048）
- **图文同空间**是关键：图片编码和文本编码落在同一个 2048 维空间里，所以一张图的向量既能匹配图片也能匹配文本
- 图片与文本数据共用同一集合 `book_search`，靠标量字段（title/category/description）区分
- `model.encode()` 输入灵活：PIL 图片列表 / 单张图片 / 文本字符串 / 列表都行
- 批量编码（传列表）比逐张编码快很多，插入场景优先批量
- 图片路径基于 `PROJECT_ROOT`（脚本在子目录，取上一级），资源统一放 `resources/images/`
