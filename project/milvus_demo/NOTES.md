# Milvus 向量数据库 Demo 笔记

项目地址：`project/milvus_demo`

---

## 项目结构

```
milvus_demo/
├── config.py                  # 公共常量
├── requirements.txt           # 依赖
├── create_collection_demo.py  # 1. 创建集合
├── insert_demo.py             # 2. 插入数据
├── vector_search_demo.py      # 3. 向量语义搜索
└── scalar_search_demo.py      # 4. 标量过滤查询
```

### 依赖（requirements.txt）

```
pymilvus>=2.5.0                  # Milvus 官方 Python SDK
sentence-transformers>=3.0.0      # 文本 embedding 模型
python-dotenv>=1.0.0              # 环境变量管理
```

### 公共常量（config.py）

```python
MILVUS_URI = "http://localhost:19530"       # Milvus 连接地址
COLLECTION_NAME = "book_search"             # 集合名称
VECTOR_FIELD = "embedding"                  # 向量字段名
MODEL_NAME = "intfloat/multilingual-e5-large"  # embedding 模型
VECTOR_DIM = 1024                           # 向量维度
```

---

## 1. 创建集合（create_collection_demo.py）

**核心流程：** 连接 → 定义 Schema → 创建集合（含索引+加载）

### 关键代码

```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

# 定义 Schema — 手动添加每个字段，精确控制类型
schema = client.create_schema(
    auto_id=True,               # 主键自动生成
    enable_dynamic_field=True,  # 允许插入未预定义的字段
)

schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("description", DataType.VARCHAR, max_length=2048)
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)

# 定义向量索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",     # Milvus 自动选择最优索引
    metric_type="COSINE",       # 余弦相似度
)

# 创建集合 — 创建、建索引、加载一步完成
client.create_collection(
    collection_name="book_search",
    schema=schema,
    index_params=index_params,
)
```

### 要点

- `MilvusClient` 是官方推荐的高层 API，比 `connections` + ORM 更简洁
- 传入 `schema` 走精细创建路径，`create_collection` 内部自动完成 `create_index` + `load_collection`
- 不传 `schema` 只传 `dimension` 则走快速创建，只生成主键+向量字段
- `metric_type` 三种选项：`COSINE`（语义搜索）、`IP`（内积）、`L2`（欧氏距离）
- `AUTOINDEX` 让 Milvus 根据数据规模自动选索引（小数据用 FLAT，大数据自动切 HNSW）

---

## 2. 插入数据（insert_demo.py）

**核心流程：** 连接 + 加载模型 → 准备数据 → 模型编码 → 插入 → 验证

### 关键代码

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# 准备原始数据
books = [
    {"title": "算法导论", "category": "计算机科学",
     "description": "算法领域的经典教材，系统讲解排序、图算法..."},
    {"title": "Python 编程：从入门到实践", "category": "编程语言",
     "description": "面向初学者的 Python 教程..."},
    # ...
]

# 批量编码 — 把所有 description 转成向量（比逐条快很多）
descriptions = [b["description"] for b in books]
embeddings = model.encode(descriptions)  # → ndarray, shape: (5, 1024)

# 组装数据 — 将 ndarray 每行转为 list
data = [
    {**b, "embedding": emb.tolist()} for b, emb in zip(books, embeddings)
]

# 插入
result = client.insert(collection_name="book_search", data=data)
# → {"insert_count": 5, "ids": [468779015314108449, ...]}
```

### 要点

- `model.encode()` 接受列表返回 ndarray，即使单条也要包成 `["文本"]`
- 批量编码比逐条调用快很多
- 向量需要用 `.tolist()` 转为 Python 列表再传给 Milvus
- `insert()` 返回生成的 `insert_count` 和主键 `ids` 列表
- `milvus-lite` 已从 `requirements.txt` 移除（Docker 运行不需要）

---

## 3. 向量语义搜索（vector_search_demo.py）

**核心流程：** 连接 + 加载模型 → 输入查询 → 编码 → 纯向量搜索 → 向量+标量混合搜索

### 纯向量搜索

```python
queries = [
    "如何设计可复用的代码？",
    "有哪些排序算法？",
    "怎样用 Python 写网站？",
]

query_embeddings = model.encode(queries)

for query, embedding in zip(queries, query_embeddings):
    results = client.search(
        collection_name="book_search",
        data=[embedding.tolist()],
        limit=3,
        output_fields=["title", "description"],
    )

    for hit in results[0]:
        print(f"{hit['entity']['title']}  (相似度: {hit['distance']:.4f})")
```

### 向量 + 标量混合搜索

```python
query = "算法相关的书"
query_embedding = model.encode([query])

results = client.search(
    collection_name="book_search",
    data=[query_embedding[0].tolist()],
    filter='category == "计算机科学"',  # 先标量过滤，再向量匹配
    limit=3,
    output_fields=["title", "category", "description"],
)
```

### 要点

- `search()` 的 `data` 参数是列表，`[向量]` 表示一个查询
- `results` 是嵌套列表：外层每条查询，内层每个结果；单个查询时 `results[0]` 才是结果列表
- `hit['distance']` 是 COSINE 相似度，越接近 1 越相关
- `filter` 参数实现**混合搜索**：先在标量维度缩小候选集，再在候选集内做向量相似度匹配，两步都在 Milvus 内部完成
- `search()` 走向量索引做语义匹配，`query()` 走标量过滤做精确查找，两者互补

---

## 4. 标量过滤查询（scalar_search_demo.py）

**核心流程：** 连接 → 按分类字段做各种过滤查询

### 关键代码

```python
# 精确匹配
client.query(collection_name="book_search",
             filter='category == "计算机科学"',
             output_fields=["id", "title", "category"])

# 模糊匹配
client.query(collection_name="book_search",
             filter='category like "%工程%"',
             output_fields=["id", "title", "category"])

# 组合过滤
client.query(collection_name="book_search",
             filter='category == "人工智能" and title like "%学习%"',
             output_fields=["id", "title", "category"])

# IN 查询
client.query(collection_name="book_search",
             filter='category in ["计算机科学", "编程语言"]',
             output_fields=["id", "title", "category"])

# 按主键列表查（比手写 filter 更简洁）
client.query(collection_name="book_search",
             ids=[1, 2, 3],
             output_fields=["id", "title"])
```

### 要点

- `query()` 是标量过滤，不走向量索引，适合精确查找
- 字符串值在 filter 中用双引号：`'title == "算法导论"'`
- 通配符 `%`：`like "%Python%"` 匹配包含 Python 的任意字符串
- 支持 `and` / `or` 组合多个条件
- `ids` 参数比手写 `filter` 更简洁，按主键批量查
- `query()` 和 `search()` 互补：精确查找走 query，语义搜索走 search

---

## 关键概念

### connections 单例机制

`connections.connect()` 不是线程变量，而是模块级单例：

```python
# pymilvus/orm/connections.py
class Connections(metaclass=SingleInstanceMetaClass):
    def __init__(self):
        self._alias_handlers = {}  # alias → GrpcHandler

connections = Connections()  # 模块级单例，import 时创建
```

`connect()` 把 gRPC 连接存入 `_alias_handlers` 字典，后续 `Collection`、`utility` 等所有类都引用同一个单例，通过 alias（默认 `"default"`）取出 gRPC handler 通信。

### DataType 枚举值

`describe_collection()` 返回的字段类型是数字，对应 protobuf 枚举：

| 数字 | 类型 |
|------|------|
| 5 | INT64 |
| 21 | VARCHAR |
| 101 | FLOAT_VECTOR |

用 `DataType(f['type']).name` 可转为可读名称。

### embedding 模型选型

常用模型比较：

| 模型 | 维度 | 大小 | 中文效果 |
|------|------|------|----------|
| `all-MiniLM-L6-v2` | 384 | 80MB | 一般 |
| `intfloat/multilingual-e5-large` | 1024 | 2.2GB | 多语言，中英都好 |
| `BAAI/bge-m3` | 1024 | 2.2GB | 多语言，中英都强 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | 纯中文优秀 |

模型查找渠道：HuggingFace Models（`library:sentence-transformers`）、MTEB Leaderboard（性能榜单）、ModelScope（国内下载快）。