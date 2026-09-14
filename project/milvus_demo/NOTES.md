# Milvus 向量数据库 Demo 笔记

项目地址：`project/milvus_demo`

---

## 项目结构

```
milvus_demo/
├── config.py                  # 公共常量
├── requirements.txt           # 依赖
├── create_collection_demo.py  # 1. 创建集合（稠密+稀疏双向量）
├── insert_demo.py             # 2. 插入数据
├── vector_search_demo.py      # 3. 搜索：稠密语义 / 稀疏 BM25 / 混合
├── scalar_search_demo.py      # 4. 标量过滤查询
├── sort_demo.py               # 5. 排序查询
└── aggregate_demo.py          # 6. 聚合统计
```

### 依赖（requirements.txt）

```
pymilvus>=2.5.0                  # Milvus 官方 Python SDK
sentence-transformers>=3.0.0      # 文本 embedding 模型
python-dotenv>=1.0.0              # 环境变量管理
```

### 公共常量（config.py）

```python
MILVUS_URI = "http://localhost:19530"           # Milvus 连接地址
COLLECTION_NAME = "book_search"                 # 集合名称
VECTOR_FIELD = "embedding"                      # 稠密向量字段
SPARSE_VECTOR_FIELD = "sparse_embedding"        # 稀疏向量字段（BM25 自动生成）
TEXT_FIELD = "description"                      # 文本字段（BM25 Function 输入）
MODEL_NAME = "intfloat/multilingual-e5-large"   # embedding 模型
VECTOR_DIM = 1024                               # 向量维度
```

---

## 1. 创建集合（create_collection_demo.py）

**核心流程：** 连接 → 定义 Schema（稠密+稀疏双向量 + BM25 Function）→ 创建集合（含索引+加载）

### 关键代码

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType

client = MilvusClient(uri="http://localhost:19530")

# 定义 Schema — 手动添加每个字段，精确控制类型
schema = client.create_schema(
    auto_id=True,               # 主键自动生成
    enable_dynamic_field=True,  # 允许插入未预定义的字段
)

schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)
schema.add_field("category", DataType.VARCHAR, max_length=64)
schema.add_field("price", DataType.DOUBLE, nullable=True)

# 文本字段 — 作为 BM25 Function 的输入，必须开启 analyzer；中文用 jieba 分词
schema.add_field("description", DataType.VARCHAR, max_length=2048,
                 enable_analyzer=True,
                 analyzer_params={"tokenizer": "jieba"})

# 稠密向量 — 语义搜索用，客户端模型编码后插入
schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1024)
# 稀疏向量 — BM25 Function 输出，服务端自动生成，无需 dim
schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)

# 注册 BM25 Function：服务端自动完成「文本 → 稀疏向量」的转换
schema.add_function(Function(
    name="text_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["description"],
    output_field_names=["sparse_embedding"],
))

# 定义向量索引 — 稠密和稀疏分别建
index_params = client.prepare_index_params()
index_params.add_index(field_name="embedding",
                       index_type="AUTOINDEX", metric_type="COSINE")
# 稀疏索引 metric_type 必须是 BM25（配合 BM25 Function）
index_params.add_index(field_name="sparse_embedding",
                       index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

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
- BM25 Function 三要素：输入字段开启 `enable_analyzer`、注册 Function、稀疏索引 `metric_type="BM25"`
- 中文必须配置分词器（如 `{"tokenizer": "jieba"}`），否则整句被当成一个 token 无法匹配

---

## 2. 插入数据（insert_demo.py）

**核心流程：** 连接 + 加载模型 → 准备数据 → 模型编码 → 插入 → 验证

### 关键代码

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

# 准备原始数据
books = [
    {"title": "算法导论", "category": "计算机科学", "price": 128.00,
     "description": "算法领域的经典教材，系统讲解排序、图算法..."},
    {"title": "Python 编程：从入门到实践", "category": "编程语言", "price": 89.00,
     "description": "面向初学者的 Python 教程..."},
    # ...
]

# 批量编码 — 把所有 description 转成向量（比逐条快很多）
# multilingual-e5 模型要求文本加 passage: 前缀
descriptions = [b["description"] for b in books]
embeddings = model.encode([f"passage: {d}" for d in descriptions])

# 组装数据 — 将 ndarray 每行转为 list
# 稀疏向量不用管 — BM25 Function 在服务端从 description 自动生成
data = [
    {**b, "embedding": emb.tolist()} for b, emb in zip(books, embeddings)
]

# 插入
result = client.insert(collection_name="book_search", data=data)
# → {"insert_count": 5, "ids": [468779015314108449, ...]}

client.flush("book_search")  # 刷盘后 query 立即可见
```

### 要点

- `model.encode()` 接受列表返回 ndarray，即使单条也要包成 `["文本"]`
- 批量编码比逐条调用快很多
- 向量需要用 `.tolist()` 转为 Python 列表再传给 Milvus
- `insert()` 返回生成的 `insert_count` 和主键 `ids` 列表
- `milvus-lite` 已从 `requirements.txt` 移除（Docker 运行不需要）
- multilingual-e5 模型要求文本加前缀：插入用 `passage:`，查询用 `query:`
- 插入数据只传文本和稠密向量，稀疏向量由服务端 BM25 Function 自动生成
- Function 输出字段（sparse_embedding）是服务端内部数据，不允许通过 query() 直接读取
- `flush()` 让数据从增长段落盘，之后 `query()` 才能查到（insert 后立即可 search，但 query 有延迟）

---

## 3. 搜索查询（vector_search_demo.py）— 稠密 / 稀疏 / 混合

**核心流程：** 连接 + 加载模型 → 输入查询 → 编码 → 三种搜索方式对比 → 批量查询

### 稠密向量搜索（语义）

用 embedding 模型编码查询，适合交互式场景：

```python
for query, embedding in zip(queries, query_embeddings):
    results = client.search(
        collection_name="book_search",
        data=[embedding.tolist()],         # 单个向量
        anns_field="embedding",            # 集合有多个向量字段，必须显式指定
        filter='category == "计算机科学"',  # 标量过滤
        limit=3,
        output_fields=["title", "category", "description"],
    )

    for hit in results[0]:  # results[0] 是当前查询的结果列表
        print(f"{hit['entity']['title']}  (相似度: {hit['distance']:.4f})")
```

### 稀疏向量搜索（BM25 关键词）

data 直接传查询文本字符串，服务端 BM25 Function 自动完成分词和权重计算：

```python
results = client.search(
    collection_name="book_search",
    data=[query],                    # 直接传文本！
    anns_field="sparse_embedding",
    limit=3,
    output_fields=["title", "category"],
)
# hit['distance'] 是 BM25 分数，越高越相关
```

### 混合搜索（稠密 + 稀疏，ranker 融合）

通过 AnnSearchRequest 分别定义每个向量字段的搜索请求，再用 ranker 融合：

```python
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker

dense_req = AnnSearchRequest(data=[dense_vec], anns_field="embedding",
                             param={"metric_type": "COSINE"}, limit=3)
sparse_req = AnnSearchRequest(data=[query], anns_field="sparse_embedding",
                              param={"metric_type": "BM25"}, limit=3)

# WeightedRanker — 加权求和，手动调权重（稠密 0.7 + 稀疏 0.3）
client.hybrid_search(collection_name="book_search", reqs=[dense_req, sparse_req],
                     ranker=WeightedRanker(0.7, 0.3), limit=3, ...)

# RRFRanker — 倒数排名融合，无需调参
client.hybrid_search(collection_name="book_search", reqs=[dense_req, sparse_req],
                     ranker=RRFRanker(k=60), limit=3, ...)
```

### 批量查询（向量 + 标量混合搜索）

一次传多个向量，单次网络请求完成所有查询，比逐条高效：

```python
# 所有查询向量一次传入
results = client.search(
    collection_name="book_search",
    data=[emb.tolist() for emb in query_embeddings],  # 多个向量
    anns_field="embedding",                           # 多向量字段必须显式指定
    filter='category == "计算机科学"',  # 先标量过滤，再向量匹配
    limit=3,
    output_fields=["title", "category", "description"],
)

# results[i] 对应 queries[i] 的结果
for i, (query, hits) in enumerate(zip(queries, results)):
    for hit in hits:  # 直接遍历，不用 results[0]
        print(f"{hit['entity']['title']}  (相似度: {hit['distance']:.4f})")
```

### 要点

- `search()` 的 `data` 参数是**列表**，单向量 `[vec]`，多向量 `[vec1, vec2, ...]`
- 集合有多个向量字段时必须显式传 `anns_field`，否则报 `multiple anns_fields exist`
- `results` 是嵌套列表：`results[i]` 对应第 i 条查询的结果
- `hit['distance']`：稠密搜索是 COSINE 相似度（越接近 1 越相关），稀疏搜索是 BM25 分数（越高越相关）
- `filter` 参数实现**混合搜索**：先在标量维度缩小候选集，再在候选集内做向量相似度匹配，两步都在 Milvus 内部完成
- 稠密搜索用模型编码查询（multilingual-e5 需 `query:` 前缀）；稀疏搜索直接传文本即可
- 混合搜索在 pymilvus 3.x 中使用 `hybrid_search()` + `AnnSearchRequest`（不是 `search()`）
- `WeightedRanker` 可调权重控制两种向量贡献比例；`RRFRanker` 自动融合排名
- 稠密权重高 → 偏语义理解；稀疏权重高 → 偏关键词精确匹配
- 逐条查询适合交互式（用户输入一条查一条），批量查询适合离线处理（一次查完所有）

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

## 5. 排序查询（sort_demo.py）

**核心流程：** 连接 → 按 id 升序/降序 → 按字符串字段排序 → 多字段排序

### 关键代码

```python
# 按 id 升序
client.query(
    collection_name="book_search",
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    order_by="id:asc",
    limit=10,
)

# 按 id 降序
client.query(
    collection_name="book_search",
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    order_by="id:desc",
    limit=10,
)

# 按字符串字段排序
client.query(
    collection_name="book_search",
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    order_by="category:asc",
    limit=10,
)

# 多字段排序 — 先按 category 升序，再按 id 降序
client.query(
    collection_name="book_search",
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    order_by=["category:asc", "id:desc"],
    limit=10,
)
```

### 要点

- `order_by` 格式为 `"字段名:asc"` 或 `"字段名:desc"`（冒号分隔）
- 多字段排序：`order_by=["category:asc", "id:desc"]`，按列表顺序依次应用
- 字符串字段按字典序排列

---

## 6. 聚合统计（aggregate_demo.py）

**核心流程：** 连接 → 按分类统计数量 → 数值聚合（sum/avg/min/max）

需要 Milvus 2.4+ 服务端支持。

### 关键代码

```python
# 按分类统计书籍数量
client.query(
    collection_name="book_search",
    filter="",
    output_fields=["count(*)", "category"],
    group_by_fields=["category"],
    limit=10,
)

# 按分类统计 price 的各类聚合值
client.query(
    collection_name="book_search",
    filter="",
    output_fields=[
        "count(*)", "sum(price)", "avg(price)",
        "min(price)", "max(price)", "category",
    ],
    group_by_fields=["category"],
    limit=10,
)
```

### 支持的聚合函数

| 函数 | 说明 | 适用字段类型 |
|------|------|------------|
| `count(*)` | 每组记录数 | — |
| `sum(字段)` | 求和 | 数值类型 |
| `avg(字段)` | 平均值 | 数值类型 |
| `min(字段)` | 最小值 | 数值类型 |
| `max(字段)` | 最大值 | 数值类型 |

### 要点

- `group_by_fields` 按指定字段分组，`output_fields` 中写聚合函数取别名
- 聚合结果中函数名即为 key，如 `row['sum(price)']`、`row['avg(price)']`
- `filter=""` 表示不筛选，全量统计
- 聚合函数只能作用于 Schema 中显式定义的字段，动态字段不支持
- 聚合功能需要 Milvus 2.4+ 服务端，旧版本会报错

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
| `Qwen/Qwen3-VL-Embedding-2B` | 2048 | ~4GB | 多模态语义理解，中英都好 |
| `intfloat/multilingual-e5-large` | 1024 | 2.2GB | 多语言，中英都好 |
| `BAAI/bge-m3` | 1024 | 2.2GB | 多语言，中英都强 |
| `BAAI/bge-large-zh-v1.5` | 1024 | 1.3GB | 纯中文优秀 |

模型查找渠道：HuggingFace Models（`library:sentence-transformers`）、MTEB Leaderboard（性能榜单）、ModelScope（国内下载快）。