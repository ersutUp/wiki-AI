"""Milvus 向量数据库示例 — 使用 MilvusClient + 自定义 Schema 创建集合。

集合同时包含稠密向量（语义搜索）和稀疏向量（BM25 关键词搜索）：
  - 稠密向量 embedding：客户端用 embedding 模型编码后插入
  - 稀疏向量 sparse_embedding：注册 BM25 Function，服务端从 description
    文本自动生成，插入时只传文本即可
"""

from pymilvus import MilvusClient, DataType, Function, FunctionType
from config import (
    MILVUS_URI,
    COLLECTION_NAME,
    VECTOR_FIELD,
    VECTOR_DIM,
    SPARSE_VECTOR_FIELD,
    TEXT_FIELD,
)

# ──────────────────────────────────────────────────────────────────────
# 1. 创建 Client（自动连接）
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

# ──────────────────────────────────────────────────────────────────────
# 2. 定义 Schema
# ──────────────────────────────────────────────────────────────────────
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"⚠️  已删除旧集合: {COLLECTION_NAME}")

# 创建空的 Schema 模板
schema = client.create_schema(
    auto_id=True,  # 主键自动生成
    enable_dynamic_field=True,  # 允许插入未预先定义的字段
    description="书籍信息集合，同时支持语义搜索（稠密向量）和关键词搜索（稀疏向量）",
)

# 添加标量字段和向量字段
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)  # 书名
schema.add_field("category", DataType.VARCHAR, max_length=64)  # 分类
schema.add_field("price", DataType.DOUBLE, nullable=True)  # 价格

# 文本字段 — 作为 BM25 Function 的输入，必须开启 analyzer（分词器）
# 中文用 jieba 分词，否则整句会被当成一个 token 无法匹配
schema.add_field(
    TEXT_FIELD,
    DataType.VARCHAR,
    max_length=2048,  # 内容摘要
    enable_analyzer=True,
    analyzer_params={"tokenizer": "jieba"},
)

# 稠密向量 — 语义级匹配，客户端模型编码后插入
schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

# 稀疏向量 — 作为 BM25 Function 的输出，服务端自动生成，无需指定 dim
schema.add_field(SPARSE_VECTOR_FIELD, DataType.SPARSE_FLOAT_VECTOR)

# ──────────────────────────────────────────────────────────────────────
# 3. 注册 BM25 Function — 服务端自动完成「文本 → 稀疏向量」的转换
# ──────────────────────────────────────────────────────────────────────
schema.add_function(Function(
    name="text_bm25",                          # 函数名（唯一即可）
    function_type=FunctionType.BM25,
    input_field_names=[TEXT_FIELD],            # 输入：原始文本字段
    output_field_names=[SPARSE_VECTOR_FIELD],  # 输出：稀疏向量字段
))

# ──────────────────────────────────────────────────────────────────────
# 4. 创建集合 + 索引 + 加载（一步完成）
# ──────────────────────────────────────────────────────────────────────
# 定义索引参数：稠密和稀疏向量分别建索引
index_params = client.prepare_index_params()

# 稠密向量索引
index_params.add_index(
    field_name=VECTOR_FIELD,
    index_type="AUTOINDEX",  # Milvus 自动选择最优索引类型
    metric_type="COSINE",  # 相似度度量：COSINE / IP / L2
)

# 稀疏向量索引 — 倒排索引；metric_type 必须是 BM25（配合 BM25 Function）
index_params.add_index(
    field_name=SPARSE_VECTOR_FIELD,
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)

# 传入 schema 时走精细创建路径，索引和加载自动完成
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)
print(f"✅ 集合已创建: {COLLECTION_NAME}")

# ──────────────────────────────────────────────────────────────────────
# 5. 查看集合信息
# ──────────────────────────────────────────────────────────────────────
info = client.describe_collection(COLLECTION_NAME)
print(f"\n📊 集合信息:")
print(f"   名称: {info['collection_name']}")
print(f"   ID: {info['collection_id']}")
print(f"   字段:")
for f in info["fields"]:
    type_name = DataType(f["type"]).name
    if type_name == "SPARSE_FLOAT_VECTOR":
        print(f"     {f['name']}: SPARSE_FLOAT_VECTOR（BM25 Function 自动生成）")
    elif type_name == "FLOAT_VECTOR":
        print(f"     {f['name']}: FLOAT_VECTOR dim={f.get('params', {}).get('dim', '?')}")
    else:
        print(f"     {f['name']}: {type_name}")
print(f"   Functions: {[(fn['name'], fn['type']) for fn in info.get('functions', [])]}")
print(f"   已有索引: {[idx for idx in client.list_indexes(COLLECTION_NAME)]}")

# ──────────────────────────────────────────────────────────────────────
# 6. 列出所有集合
# ──────────────────────────────────────────────────────────────────────
print(f"\n📋 所有集合: {client.list_collections()}")
