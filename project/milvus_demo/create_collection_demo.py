"""Milvus 向量数据库示例 — 使用 MilvusClient + 自定义 Schema 创建集合。"""

from pymilvus import MilvusClient, DataType
from config import MILVUS_URI, COLLECTION_NAME, VECTOR_FIELD, VECTOR_DIM

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
    description="书籍信息集合，用于语义搜索 demo",
)

# 添加标量字段和向量字段
schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field("title", DataType.VARCHAR, max_length=512)  # 书名
schema.add_field("category", DataType.VARCHAR, max_length=64)  # 分类
schema.add_field("description", DataType.VARCHAR, max_length=2048)  # 内容摘要
schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

# ──────────────────────────────────────────────────────────────────────
# 3. 创建集合 + 索引 + 加载（一步完成）
# ──────────────────────────────────────────────────────────────────────
# 定义索引参数
index_params = client.prepare_index_params()
index_params.add_index(
    field_name=VECTOR_FIELD,
    index_type="AUTOINDEX",  # Milvus 自动选择最优索引类型
    metric_type="COSINE",  # 相似度度量：COSINE / IP / L2
)

# 传入 schema 时走精细创建路径，索引和加载自动完成
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)
print(f"✅ 集合已创建: {COLLECTION_NAME}")

# ──────────────────────────────────────────────────────────────────────
# 4. 查看集合信息
# ──────────────────────────────────────────────────────────────────────
info = client.describe_collection(COLLECTION_NAME)
print(f"\n📊 集合信息:")
print(f"   名称: {info['collection_name']}")
print(f"   ID: {info['collection_id']}")
print(f"   字段:")
for f in info["fields"]:
    print(f"     {f['name']}: {DataType(f['type']).name} {f.get('params', '')}")
print(f"   已有索引: {[idx for idx in client.list_indexes(COLLECTION_NAME)]}")

# ──────────────────────────────────────────────────────────────────────
# 5. 列出所有集合
# ──────────────────────────────────────────────────────────────────────
print(f"\n📋 所有集合: {client.list_collections()}")