"""创建 graphRAG OCR 文档向量集合。

字段对应 VectorFild TypedDict：
  - category / file_path / pageNo / source / doc_type / chapter：标量元数据
  - content：原始文本，同时作为 BM25 稀疏向量的输入
  - embeddings：稠密向量（Qwen3-VL-Embedding-2B，dim=2048）
  - sparse_embeddings：BM25 Function 自动生成的稀疏向量
"""

from pymilvus import MilvusClient, DataType, Function, FunctionType

from schema import OcrDocument

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "ocr_documents"
F = OcrDocument.F
VECTOR_DIM = 2048  # Qwen3-VL-Embedding-2B 输出维度

# ──────────────────────────────────────────────────────────────────────
# 1. 连接
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

# ──────────────────────────────────────────────────────────────────────
# 2. Schema
# ──────────────────────────────────────────────────────────────────────
if client.has_collection(COLLECTION_NAME):
    client.drop_collection(COLLECTION_NAME)
    print(f"⚠️  已删除旧集合: {COLLECTION_NAME}")

schema = client.create_schema(
    auto_id=True,
    enable_dynamic_field=False,
    description="OCR 文档向量集合，支持语义搜索（稠密）和关键词搜索（BM25 稀疏）",
)

schema.add_field("id", DataType.INT64, is_primary=True)
schema.add_field(F.FILE_TYPE,  DataType.VARCHAR, max_length=256,   nullable=True)
schema.add_field(F.FILE_PATH, DataType.VARCHAR, max_length=1024,    nullable=True)
schema.add_field(F.PAGE_NO,   DataType.VARCHAR, max_length=64,    nullable=True)
schema.add_field(F.SOURCE,    DataType.VARCHAR, max_length=512,   nullable=True)
schema.add_field(F.DOC_TYPE,  DataType.VARCHAR, max_length=64)
schema.add_field(F.CHAPTER,   DataType.VARCHAR, max_length=512,   nullable=True)

# content 同时是 BM25 Function 的输入，必须开启 jieba 分词
schema.add_field(
    F.CONTENT,
    DataType.VARCHAR,
    max_length=65535,
    enable_analyzer=True,
    analyzer_params={"tokenizer": "jieba"},
)

# 稠密向量 — 客户端编码后插入
schema.add_field(F.EMBEDDINGS, DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

# 稀疏向量 — BM25 Function 自动生成，无需指定 dim
schema.add_field(F.SPARSE, DataType.SPARSE_FLOAT_VECTOR)

# ──────────────────────────────────────────────────────────────────────
# 3. BM25 Function
# ──────────────────────────────────────────────────────────────────────
schema.add_function(Function(
    name="content_bm25",
    function_type=FunctionType.BM25,
    input_field_names=[F.CONTENT],
    output_field_names=[F.SPARSE],
))

# ──────────────────────────────────────────────────────────────────────
# 4. 索引 + 创建集合
# ──────────────────────────────────────────────────────────────────────
index_params = client.prepare_index_params()

index_params.add_index(
    field_name=F.EMBEDDINGS,
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

index_params.add_index(
    field_name=F.SPARSE,
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)

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
print(f"   字段:")
for f in info["fields"]:
    type_name = DataType(f["type"]).name
    if type_name == "SPARSE_FLOAT_VECTOR":
        print(f"     {f['name']}: SPARSE_FLOAT_VECTOR（BM25 自动生成）")
    elif type_name == "FLOAT_VECTOR":
        print(f"     {f['name']}: FLOAT_VECTOR dim={f.get('params', {}).get('dim', '?')}")
    else:
        print(f"     {f['name']}: {type_name}")
print(f"   Functions: {[(fn['name'], fn['type']) for fn in info.get('functions', [])]}")
print(f"   索引: {client.list_indexes(COLLECTION_NAME)}")
print(f"\n📋 所有集合: {client.list_collections()}")
