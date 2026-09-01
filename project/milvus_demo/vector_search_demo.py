"""Milvus 向量数据库示例 — 语义搜索查询。"""

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from config import MILVUS_URI, COLLECTION_NAME, VECTOR_FIELD, MODEL_NAME

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus + 加载 embedding 模型
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

model = SentenceTransformer(MODEL_NAME)
print("✅ embedding 模型已加载")

# ──────────────────────────────────────────────────────────────────────
# 2. 输入查询
# ──────────────────────────────────────────────────────────────────────
queries = [
    "如何设计可复用的代码？",
    "有哪些排序算法？",
    "怎样用 Python 写网站？",
]

# 将查询文本转为向量（批量编码）
query_embeddings = model.encode(queries)  # shape: (3, 1024)

# ──────────────────────────────────────────────────────────────────────
# 3. 逐条查询 — 每次只查一个向量，适合交互式场景
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("📌 逐条查询（纯向量搜索）")
print("=" * 60)

for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding.tolist()],  # 单个向量
        filter='category == "计算机科学"',  # 只在计算机科学分类中搜索
        limit=3,
        output_fields=["title", "category", "description"],
    )

    print(f"\n🔍 [{i+1}] {query}+ filter: category == \"计算机科学\"")
    for j, hit in enumerate(results[0]):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  (相似度: {hit['distance']:.4f})")

# ──────────────────────────────────────────────────────────────────────
# 4. 批量查询 — 一次传多个向量，比逐条更高效，适合批量处理
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("📌 批量查询（向量 + 标量混合搜索）")
print("=" * 60)

results = client.search(
    collection_name=COLLECTION_NAME,
    data=[emb.tolist() for emb in query_embeddings],  # 多个向量，一次请求
    filter='category == "计算机科学"',  # 只在计算机科学分类中搜索
    limit=3,
    output_fields=["title", "category", "description"],
)

# results 是嵌套列表：results[i] 对应 queries[i] 的结果
for i, (query, hits) in enumerate(zip(queries, results)):
    print(f"\n🔍 [{i+1}] {query} + filter: category == \"计算机科学\"")
    for j, hit in enumerate(hits):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  (相似度: {hit['distance']:.4f})")