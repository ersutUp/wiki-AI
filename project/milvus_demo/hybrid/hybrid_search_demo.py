"""Milvus 向量数据库示例 — 混合搜索：稠密 + 稀疏联合检索，通过 ranker 融合。

混合搜索通过 AnnSearchRequest 分别定义每个向量字段的搜索请求：
  - WeightedRanker：加权求和，可调权重控制两种向量的贡献比例
  - RRFRanker：倒数排名融合，无需调权，适合不确定权重的场景
"""

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, RRFRanker
from sentence_transformers import SentenceTransformer
from config import (
    MILVUS_URI,
    COLLECTION_NAME,
    VECTOR_FIELD,
    SPARSE_VECTOR_FIELD,
    MODEL_NAME,
)

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

# 将查询文本转为稠密向量
query_embeddings = model.encode(queries)

# ──────────────────────────────────────────────────────────────────────
# 3. 逐条混合查询 — WeightedRanker / RRFRanker 两种融合方式对比
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)

for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
    dense_vec = embedding.tolist()
    print(f"\n🔍 [{i+1}] {query}")
    print(f"   {'─' * 50}")

    # 为稠密和稀疏分别创建 AnnSearchRequest：
    #   稠密 → 传编码后的向量
    #   稀疏 → 直接传查询文本（BM25 Function 自动处理）
    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "COSINE"},
        limit=3,
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field=SPARSE_VECTOR_FIELD,
        param={"metric_type": "BM25"},
        limit=3,
    )

    # ── 3a. WeightedRanker 加权融合：稠密权重 0.7，稀疏权重 0.3 ──
    weighted_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.7, 0.3),
        limit=3,
        output_fields=["title", "category"],
    )
    print(f"   📌 混合搜索 Weighted（稠密×0.7 + 稀疏×0.3）:")
    for j, hit in enumerate(weighted_results[0]):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  "
              f"(融合分: {hit['distance']:.4f})")

    # ── 3b. RRFRanker，无需调权重 ──
    rrf_results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=3,
        output_fields=["title", "category"],
    )
    print(f"   📌 混合搜索 RRF（倒数排名融合）:")
    for j, hit in enumerate(rrf_results[0]):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  "
              f"(融合分: {hit['distance']:.4f})")

# ──────────────────────────────────────────────────────────────────────
# 4. 小结
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("\n💡 小结:")
print("   - 混合搜索通过 AnnSearchRequest + hybrid_search() 融合两种向量")
print("     • WeightedRanker(0.7, 0.3) — 手动调权重，稠密为主、稀疏辅助")
print("     • RRFRanker(k=60) — 自动融合排名，无需调参")
print("   - 稠密权重高 → 偏语义理解；稀疏权重高 → 偏关键词精确匹配")
