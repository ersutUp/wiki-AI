"""Milvus 向量数据库示例 — 稀疏向量（BM25）查询与稠密向量查询对比。

两种查询方式对比：
  1. 稀疏向量搜索 — 直接传查询文本，服务端 BM25 Function 自动编码（关键词匹配）
  2. 稠密向量搜索 — 用 embedding 模型编码查询，语义相似度匹配（理解意思）

对比要点：
  - 稀疏（BM25）按关键词精确匹配，命中词越多分越高
  - 稠密按语义匹配，能理解同义词/上下文（如「写网站」≈「Web 开发」）
"""

from pymilvus import MilvusClient
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

# 稠密查询需要模型编码
query_embeddings = model.encode(queries)

# ──────────────────────────────────────────────────────────────────────
# 3. 逐条查询 — 稀疏（BM25）vs 稠密（语义）对比
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)

for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
    dense_vec = embedding.tolist()
    print(f"\n🔍 [{i+1}] {query}")
    print(f"   {'─' * 50}")

    # ── 3a. 稀疏向量搜索（BM25 关键词）──
    # data 直接传文本字符串！服务端 BM25 Function 自动完成分词和权重计算
    sparse_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query],
        anns_field=SPARSE_VECTOR_FIELD,
        limit=3,
        output_fields=["title", "category"],
    )
    print(f"   📌 稀疏向量（BM25 关键词搜索，直接传文本）:")
    for j, hit in enumerate(sparse_results[0]):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  "
              f"(BM25: {hit['distance']:.4f})")

    # ── 3b. 稠密向量搜索（语义）──
    dense_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[dense_vec],
        anns_field=VECTOR_FIELD,
        filter='category == "计算机科学"',  # 标量过滤照常可用
        limit=3,
        output_fields=["title", "category"],
    )
    print(f"   📌 稠密向量（语义搜索）+ filter:")
    for j, hit in enumerate(dense_results[0]):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  "
              f"(相似度: {hit['distance']:.4f})")

# ──────────────────────────────────────────────────────────────────────
# 4. 批量稀疏查询 — 一次传多条查询文本
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("📌 批量稀疏查询（BM25，一次传多条文本）")
print("=" * 60)

results = client.search(
    collection_name=COLLECTION_NAME,
    data=queries,  # 多条文本，一次请求
    anns_field=SPARSE_VECTOR_FIELD,
    limit=3,
    output_fields=["title", "category", "description"],
)

# results 是嵌套列表：results[i] 对应 queries[i] 的结果
for i, (query, hits) in enumerate(zip(queries, results)):
    print(f"\n🔍 [{i+1}] {query}")
    for j, hit in enumerate(hits):
        print(f"        [结果{j+1}] {hit['entity']['category']} / {hit['entity']['title']}  (BM25: {hit['distance']:.4f})")

# ──────────────────────────────────────────────────────────────────────
# 5. 小结
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("\n💡 小结:")
print("   - 稀疏搜索 data 直接传文本，无需模型编码，服务端 BM25 自动处理")
print("   - 稠密搜索需要先模型编码，但能理解语义（同义词、换个说法也能匹配）")
print("   - 关键词完全命中时 BM25 分数高；语义相关但用词不同时稠密更有优势")
