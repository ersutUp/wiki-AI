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
# 3. 执行语义搜索
# ──────────────────────────────────────────────────────────────────────
for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[embedding.tolist()],  # 查询向量，包在列表里
        filter='category == "计算机科学"',  # 只在计算机科学分类中搜索
        limit=3,  # 返回 top-3
        output_fields=["title", "category", "description"],  # 返回的标量字段
    )

    print(f"\n{'=' * 60}")
    print(f"🔍 混合搜索: \"{query}\" + filter: category == \"计算机科学\"")
    print(f"{'=' * 60}")
    for j, hit in enumerate(results[0]):
        print(f"  [{j + 1}] {hit['entity']['category']} / {hit['entity']['title']}  (相似度: {hit['distance']:.4f})")
        print(f"      {hit['entity']['description']}")