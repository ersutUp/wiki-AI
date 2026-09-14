"""Milvus 向量数据库示例 — 图片查询（以图搜图 + 文本搜图片）。"""

import os

from PIL import Image
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from config import MILVUS_URI, COLLECTION_NAME, MODEL_NAME

# 项目根目录：脚本在 image/ 子目录下，往上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus + 加载多模态 embedding 模型
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

model = SentenceTransformer(MODEL_NAME)
print("✅ embedding 模型已加载")


def search_by_vector(query_emb, label):
    """用向量搜索集合并打印结果。"""
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_emb.tolist()],
        limit=3,
        output_fields=["id", "title", "description"],
    )
    print(f"\n📊 搜索结果（{label}）:")
    for hit in search_results[0]:
        entity = hit["entity"]
        print(f"   [{entity['id']}] {entity['title']}  (相似度: {hit['distance']:.4f})")


# ──────────────────────────────────────────────────────────────────────
# 2. 以图搜图
# ──────────────────────────────────────────────────────────────────────
for query_name in ["resources/images/猫狗一起玩2.jpeg", "resources/images/猫运动.jpeg"]:
    query_image = Image.open(os.path.join(PROJECT_ROOT, query_name)).convert("RGB")
    query_emb = model.encode(query_image)
    search_by_vector(query_emb, f"以图搜图, 查询图: {query_name}")

# ──────────────────────────────────────────────────────────────────────
# 3. 文本搜图片（跨模态搜索）
# ──────────────────────────────────────────────────────────────────────
query_text = "一只猫"
query_text_emb = model.encode(query_text)
search_by_vector(query_text_emb, f"文本搜图片, 查询: '{query_text}'")
