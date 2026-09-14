"""Milvus 向量数据库示例 — 图文融合向量查询（纯文本搜索 + 图文融合搜索）。"""

import os

from PIL import Image
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from config import MILVUS_URI, COLLECTION_NAME, MODEL_NAME

# 项目根目录：脚本在 multimodal/ 子目录下，往上取一级
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
        limit=5,
        output_fields=["id", "title", "category", "description"],
    )
    print(f"\n📊 搜索结果（{label}）:")
    for hit in search_results[0]:
        entity = hit["entity"]
        print(f"   [{entity['id']}] {entity['title']} ({entity['category']})"
              f"  (相似度: {hit['distance']:.4f})")


# ──────────────────────────────────────────────────────────────────────
# 2. 纯文本搜索多模态向量
# ──────────────────────────────────────────────────────────────────────
query_text = "一只橘猫趴在窗台上，懒洋洋地吃着碗里的猫粮"
query_text_emb = model.encode(query_text)
search_by_vector(query_text_emb, f"文本搜多模态, 查询: '{query_text}'")

# ──────────────────────────────────────────────────────────────────────
# 3. 图文融合查询搜索多模态向量
# ──────────────────────────────────────────────────────────────────────
query_text = "猫和狗在草地上追逐打闹，一起嬉戏玩耍"
query_name = "resources/images/猫狗一起玩2.jpeg"
query_image = Image.open(os.path.join(PROJECT_ROOT, query_name)).convert("RGB")
multimodal_query = {"text": query_text, "image": query_image}
multimodal_query_emb = model.encode(multimodal_query)

search_by_vector(
    multimodal_query_emb,
    f"图文融合搜索, 查询: 图 '{query_name}' + '{query_text}'",
)
