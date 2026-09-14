"""Milvus 向量数据库示例 — 图片向量化并插入（写入部分）。"""

import os

from PIL import Image
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from config import MILVUS_URI, COLLECTION_NAME, VECTOR_FIELD, MODEL_NAME

# 项目根目录：脚本在 image/ 子目录下，往上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus + 加载多模态 embedding 模型
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

model = SentenceTransformer(MODEL_NAME)
print("✅ embedding 模型已加载")

# ──────────────────────────────────────────────────────────────────────
# 2. 准备数据
# ──────────────────────────────────────────────────────────────────────
images = [
    {
        "path": "resources/images/两只狗.jpeg",
        "title": "两只狗",
        "category": "动物",
        "description": "两只狗在一起",
    },
    {
        "path": "resources/images/汽车.jpeg",
        "title": "汽车",
        "category": "交通工具",
        "description": "一辆汽车",
    },
    {
        "path": "resources/images/狗吃东西.jpeg",
        "title": "狗吃东西",
        "category": "动物",
        "description": "一只狗正在吃东西",
    },
    {
        "path": "resources/images/狗运动.jpeg",
        "title": "狗运动",
        "category": "动物",
        "description": "一只狗在奔跑运动",
    },
    {
        "path": "resources/images/猫吃东西.jpeg",
        "title": "猫吃东西",
        "category": "动物",
        "description": "一只猫正在吃东西",
    },
    {
        "path": "resources/images/猫狗一起玩.jpeg",
        "title": "猫狗一起玩",
        "category": "动物",
        "description": "猫和狗在一起玩耍",
    },
    {
        "path": "resources/images/车-马路.jpeg",
        "title": "车-马路",
        "category": "交通工具",
        "description": "一辆车行驶在马路上",
    },
    {
        "path": "resources/images/猫上树.jpeg",
        "title": "猫上树",
        "category": "动物",
        "description": "一只猫爬上树",
    },
]

# 批量加载图片 → 批量生成 embedding（比逐张编码快很多）
pil_images = [Image.open(os.path.join(PROJECT_ROOT, img["path"])).convert("RGB") for img in images]
embeddings = model.encode(pil_images)  # 返回 ndarray, shape: (8, 2048)

# 组装插入数据：将 ndarray 每行转为 list
data = [
    {**img, VECTOR_FIELD: emb.tolist()} for img, emb in zip(images, embeddings)
]

print(f"✅ {len(images)} 张图片向量化完毕")

# ──────────────────────────────────────────────────────────────────────
# 3. 插入数据
# ──────────────────────────────────────────────────────────────────────
result = client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"✅ 已插入 {result['insert_count']} 条图片向量")
print(f"   生成的 ID: {result['ids']}")
