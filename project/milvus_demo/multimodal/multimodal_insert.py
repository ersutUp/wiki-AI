"""Milvus 向量数据库示例 — 图文融合向量化插入（一张图 + 一段描述 → 一个融合向量）。"""

import os

from PIL import Image
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from config import MILVUS_URI, COLLECTION_NAME, VECTOR_FIELD, MODEL_NAME

# 项目根目录：脚本在 multimodal/ 子目录下，往上取一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus + 加载多模态 embedding 模型
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

model = SentenceTransformer(MODEL_NAME)
print("✅ embedding 模型已加载")

# ──────────────────────────────────────────────────────────────────────
# 2. 准备图片数据（description 暂时为空，后续由大模型自动生成）
# ──────────────────────────────────────────────────────────────────────
items = [
    {
        "path": "resources/images/狗吃东西.jpeg",
        "title": "狗吃东西",
        "category": "动物",
        "description": "",
    },
    {
        "path": "resources/images/狗运动.jpeg",
        "title": "狗运动",
        "category": "动物",
        "description": "",
    },
    {
        "path": "resources/images/猫吃东西.jpeg",
        "title": "猫吃东西",
        "category": "动物",
        "description": "",
    },
    {
        "path": "resources/images/猫狗一起玩.jpeg",
        "title": "猫狗一起玩",
        "category": "动物",
        "description": "",
    },
    {
        "path": "resources/images/车-马路.jpeg",
        "title": "车-马路",
        "category": "交通工具",
        "description": "",
    },
    {
        "path": "resources/images/猫上树.jpeg",
        "title": "猫上树",
        "category": "动物",
        "description": "",
    },
]

# ──────────────────────────────────────────────────────────────────────
# 3. 大模型描述图片，填充 description 字段
# ──────────────────────────────────────────────────────────────────────
from llm.my_llm import qwen3_vl_llm
from langchain_core.messages import HumanMessage
import base64

for item in items:
    image_path = os.path.join(PROJECT_ROOT, item["path"])
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "请用一句话描述这张图片的内容，不超过 30 个字。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    )
    item["description"] = qwen3_vl_llm.invoke([message]).content
    print(f"   {item['title']}: {item['description']}")

# ──────────────────────────────────────────────────────────────────────
# 4. 图文融合编码
# ──────────────────────────────────────────────────────────────────────
# 组装 MultimodalInput：{"text": 描述, "image": PIL图片} → 模型将图文语义融合为一个向量
multimodal_inputs = [
    {"text": item["description"], "image": Image.open(os.path.join(PROJECT_ROOT, item.pop("path"))).convert("RGB")}
    for item in items
]
embeddings = model.encode(multimodal_inputs)  # 返回 ndarray, shape: (6, 2048)

# 组装插入数据：将 ndarray 每行转为 list
data = [
    {**item, VECTOR_FIELD: emb.tolist()} for item, emb in zip(items, embeddings)
]

print(f"✅ {len(data)} 条图文融合向量已就绪")

# ──────────────────────────────────────────────────────────────────────
# 5. 插入数据
# ──────────────────────────────────────────────────────────────────────
result = client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"✅ 已插入 {result['insert_count']} 条多模态融合向量")
print(f"   生成的 ID: {result['ids']}")
