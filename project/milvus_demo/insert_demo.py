"""Milvus 向量数据库示例 — 使用 embedding 模型将文本转为向量后插入。"""

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
# 2. 准备数据
# ──────────────────────────────────────────────────────────────────────
books = [
    {
        "title": "深入理解计算机系统",
        "category": "计算机科学",
        "price": 139.00,
        "description": "从程序员视角全面剖析计算机系统的实现细节，涵盖处理器架构、存储器层次、链接、异常控制流等核心主题。",
    },
    {
        "title": "算法导论",
        "category": "计算机科学",
        "price": 128.00,
        "description": "算法领域的经典教材，系统讲解排序、图算法、动态规划、数据结构等核心算法设计与分析方法。",
    },
    {
        "title": "设计模式：可复用面向对象软件的基础",
        "category": "软件工程",
        "price": 79.00,
        "description": "GoF 经典之作，介绍了 23 种经典设计模式，帮助开发者写出可复用、可维护的面向对象代。",
    },
    {
        "title": "Python 编程：从入门到实践",
        "category": "编程语言",
        "price": 89.00,
        "description": "面向初学者的 Python 教程，涵盖基础语法、函数、类、文件操作，以及 Django Web 应用开发实战。",
    },
    {
        "title": "机器学习实战",
        "category": "人工智能",
        "price": 99.00,
        "description": "基于 Python 和 scikit-learn 的机器学习入门书，涵盖分类、回归、聚类、降维等常用算法。",
    },
]

# 提取所有 description 文本，批量生成 embedding（比逐条编码快很多）
descriptions = [b["description"] for b in books]
embeddings = model.encode(descriptions)  # 返回 ndarray，shape: (5, 384)

# 组装插入数据：将 ndarray 每行转为 list
data = [
    {**b, VECTOR_FIELD: emb.tolist()} for b, emb in zip(books, embeddings)
]

print("✅ 数据已准备")

# ──────────────────────────────────────────────────────────────────────
# 3. 插入数据
# ──────────────────────────────────────────────────────────────────────
result = client.insert(collection_name=COLLECTION_NAME, data=data)
print(f"✅ 已插入 {result['insert_count']} 条数据")
print(f"   生成的 ID: {result['ids']}")

# ──────────────────────────────────────────────────────────────────────
# 4. 验证：查询插入的数据
# ──────────────────────────────────────────────────────────────────────
query_result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "title", "description"],
    limit=5,
)

print(f"\n📊 集合中共 {len(query_result)} 条数据:")
for row in query_result:
    print(f"   [{row['id']}] {row['title']}")
    print(f"       {row['description'][:50]}...")