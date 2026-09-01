"""Milvus 向量数据库示例 — 标量过滤查询（不使用向量）。"""

from pymilvus import MilvusClient
from config import MILVUS_URI, COLLECTION_NAME

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

# ──────────────────────────────────────────────────────────────────────
# 2. 按分类精确匹配
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print('📌 category == "计算机科学"')
result = client.query(
    collection_name=COLLECTION_NAME,
    filter='category == "计算机科学"',
    output_fields=["id", "title", "category"],
    limit=10,
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 3. 按分类模糊匹配
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print('📌 category like "%工程%"')
result = client.query(
    collection_name=COLLECTION_NAME,
    filter='category like "%工程%"',
    output_fields=["id", "title", "category"],
    limit=10,
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 4. 分类 + 关键词组合过滤
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print('📌 category == "人工智能" and title like "%学习%"')
result = client.query(
    collection_name=COLLECTION_NAME,
    filter='category == "人工智能" and title like "%学习%"',
    output_fields=["id", "title", "category"],
    limit=10,
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 5. 多分类 IN 查询（列出所有分类，然后按多个分类过滤）
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print('📌 category in ["计算机科学", "编程语言"]')
result = client.query(
    collection_name=COLLECTION_NAME,
    filter='category in ["计算机科学", "编程语言"]',
    output_fields=["id", "title", "category"],
    limit=10,
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 6. 列出所有分类（去重）
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📌 所有分类:")
result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["category"],
    limit=100,
)
categories = set(row["category"] for row in result)
for c in categories:
    print(f"   - {c}")