"""Milvus 向量数据库示例 — 排序查询。"""

from pymilvus import MilvusClient
from config import MILVUS_URI, COLLECTION_NAME

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

# ──────────────────────────────────────────────────────────────────────
# 2. 按 id 升序
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("📌 按 id 升序")
print("=" * 60)

result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    limit=10,
    order_by="id:asc",
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 3. 按 id 降序
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📌 按 id 降序")
print("=" * 60)

result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    limit=10,
    order_by="id:desc",
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 4. 按字符串字段排序
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📌 按 category 升序")
print("=" * 60)

result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    limit=10,
    order_by="category:asc",
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")

# ──────────────────────────────────────────────────────────────────────
# 5. 多字段排序 — 先按 category 升序，再按 id 降序
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📌 多字段排序 — category 升序 + id 降序")
print("=" * 60)

result = client.query(
    collection_name=COLLECTION_NAME,
    filter="id >= 0",
    output_fields=["id", "title", "category"],
    limit=10,
    order_by=["category:asc", "id:desc"],
)
for row in result:
    print(f"   [{row['id']}] {row['category']} / {row['title']}")