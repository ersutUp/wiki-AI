"""Milvus 向量数据库示例 — 聚合统计。"""

from pymilvus import MilvusClient
from config import MILVUS_URI, COLLECTION_NAME

# ──────────────────────────────────────────────────────────────────────
# 1. 连接 Milvus
# ──────────────────────────────────────────────────────────────────────
client = MilvusClient(uri=MILVUS_URI)
print("✅ 已连接到 Milvus")

# ──────────────────────────────────────────────────────────────────────
# 2. 基础聚合 — 按分类统计数量
# ──────────────────────────────────────────────────────────────────────
# 注意：需要 Milvus 2.4+ 服务端支持 GROUP BY
print("=" * 60)
print("📌 count(*) — 按分类统计数量")
print("=" * 60)

try:
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["count(*)", "category"],
        group_by_fields=["category"],
        limit=10,
    )
    print("\n各分类书籍数量:")
    for row in result:
        print(f"   {row['category']}: {row['count(*)']} 本")
except Exception as e:
    print(f"\n⚠️  GROUP BY 执行失败（可能服务端版本过低）: {e}")

# ──────────────────────────────────────────────────────────────────────
# 3. 数值聚合 — 按分类统计 price 的总和、平均值、最大最小值
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📌 sum / avg / min / max — 按分类统计 price")
print("=" * 60)

try:
    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["count(*)", "sum(price)", "avg(price)", "min(price)", "max(price)", "category"],
        group_by_fields=["category"],
        limit=10,
    )
    print(f"\n{'分类':<12} {'数量':<6} {'总价':<10} {'均价':<10} {'最低':<10} {'最高':<10}")
    print("-" * 58)
    for row in result:
        print(
            f"   {row['category']:<10} "
            f"{row['count(*)']:<6} "
            f"{row['sum(price)']:<10.2f} "
            f"{row['avg(price)']:<10.2f} "
            f"{row['min(price)']:<10.2f} "
            f"{row['max(price)']:<10.2f}"
        )
except Exception as e:
    print(f"\n⚠️  数值聚合执行失败: {e}")