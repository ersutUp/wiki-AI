# 稀疏向量（BM25）搜索笔记 — sparse_search_demo.py

**核心流程：** 连接 + 加载模型 → 输入查询 → 逐条对比稀疏/稠密 → 批量稀疏查询

---

## 稀疏向量搜索（BM25 关键词）

`data` 直接传查询**文本字符串**，服务端 BM25 Function 自动完成分词和权重计算：

```python
results = client.search(
    collection_name="book_search",
    data=[query],                    # 直接传文本！无需模型编码
    anns_field="sparse_embedding",
    limit=3,
    output_fields=["title", "category"],
)
# hit['distance'] 是 BM25 分数，越高越相关
```

## 稠密向量搜索（语义）对比

稠密搜索需要先用模型编码查询：

```python
query_embeddings = model.encode(queries)  # 批量编码

dense_results = client.search(
    collection_name="book_search",
    data=[embedding.tolist()],
    anns_field="embedding",
    filter='category == "计算机科学"',  # 标量过滤照常可用
    limit=3,
    output_fields=["title", "category"],
)
# hit['distance'] 是 COSINE 相似度
```

## 批量稀疏查询

一次传多条查询文本，单次网络请求完成所有查询：

```python
results = client.search(
    collection_name="book_search",
    data=queries,                     # 多条文本列表
    anns_field="sparse_embedding",
    limit=3,
    output_fields=["title", "category", "description"],
)

# results 是嵌套列表：results[i] 对应 queries[i] 的结果
for i, (query, hits) in enumerate(zip(queries, results)):
    for hit in hits:
        print(f"{hit['entity']['title']}  (BM25: {hit['distance']:.4f})")
```

---

## 要点

- 稀疏搜索 `data` 直接传文本，Milvus 服务端的 BM25 Function 在**查询时**自动对文本做分词和打分
- 稠密搜索需要客户端先模型编码；稀疏搜索零编码成本，查询快
- 两种搜索的 `hit['distance']` 含义不同：
  - 稠密 → COSINE 相似度（越接近 1 越相关）
  - 稀疏 → BM25 分数（无上界，越高越相关）
- BM25 是**词频-based**：查询词在文档中命中越多、越稀有，分数越高；完全没命中该词则无贡献
- 语义相关但用词不同时（如「写网站」vs「Web 开发」），BM25 抓不到、稠密能抓到
- 关键词精确匹配场景（型号、人名、专有名词），BM25 往往比稠密更准
- `anns_field` 必须显式指定（集合有稠密、稀疏两个向量字段）
