# 混合搜索（Hybrid Search）笔记 — hybrid_search_demo.py

**核心流程：** 连接 + 加载模型 → 编码查询 → 组装 AnnSearchRequest → hybrid_search + ranker 融合

---

## 混合搜索三步走

### 1. 为每个向量字段创建 AnnSearchRequest

稠密传编码后的向量，稀疏直接传查询文本：

```python
from pymilvus import AnnSearchRequest, WeightedRanker, RRFRanker

dense_req = AnnSearchRequest(
    data=[dense_vec],              # 模型编码后的向量
    anns_field="embedding",
    param={"metric_type": "COSINE"},
    limit=3,                        # 每路先各自取 top 3
)
sparse_req = AnnSearchRequest(
    data=[query],                   # 直接传文本，BM25 Function 自动处理
    anns_field="sparse_embedding",
    param={"metric_type": "BM25"},
    limit=3,
)
```

### 2. 用 ranker 融合两路结果

```python
# 方式一：WeightedRanker 加权融合（稠密×0.7 + 稀疏×0.3）
results = client.hybrid_search(
    collection_name="book_search",
    reqs=[dense_req, sparse_req],
    ranker=WeightedRanker(0.7, 0.3),
    limit=3,
    output_fields=["title", "category"],
)

# 方式二：RRFRanker 倒数排名融合，无需调参
results = client.hybrid_search(
    collection_name="book_search",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),
    limit=3,
    output_fields=["title", "category"],
)
```

---

## WeightedRanker 计算原理

用**分数本身**计算。服务端先把各路分数归一化到 [0,1]，再加权求和：

```
最终分 = w₁ × 归一化(稠密分) + w₂ × 归一化(稀疏分)
```

- 归一化方式跟 metric 有关：COSINE 用 `(1+s)/2`、BM25 用 `arctan` 压缩（分数无上界）、L2 用 `1/(1+d)`
- `norm_score` 参数默认 `True`（开归一化）；传 `norm_score=False` 直接用原始分数加权，仅当各路分数同量纲时才安全
- 权重需要手动调：`WeightedRanker(0.7, 0.3)` = 稠密占 70%、稀疏占 30%

## RRFRanker 计算原理

不用分数，只用**排名**：

```
最终分 = 1/(k + r₁) + 1/(k + r₂)
```

- `r` 是该结果在每一路里的排名（从 1 开始）
- 例：某结果稠密排第 1、稀疏排第 5 → `1/(60+1) + 1/(60+5) = 0.0318`
- `k=60` 是平滑常数（论文经验值）：k 越小头部排名权重越大，k 越大排名差异被抹平
- 完全免疫两路分数量纲差异，无需调参

## 两者对比

| | WeightedRanker | RRFRanker |
|---|---|---|
| 输入 | 归一化后的分数 | 只看排名 |
| 调参 | 要调权重 | k=60 通用 |
| 敏感度 | 对量纲差异敏感（靠归一化解决） | 完全免疫量纲 |
| 适用 | 明确知道哪路更可靠 | 不确定、想省事、分数不可比 |

---

## 要点

- 混合搜索是单独的一个方法 `hybrid_search()`，普通搜索的 `search()` 干不了这事
- 权重是「按顺序配对」的：`reqs=[稠密, 稀疏]` 配 `WeightedRanker(0.7, 0.3)` 就是稠密拿 0.7；要是把 reqs 顺序换了权重没换，含义就反了，还不报错
- 每路的 `limit=3` 是「这一路先各自挑出 3 个候选」，最后外层的 `limit=3` 才是从融合后的结果里取几个
- 想要「懂意思」就把稠密权重调大；想要「字对字匹配」就把稀疏权重调大
- 混合搜索最有用的是：一个结果既懂意思又带关键词，两边都说它好，那它大概率真的相关
