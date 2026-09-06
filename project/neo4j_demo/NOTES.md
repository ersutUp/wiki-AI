# Neo4j 知识图谱 Demo 笔记

项目地址：`project/neo4j_demo`

---

## 项目结构

```
neo4j_demo/
├── docker-compose.yml           # Neo4j 容器编排
├── .env / .env.example          # 环境变量（连接配置、API Key）
├── env_utils.py                 # 配置加载（含 get_neo4j_config）
├── my_llm.py                    # LLM 实例（glm/claude/deepseek/gpt55）
├── load_wikipedia_datas.py      # 维基百科数据获取与 JSON 持久化
├── load_documents_to_neo4j.py   # 核心 Demo：LLM 抽图 + 写入 Neo4j
├── langchain_query_neo4j.py     # 知识图谱问答：GraphCypherQAChain（自然语言→Cypher→回答）
├── wikipedia_docs.json          # 本地缓存文档数据
└── requirements.txt             # 依赖
```

### 依赖（requirements.txt 核心）

```
neo4j==6.3.0                        # Neo4j 官方 Python 驱动
langchain==1.4.0                    # LLM 框架
langchain-experimental>=0.4.2       # LLMGraphTransformer
langchain-neo4j>=0.10.0             # Neo4jGraph 写入封装
opencc-python-reimplemented>=0.1.7  # 实体名繁简转换
```

### 环境变量（.env）

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=change-me
NEO4J_DATABASE=neo4j
```

---

## 整体流程

```
维基百科文档(JSON) → LLMGraphTransformer 抽图 → 实体名标准化 → Neo4jGraph 写入 → 验证查询
```

三步：
1. **大模型抽图**：`LLMGraphTransformer` 把每篇文档转成 `GraphDocument`（nodes=实体，relationships=关系）
2. **标准化**：繁转简/全角转半角/去空白，让"馬雲"和"马云"合并为同一实体
3. **写入**：`add_graph_documents` 用 MERGE 幂等写入，重复运行不产生重复数据

---

## 1. 启动 Neo4j（docker-compose.yml）

```bash
docker compose up -d
```

要点：
- `NEO4J_PLUGINS=["apoc"]`：容器**首次启动自动从 GitHub 下载 APOC Core**（免费开源版），jar 落在挂载的 `./neo4j/plugins/` 持久化，之后不再下载
- 密码通过 `${NEO4J_PASSWORD:?...}` 从 `.env` 注入，Compose 在**宿主机侧**替换；healthcheck 里的 `$$` 才是传给容器内 shell 的
- healthcheck 用无认证的 `wget -q http://localhost:7474`，避免把密码作为环境变量传进容器（`docker inspect` 可见）

浏览器访问：`http://localhost:7474`

---

## 2. LLM 抽图（LLMGraphTransformer）

```python
transformer = LLMGraphTransformer(
    llm=gpt55_llm,
    allowed_nodes=ALLOWED_NODES,              # 实体类型白名单
    allowed_relationships=ALLOWED_RELS,       # 关系三元组白名单
)
graph_docs = transformer.convert_to_graph_documents(docs)
```

### allowed_nodes：限制实体标签

只写字符串列表即可。不限制的话大模型会造出各种奇怪标签。中文标签完全支持（`人物/公司/地点`...），对中文语料更自然。

### allowed_relationships：建议用三元组

```python
# 只写字符串 → 只限制关系名，端点随便配
("FOUNDER_OF",)

# 三元组 → 同时锁定两端实体类型搭配
("人物", "创办", "公司")   # 马云 -> 阿里巴巴
```

三元组能过滤掉 `(公司)-[:创办]->(城市)` 这类错误搭配。**限制是硬性的**：不在白名单里的关系直接丢弃，不重写。

### convert_to_graph_documents 一次传全部 vs 循环传单个

源码内部就是 `[self.process_response(d) for d in documents]`——**逐篇调 LLM 的循环**，两种写法完全等价。一次传全部更简洁，缺点是等待期间没有逐篇进度。大批量可用异步版 `aconvert_to_graph_documents`（`asyncio.gather` 并发，快但注意限流）。

---

## 3. 实体名标准化

```python
def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)  # 全角→半角、兼容字符归一（㎡→m2）
    name = cc.convert(name)                      # 繁→简（OpenCC，"馬雲"→"马云"）
    return re.sub(r"\s+", "", name).strip()      # 去所有空白
```

- **NFKC vs 手写全角表**：NFKC 是标准库内置的 Unicode 兼容归一化，覆盖面远大于手写的 `str.maketrans` 全角表（还能处理 `Ｎｅｏ４ｊ→Neo4j`、`①→1`），首选 NFKC
- **节点和关系端点要分别标准化**：`LLMGraphTransformer` 构建关系时会为端点**新建 Node 对象**（源码 `map_to_base_relationship`），并不引用 `graph_doc.nodes` 里的节点——只改 nodes 的 id，关系端点还是旧名字，写入会 MERGE 出重复实体：

```python
for node in graph_doc.nodes:
    node.id = normalize_name(node.id)
for rel in graph_doc.relationships:      # 容易漏掉！
    rel.source.id = normalize_name(rel.source.id)
    rel.target.id = normalize_name(rel.target.id)
```

- 跨文档同名实体交给写入时的 MERGE 自动合并

---

## 4. 写入 Neo4j（Neo4jGraph）

```python
graph = Neo4jGraph(url=..., username=..., password=..., database=...)
graph.add_graph_documents(graph_docs, include_source=False, baseEntityLabel=True)
```

### 两个关键参数

| 参数 | 作用 |
|---|---|
| `baseEntityLabel=True` | 给每个实体加 `__Entity__` 标签并自动建唯一索引，提升导入和查询性能 |
| `include_source=True` | 把源文档存为 `Document` 节点，用 `MENTIONS` 关系连到实体（可溯源"实体出自哪篇文章"） |

### Document 节点的 id 规则（langchain_neo4j 源码行为）

```python
MERGE (d:Document {id: $document.metadata.id})
# metadata.id 不存在时回退为 md5(page_content)
```

- **metadata.id 缺失 → 用正文 MD5**：正文一变节点就"分身"（新旧并存）
- **手动设置稳定 id**：对来源 URL 取 MD5，正文更新时 id 不变，始终 MERGE 到同一节点：

```python
stable_key = metadata.get("source") or metadata.get("title") or ""
metadata["id"] = md5(stable_key.encode("utf-8")).hexdigest()
```

注意：`include_source=False` 时 Document 节点不写库，此 id 无消费方。

---

## 5. 索引与唯一约束

### 为什么要建：`__Entity__.id` 唯一约束

写入查询的核心是 `MERGE (n:__Entity__ {id: row.id})`，唯一约束是它可靠工作的前提：

1. **正确性**：MERGE 的"查找→不存在→创建"不是原子的。单线程串行导入暂时没事，但**并发/多进程导入时**两个事务可能同时发现 id 不存在、同时创建 → 重复节点。唯一约束让后一个事务直接报错回滚，挡住竞态
2. **性能**：无索引时按 id 的 MERGE/MATCH 退化为**标签全扫描**，导入复杂度 O(N²)——233 个节点无感，几万节点后肉眼可见变慢

### 创建方式（两层保障）

```python
# 显式创建（推荐，写在 add_graph_documents 之前）：
# IF NOT EXISTS 幂等，直接对库执行，且具名好辨认
graph.query(
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
)

# 库的自动创建：add_graph_documents(baseEntityLabel=True) 时
# 检查缓存 schema，没有就 CREATE CONSTRAINT IF NOT EXISTS
```

自动创建有个坑：它检查的是 `Neo4jGraph` **初始化时缓存的 schema 快照**（`self.structured_schema`），不是实时查库。连接建立后约束被删了，缓存还显示"存在"，就会跳过创建。所以显式语句是主、自动创建是兜底。

### 唯一约束 ≠ 手动索引

唯一约束自动带一个 **backing index**（背衬索引），删约束时索引一起消失。`owningConstraint` 非空的索引不要单独删：

```cypher
-- 查
SHOW INDEXES       -- 所有索引（含约束背衬索引、系统自带 LOOKUP 索引）
SHOW CONSTRAINTS   -- 所有约束

-- 看清哪些是背衬索引
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, owningConstraint

-- 删：约束背衬的索引删约束即可（会连索引一起删）
DROP CONSTRAINT 约束名    -- 如 DROP CONSTRAINT document_id
DROP INDEX 索引名         -- 只对手动 CREATE INDEX 建的普通索引用
```

系统每个库自带的 LOOKUP 索引（`index_343aff4e` 之类）不可删也不用管。

---

## 6. 踩坑记录

### 实体名存在 id 属性，不是 name

`LLMGraphTransformer` 写入的实体节点，实体名放在 **`id` 属性**里。查询用 `a.name` 会返回 `None`：

```cypher
-- 错：a.name 全是 None
MATCH (a:Person) RETURN a.name

-- 对
MATCH (a:`人物`) RETURN a.id          -- 中文标签要加反引号
```

### 内容审核拦截

维基百科正文（含敏感人物/事件）会被国内 LLM 服务的审核拦截（GLM 报 code 1301、FUNCLOUD 报 data_inspection_failed）。换对内容更宽容的模型通道（如 FUNCLOUD 的 gpt55/claude）可绕过。

### 切换标签/模型后要清库

从英文标签切中文标签、或换抽取模型后，直接重跑会让**新旧两套标签并存**。重跑前清库：

```cypher
MATCH (n) DETACH DELETE n
```

### Compose 变量替换的两种语法

```yaml
environment:
  - NEO4J_AUTH=${NEO4J_PASSWORD}     # Compose 宿主机侧替换（读 .env）
healthcheck:
  test: ["CMD-SHELL", "echo $${VAR}"]  # $$ 转义 → 容器内 shell 运行时展开
```

列表格式（`- KEY=VALUE`）下外层单引号会原样传进容器，`NEO4J_PLUGINS='["apoc"]'` 会失败，应写 `NEO4J_PLUGINS=["apoc"]`。

---

## 7. 知识图谱问答（GraphCypherQAChain）

```python
chain = GraphCypherQAChain.from_llm(
    llm=glm_llm,
    graph=graph,
    cypher_prompt=CYPHER_PROMPT,   # 自定义 Cypher 生成提示词
    verbose=True,                  # 打印生成的 Cypher，调试必开
    allow_dangerous_requests=True, # 安全确认开关，必须显式 True
    return_intermediate_steps=True,# 返回中间步骤（生成的 Cypher、查询结果）
)
chain.invoke({"query": "阿里巴巴的创始人是谁？"})
```

### 链的执行流程

```
问题 ──→ ① cypher_prompt(问题+Schema) ──→ LLM 生成 Cypher
      ──→ ② 执行 Cypher 查图谱 ──→ ③ QA prompt(问题+查询结果) ──→ LLM 生成回答
```

`Neo4jGraph` 初始化时自动读库里的标签/关系类型/属性生成 Schema，注入到提示词里，LLM 据此"知道"图里有什么可以查。

### 中文标签必须自定义 cypher_prompt

默认提示词是英文的，模型看到中文语料的问题会**顺着生成英文标签**（`MATCH (p:Person)-[:FOUNDER_OF]->...`），而库里是 `人物/创办`，查出来永远是空。自定义提示词两个要点：

1. 明确要求"标签和关系类型与 Schema 完全一致，不要翻译成英文"
2. 给一个实体匹配示例：实体名在 `id` 属性上（不是 name，见踩坑记录），`MATCH (p:人物 {id: "马云"})`

模板必须包含 `{schema}` 和 `{question}` 两个占位符（链内部填入），写其他占位符运行时报错。`PromptTemplate` 里字面量花括号要写成 `{{id: "马云"}}` 转义。

### allow_dangerous_requests：不是权限开关，是确认开关

设 `False` 和不设一样直接 `ValueError` 拒绝运行——它表达的是"我知晓此链会执行 LLM 生成的任意 Cypher"，只有 `True` 一个合法值。真正的防护在生产环境做：给数据库账号收窄权限（只读）或校验生成的 Cypher。

### 验证时先确认图谱非空

链本身正常但查询全空时，先查库排查（本次就是库里 0 节点——`load_documents_to_neo4j.py` 没在此库上跑过）：

```python
g.query("MATCH (n) RETURN labels(n)[0], count(n)")
```

---

## 8. 运行

```bash
# 1. 启动数据库
docker compose up -d

# 2. 构建图谱（首次无 JSON 会自动联网抓取维基百科，需代理）
python load_documents_to_neo4j.py

# 3. 图谱问答
python langchain_query_neo4j.py

# 4. 浏览器查看图谱
open http://localhost:7474
# 示例查询：MATCH (a:`人物`)-[r]->(b) RETURN a,r,b LIMIT 50
```
