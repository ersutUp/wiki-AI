"""
文档实体抽取与 Neo4j 入库模块

流程：
1. 将切分后的 Document 列表（或原始文本）通过 LLMGraphTransformer 抽取实体和关系
2. 标准化实体名（繁转简 / 全角转半角 / 去空白）
3. 用 MERGE 幂等写入 Neo4j
"""
import re
import unicodedata

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument

from config.logger import get_app_logger
from llm.llm_all import deepseek_v4_pro, claude_sonnet_4_6
from utils.env_utils import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)

_logger = get_app_logger("neo4j_graph")

# ---------- 技术文档专用实体类型 ----------
# 限制实体类型：不加白名单时大模型会造出大量噪声标签（如"步骤"、"示例"），
# 导致图谱碎片化，查询时难以按类型筛选。
ALLOWED_NODES = [
    "技术",       # 技术栈、框架、语言，如 ClickHouse、Kafka、Python
    "组件",       # 系统内部模块/组件，如 MergeTree、ReplicatedMergeTree
    "概念",       # 领域术语/抽象概念，如 分片、副本、OLAP
    "操作",       # 具体的操作/命令/SQL 语句，如 INSERT、OPTIMIZE
    "配置项",     # 配置参数，如 max_memory_usage、compression_codec
    "工具",       # 辅助工具，如 clickhouse-client、clickhouse-copier
    "公司",       # 相关公司/组织，如 Yandex、Apache
    "版本",       # 软件版本号，如 ClickHouse 22.8
]

# 限制关系三元组 (头实体类型, 关系, 尾实体类型)：
# 同时约束关系名和两端的实体类型搭配，避免出现 (配置项)-[:包含]->(技术) 这类语义错误的边。
# 不在此列表中的三元组组合，即使模型尝试生成也会被 LLMGraphTransformer 过滤掉。
ALLOWED_RELS = [
    ("技术",   "包含",     "组件"),    # ClickHouse 包含 MergeTree
    ("技术",   "依赖",     "技术"),    # ClickHouse 依赖 ZooKeeper
    ("技术",   "支持",     "概念"),    # ClickHouse 支持 列式存储
    ("技术",   "属于",     "公司"),    # ClickHouse 属于 Yandex
    ("技术",   "有版本",   "版本"),    # ClickHouse 有版本 22.8
    ("组件",   "实现",     "概念"),    # MergeTree 实现 列式存储
    ("组件",   "关联",     "组件"),    # 组件之间的依赖/关联
    ("操作",   "作用于",   "组件"),    # INSERT 作用于 MergeTree
    ("操作",   "作用于",   "技术"),    # OPTIMIZE 作用于 ClickHouse
    ("配置项", "属于",     "技术"),    # max_memory_usage 属于 ClickHouse
    ("配置项", "属于",     "组件"),    # 配置项 属于 组件
    ("工具",   "用于",     "技术"),    # clickhouse-client 用于 ClickHouse
    ("概念",   "相关",     "概念"),    # OLAP 相关 列式存储
]


def normalize_name(name: str) -> str:
    """实体名标准化：Unicode 兼容归一（全角→半角）、去所有空白。

    不做繁简转换——技术文档基本都是简体，繁简混用极少，
    引入 opencc 依赖会增加环境复杂度，性价比不高。
    """
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", "", name).strip()


def _normalize_graph_doc(graph_doc: GraphDocument) -> None:
    """标准化一个 GraphDocument 里所有实体名。

    LLMGraphTransformer 构建关系时，会为关系的 source/target 端点新建独立的 Node 对象
    （见 langchain_experimental 源码 map_to_base_relationship），而不是引用 nodes 列表
    中已有的节点。因此必须对 nodes 列表和关系端点分别做标准化，否则同一实体会以
    不同字符串形式出现，写入时 MERGE 会产生重复节点。
    """
    for node in graph_doc.nodes:
        node.id = normalize_name(node.id)
    for rel in graph_doc.relationships:
        rel.source.id = normalize_name(rel.source.id)
        rel.target.id = normalize_name(rel.target.id)


def extract_and_store(
    docs: list[Document],
    uri: str = NEO4J_URI,
    username: str = NEO4J_USERNAME,
    password: str = NEO4J_PASSWORD,
    database: str = NEO4J_DATABASE,
) -> None:
    """从文档列表中抽取实体/关系并写入 Neo4j。

    参数：
        docs: 待抽取的文档列表（每个 Document 的 page_content 为正文）
        uri / username / password / database: Neo4j 连接参数
    """
    if not docs:
        _logger.warning("文档列表为空，跳过 Neo4j 入库")
        return

    _logger.info("开始实体抽取，文档数: %d", len(docs))

    # LLMGraphTransformer 内部逐篇调用 LLM，不并发，文档数多时耗时较长
    transformer = LLMGraphTransformer(
        llm=claude_sonnet_4_6,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELS,
    )

    _logger.info("连接 Neo4j: %s，数据库: %s", uri, database)
    graph = Neo4jGraph(url=uri, username=username, password=password, database=database)

    # 唯一约束是 MERGE 幂等写入的前提：无约束时并发导入会产生重复节点，
    # 按 id 匹配也会退化为全标签扫描。IF NOT EXISTS 保证可重复执行。
    graph.query(
        "CREATE CONSTRAINT entity_id IF NOT EXISTS "
        "FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
    )

    for doc in docs:
        source = doc.metadata.get("source") or ""
        title = source + " > ".join(
            h for h in [doc.metadata.get("h1"), doc.metadata.get("h2"), doc.metadata.get("h3")] if h
        ) or "未知"
        _logger.info("开始抽取《%s》", title)
        graph_docs = transformer.convert_to_graph_documents([doc])
        graph_doc = graph_docs[0]
        _normalize_graph_doc(graph_doc)
        _logger.info("《%s》抽出 %d 个实体、%d 个关系", title, len(graph_doc.nodes), len(graph_doc.relationships))
        # baseEntityLabel=True：给所有实体节点加 __Entity__ 标签并建唯一索引，便于跨类型查询；
        # include_source=False：不把源文档存为 Document 节点，减少图谱噪声（溯源由 Milvus 侧负责）
        graph.add_graph_documents(graph_docs, include_source=False, baseEntityLabel=True)

    _logger.info("Neo4j 入库完成")

if __name__ == "__main__":
    from config.logger import configure_logging
    configure_logging()

    test_docs = [
        Document(
            page_content=(
                "ClickHouse 是由 Yandex 开发的列式数据库管理系统，专为 OLAP 场景设计。"
                "其核心存储引擎 MergeTree 支持数据分片和副本机制，可通过 ZooKeeper 协调多节点写入。"
                "常用工具 clickhouse-client 可执行 INSERT、OPTIMIZE 等操作，"
                "配置项 max_memory_usage 用于限制单次查询的内存上限。"
            ),
            metadata={"source": "test_clickhouse"},
        )
    ]

    extract_and_store(test_docs)
    print("测试完成，请在 Neo4j Browser 执行：MATCH (n) RETURN n LIMIT 50 查看结果")