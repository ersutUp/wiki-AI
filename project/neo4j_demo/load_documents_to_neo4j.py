"""用大模型从维基百科文档抽取实体和关系，构建知识图谱写入 Neo4j 的 Demo。"""
import re
import unicodedata
from hashlib import md5

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.graphs.graph_document import GraphDocument
from opencc import OpenCC

from env_utils import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from load_wikipedia_datas import load
from my_llm import claude_llm, gpt55_llm

# 限制抽取的实体类型，让图谱更干净（不限的话大模型会造出各种奇怪标签）
ALLOWED_NODES = [
    "人物",    # Person
    "公司",    # Company
    "组织",    # Organization
    "地点",    # Location
    "产品",    # Product
    "事件",    # Event
    "日期",    # Date
    "奖项",    # Award
]
# ALLOWED_NODES = ["Person", "Company", "Organization", "Location", "Product", "Event", "Date", "Award"]

# 限制抽取的关系：三元组 (头实体类型, 关系, 尾实体类型)，
# 同时约束关系名和两端的实体类型搭配，避免出现 (公司)-[:创始人]->(城市) 这类错误
ALLOWED_RELS = [
    ("人物", "创办", "公司"),        # FOUNDER_OF：马云 -> 阿里巴巴
    ("人物", "担任CEO", "公司"),     # CEO_OF：马化腾 -> 腾讯
    ("人物", "担任董事长", "公司"),  # CHAIRMAN_OF：董事局主席
    ("人物", "任职于", "公司"),      # EMPLOYEE_OF
    ("人物", "成员", "组织"),        # MEMBER_OF：人物 -> 组织
    ("人物", "毕业于", "组织"),      # EDUCATED_AT：雷军 -> 武汉大学
    ("人物", "出生于", "地点"),      # BORN_IN
    ("人物", "祖籍", "地点"),        # ANCESTRAL_ORIGIN：马云 -> 浙江嵊州
    ("人物", "出席", "事件"),        # ATTENDED：马云 -> 民营企业座谈会
    ("人物", "获得", "奖项"),        # AWARDED
    ("公司", "位于", "地点"),        # LOCATED_IN：阿里巴巴 -> 杭州
    ("公司", "隶属于", "公司"),      # SUBSIDIARY_OF：支付宝 -> 阿里巴巴
    ("公司", "投资", "公司"),        # INVESTOR_IN：投资方 -> 被投公司
    ("公司", "收购", "公司"),        # ACQUIRED：收购方 -> 被收购方
    ("公司", "生产", "产品"),        # PRODUCED：腾讯 -> QQ
]
# ALLOWED_RELS = [
#     ("Person", "FOUNDER_OF", "Company"),        # 创始人：马云 -> 阿里巴巴
#     ("Person", "CEO_OF", "Company"),            # 首席执行官：马化腾 -> 腾讯
#     ("Person", "CHAIRMAN_OF", "Company"),       # 董事长/董事局主席
#     ("Person", "EMPLOYEE_OF", "Company"),       # 任职于
#     ("Person", "MEMBER_OF", "Organization"),    # 是…成员：人物 -> 组织
#     ("Person", "EDUCATED_AT", "Organization"),  # 毕业于：雷军 -> 武汉大学
#     ("Person", "BORN_IN", "Location"),          # 出生于
#     ("Person", "ANCESTRAL_ORIGIN", "Location"), # 祖籍：马云 -> 浙江嵊州
#     ("Person", "ATTENDED", "Event"),            # 出席：马云 -> 民营企业座谈会
#     ("Person", "AWARDED", "Award"),             # 获得：获奖
#     ("Company", "LOCATED_IN", "Location"),      # 位于：阿里巴巴 -> 杭州
#     ("Company", "SUBSIDIARY_OF", "Company"),    # 是…子公司：支付宝 -> 阿里巴巴
#     ("Company", "INVESTOR_IN", "Company"),      # 投资：投资方 -> 被投公司
#     ("Company", "ACQUIRED", "Company"),         # 收购：收购方 -> 被收购方
#     ("Company", "PRODUCED", "Product"),         # 生产/推出：腾讯 -> QQ
# ]

# 繁体转简体：维基百科正文繁简混杂，"馬雲"和"马云"应是同一个实体
cc = OpenCC("t2s")


def normalize_name(name: str) -> str:
    """实体名标准化：Unicode 兼容归一（全角转半角等）、繁转简、去所有空白"""
    name = unicodedata.normalize("NFKC", name)
    name = cc.convert(name)
    return re.sub(r"\s+", "", name).strip()


def normalize_graph_doc(graph_doc: GraphDocument) -> None:
    """标准化实体名（繁转简/全角转半角/去空白）。

    注意：LLMGraphTransformer 构建关系时会为端点新建 Node 对象（源码
    map_to_base_relationship），并非引用 nodes 列表里的节点，
    所以节点和关系端点要分别标准化，否则会 MERGE 出重复实体。
    跨文档的同名实体由写入时的 MERGE 自动合并。
    """
    for node in graph_doc.nodes:
        node.id = normalize_name(node.id)
    for rel in graph_doc.relationships:
        rel.source.id = normalize_name(rel.source.id)
        rel.target.id = normalize_name(rel.target.id)


def unique_entity_count(graph_docs: list) -> int:
    """统计标准化后跨文档唯一实体数"""
    return len({(node.type, node.id) for gd in graph_docs for node in gd.nodes})


def load_docs() -> list[Document]:
    """加载维基百科文档：本地 JSON 存在则直接读取，否则联网抓取（复用 load_wikipedia_datas）"""
    docs = load()
    for doc in docs:
        metadata = doc.metadata
        # 用 MD5 作为稳定 id：对来源 URL（无则用标题）取哈希，
        # 正文变化时 id 不变，始终 MERGE 到同一 Document 节点
        # 如果不存原文，可以忽略这里。
        stable_key = metadata.get("source") or metadata.get("title") or ""
        metadata["id"] = md5(stable_key.encode("utf-8")).hexdigest()
    return docs


def main() -> None:
    docs = load_docs()

    # 1. 大模型抽图：把每篇文档转成 GraphDocument（节点 = 实体，关系 = 实体间的联系）
    transformer = LLMGraphTransformer(
        llm=gpt55_llm, allowed_nodes=ALLOWED_NODES, allowed_relationships=ALLOWED_RELS
    )
    # 内部是逐篇调 LLM 的 for 循环，传全部文档与外部循环等价
    graph_docs = transformer.convert_to_graph_documents(docs)
    for doc, graph_doc in zip(docs, graph_docs):
        normalize_graph_doc(graph_doc)
        print(f"  《{doc.metadata.get('title')}》抽出 {len(graph_doc.nodes)} 个实体、{len(graph_doc.relationships)} 个关系")

    # 实体名标准化：跨文档唯一实体数（写入时 MERGE 自动合并同名实体）
    print(f"\n标准化后跨文档唯一实体数：{unique_entity_count(graph_docs)}")

    # 2. 写入 Neo4j：baseEntityLabel=True 给实体加 __Entity__ 标签并建唯一索引
    # include_source=True 时会把源文档存为 Document 节点并用 MENTIONS 连到实体（可溯源）
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    # 显式确保唯一约束：MERGE 幂等的基础（无约束时并发导入会产生重复节点，
    # 按 id 匹配也会退化为全标签扫描）；
    # baseEntityLabel=True 本身也会自动建
    graph.query(
        "CREATE CONSTRAINT entity_id IF NOT EXISTS "
        "FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
    )

    graph.add_graph_documents(graph_docs, include_source=False, baseEntityLabel=True)

    # 3. 验证输出
    stats = graph.query(
        """
        MATCH (n) RETURN labels(n) AS labels, count(n) AS count
        ORDER BY count DESC LIMIT 15
        """
    )
    print("\n库中节点统计：")
    for row in stats:
        print(f"  {row['labels']}: {row['count']}")

    rel_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
    print(f"\n关系总数：{rel_count}")

    # 抽查一个实体的关系（实体名存在节点的 id 属性里）
    sample = graph.query(
        """
        MATCH (a:`人物`)-[r]->(b)
        RETURN a.id AS from, type(r) AS rel, coalesce(b.id, b.title) AS to
        LIMIT 10
        """
    )
    print("\n实体关系示例：")
    for row in sample:
        print(f"  ({row['from']})-[:{row['rel']}]->({row['to']})")


if __name__ == "__main__":
    main()
