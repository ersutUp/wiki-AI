"""基于 GraphCypherQAChain 的知识图谱问答 Demo。

流程：自然语言问题 → LLM 根据 Neo4j 图谱 Schema 生成 Cypher → 执行查询 → LLM 根据查询结果生成回答。
依赖 load_documents_to_neo4j.py 已构建好的图谱（节点/关系标签为中文，如 人物、创办）。
"""
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

from env_utils import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from my_llm import glm_llm, claude_llm

# 自定义 Cypher 生成提示词：本图谱的标签是中文（人物/公司/创办），
# 默认英文提示词下模型容易生成英文标签（如 Person、FOUNDER_OF）导致查不到数据
CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""任务：根据图谱结构（Schema）把用户问题翻译成一条 Cypher 查询语句。

图谱结构：
{schema}

要求：
1. 节点标签和关系类型必须与 Schema 中的完全一致（中文，如 人物、公司、创办），不要翻译成英文。
2. 实体名存放在节点的 id 属性上，用 {{id: "实体名"}} 匹配，例如 MATCH (p:人物 {{id: "马云"}})。
3. 不要查询 Document 节点。
4. 只输出一条 Cypher 语句，不要解释，不要用 markdown 代码块包裹。

用户问题：{question}
Cypher：""",
)


def main() -> None:
    # 连接时自动读取库里的标签/关系类型/属性，生成 Schema 注入提示词
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    # 生成 Cypher 语句的大模型 与 问答的大模型使用同一个
    chain = GraphCypherQAChain.from_llm(
        llm=glm_llm,
        graph=graph,
        cypher_prompt=CYPHER_PROMPT,
        verbose=True,  # 打印生成的 Cypher，方便调试
        # 安全确认开关：此链会执行 LLM 生成的任意 Cypher，必须显式设 True 才能运行。
        # 生产环境应给数据库账号收窄权限（只读）或自行校验生成的 Cypher
        allow_dangerous_requests=True,
        return_intermediate_steps=True
    )

    # 生成 Cypher 语句 和 问答 使用不同的大模型
    # chain = GraphCypherQAChain.from_llm(
    #     cypher_llm=glm_llm,
    #     qa_llm=claude_llm,
    #     graph=graph,
    #     cypher_prompt=CYPHER_PROMPT,
    #     verbose=True,  # 打印生成的 Cypher，方便调试
    #     # 安全确认开关：此链会执行 LLM 生成的任意 Cypher，必须显式设 True 才能运行。
    #     # 生产环境应给数据库账号收窄权限（只读）或自行校验生成的 Cypher
    #     allow_dangerous_requests=True,
    #     return_intermediate_steps=True
    # )

    questions = [
        "阿里巴巴的创始人是谁？",
        "腾讯生产过哪些产品？",
        "有哪些公司位于杭州？",
        "雷军毕业于哪所学校？",
        "详细介绍一下马云",
        "雷军和马云谁牛逼？"
    ]
    for question in questions:
        print(f"\n{'=' * 50}\n问题：{question}")
        result = chain.invoke({"query": question})
        print(f"回答：{result['result']}")


if __name__ == "__main__":
    main()
