

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver

from env_utils import MYSQL_PASSWORD
from my_llm import funcloude_claude_llm



DB_URI = f"mysql://root:{MYSQL_PASSWORD}@10.52.25.32:3306/langchain"
with PyMySQLSaver.from_conn_string(DB_URI) as checkpointer:
    # 创建数据库表
    checkpointer.setup()

    agent =create_agent(
        model=funcloude_claude_llm,
        system_prompt="你是聊天助手",
        # 开启记忆 保存 mysql 数据库
        checkpointer=checkpointer,
    )
    #保持记忆的关键固定格式，  configurable.thread_id
    config = {"configurable": {"thread_id": "chat1"}}

    # 流式输出
    resp = agent.invoke({"messages": [HumanMessage("你好，我是王总")]},stream_mode="checkpoints",config=config)
    print(resp)
    print(resp[-1]["values"]["messages"][-1].content)

    print("-"*50)

    resp = agent.invoke({"messages": [HumanMessage("我是谁")]},stream_mode="checkpoints",config=config)
    print(resp)
    print(resp[-1]["values"]["messages"][-1].content)

