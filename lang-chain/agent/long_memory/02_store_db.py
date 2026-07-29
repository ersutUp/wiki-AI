"""
store 长期记忆结合 agent的使用
"""
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, ArgsSchema
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mysql.pymysql import PyMySQLSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.store.mysql import PyMySQLStore
from pydantic import Field, BaseModel

from env_utils import MYSQL_PASSWORD
from my_llm import glm_llm

namespace = ("user", "info")

class UserContext(BaseModel):
    userId: int

class UserInfo(BaseModel):
    name: str = Field(description="用户姓名")
    sex: str = Field(description="用户性别")
    age: int = Field(description="用户年龄")

@tool(args_schema=UserInfo)
def set_user_info(runtime: ToolRuntime, name: str, sex: str, age: int) -> str:
    """
    设置用户的个人信息。
    """
    user_id = runtime.context.userId
    if user_id is None:
        return "缺少用户 id上下文"
    runtime.store.put(namespace, user_id, {
        "name": name,
        "sex": sex,
        "age": age,
    })
    return f"已设置用户 {name} 的个人信息为 {sex} {age} 岁"

@tool
def get_user_info(runtime: ToolRuntime) -> str:
    """
    获取用户的个人信息。
    """

    context = runtime.context
    if context is None:
        return "缺少上下文"

    user_id = context.userId
    if user_id is None:
        return "缺少用户 id上下文"

    user_info = runtime.store.get(namespace, user_id)
    if user_info:
        return f"用户 {user_info.value['name']} 的个人信息为 {user_info.value['sex']} {user_info.value['age']} 岁"
    else:
        return "用户未设置个人信息"




DB_URI = f"mysql://root:{MYSQL_PASSWORD}@10.52.25.32:3306/langchain"
with (
        PyMySQLSaver.from_conn_string(DB_URI) as checkpointer,
        PyMySQLStore.from_conn_string(DB_URI) as store
     ):

    checkpointer.setup()
    store.setup()

    agent = create_agent(
        model=glm_llm,
        system_prompt="你是个人信息管理助手。",
        tools=[set_user_info, get_user_info],
        store=store,
        context_schema=UserContext,
    )

    resp2 = agent.invoke(
        {"messages": [HumanMessage("设置用户信息为张三，男，25岁")]},
        context={"userId": 15},
    )
    print(resp2["messages"][-1].content)

    print("-----------------"*50)

    resp2 = agent.invoke({"messages": [HumanMessage("获取用户张三的个人信息")]})
    print(resp2["messages"][-1].content)

    print("-----------------"*50)
    resp3 = agent.invoke({
        "messages": [HumanMessage("获取用户张三的个人信息")]},
        context={"userId": 15},
    )
    print(resp3["messages"][-1].content)


