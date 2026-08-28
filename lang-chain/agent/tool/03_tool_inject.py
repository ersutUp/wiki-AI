"""
演示 @tool 注解的自动注入功能：
- InjectedToolCallId：自动注入当前工具调用 ID
- InjectedState：自动注入 Agent 状态中的指定字段
- InjectedStore：自动注入长期记忆存储（BaseStore）
"""
from typing import Annotated
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from my_llm import glm_llm


# 创建一个内存存储，用于演示 InjectedStore
store = InMemoryStore()


@tool
def remember_user_info(
    name: str,
    hobby: str,
    # 以下三个参数由框架自动注入，大模型不需要、也看不到这些参数
    call_id: Annotated[str, InjectedToolCallId],
    msgs: Annotated[list, InjectedState("messages")],
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """记住用户的姓名和爱好，以便后续对话中使用。"""
    # InjectedToolCallId：框架自动注入当前工具调用的唯一 ID
    print(f"[自动注入] 工具调用 ID: {call_id}")

    # InjectedState("messages")：框架自动注入当前 Agent 状态中的 messages 列表
    msg_count = len(msgs)
    print(f"[自动注入] 当前消息数量: {msg_count}")

    # InjectedStore：框架自动注入 Store 实例，用于存取长期记忆
    user_ns = ("users", name)
    store.put(user_ns, "name", {"name": name})
    store.put(user_ns, "hobby", {"hobby": hobby})
    print(f"[自动注入] 已将用户信息存入 Store: {user_ns}")

    return f"已记住用户 {name} 的爱好是 {hobby}（当前对话共 {msg_count} 条消息）"


@tool
def get_user_info(
    name: str,
    call_id: Annotated[str, InjectedToolCallId],
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """查询已记住的用户姓名和爱好。"""
    print(f"[自动注入] 工具调用 ID: {call_id}")

    user_ns = ("users", name)
    stored_name = store.get(user_ns, "name")
    stored_hobby = store.get(user_ns, "hobby")

    if stored_name and stored_hobby:
        return f"用户 {stored_name.value['name']} 的爱好是 {stored_hobby.value['hobby']}"
    else:
        return f"未找到用户 {name} 的信息"


# 创建带有 store 的 Agent
# 注意：注入参数（call_id, msgs, store）不在 args_schema 中暴露给大模型
agent = create_agent(
    model=glm_llm,
    tools=[remember_user_info, get_user_info],
    store=store,
    system_prompt="你是用户信息管理助手。用户告诉你信息时，用 remember_user_info 记住；用户查询时，用 get_user_info 查询。",
)

print("=" * 50)
print("第一轮对话：记住用户信息")
print("=" * 50)
res = agent.invoke(
    {"messages": [HumanMessage("我叫小明，我喜欢打篮球")]},
    config={"configurable": {"thread_id": "demo-1"}},
)
print("Agent 回复:", res["messages"][-1].content)

print()
print("=" * 50)
print("第二轮对话：查询用户信息")
print("=" * 50)
res = agent.invoke(
    {"messages": [HumanMessage("查询小明的信息")]},
    config={"configurable": {"thread_id": "demo-1"}},
)
print("Agent 回复:", res["messages"][-1].content)