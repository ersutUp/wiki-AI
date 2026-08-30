"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Any, Dict, Annotated

from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.my_llm import glm_llm, deepseek_flash4_llm

@tool
def saveUser(username: str,
             state: Annotated[dict[str, Any],  InjectedState],
             tool_call_id: Annotated[str, InjectedToolCallId],
             ) -> Command:
    """
    保存用户信息

    Args:
        username: 用户名

    Returns:
        str: 保存结果
    """
    tool_count = state.get("tool_count", 0)

    return Command(
        update={
            "username": username,
            "tool_count": tool_count + 1,
            "messages": [ToolMessage(
                tool_call_id=tool_call_id,
                content=f"用户 {username} 已保存"
            )]
        },
    )

@tool
def getUser(state: Annotated[dict[str, Any], InjectedState],
             tool_call_id: Annotated[str, InjectedToolCallId],
             ) -> Command:
    """
    获取用户信息

    Returns:
        str: 用户名
    """

    username = state.get("username", "")
    tool_count = state.get("tool_count", 0)

    content = "暂无用户信息"
    if username != "":
        content = f"当前用户是 {username}"

    return Command(
        update={
            "tool_count": tool_count + 1,
            "messages": [ToolMessage(
                tool_call_id=tool_call_id,
                content=content
            )]
        },
    )


class State(MessagesState):
    """
    状态定义
    """
    tool_count: int
    username: str



async def call_model(state: State, runtime: Runtime) -> Dict[str, Any]:
    """
    调用模型
    """
    print("===" * 100)
    print(state)

    return {
        "messages": await llm_with_tools.ainvoke(state["messages"]),
    }

toolNode = ToolNode(tools=[
    getUser,
    saveUser,
])

# 将工具绑定到模型，LLM 才知道可以调用这些工具
llm_with_tools = deepseek_flash4_llm.bind_tools([getUser, saveUser])

state = State()
state["messages"] = [SystemMessage(content="你是一个智能信息管理助手")]

# Define the graph
graph = (
    StateGraph(State)
    .add_node(call_model)
    .add_node("tools", toolNode)

    .add_edge(START, "call_model")


    .add_conditional_edges("call_model", tools_condition)

    .add_edge("tools", "call_model")
    .compile(name="New Graph", interrupt_before=["tools"])
)
