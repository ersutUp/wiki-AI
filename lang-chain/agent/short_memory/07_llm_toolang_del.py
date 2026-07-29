"""
上下问太长的解决方案 删除消息
"""
from typing import Any, Iterator

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from my_llm import glm_llm, funcloude_claude_llm

def print_chunk(respChunks: Iterator[dict[str, Any] | Any]) -> None:
    chunkMessages = None
    for chunk in respChunks:
        # print(chunk)
        msg = chunk.get("messages")
        if msg:
            content = msg[-1].content
        else:
            content = "无"
        print(f"type: {type(msg[-1])}, content: {content}")
        chunkMessages = msg

    if chunkMessages:
        print(f"剩余消息: {chunkMessages}")

@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(date: str, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"


@tool
def clear_messages(runtime: ToolRuntime, clear_messages: bool) -> Command:
    """清除所有消息。"""
    if clear_messages:
        return Command(update = {
            "clear_messages": True,
            "messages": [ToolMessage(
                content="已更新清除消息状态",
                tool_call_id=runtime.tool_call_id,
            )]
        })
    else:
        return None


#截断消息
@before_model
def delete_messages_by_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]|None:
    """删除消息"""

    clear_messages = state.get("clear_messages", False)
    if clear_messages:
        print(f">>> state: {state}")
        return {
            "clear_messages": False,
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                # AIMessage(content="已成功删除了所有消息,回答给用户，我已清除所有消息，您可以继续与我对话。")
                SystemMessage(content="已成功删除了所有消息,回答给用户，我已清除所有消息，您可以继续与我对话。")
            ]
        }
    else:
        return None

class MyState(AgentState):
    clear_messages: bool

agent = create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是聊天助手",
    tools=[get_current_location, get_weather, clear_messages],
    # 中间件 截断消息
    middleware=[delete_messages_by_middleware],
    # 开启记忆 保存 内存
    checkpointer=InMemorySaver(),
    state_schema=MyState,
)


config = {"configurable": {"thread_id": "chat1"}}

respChunks = agent.stream({"messages": [HumanMessage("你好，我想知道今天的天气怎么样")]}, stream_mode="values",config=config)
print_chunk(respChunks)

respChunks = agent.stream({"messages": [HumanMessage("清除消息")]}, stream_mode="values",config=config)
print_chunk(respChunks)

