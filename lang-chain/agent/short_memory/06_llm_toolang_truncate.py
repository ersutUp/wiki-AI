"""
上下问太长的解决方案 通过中间件截断消息
"""
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from my_llm import glm_llm, funcloude_claude_llm


@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(date: str, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"


#截断消息
@before_model
def truncate_messages_by_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any]|None:
    """截断消息"""
    max_msg = 3
    msg = state["messages"]
    if len(msg) > max_msg:
        print(f">>> state: {state}")
        # 这里是避免 ToolMessage 作为第一个
        if isinstance(msg[-max_msg], ToolMessage):
            print(f">>> 避免 ToolMessage 作为第一个, {msg[-max_msg]}")
            max_msg -= 1
        return {"messages": [RemoveMessage(id = msg.id) for msg in msg[:-max_msg]]}
    else:
        return None


agent = create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是聊天助手",
    tools=[get_current_location, get_weather],
    # 中间件 截断消息
    middleware=[truncate_messages_by_middleware],
)

respChunks = agent.stream({"messages": [HumanMessage("你好，我想知道今天的天气怎么样")]}, stream_mode="values")
chunkMessages = None
for chunk in respChunks:
    # print(chunk)
    msg = chunk.get("messages")
    if msg:
        content = msg[-1].content
    else:
        content = "无"
    print(f"content: {content}")
    chunkMessages = msg

if chunkMessages:
    print(f"剩余消息: {chunkMessages}")
