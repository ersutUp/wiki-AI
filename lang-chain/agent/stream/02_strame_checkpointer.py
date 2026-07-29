from datetime import datetime
from typing import Optional, Union, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from my_llm import glm_llm, funcloude_claude_llm

from pydantic import BaseModel, Field

@tool
def get_current_location() -> str:
    """获取当前位置。"""

    return "当前位置为北京市。"

@tool
def get_weather(date: datetime, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"


agent =create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是天气查询助手",
    tools=[get_current_location, get_weather],
    # 开启记忆 保存在内存中
    checkpointer=InMemorySaver(),
)
#保持记忆的关键固定格式，  configurable.thread_id
config = {"configurable": {"thread_id": "chat1"}}

# 流式输出
resChunk = agent.stream({"messages": [HumanMessage("今天天气怎么样")]},stream_mode="checkpoints",config=config)

for chunk in resChunk:
    print(chunk)

print("="*50)

resChunk = agent.stream({"messages": [HumanMessage("我刚刚看的哪天的天气？")]},stream_mode="checkpoints",config=config)

for chunk in resChunk:
    print(chunk)