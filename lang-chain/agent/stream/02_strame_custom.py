import time
from datetime import datetime
from typing import Optional, Union, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

from my_llm import glm_llm, funcloude_claude_llm

from pydantic import BaseModel, Field

@tool
def get_current_location() -> str:
    """获取当前位置。"""
    writer = get_stream_writer()
    time.sleep(1)
    writer("查询位置进度 20%")
    time.sleep(1)
    writer("查询位置进度 40%")
    time.sleep(1)
    writer("查询位置进度 60%")
    time.sleep(1)
    writer("查询位置进度 80%")
    time.sleep(1)
    writer("查询位置进度 100%")

    return "当前位置为北京市。"

@tool
def get_weather(date: datetime, city: str) -> str:
    """获取指定城市的天气信息。"""
    writer = get_stream_writer()
    time.sleep(1)
    writer("查询天气进度 20%")
    time.sleep(1)
    writer("查询天气进度 40%")
    time.sleep(1)
    writer("查询天气进度 60%")
    time.sleep(1)
    writer("查询天气进度 80%")
    time.sleep(1)
    writer("查询天气进度 100%")

    return f"{date} {city}的天气为晴朗，25°C。"


agent =create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是天气查询助手",
    tools=[get_current_location, get_weather],
)

# 使用 custom 加 updates 输出
resChunk = agent.stream({"messages": [HumanMessage("今天天气怎么样")]},stream_mode=["custom","updates"])

for chunk in resChunk:
    print(chunk)
