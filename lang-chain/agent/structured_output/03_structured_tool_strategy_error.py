from datetime import datetime
from typing import Optional, Union, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

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

class structuredRes(BaseModel):
    #位置 枚举是引导报错
    location: Literal["石家庄市"] = Field(description="位置",)
    #温度 范围是引导报错
    temperature: Optional[float] = Field(None, description="温度,范围10-20摄氏度之间", ge=10.0, le=20.0)
    #天气描述
    weather_description: Optional[str] = Field(None, description="天气描述")
    #日期
    date: Optional[datetime] = Field(None, description="日期")


agent =create_agent(
    model=funcloude_claude_llm,
    system_prompt="你是天气查询助手",
    tools=[get_current_location, get_weather],
    # 结构化输出
    response_format=ToolStrategy(
        structuredRes,
        tool_message_content="处理完成",
        handle_errors="出错了～"
    ),
)
resChunk = agent.stream({"messages": [HumanMessage("今天天气怎么样")]})


for chunk in resChunk:
    for item, value in chunk.items():
        # print(f"{item}: {value}")
        value["messages"][-1].pretty_print()

        # structured_response 是结构化的数据
        if value.get("structured_response"):
            print(f"这是结构化数据：{value['structured_response']}")
