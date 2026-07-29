from datetime import datetime
from typing import Optional, Union

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import glm_llm, funcloude_llm, funcloude_deepseek_llm

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
    #位置
    location: Optional[str] = Field(None, description="位置")
    #温度
    temperature: Optional[float] = Field(None, description="温度")
    #天气描述
    weather_description: Optional[str] = Field(None, description="天气描述")
    #日期
    date: Optional[datetime] = Field(None, description="日期")


class workRes(BaseModel):
    """工作数据"""
    #工作内容
    work: Optional[str] = Field(None, description="工作内容")
    #结果
    result: Optional[str] = Field(None, description="工作结果")


agent = create_agent(
    model= funcloude_deepseek_llm,
    tools=[get_current_location, get_weather],
    system_prompt="你是天气查询助手，优先处理用户关于天气的问题",
    # 多个结构化 使用 Union
    response_format=ToolStrategy(Union[structuredRes, workRes], tool_message_content="处理完成"),
)



res = agent.invoke({"messages": [HumanMessage("今天工作心情愉快，完成代码，非常完美，看下今天天气怎么样")]})
print(res)
# structured_response 是结构化的数据
if "structured_response" in res:
    print(res["structured_response"])

