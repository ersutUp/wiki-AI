"""
动态切换工具
"""
import json
from datetime import datetime

from typing import Optional, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import Field, BaseModel, field_validator

from my_llm import glm_llm

@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(date: datetime, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"

class UserInfo(BaseModel):
    userRole: str = None

# 通过 request.override(tools=tools) 动态切换工具
@wrap_model_call
def dynamic_tool_selection(request: ModelRequest[UserInfo], handler) -> ModelResponse:
    """动态选择工具。"""
    context = request.runtime.context
    # print(context)

    # 非 vip 用户 只能调用天气工具
    use_tool_names = [get_weather.name]
    tools = [tool for tool in request.tools if tool.name in use_tool_names]

    if context is None:
        request = request.override(tools=tools)
        return handler(request)
    elif context.userRole is None:
        request = request.override(tools=tools)
    elif context.userRole == "vip":
        # vip 用户 可以调用所有工具
        pass

    print( [t.name for t in request.tools])
    return handler(request)

agent = create_agent(
    model=glm_llm,
    tools=[get_current_location, get_weather],
    middleware=[dynamic_tool_selection],
    context_schema= UserInfo,
    system_prompt="你是天气查询助手，可以通过工具获取天气和位置",
)

res = agent.invoke(
    {"messages": [HumanMessage("查询天气信息")]},
    context=UserInfo(userRole="vip")
)
# print(res)
print(res["messages"][-1].content)

print("----------------"*20)



res = agent.invoke(
    {"messages": [HumanMessage("查询天气信息")]},
    context=UserInfo()
)
# print(res)
print(res["messages"][-1].content)





