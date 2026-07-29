from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_tool_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from my_llm import glm_llm

@tool
def get_current_location() -> str:
    """获取当前位置。"""

    raise ValueError("获取当前位置失败")

    return "当前位置为北京市。"

@tool
def get_weather(date: datetime, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"

#中间件
@wrap_tool_call
def agent_tool_call(request: ToolCallRequest,handler ) -> ToolMessage:
    print(request)

    try:
        return handler(request)
    except Exception as e:
        print(e)
        return ToolMessage(
            content="工具出错",
            tool_call_id=request.state["messages"][0].id,
        )

agent = create_agent(
    model=glm_llm,
    tools=[get_current_location, get_weather],
    middleware=[agent_tool_call],
    system_prompt="你是天气查询助手",
)

m = HumanMessage("今天天气怎么样")
res = agent.invoke(
    {"messages": [m]},
)
print(res)
print(res["messages"][-1].content)

print("\n--------已有位置----------")
m = HumanMessage("石家庄今天天气怎么样")
res = agent.invoke(
    {"messages": [m]},
)
print(res)
print(res["messages"][-1].content)
