from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import glm_llm

@tool
def get_current_location() -> str:
    """获取当前位置。"""
    return "当前位置为北京市。"

@tool
def get_weather(date: str, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"

#中间件
@dynamic_prompt
def agent_dynamic_prompt(request: ModelRequest) -> str:
    print(request)

    role = request.runtime.context.get("role")
    if role is None:
        return "你是普通查询助手，只能查询当天的天气"
    if role == "vip":
        return "你是会员查询助手，你可以查询今天的天气和历史天气。"
    else:
        return "你是普通查询助手，只能查询当天的天气"

agent = create_agent(
    model=glm_llm,
    tools=[get_current_location, get_weather],
    middleware=[agent_dynamic_prompt],
    system_prompt="你是天气查询助手",
)

m = HumanMessage("昨天天气怎么样")

res = agent.invoke(
    {"messages": [m]},
    context={"role": "vip"}
)
print(res["messages"][-1].content)

print("-----------------")

res = agent.invoke(
    {"messages": [m]},
    context={"role": "user"}
)
print(res["messages"][-1].content)
