from datetime import datetime
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.stores import InMemoryStore
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from my_llm import glm_llm

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
    model=glm_llm,
    tools=[get_current_location, get_weather],
    system_prompt="你是天气查询助手",
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            description_prefix="请确认以下信息是否正确：",
            interrupt_on={
                "get_weather": True,
                "get_current_location": False,
            }
        )
    ]
)

config = {"configurable": {"thread_id": "chat1"}}

res = agent.invoke(
    {"messages": [HumanMessage("今天天气怎么样")]},
    config=config,
    version='v2'
)
print(res)
#是否中断
interrupts = res.interrupts
if interrupts:
    # 打印中断信息
    action_requests = interrupts[0].value["action_requests"]
    review_configs = interrupts[0].value["review_configs"]
    # 遍历 action_requests 和 review_configs，打印描述和工具名称
    for action_request, review_config in zip(action_requests, review_configs):
        description = action_request["description"]
        tool_name = action_request["name"]
        print(f"工具名称：{tool_name}")
        print(f"描述：{description}")
        allowed_decisions = review_config['allowed_decisions']
        print(f"可操作选项：{','.join(allowed_decisions) if allowed_decisions else '无'}")
else:
    print(res.value["messages"][-1].content)
