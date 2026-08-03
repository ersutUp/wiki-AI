import asyncio
import json

from langchain.agents import create_agent, AgentState
from langchain_core.callbacks.manager import handle_event
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langgraph.types import Command
from pydantic import BaseModel

from my_llm import glm_llm

mcp_client = MultiServerMCPClient(
    {
        "http_demo": {
            "transport": "http",
            "url": "http://localhost:28008/my-mcp",
        },
    },
)


async def getUserResource():
    resources = await mcp_client.get_resources("http_demo",uris=["user://get_user/1,2"])
    user = resources[0].as_string()
    print(user)
    return user


async def agent():
    user_str = await getUserResource()
    agent = create_agent(
        model=glm_llm,
        system_prompt=f"""
            你是用户信息管理助手，
            这是用户信息 {user_str}
        """,
    )
    #
    m = HumanMessage("展示所有用户信息？")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await agent.ainvoke(
        {"messages": [m]},
    )
    print(res)
    print(res["messages"][-1].content)



asyncio.run(agent())

