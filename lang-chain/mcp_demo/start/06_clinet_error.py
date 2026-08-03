import asyncio
import json

from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langgraph.types import Command
from pydantic import BaseModel

from my_llm import glm_llm


async def getMcpTools():
    mcp_client = MultiServerMCPClient(
        {
            "http_demo": {
                "transport": "http",
                "url": "http://localhost:28008/my-mcp",
            },
            
        },
    )

    tools = await mcp_client.get_tools()
    print(tools)

    return tools


async def agent():
    tools = await getMcpTools()
    agent = create_agent(
        model=glm_llm,
        tools=tools,
        system_prompt="""
            你是用户信息管理查询助手，
            若网络抖动等错误，请重试5次。
        """,
    )
    #
    # m = HumanMessage("查询下用户 99 登记的信息？")
    #
    # # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    # res = await agent.ainvoke(
    #     {"messages": [m]},
    # )
    # print(res)
    # print(res["messages"][-1].content)

    m = HumanMessage("查询下用户 张三 登记的信息？")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await agent.ainvoke(
        {"messages": [m]},
    )
    print(res)
    print(res["messages"][-1].content)


asyncio.run(agent())

