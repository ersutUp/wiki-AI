import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# 获取当前项目的根目录
import os

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from pydantic import BaseModel

from my_llm import glm_llm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UserContext(BaseModel):
    user_id: int



async def http_tool_interceptor(request: MCPToolCallRequest, handler) -> MCPToolCallResult:
    """
    从上下文获取用户ID，并将其添加到工具调用参数中
    """
    print(f"request: {request}")
    userContext = request.runtime.context
    if userContext is None:
        pass
    elif userContext.user_id == 0:
        pass
    else:
        user_id = userContext.user_id
        request = request.override(args={**request.args, "user_id": user_id})
        print(f"request: {request}")

    return await handler(request)


async def getMcpTools():
    mcp_client = MultiServerMCPClient(
        {
            "http_demo": {
                "transport": "http",
                "url": "http://localhost:28008/my-mcp",
            },
        },
        # 拦截器
        tool_interceptors=[http_tool_interceptor]
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
            你是用户信息管理查询助手
            在MCP拦截器中，已经从上下文中配置了用户ID，所以在工具只需要用户 id 参数可以直接调用工具。
        """,
        context_schema=UserContext
    )

    m = HumanMessage("查询下我登记的信息？")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await agent.ainvoke(
        {"messages": [m]},
        context=UserContext(user_id="3")
    )
    print(res)
    print(res["messages"][-1].content)


asyncio.run(agent())

