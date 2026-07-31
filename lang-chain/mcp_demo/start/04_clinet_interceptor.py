import asyncio
import json

from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langgraph.types import Command
from pydantic import BaseModel

from my_llm import glm_llm


class UserInfoState(AgentState):
    user_id: int
    name: str
    age: int
    email: str



class UserContext(BaseModel):
    user_id: int



async def http_tool_interceptor(request: MCPToolCallRequest, handler) -> MCPToolCallResult:
    """
    从上下文获取用户ID，并将其添加到工具调用参数中
    """
    # print(f"request: {request}")
    userContext = request.runtime.context
    if userContext is None:
        pass
    elif userContext.user_id == 0:
        pass
    else:
        user_id = userContext.user_id
        request = request.override(args={**request.args, "user_id": user_id})

    response = await handler(request)

    if request.name == "get_user_by_id":

        json_str = response.content[0].text
        print("MCP response:user_info:", json_str)
        json_obj = json.loads(json_str)

        # 转换为 ToolMessage
        tool_msg = ToolMessage(
            content=json_str,
            tool_call_id=request.runtime.tool_call_id,
        )
        # 返回 Command 同时更新状态
        return Command(
            update={
                "messages": [tool_msg],
                **json_obj
            }
        )


    return response


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
        context_schema=UserContext,
        state_schema=UserInfoState,
    )

    m = HumanMessage("查询下我登记的信息？")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await agent.ainvoke(
        {"messages": [m]},
        context=UserContext(user_id=3),
    )
    print(res)
    print(res["messages"][-1].content)


asyncio.run(agent())

