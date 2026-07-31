import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# 获取当前项目的根目录
import os

from my_llm import glm_llm

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



async def getMcpTools():
    mcp_client = MultiServerMCPClient({
        "stdio_demohhh": {
            "transport": "stdio",
            "command": project_root+"/.venv/bin/python",
            "args": [project_root+"/mcp_demo/start/01_server_stdio.py"],
        },
    })
    tools = await mcp_client.get_tools()
    print(tools)

    return tools

async def agent():
    tools = await getMcpTools()
    agent = create_agent(
        model=glm_llm,
        tools=tools,
        # middleware=[agent_dynamic_prompt],
        system_prompt="你是天气查询助手",
    )

    m = HumanMessage("北京天气怎么样")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await agent.ainvoke(
        {"messages": [m]},
    )
    print(res)
    print(res["messages"][-1].content)


asyncio.run(agent())

