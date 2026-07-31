import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from my_llm import glm_llm


async def getMcpTools():
    mcp_client = MultiServerMCPClient({
        "http_demo": {
            "transport": "http",
            "url": "http://localhost:28008/my-mcp",
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

