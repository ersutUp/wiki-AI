import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext, LoggingMessageNotificationParams
from langchain_mcp_adapters.client import MultiServerMCPClient

from my_llm import glm_llm

async def progress_handler(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext):
    print(f"[进度] {progress/total*100}% , 信息：{message}, 当前MCP：{context.server_name}, 当前工具：{context.tool_name}")

async def mcp_log_handler(
        params: LoggingMessageNotificationParams,
        context: CallbackContext):
    # print(params)
    print(f"[日志({params.level})] {params.data["msg"]} ,当前MCP：{context.server_name}, 当前工具：{context.tool_name},")

mcp_client = MultiServerMCPClient(
    {
        "http_demo": {
            "transport": "http",
            "url": "http://localhost:28009/my-mcp",
        },
    },
    callbacks=Callbacks(
        on_progress= progress_handler,
        on_logging_message=mcp_log_handler
    )
)


async def getMCPTools():
    tools = await mcp_client.get_tools()
    print(tools)
    return tools


async def agent():
    tools = await getMCPTools()
    my_agent = create_agent(
        model=glm_llm,
        tools=tools,
        system_prompt=f"""
            你是用户信息管理助手
        """,
    )
    #
    m = HumanMessage("查询用户 1 的数据？")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await my_agent.ainvoke(
        {"messages": [m]},
    )
    print(res)
    print(res["messages"][-1].content)



asyncio.run(agent())

