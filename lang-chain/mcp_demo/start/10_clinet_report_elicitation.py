import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext, LoggingMessageNotificationParams, \
    ElicitRequestParams
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.context import RequestContext
from mcp.types import ElicitResult

from my_llm import glm_llm


async def elicitation_handler(
        mcp_context: RequestContext,
        params: ElicitRequestParams,
        context: CallbackContext,
) -> ElicitResult:
    # 打印服务端发来的提示信息
    print("\n" + "=" * 40)
    print(f"[服务端询问] {params.message}")
    print(f"(来自 MCP：{context.server_name}, 工具：{context.tool_name})")
    print("=" * 40)

    # 用 asyncio.to_thread 包装 input，避免阻塞事件循环（agent.ainvoke 运行在异步上下文里）
    def _read(prompt: str) -> str:
        return input(prompt).strip()

    # 展示操作选项，让用户选择
    while True:
        choice = _read(
            "请选择操作 [1=填写(accept) / 2=拒绝(decline) / 3=取消(cancel)]，默认 1：",
        )
        if choice in ("", "1"):
            action = "accept"
            break
        if choice == "2":
            action = "decline"
            break
        if choice == "3":
            action = "cancel"
            break
        print("输入有误，请重新选择。")

    # 接受时才需要用户填写内容
    if action == "accept":
        # 服务端用的是 response_type=str，对应表单里的 "value" 字段
        value = _read("请输入内容：")
        return ElicitResult(action="accept", content={"value": value})

    # 拒绝/取消时无需提交表单内容
    return ElicitResult(action=action)


mcp_client = MultiServerMCPClient(
    {
        "http_demo": {
            "transport": "http",
            "url": "http://localhost:28019/my-mcp",
        },
    },
    callbacks=Callbacks(
        on_elicitation= elicitation_handler
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
    m = HumanMessage("增加一个用户，姓名为赵六，年纪 37。添加后展示所有用户。")

    # 这里需要异步（ainvoke） 因为 mcp 工具是异步的
    res = await my_agent.ainvoke(
        {"messages": [m]},
    )
    print(res)
    print(res["messages"][-1].content)



asyncio.run(agent())

