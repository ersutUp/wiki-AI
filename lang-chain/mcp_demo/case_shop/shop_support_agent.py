import asyncio
from typing import Callable, Awaitable

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext, LoggingMessageNotificationParams, \
    ElicitRequestParams
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Interrupt, Command
from mcp.shared.context import RequestContext
from mcp.types import ElicitResult
from pydantic import BaseModel
from watchfiles import awatch

from my_llm import glm_llm


async def chunks_handle(respChunks, agent, config, context):
    """处理响应 chunk"""
    async for chunk in respChunks:
        # print(chunk)
        # 处理 updates 类型的 chunk
        chunkType = chunk[0]
        if chunkType == "updates":
            chunkData = chunk[1]
            if chunkData.get('model', None) is None:
                print("")
            # print(chunk)
            # 检查是否中断
            if chunkData and "__interrupt__" in chunkData:
                decisions = interrupts_handle(chunkData["__interrupt__"][0])
                # 恢复中断
                respChunksTool = agent.astream(
                    Command(
                        resume={"decisions": decisions},
                    ),
                    stream_mode=["updates", "messages"],
                    config=config,
                    context=context,
                )
                await chunks_handle(respChunksTool, agent, config, context)
        elif chunkType == "messages":
            chunkData = chunk[1]
            message = chunkData[0]
            if isinstance(message, AIMessageChunk):
                print(message.content,end="")


def interrupts_handle(interrupt: Interrupt) -> list:
    """处理中断信息"""
    print("interrupt", interrupt)

    decisions = []
    action_requests = interrupt.value["action_requests"]
    review_configs = interrupt.value["review_configs"]
    print(f"\n需要处理{len(review_configs)}条数据")
    # 遍历 action_requests 和 review_configs，打印描述和工具名称
    i = 0
    for action_request, review_config in zip(action_requests, review_configs):
        i += 1
        print(f"\n========================第{i}条数据=========================")
        description = action_request["description"]
        tool_name = action_request["name"]
        tool_args = action_request["args"]
        print(f">>>工具名称：{tool_name}")
        print(f">>>参数：{tool_args}")
        print(f">>>描述：{description}")
        allowed_decisions = review_config['allowed_decisions']


        if len(allowed_decisions) == 1 and allowed_decisions[0] == "respond":
            respond_message = input("<<<请输入回复内容：")
            decisions.append({"type": "respond", "message": respond_message})
        else:
            while True:
                decision = input(f"<<<按选择操作（{','.join(allowed_decisions) if allowed_decisions else '无'}）：")
                if decision == "approve":
                    decisions.append({"type": "approve"})
                    break
                elif decision == "reject":
                    reject_reason = input("<<<请输入拒绝原因：")
                    decisions.append({"type": "reject", "message": reject_reason})
                    break
                elif decision == "edit":
                    tool_new_args = {}
                    for arg, value in tool_args.items():
                        tool_new_args[arg] = input(f"<<<请输入{arg}的值(原值为：{value})：")

                    decisions.append({"type": "edit", "edited_action": {"name": tool_name, "args": tool_new_args}})
                    break
                elif decision == "respond":
                    respond_message = input("<<<请输入回复内容：")
                    decisions.append({"type": "respond", "message": respond_message})
                    break
                else:
                    print("无效操作，请重新输入")

    print("AI处理中（恢复中断）")
    # print(decisions)
    return decisions


# 将用户 id 放入 tool 的参数中
def user_info_interceptors(
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],) -> MCPToolCallResult:

    if request.name == "refund_order":
        my_agent_context: MyAgentContext | None = request.runtime.context
        if my_agent_context is not None:
            request = request.override(args={ **request.args, "user_id": my_agent_context.user_id})

    return handler(request)

async def progress_handler(
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
):
    print(f"[进度] {progress/total*100}% , 信息：{message}, 当前MCP：{context.server_name}, 当前工具：{context.tool_name}")

async def logging_message_handler(
        params: LoggingMessageNotificationParams,
        context: CallbackContext,
):
    print(f"[日志({params.level})] {params.data["msg"]} ,当前MCP：{context.server_name}, 当前工具：{context.tool_name}")


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
        "order_mcp": {
            "transport": "http",
            "url": "http://localhost:28032/my-mcp",
        },
        "notify": {
            "transport": "http",
            "url": "http://localhost:28031/my-mcp",
        },
    },
    callbacks=Callbacks(
        on_elicitation= elicitation_handler,
        on_progress= progress_handler,
        on_logging_message=logging_message_handler
    ),
    tool_interceptors= [
        user_info_interceptors
    ]
)


#获取 MCP 所有工具
async def getMCPTools():
    tools = await mcp_client.get_tools()
    print(tools)
    return tools

#获取退款成功消息模板提示词
async def get_refund_prompt() -> str:
    pp_result = await mcp_client.get_prompt(
        "notify",
        "refund_response",
        arguments={"order_id": "{订单号}" , "amount": "金额" })
    return pp_result[0].content

#获取退款政策资源
async def get_doc_refund() -> str:
    resources = await mcp_client.get_resources("notify", uris=["doc://refund"])
    return resources[0].as_string()

#上下文
class MyAgentContext(BaseModel):
    user_id:str

async def shop_agent():
    tools = await getMCPTools()
    my_agent = create_agent(
        model=glm_llm,
        tools=tools,
        system_prompt=f"""
            你是商城客服
            
            这是退款政策
            {await get_doc_refund()}
            
            这是退款成功消息的模板
            {await get_refund_prompt()}
            
            用户信息已经放在上下文了，并传给了工具,你无需确认用户身份
        """,
        checkpointer=InMemorySaver(),
        context_schema=MyAgentContext,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "refund_order": InterruptOnConfig(
                        description="确定退款吗？",
                        allowed_decisions=["approve", "reject"]
                    )
                }
            )
        ]
    )

    config = RunnableConfig(configurable={"thread_id": "chat1"})
    my_context = MyAgentContext(user_id="3")


    while True:
        msg = input("你：")
        if msg in ("exit", "quit", "q"):
            break
        m = HumanMessage(msg)
        respChunks = my_agent.astream(
            {"messages": [m]},
            config=config,
            stream_mode=["updates", "messages"],
            context=my_context,
        )
        print("="*50)
        print("客服：", end="")
        await chunks_handle(respChunks, my_agent, config, my_context)


asyncio.run(shop_agent())

