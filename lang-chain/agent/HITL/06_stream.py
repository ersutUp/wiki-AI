"""
回复
"""


from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, GraphOutput, Interrupt

from my_llm import glm_llm

data_list = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "桃子", "橘子"]

def chunks_handle(respChunks, agent):
    """处理响应 chunk"""
    for chunk in respChunks:
        # print(chunk)
        # 处理 updates 类型的 chunk
        chunkType = chunk[0]
        if chunkType == "updates":
            # print(chunk)
            chunkData = chunk[1]
            # 检查是否中断
            if chunkData and "__interrupt__" in chunkData:
                decisions = interrupts_handle(chunkData["__interrupt__"][0])
                # 恢复中断
                respChunksTool = agent.stream(
                    Command(
                        resume={"decisions": decisions},
                    ),
                    config=config,
                    stream_mode=["updates", "messages"]
                )
                chunks_handle(respChunksTool, agent)
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

@tool
def query_all_data() -> str:
    """查询所有数据。"""
    return f"所有数据为：{', '.join(data_list)}"

@tool
def delete_data(data: str) -> str:
    """
    删除指定数据， 用户可能中断该工具，用户可以指定删除的数据。
    """
    if data not in data_list:
        return f"数据{data}不存在"
    data_list.remove(data)
    return f"已删除数据：{data}\n"



@tool
def placeholder_tool(msg: str) -> str:
    """
    需要用户提供信息时调用这个工具
    补充数据时调用这个工具
    确认信息时调用这个工具
    Args:
        msg: 用户提供的信息
    """
    raise RuntimeError("错误不应该运行到这里")

agent =create_agent(
    model=glm_llm,
    tools=[query_all_data, delete_data, placeholder_tool],
    system_prompt="""
        你是数据管理助手
        delete_data工具可能会被用户中断，用户可以指定删除的数据。
        如果用户修改数据，你无需重新调用删除
        需要用户提供信息或者用户对话中缺少关键信息时调用placeholder_tool工具，让其提供信息
                  """,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            description_prefix="请确认以下信息是否正确：",
            interrupt_on={
                "placeholder_tool": InterruptOnConfig(
                    allowed_decisions=["respond"],
                    description="需要输入信息"
                ),
                "delete_data": InterruptOnConfig(
                    allowed_decisions=["approve", "reject", "edit"],
                    description="需要确认删除数据"
                ),
                "query_all_data": False,
            }
        )
    ]
)

config = RunnableConfig(configurable={"thread_id": "chat1"})

respChunks = agent.stream(
    # {"messages": [HumanMessage("帮我删除数据，删除后展示所有数据")]},# 调用 placeholder_tool 后再调用 delete_data 这两个不是同时的
    {"messages": [HumanMessage("帮我删除苹果、香蕉数据,同时删除，删除后展示所有数据")]}, #多工具同时调用
    config=config,
    stream_mode=["updates", "messages"]
)
chunks_handle(respChunks, agent)
#
# #是否中断
# interrupts = res.interrupts
#
# if interrupts:
#     resp = interrupts_handle(interrupts[0], agent)
#     print(f"resp:{resp}")
#     res = resp
#
# print(res.value["messages"][-1].content)
