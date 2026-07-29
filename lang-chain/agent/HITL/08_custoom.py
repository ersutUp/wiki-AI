"""
根据不同的水果，执行不同的操作
- 如果是 ignore_fruits 中的水果，直接删除
- 如果不是 ignore_fruits 中的水果，需要确认是否删除
"""
from typing import Dict, Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, after_model
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command, GraphOutput, Interrupt, interrupt

from my_llm import glm_llm

data_list = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "桃子", "橘子"]
# 删除时忽略的水果
ignore_fruits = ["苹果", "香蕉"]

# 处理中断信息
def interrupts_handle(interrupt: Interrupt, agent) -> GraphOutput:
    """处理中断信息"""

    while True:
        decisions = []
        action_requests = interrupt.value["action_requests"]
        review_configs = interrupt.value["review_configs"]
        print(f"需要处理{len(review_configs)}条数据")
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

        print("处理中（恢复中断）")
        print(decisions)
        # 恢复中断
        resp = agent.invoke(
            Command(
                resume={"decisions": decisions},
            ),
            config=config,
            version='v2'
        )

        if len(resp.interrupts) == 0:
            break
        else:
            interrupt = resp.interrupts[0]

    return resp
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
    return f"已删除数据：{data}"



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

# 自定义 HumanInTheLoopCustom
@after_model
def HumanInTheLoopCustom(state: AgentState, runtime: Runtime) -> Dict[str, Any] | None:
    """
    自定义 HumanInTheLoopCustom
    处理中断信息，根据用户输入判断是否执行 tool_call
    """
    print(state)
    last_message = state["messages"][-1]
    # print(type(last_message))
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
        inter = None
        inter_tool_calls = []
        action_requests = []
        review_configs = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")
            # 如果是删除工具，且数据不在忽略列表中，需要中断
            if tool_name == "delete_data" and tool_args.get("data") not in ignore_fruits:
                # value={'action_requests': [{'name': 'delete_data', 'args': {'data': '苹果'}, 'description': '需要确认删除数据'}, {'name': 'delete_data', 'args': {'data': '香蕉'}, 'description': '需要确认删除数据'}], 'review_configs': [{'action_name': 'delete_data', 'allowed_decisions': ['approve', 'reject', 'edit']}, {'action_name': 'delete_data', 'allowed_decisions': ['approve', 'reject', 'edit']}]}
                action_requests.append({'name': tool_name, 'args': tool_args, 'description': '需要确认删除数据'})
                review_configs.append({"action_name": tool_name, "allowed_decisions": ["approve", "reject"]})
                # 保持 tool_call 的顺序！！！！
                inter_tool_calls.append(tool_call)

        # 如果有需要中断的数据，中断
        if len(action_requests) > 0:
            # 中断
            inter = interrupt(
                value={
                    "action_requests": action_requests,
                    "review_configs": review_configs
                }
            )
        if inter:
            msg = []
            decisions = inter["decisions"]
            # 遍历 tool_call 和 decision，根据 decision 决定是否执行 tool_call
            for tool_call, decision in zip(inter_tool_calls,decisions):
                if decision["type"] == "approve":
                    continue
                elif decision["type"] == "reject":
                    msg.append(ToolMessage(
                            content="用户拒绝执行删除工具",
                            tool_call_id=tool_call["id"],
                        ))

            if len(msg) > 0:
                return { "messages": msg }
    return None


agent =create_agent(
    model=glm_llm,
    tools=[query_all_data, delete_data],
    system_prompt="""
        你是数据管理助手
        delete_data工具可能会被用户中断，用户可以指定删除的数据。
        如果用户修改数据，你无需重新调用删除
        需要用户提供信息或者用户对话中缺少关键信息时调用placeholder_tool工具，让其提供信息
                  """,
    checkpointer=InMemorySaver(),
    middleware=[ HumanInTheLoopCustom ]
        # HumanInTheLoopMiddleware(
        #     description_prefix="请确认以下信息是否正确：",
        #     interrupt_on={
        #         "placeholder_tool": InterruptOnConfig(
        #             allowed_decisions=["respond"],
        #             description="需要输入信息"
        #         ),
        #         "delete_data": InterruptOnConfig(
        #             allowed_decisions=["approve", "reject", "edit"],
        #             description="需要确认删除数据"
        #         ),
        #         "query_all_data": False,
        #     }
        # )
    # ]
)

config = RunnableConfig(configurable={"thread_id": "chat1"})

res = agent.invoke(
    # {"messages": [HumanMessage("帮我删除数据，删除后展示所有数据")]},
    {"messages": [HumanMessage("帮我删除葡萄，苹果、香蕉、西瓜,同时删除，删除后展示所有数据")]},
    config=config,
    version='v2'
)

#是否中断
interrupts = res.interrupts

if interrupts:
    resp = interrupts_handle(interrupts[0], agent)
    print(f"resp:{resp}")
    res = resp

print(res.value["messages"][-1].content)
