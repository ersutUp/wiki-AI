"""
回复
"""


from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from my_llm import glm_llm

data_list = ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "桃子", "橘子"]

@tool
def query_all_data() -> str:
    """查询所有数据。"""
    return f"所有数据为：{', '.join(data_list)}"

@tool
def delete_data(data: str) -> str:
    """
    删除指定数据， 用户可能中断该工具，用户可以指定删除的数据。
    """
    data_list.remove(data)
    return f"已删除数据：{data}"


agent =create_agent(
    model=glm_llm,
    tools=[query_all_data, delete_data],
    system_prompt="你是数据管理助手"
                  "delete_data工具可能会被用户中断，用户可以指定删除的数据。"
                  "如果用户修改数据，你无需重新调用删除",
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            description_prefix="请确认以下信息是否正确：",
            interrupt_on={
                "delete_data": InterruptOnConfig(
                    allowed_decisions=["approve","reject","edit","respond"],
                    description="是否删除数据？"
                ),
                "query_all_data": False,
            }
        )
    ]
)

config = RunnableConfig(configurable={"thread_id": "chat1"})

res = agent.invoke(
    {"messages": [HumanMessage("查询数据，如果有橘子就删除了，最后展示出所有数据")]},
    config=config,
    version='v2'
)
#是否中断
interrupts = res.interrupts

decisions = []

if interrupts:
    print(res)
    # 打印中断信息
    action_requests = interrupts[0].value["action_requests"]
    review_configs = interrupts[0].value["review_configs"]

    description = action_requests[0]["description"]
    tool_name = action_requests[0]["name"]
    tool_args = action_requests[0]["args"]
    print(f"工具名称：{tool_name}")
    print(f"参数：{tool_args}")
    print(f"描述：{description}")

    allowed_decisions = review_configs[0]['allowed_decisions']

    while True:
        decision = input(f"按选择操作（{','.join(allowed_decisions) if allowed_decisions else '无'}）：")
        if decision == "approve":
            decisions.append({"type": "approve"})
            break
        elif decision == "reject":
            reject_reason = input("请输入拒绝原因：")
            decisions.append({"type": "reject", "message": reject_reason})
            break
        elif decision == "edit":
            tool_new_args = {}
            for arg, value in tool_args.items():
                tool_new_args[arg] = input(f"请输入{arg}的值(原值为：{value})：")

            decisions.append({"type": "edit", "edited_action": { "name": tool_name , "args": tool_new_args }})
            break
        elif decision == "respond":
            respond_message = input("请输入回复内容：")
            decisions.append({"type": "respond", "message": respond_message})
            break
        else:
            print("无效操作，请重新输入")

    print("处理中（恢复中断）")
    # 恢复中断
    res = agent.invoke(
        Command(
            resume={"decisions": decisions},
        ),
        config=config,
        version='v2'
       )

print(res)
if len(res.interrupts) == 0:
    print(res.value["messages"][-1].content)
else:
    print("又调用工具")
