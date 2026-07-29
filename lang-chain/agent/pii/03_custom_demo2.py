"""
信息安全处理, 通过 大模型 检查 大模型 生成的内容是否存在敏感信息
"""
from pyexpat.errors import messages
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import PIIMiddleware, before_model, after_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from my_llm import glm_llm, funcloude_claude_llm


@after_model(can_jump_to=["end"])
def check_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if messages:
        content = messages[0].content
        tmpAgent = create_agent(
            model=glm_llm,
            system_prompt="""
                你是信息检查助手，检查信息内是否包含份量、重量、体重等信息。这些信息一旦出现就返回“不安全”，你只返回“安全”或者“不安全”两种值。
            """,
        )
        tmpRes = tmpAgent.invoke(
            {"messages": [HumanMessage(content=content)]},
        )
        #
        if tmpRes["messages"][-1].content == "不安全":
            return {
                "messages": [{
                        "role": "assistant",
                        "content": "您输入的内容包含不安全的信息，无法处理",
                    }],
                "jump_to": "end",
            }
    return None



agent = create_agent(
    model=funcloude_claude_llm,
    middleware=[
        check_messages,
    ],
    system_prompt="你是减肥规划师",
)

m = HumanMessage("我要减肥，帮我制定一个简单的用餐计划，包含具体的分量或者重量。限制在50字以内。")

res = agent.invoke(
    {"messages": [m]},
)
print(res)
print("+" * 60)
for msg in res["messages"]:
    print(type(msg).__name__,end=" ")
    print(msg.content)
