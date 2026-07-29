"""
信息安全处理, 棅查用户输入是否包含禁用词
"""
from pyexpat.errors import messages
from typing import Any

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import PIIMiddleware, before_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from my_llm import glm_llm


#禁用词
banned_words = ["滚蛋","尼玛"]


@before_model(can_jump_to=["end"])
def check_banned_words(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    if messages:
        content = messages[0].content
        for word in banned_words:
            # 检查是否存在
            i = content.index(word)
            if i > -1:
                return {
                    "messages": [{
                            "role": "assistant",
                            "content": "您输入的内容包含禁用词，无法处理",
                        }],
                    "jump_to": "end",
                }
    return None



agent = create_agent(
    model=glm_llm,
    middleware=[
        check_banned_words,
    ],
    system_prompt="你是感情助手",
)

m = HumanMessage("滚蛋，我不要你")

res = agent.invoke(
    {"messages": [m]},
)
print(res)
print("+" * 60)
for msg in res["messages"]:
    print(type(msg).__name__,end=" ")
    print(msg.content)
