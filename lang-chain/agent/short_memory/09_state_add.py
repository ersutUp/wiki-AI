"""
演示自定义状态中 Annotated[list, add] 的追加效果

add reducer（来自 operator.add）将每次更新追加到列表末尾，而非替换。
例如：已有 ["a", "b"]，更新 ["c"] → 结果 ["a", "b", "c"]
"""
from operator import add
from typing import Annotated

from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from my_llm import glm_llm


class NoteState(AgentState):
    """自定义状态：notes 使用 add reducer，每次更新会追加而非替换"""
    notes: Annotated[list[str], add]


@tool
def add_note(
    content: str,
    call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """添加一条笔记。"""
    # 更新 notes 字段时，add reducer 会自动将新内容追加到已有列表末尾
    return Command(update={
        "notes": [content],
        "messages": [ToolMessage(
            content=f"笔记已添加",
            tool_call_id=call_id,
        )],
    })


agent = create_agent(
    model=glm_llm,
    system_prompt="你是笔记助手。用户说的事情，用 add_note 记录下来。",
    tools=[add_note],
    checkpointer=InMemorySaver(),
    state_schema=NoteState,
)

config = {"configurable": {"thread_id": "note-demo"}}

# 第一轮：添加两条笔记
print("=" * 40)
print("第一轮：添加笔记")
print("=" * 40)
agent.invoke(
    {"messages": [HumanMessage("记住：下周一下午3点开会，另外别忘了买牛奶")]},
    config=config,
)
state = agent.get_state(config=config)
print(f"notes = {state.values['notes']}")
# 输出示例：notes = ['下周一下午3点开会', '别忘了买牛奶']

# 第二轮：再添加一条
print()
print("=" * 40)
print("第二轮：追加更多笔记")
print("=" * 40)
agent.invoke(
    {"messages": [HumanMessage("再记一条：周五前提交报告")]},
    config=config,
)
state = agent.get_state(config=config)
print(f"notes = {state.values['notes']}")
# 输出示例：notes = ['下周一下午3点开会', '别忘了买牛奶', '周五前提交报告']

# 对比：普通字段（无 add reducer）的更新是替换，而 notes 是追加
print()
print("关键区别：")
print("  普通字段更新 → 替换旧值")
print("  Annotated字段更新 → 追加到旧列表末尾")