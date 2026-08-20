'''
流式输出时通过 updates 模式获取中断信息
以下代码自动审批是为了演示
'''

import asyncio
from datetime import datetime
from typing import Optional, Union, Literal, AsyncIterator, Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from my_llm import glm_llm, funcloude_claude_llm, funcloude_deepseek_llm

from pydantic import BaseModel, Field

@tool
def get_current_location() -> str:
    """获取当前位置。"""

    return "当前位置为北京市。"

@tool
def get_weather(date: datetime, city: str) -> str:
    """获取指定城市的天气信息。"""
    return f"{date} {city}的天气为晴朗，25°C。"


config = {"configurable": {"thread_id": "chat1"}}
stream_mode = ['messages', 'updates']
version = "v2"

agent =create_agent(
    model=funcloude_deepseek_llm,
    system_prompt="你是天气查询助手",
    tools=[get_current_location, get_weather],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_current_location": InterruptOnConfig(
                    allowed_decisions=["approve", "reject"],
                    description="是否允许获取当前位置？",
                )
            }
        )
    ],
    checkpointer=InMemorySaver()
)


async def process_chunk(chunks: AsyncIterator[dict[str, Any] | Any]):
    async for chunk in chunks:
        data = chunk["data"]
        match chunk["type"]:
            case "updates":
                interrupts = data.get("__interrupt__",None)
                if interrupts:
                    print(f"中断：{interrupts}")
                    decisions = []
                    for interrupt in interrupts:
                        decisions.append({"type": "approve"})
                    resChunk = agent.astream(
                        Command(
                            resume={"decisions": decisions},
                        ),
                        config=config,
                        stream_mode = stream_mode,
                        version = version
                    )
                    await process_chunk(resChunk)
            case "messages":
                # print(chunk)
                msg = data[0]
                if isinstance(msg, AIMessageChunk) and msg.content != "":
                    print(msg.content, end="")

async def main():

    # 流式输出
    resChunk = agent.astream(
        {"messages": [HumanMessage("今天天气怎么样")]},
        config=config,
        stream_mode = stream_mode,
        version = version
    )
    await process_chunk(resChunk)




if __name__ == "__main__":
    asyncio.run(main())