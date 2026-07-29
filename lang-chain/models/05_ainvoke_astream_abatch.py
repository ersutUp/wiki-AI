"""
异步执行
"""
import asyncio

from langchain.chat_models import init_chat_model

from env_utils import OPENAI_API_KEY, OPENAI_BASE_URL

agent = init_chat_model(
    model="glm-5.1",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


async def amethod():

    ares = agent.ainvoke("你几岁了？")

    astreamRes = agent.astream("你是谁")

    abatchRes = agent.abatch([
        "你几岁了？",
        "你性别是？",
        "你来自哪里？",
    ])

    for i in range(10):
        await asyncio.sleep(1)
        print("异步执行中")

    async for chunk in astreamRes:
        print(chunk.content, end="")

    print()


    batchRes = await abatchRes

    for chunk in batchRes:
        print(chunk.content, end="")
    print()

    res = await ares
    print(res)

    print("执行结束")

asyncio.run(amethod())