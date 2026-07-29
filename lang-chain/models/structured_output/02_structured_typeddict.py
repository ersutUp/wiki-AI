from typing import Annotated, TypedDict

from my_llm import chat_llm


class Movie(TypedDict):
    title: Annotated[str, "电影标题"]
    director: Annotated[str, "导演"]
    year: Annotated[int, "上映年份"]
    actors: Annotated[list[str], "主要演员列表"]

structured_agent = chat_llm.with_structured_output(Movie,  method="function_calling")
res = structured_agent.invoke("你好，我想知道《速度与激情》的导演和演员")
print(type(res))
print(res)


