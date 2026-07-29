from langchain_core.messages import tool
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from my_llm import chat_llm
from pydantic import BaseModel, Field


class Actor(BaseModel):
    name: str = Field(description="演员姓名")
    role: str = Field(description="演员角色")

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演")
    year: int = Field(description="上映年份")
    # 演员
    actors: list[Actor] = Field(description="主要演员列表")

# 定义提示模板
prompt = ChatPromptTemplate.from_template("""
你是一个专业的电影信息助手，你的任务是根据用户的问题，返回电影的详细信息。
请严格按照下面的格式说明输出，只输出 JSON，不要输出任何其他内容。

用户的问题：介绍《{input}》
""")

chain = prompt | chat_llm | JsonOutputParser(pydantic_object=Movie)

res = chain.invoke({"input": "速度与激情"})
print(type(res))
print(res)
