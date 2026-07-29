from typing import Annotated, TypedDict

from my_llm import chat_llm

json_schema = {
    "title": "movieInfo",
    "description": "电影信息",
    "type": "object",
    "properties": {
        "title": { "type": "string", "description": "电影标题", },
        "director": { "type": "string", "description": "电影导演", },
        "year": { "type": "integer", "description": "上映年份", },
        "actors": { "type": "array", "description": "主要演员列表", },
        "items": { "type": "string", "description": "演员姓名", },
    },
    "required": ["title", "director", "year", "actors"],
}

json_schema_agent = chat_llm.with_structured_output(json_schema,  method="function_calling")
res  = json_schema_agent.invoke("你好，我想知道《速度与激情》的导演和演员")
print(type(res))
print(res)
