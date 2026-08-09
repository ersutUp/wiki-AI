import json
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import TavilyClient

from agent.env_utils import TAVILY_API_KEY
from agent.multi_agent.yaml_loader import register_tool


@tool(parse_docstring=True)
def websearch(query: str) -> str:
    """
    网络搜索内容工具

    Args:
        query: 搜索查询

    Returns:
        搜索结果的 JSON字符串
    """

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query)
    results = response.get("results", None)

    if results is None:
        return "没有搜索到相关结果"

    if len(results) == 0:
        return "没有搜索到相关结果"

    return json.dumps(results, ensure_ascii=False)

