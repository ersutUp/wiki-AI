"""
MCP 服务器 通过 stdio 接收请求
"""
import random

from fastmcp import FastMCP

mcp_stdio = FastMCP("stdio_demo")


@mcp_stdio.tool
def get_weather(city: str) -> str:
    """获取天气"""
    # 随机生成天气
    weather = random.choice(["晴朗的", "阴的", "雨的"])
    return f"{city}的天气是 {weather}"


if __name__ == "__main__":
    mcp_stdio.run(
        transport="stdio",
    )