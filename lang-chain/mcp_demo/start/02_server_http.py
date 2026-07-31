"""
MCP 服务器 通过 stdio 接收请求
"""
import random

from fastmcp import FastMCP

mcp_stdio = FastMCP("http_demo")


@mcp_stdio.tool
def get_weather(city: str) -> str:
    """获取天气"""
    # 随机生成天气
    weather = random.choice(["晴朗的", "阴的", "雨的"])
    return f"{city}的天气是 {weather}"


if __name__ == "__main__":
    mcp_stdio.run(
        transport="http",
        # 启动 HTTP 服务器的端口
        port=28008,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )