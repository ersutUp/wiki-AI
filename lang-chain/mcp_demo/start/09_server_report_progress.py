"""
MCP 服务器 通过 http 接收请求
"""
import asyncio
import json

from fastmcp import FastMCP, Context

mcp = FastMCP("http_demo")

# 内存中的用户存储，key 为用户 id，value 为用户信息
# 注意：仅用于 demo，进程重启后数据丢失
_users: dict[int, dict] = {
    1: {"user_id": 1, "name": "张三", "age": 25, "email": "zhangsan@example.com"},
    2: {"user_id": 2, "name": "李四", "age": 30, "email": "lisi@example.com"},
    3: {"user_id": 3, "name": "王五", "age": 28, "email": "wangwu@example.com"},
}




@mcp.tool
async def get_user_by_id(user_id: int, ctx: Context) -> str:
    """根据用户 id 查询用户信息

    Args:
        user_id: 用户 id

    Returns:
        用户信息字符串，不存在时返回提示
    """

    #日志
    await ctx.info("info日志")
    await ctx.warning("报警日志")
    await ctx.error("错误日志")
    await ctx.debug("debug日志")

    # report_progress 和 sleep 都是协程，必须在 async 函数里 await 才会真正执行，
    # 否则只生成协程对象被丢弃，进度通知发不出去，客户端回调也不会触发。
    await ctx.report_progress(25, 100, "开始处理")
    await asyncio.sleep(1)
    await ctx.report_progress(50, 100, "查询中")
    user = _users.get(user_id)
    await asyncio.sleep(1)
    await ctx.report_progress(75, 100, "查询完成")

    if user is None:
        return json.dumps({}, ensure_ascii=False)

    await asyncio.sleep(1)
    await ctx.report_progress(100, 100, "处理完成")
    return json.dumps(user, ensure_ascii=False)


@mcp.tool
def list_users() -> str:
    """列出所有用户信息

    Returns:
        所有用户的列表字符串
    """
    if not _users:
        return "当前没有任何用户"

    return str(list(_users.values()))



if __name__ == "__main__":
    mcp.run(
        transport="http",
        # 启动 HTTP 服务器的端口
        port=28009,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )