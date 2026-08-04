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

# 下一个可用的用户 id（自增主键）
_next_id = 4

@mcp.tool
async def add_user( ctx: Context, name: str, age: int, email: str = None) -> str:
    """添加用户，id 由系统自动分配

    Args:
        name: 用户名
        age: 年龄
        email: 邮箱（可选）

    Returns:
        操作结果说明（包含新分配的用户 id）
    """

    if email is None:
        # response_type=str：框架自动生成带 "value" 字段的表单，返回时自动解构，
        # 所以 email_elicit.data 直接就是 str（邮箱字符串），不需要再 ["value"]
        email_elicit = await ctx.elicit("请提供邮箱", response_type=str)
        if email_elicit.action == "accept":
            email = email_elicit.data
        elif email_elicit.action == "decline":
            return "用户未填写邮箱信息，本次添加用户失败。"
        else:
            return "缺少邮箱信息"
    global _next_id
    # 用户名重复校验
    for u in _users.values():
        if u["name"] == name:
            return f"用户名 {name} 已存在，添加失败"

    user_id = _next_id




    _users[user_id] = {"user_id": user_id, "name": name, "age": age, "email": email}
    _next_id += 1
    return f"用户 {name} 添加成功，id 为 {user_id}"


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
        port=28019,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )