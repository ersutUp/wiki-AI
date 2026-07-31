"""
MCP 服务器 通过 http 接收请求
"""
import json

from fastmcp import FastMCP

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
def add_user(name: str, age: int, email: str = "") -> str:
    """添加用户，id 由系统自动分配

    Args:
        name: 用户名
        age: 年龄
        email: 邮箱（可选）

    Returns:
        操作结果说明（包含新分配的用户 id）
    """
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
def get_user_by_id(user_id: int) -> str:
    """根据用户 id 查询用户信息

    Args:
        user_id: 用户 id

    Returns:
        用户信息字符串，不存在时返回提示
    """
    user = _users.get(user_id)
    if user is None:
        return json.dumps({}, ensure_ascii=False)

    return json.dumps(user, ensure_ascii=False)


@mcp.tool
def get_user(name: str) -> str:
    """根据用户名查询用户信息

    Args:
        name: 用户名

    Returns:
        用户信息字符串，不存在时返回提示
    """
    for user in _users.values():
        if user["name"] == name:
            return str(user)

    return f"用户 {name} 不存在"


@mcp.tool
def list_users() -> str:
    """列出所有用户信息

    Returns:
        所有用户的列表字符串
    """
    if not _users:
        return "当前没有任何用户"

    return str(list(_users.values()))


@mcp.tool
def delete_user(user_id: int) -> str:
    """根据用户 id 删除指定用户

    Args:
        user_id: 用户 id

    Returns:
        操作结果说明
    """
    if user_id not in _users:
        return f"用户 id {user_id} 不存在，删除失败"

    name = _users[user_id]["name"]
    _users.pop(user_id)
    return f"用户 {name}(id={user_id}) 删除成功"


if __name__ == "__main__":
    mcp.run(
        transport="http",
        # 启动 HTTP 服务器的端口
        port=28008,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )