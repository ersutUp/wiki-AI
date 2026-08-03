"""
MCP 服务器 通过 http 接收请求
"""
import json
import random

from fastmcp import FastMCP

mcp = FastMCP("http_demo")

# 内存中的用户存储，key 为用户 id，value 为用户信息
# 注意：仅用于 demo，进程重启后数据丢失
_users: dict[int, dict] = {
    1: {"user_id": 1, "name": "张三", "age": 25, "email": "zhangsan@example.com"},
    2: {"user_id": 2, "name": "李四", "age": 30, "email": "lisi@example.com"},
    3: {"user_id": 3, "name": "王五", "age": 28, "email": "wangwu@example.com"},
}

@mcp.prompt
def prompt_user():
    return "字段：ID、用户姓名、年龄,邮箱为敏感数据要脱敏处理。"


@mcp.resource(uri="user://get_user/{user_ids}", mime_type="application/json")
def get_user(user_ids: str) -> str:
    """根据用户 id 批量查询用户信息

    Args:
        user_ids: 用户 id 列表，多个用英文逗号分隔，例如 "1,2,3"

    说明：
        资源模板（URI template）的参数 FastMCP 只支持 int/float/bool/str，
        不支持 list。所以这里用逗号分隔的字符串承接，内部再解析成 id 列表。

    用法示例：
        user://get_user/1              → 查单个
        user://get_user/1,2,3          → 批量查多个
        user://get_user/1,2,99         → 不存在的 id 会标记为 null
    """
    # 把 "1,2,3" 解析成 [1, 2, 3]，忽略空串和非数字片段
    raw_ids = [s.strip() for s in user_ids.split(",") if s.strip()]
    id_list: list[int] = []
    for s in raw_ids:
        if s.isdigit():
            id_list.append(int(s))

    # 逐个查询，按 id 聚合结果；不存在的置为 None
    result: dict[int, dict | None] = {}
    for uid in id_list:
        result[uid] = _users.get(uid)

    return json.dumps(result, ensure_ascii=False) # ensure_ascii 禁止中文转 u 码





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