"""
MCP 服务器 订单服务 通过 http 接收请求
"""
import json

from fastmcp import FastMCP, Context

mcp = FastMCP("order_demo")

# 订单状态常量
PENDING = "pending"        # 待付款
PAID = "paid"              # 已付款
SHIPPED = "shipped"        # 已发货
DELIVERED = "delivered"    # 已送达
CANCELLED = "cancelled"    # 已取消
REFUNDED = "refunded"      # 已退款


# 内存中的订单存储（demo 用，进程重启后丢失）
# 结构：key 为 order_id，value 为订单信息
# 字段说明：order_id 订单号、user_id 用户id、product 商品、
#          amount 金额、quantity 数量、status 状态、created_at 创建时间
orders: dict[str, dict] = {
    "ORD20260801001": {
        "order_id": "ORD20260801001",
        "user_id": 1,
        "product": "蓝牙耳机",
        "amount": 199.00,
        "quantity": 1,
        "status": DELIVERED,
        "created_at": "2026-08-01 09:15:00",
    },
    "ORD20260801002": {
        "order_id": "ORD20260801002",
        "user_id": 2,
        "product": "机械键盘",
        "amount": 359.00,
        "quantity": 1,
        "status": SHIPPED,
        "created_at": "2026-08-01 10:30:00",
    },
    "ORD20260802001": {
        "order_id": "ORD20260802001",
        "user_id": 1,
        "product": "充电宝",
        "amount": 89.00,
        "quantity": 2,
        "status": PAID,
        "created_at": "2026-08-02 14:20:00",
    },
    "ORD20260802002": {
        "order_id": "ORD20260802002",
        "user_id": 3,
        "product": "显示器支架",
        "amount": 129.00,
        "quantity": 1,
        "status": PENDING,
        "created_at": "2026-08-02 16:45:00",
    },
    "ORD20260803001": {
        "order_id": "ORD20260803001",
        "user_id": 2,
        "product": "无线鼠标",
        "amount": 79.00,
        "quantity": 3,
        "status": CANCELLED,
        "created_at": "2026-08-03 08:50:00",
    },
    "ORD20260803002": {
        "order_id": "ORD20260803002",
        "user_id": 3,
        "product": "USB-C 扩展坞",
        "amount": 3159.00,
        "quantity": 1,
        "status": DELIVERED,
        "created_at": "2026-08-03 11:10:00",
    },
    "ORD20260803003": {
        "order_id": "ORD20260803003",
        "user_id": 1,
        "product": "笔记本内胆包",
        "amount": 49.00,
        "quantity": 2,
        "status": SHIPPED,
        "created_at": "2026-08-03 18:25:00",
    },
}


@mcp.tool
async def get_order_by_id(order_id: str, ctx: Context) -> str:
    """根据订单号查询订单详情

    Args:
        order_id: 订单号，例如 ORD20260801001

    Returns:
        订单信息 JSON 字符串，不存在时返回空对象 {}
    """
    await ctx.info(f"按订单号查询：{order_id}")
    order = orders.get(order_id)
    if order is None:
        await ctx.warning(f"订单不存在：{order_id}")
        return json.dumps({}, ensure_ascii=False)
    await ctx.info(f"命中订单：{order['product']}，状态：{order['status']}")
    return json.dumps(order, ensure_ascii=False)


@mcp.tool
async def list_orders_by_user(user_id: int, ctx: Context) -> str:
    """根据用户 id 查询该用户的所有订单

    Args:
        user_id: 用户 id

    Returns:
        该用户的订单列表 JSON 字符串
    """
    await ctx.info(f"按用户查询订单：user_id={user_id}")
    user_orders = [o for o in orders.values() if o["user_id"] == user_id]
    await ctx.info(f"该用户共 {len(user_orders)} 笔订单")
    return json.dumps(user_orders, ensure_ascii=False)


@mcp.tool
async def list_orders_by_status(status: str, ctx: Context) -> str:
    """根据订单状态查询订单列表

    Args:
        status: 订单状态，取值为 pending(待付款)/paid(已付款)/shipped(已发货)/delivered(已送达)/cancelled(已取消)

    Return:
        匹配状态的订单列表 JSON 字符串
    """
    await ctx.info(f"按状态查询订单：status={status}")
    matched = [o for o in orders.values() if o["status"] == status]
    if not matched:
        await ctx.warning(f"没有状态为 {status} 的订单")
    return json.dumps(matched, ensure_ascii=False)


@mcp.tool
async def list_all_orders(ctx: Context) -> str:
    """列出所有订单

    Returns:
        所有订单的列表 JSON 字符串
    """
    await ctx.info(f"查询全部订单，共 {len(orders)} 笔")
    return json.dumps(list(orders.values()), ensure_ascii=False)

@mcp.tool
async def refund_order(order_id: str, user_id: int, ctx: Context) -> str:
    """订单退款

    Args:
        order_id: 订单号
        user_id: 操作用户 id（用于校验是否为本人订单）

    Returns:
        退款结果说明
    """
    await ctx.info(f"申请退款：order_id={order_id}, user_id={user_id}")

    # 获取订单（用 get 避免订单不存在时直接抛 KeyError）
    order = orders.get(order_id)

    await ctx.report_progress(33,100, "退款进度：验证订单中")
    # 判断是否存在
    if order is None:
        await ctx.warning(f"订单不存在：{order_id}")
        return "订单不存在"

    # 判断订单是否为该用户的
    if order["user_id"] != user_id:
        await ctx.warning("订单不属于该用户")
        return "订单不属于你"

    # 退款状态、待付款状态、已取消状态 不允许退款
    if order["status"] in (REFUNDED, PENDING, CANCELLED):
        await ctx.warning(f"当前状态不允许退款：{order['status']}")
        return f"订单当前状态为 {order['status']}，不允许退款"

    amount = order["amount"]

    msg = ""
    await ctx.report_progress(66,100, "退款进度：订单退款中")
    if amount <= 300:
        # 已付款 / 已发货 / 已送达 可以退款，这里直接退款
        await ctx.info(f"退款成功：{order_id}")
        msg = f"订单 {order_id} 退款成功"
    else:
        # 大额订单 与用户 再次确认情况
        ok = "确定"
        elicit_result = await ctx.elicit(f"当前订单金额为{amount},确认退款吗？",response_type=[ok,"取消"])
        if elicit_result.action == "accept" :
            if elicit_result.data == ok:
                order["status"] = REFUNDED
                msg = f"订单 {order_id} 退款成功"
            else:
                msg = f"订单 {order_id} 取消退款"
        elif elicit_result.action == "decline":
            msg = f"订单 {order_id} 取消退款({elicit_result.action})"
        else:
            msg = f"订单 {order_id} 退出了退款流程"

    await ctx.report_progress(100,100, "退款进度：订单处理完成")
    return msg

if __name__ == "__main__":
    mcp.run(
        transport="http",
        # 启动 HTTP 服务器的端口
        port=28032,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )
