"""
MCP 服务器 订单服务 通过 http 接收请求
"""
import json

from fastmcp import FastMCP, Context

mcp = FastMCP("order_demo")



@mcp.tool
async def send_sms(sms: str,  phone_number:str , ctx: Context) -> str:
    """
    发送短信到手机号

    Args:
        sms: 短信内容
        phone_number: 手机号
    Returns:
        发送情况
    """
    return "发送成功"

@mcp.resource(uri="doc://refund")
def doc_refund() :
    return """
        7天无理由退货
        退货条件：在7天内，用户可以无理由退货。
        退货流程：
            1. 用户在7天内联系客服，说明退货原因。
            2. 客服审核退货申请，确认无误后，用户可以退货。
            3. 退货完成后，询问用户是否需要发送通知短信。
        退货状态：
            1. 未付款不让退货，
            2. 取消不允许
            3. 退货成功不允许退货
    """

@mcp.prompt
def refund_response(order_id: str, amount: str) -> str:
    """退款回复模板"""
    return ( f"""
        好的，已为您处理订单 {order_id} 的退款。"
        退款金额：{amount} 元
        预计 3 个工作日内退回原支付方式。
        如有疑问可随时联系我们。
     """)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        # 启动 HTTP 服务器的端口
        port=28031,
        # 启动 HTTP 服务器的主机地址
        host="localhost",
        # 启动 HTTP 服务器的路径 默认是 /mcp
        path="/my-mcp",
    )
