"""
信息安全处理
"""

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import glm_llm


# 工具定义
@tool
def set_user_info(email: str = None, id_card: str = None, credit_card: str = None, ip: str = None, mac_address: str = None) -> str:
    """
    模拟设置用户信息工具,部分信息可以为空。

    Args:
        email: 用户邮箱
        id_card: 身份证号
        credit_card: 信用卡号
        ip: 用户IP地址
        mac_address: 用户MAC 地址
    """
    print("=" * 60)
    user_info = f"已设置用户信息: 用户的邮箱为 {email}, 身份证号为 {id_card}, 信用卡号为 {credit_card}, 用户IP地址为 {ip}, 用户MAC 地址为 {mac_address}"
    print(f"【set_user_info工具正在执行】- {user_info}")
    print("=" * 60, end="\n\n")
    return user_info



agent = create_agent(
    model=glm_llm,
    tools=[set_user_info],
    middleware=[
        # 邮箱脱敏， mask 策略是将邮箱 脱敏 例如 83887657@qq.com 脱敏为 83887657@****.com
        PIIMiddleware(
            pii_type="email",
            strategy="mask",
            apply_to_input=False, # 模型调用前处理
            apply_to_output=True, # 模型调用后处理
            apply_to_tool_results=True, # 工具调用后处理
        ),
        # 隐藏IP地址， redact 策略是将IP地址替换为占位符 [REDACTED_IP]
        PIIMiddleware(
            pii_type="ip",
            strategy="redact",
            apply_to_input=True, # 模型调用前处理
            apply_to_output=False, # 模型调用后处理
            apply_to_tool_results=True, # 工具调用后处理
        ),
        # 银行卡脱敏， mask 策略是将银行卡号 脱敏 例如 62170012345678901234567890123456 脱敏为 6217************1234
        PIIMiddleware(
            pii_type="credit_card",
            strategy="mask",
            apply_to_input=True, # 模型调用前处理
            apply_to_output=True, # 模型调用后处理
            apply_to_tool_results=True, # 工具调用后处理
        ),
        # MAC 地址脱敏， mask 策略是将 MAC 地址 脱敏 例如 00:1A:2B:3C:4D:5E 脱敏为 **:**:**:**:**:5E
        PIIMiddleware(
            pii_type="mac_address",
            strategy="mask",
            apply_to_input=True, # 模型调用前处理
            apply_to_output=True, # 模型调用后处理
            apply_to_tool_results=True, # 工具调用后处理
        ),
        # 身份证号脱敏， hash 策略是将身份证号 进行哈希处理
        PIIMiddleware(
            pii_type="id_card",
            strategy="hash",
            detector=r"\d{17}[\dXx]", # 身份证的正则表达式
            apply_to_input=True, # 模型调用前处理
            apply_to_output=True, # 模型调用后处理
            apply_to_tool_results=True, # 工具调用后处理
        ),
    ],
    system_prompt="你是用户信息管理助手",
)

m = HumanMessage("帮我录入我的信息，我的邮箱是 83887657@qq.com; 身份证号为 13013119911024876X; 信用卡号为 3530111333300000。 我的IP地址是 192.168.1.100, 我的MAC 地址是 00:1A:2B:3C:4D:5E")

res = agent.invoke(
    {"messages": [m]},
)
print(res)
print("+" * 60)
for msg in res["messages"]:
    print(type(msg).__name__,end=" ")
    print(msg.content)
