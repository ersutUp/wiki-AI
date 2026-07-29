"""
通过 @tool 注解定义一个工具
"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from my_llm import glm_llm


@tool
def get_employee_info(employee_id: str) -> str:
    # 下边这句注释是工具的描述，大模型会根据这个描述来调用这个工具
    """
    获取员工信息。
    Args:
        employee_id: 员工ID
    Returns:
        员工信息
    """
    # 模拟数据
    employee_info = {
        "1001": {"name": "张三", "age": 30},
        "1002": {"name": "李四", "age": 25},
        "1003": {"name": "王五", "age": 35},
    }
    return f"员工 {employee_id} 的信息为：{employee_info.get(employee_id, '未找到')}"

agent = create_agent(
    model=glm_llm,
    tools=[get_employee_info],
)

res = agent.invoke({"messages": [HumanMessage("查询员工1001的信息")]})
print(res)
print(res["messages"][-1].content)





