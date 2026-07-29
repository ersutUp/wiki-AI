"""
通过 @tool 注解定义一个工具
"""
import json

from typing import Optional, Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import Field, BaseModel, field_validator

from my_llm import glm_llm

class EmployeeInfoArgs(BaseModel):
    # 一定要加default=None，否则会报错，因为大模型会自动填充这个字段 可能会填充为 'null' python 不识别 null
    employee_id: Optional[str] = Field(description="员工ID", default=None)
    name: Optional[str] = Field(description="员工姓名", default=None)
    age: Optional[int] = Field(description="员工年龄", default=None)
    department: Optional[str] = Field(description="员工部门", default=None)
    #枚举
    sex: Optional[Literal['0', '1']] = Field(description="员工性别 0:女 1:男", default=None)

    @field_validator("employee_id")
    def validate_employee_id(cls, employee_id: str):
        return employee_id.upper()

@tool(args_schema=EmployeeInfoArgs)
def get_employee_info(
        employee_id: str,
        name: str,
        age: int,
        department: str,
        sex: str,
) -> str:
    # 下边这句注释是工具的描述，大模型会根据这个描述来调用这个工具
    """
    获取员工信息。
    Returns:
        员工信息
    """
    # 模拟数据
    employee_info = [
        {"employee_id": "EID1001", "name": "张三", "age": 30, "department": "销售部", "sex": "1"},
        {"employee_id": "EID1002", "name": "李四", "age": 25, "department": "技术部", "sex": "1"},
        {"employee_id": "EID1003", "name": "王五", "age": 35, "department": "销售部", "sex": "1"},
        {"employee_id": "EID1004", "name": "赵六", "age": 25, "department": "技术部", "sex": "0"},
        {"employee_id": "EID1004", "name": "廖十", "age": 25, "department": "技术部", "sex": "0"},
    ]

    # 模拟查询员工信息
    filter_employee = employee_info
    if employee_id:
        filter_employee = [item for item in filter_employee if item["employee_id"] == employee_id]
    if name:
        filter_employee = [item for item in filter_employee if item["name"] == name]
    if age:
        filter_employee = [item for item in filter_employee if item["age"] == age]
    if department:
        filter_employee = [item for item in filter_employee if item["department"] == department]
    if sex:
        filter_employee = [item for item in filter_employee if item["sex"] == sex]

    if filter_employee:
        res = {
            "count": len(filter_employee),
            "employee_info": filter_employee,
        }
        json_str = json.dumps(res, ensure_ascii=False)
        return f"员工的信息为：{json_str}"
    else:
        return f"员工不存在"

agent = create_agent(
    model=glm_llm,
    tools=[get_employee_info],
)

res = agent.invoke({"messages": [HumanMessage("查询技术部的女员工信息")]})
print(res)
print(res["messages"][-1].content)





