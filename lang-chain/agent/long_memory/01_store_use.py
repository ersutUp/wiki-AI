"""
store 长期记忆的使用
"""
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

namespace = ("user_001", "info")

# 存储记忆
store.put(
    namespace, "like",
    {
        "city": "石家庄",
        "food": "鱼香肉丝",
    }
)

# 存储记忆
store.put(
    namespace, "name",
    {
        "firstname": "张",
        "lastname": "三",
    }
)

# 获取记忆
print(store.get(namespace, key="like"))

# 获取所有记忆
print(store.search(namespace))

# 删除记忆
store.delete(namespace, key="like")

# 获取记忆
print(store.get(namespace, key="like"))

# 获取记忆
print(store.get(namespace, key="name"))
