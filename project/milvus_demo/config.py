"""项目公共常量，各 demo 文件统一引用。"""

# Milvus 连接
MILVUS_URI = "http://localhost:19530"

# 集合
COLLECTION_NAME = "book_search"
VECTOR_FIELD = "embedding"

# embedding 模型
MODEL_NAME = "intfloat/multilingual-e5-large"
VECTOR_DIM = 1024  # 与模型输出维度一致