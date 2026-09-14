"""项目公共常量，各 demo 文件统一引用。"""

# Milvus 连接
MILVUS_URI = "http://localhost:19530"

# 集合
COLLECTION_NAME = "book_search"
VECTOR_FIELD = "embedding"  # 稠密向量字段（语义搜索）
SPARSE_VECTOR_FIELD = "sparse_embedding"  # 稀疏向量字段（BM25 Function 自动生成）
TEXT_FIELD = "description"  # 文本字段（BM25 Function 输入）

# embedding 模型
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
VECTOR_DIM = 2048  # 与模型输出维度一致