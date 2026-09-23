from os import getenv

import dotenv

dotenv.load_dotenv(override=True)

LLM_API_KEY = getenv("LLM_API_KEY")
LLM_BASE_URL = getenv("LLM_BASE_URL")

# Neo4j 连接配置（.env 未设置时使用默认本地值）
NEO4J_URI = getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = getenv("NEO4J_DATABASE", "neo4j")
