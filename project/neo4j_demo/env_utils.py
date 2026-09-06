from os import getenv
from pathlib import Path

import dotenv

BASE_DIR = Path(__file__).parent

dotenv.load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = getenv("OPENAI_BASE_URL")

FUNCLOUD_API_KEY = getenv("FUNCLOUD_API_KEY")
FUNCLOUD_BASE_URL = getenv("FUNCLOUD_BASE_URL")
MYSQL_PASSWORD = getenv("MYSQL_PASSWORD")

# Neo4j 连接配置（.env 未设置时使用默认本地值）
NEO4J_URI = getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = getenv("NEO4J_DATABASE", "neo4j")

if not NEO4J_PASSWORD:
    raise ValueError("缺少环境变量：NEO4J_PASSWORD（请在 .env 中设置）")
