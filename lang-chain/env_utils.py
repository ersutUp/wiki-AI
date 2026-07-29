from os import getenv

import dotenv

dotenv.load_dotenv(override=True)

OPENAI_API_KEY = getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = getenv("OPENAI_BASE_URL")

FUNCLOUD_API_KEY = getenv("FUNCLOUD_API_KEY")
FUNCLOUD_BASE_URL = getenv("FUNCLOUD_BASE_URL")

MYSQL_PASSWORD = getenv("MYSQL_PASSWORD")