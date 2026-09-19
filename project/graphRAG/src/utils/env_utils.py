from os import getenv

import dotenv

dotenv.load_dotenv(override=True)

LLM_API_KEY = getenv("LLM_API_KEY")
LLM_BASE_URL = getenv("LLM_BASE_URL")
