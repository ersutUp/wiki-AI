from os import getenv

import dotenv

dotenv.load_dotenv(override=True)

GLM_API_KEY = getenv("GLM_API_KEY")
GLM_BASE_URL = getenv("GLM_BASE_URL")

FUNCLOUD_API_KEY = getenv("FUNCLOUD_API_KEY")
FUNCLOUD_BASE_URL = getenv("FUNCLOUD_BASE_URL")