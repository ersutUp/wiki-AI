from os import getenv

import dotenv

dotenv.load_dotenv(override=True)

FUNCLOUD_API_KEY = getenv("FUNCLOUD_API_KEY")
FUNCLOUD_BASE_URL = getenv("FUNCLOUD_BASE_URL")