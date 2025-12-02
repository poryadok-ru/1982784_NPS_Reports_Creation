import os
from dotenv import load_dotenv

load_dotenv()

# Настройки базы данных
DB_PARAMS = {
    "host": os.getenv("NPS_DB_HOST"),
    "port": int(os.getenv("NPS_DB_PORT", "5432")),
    "dbname": os.getenv("NPS_DB_NAME"),
    "user": os.getenv("NPS_DB_USER"),
    "password": os.getenv("NPS_DB_PASSWORD"),
}

# Настройки LLM
API_KEY = os.getenv("LLM_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("LITELLM_BASE_URL")

# Настройки приложения
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
PERIOD_START = os.getenv("PERIOD_START")
PERIOD_END = os.getenv("PERIOD_END")

# Настройки Bitrix24
PORADOCK_TOKEN = os.getenv("PORADOCK_LOG_TOKEN")
BITRIX_NPS_REPORTS_FOLDER_ID = int(os.getenv("BITRIX_NPS_REPORTS_FOLDER_ID"))
BITRIX_TOKEN = os.getenv("BITRIX_TOKEN")
BITRIX_USER_ID = int(os.getenv("BITRIX_USER_ID"))