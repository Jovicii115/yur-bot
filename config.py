import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Настройки YandexGPT
# YANDEX_GPT_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
# YANDEX_GPT_MODEL = "yandexgpt-lite"  # Или "yandexgpt-pro" для Pro версии

# Настройки Deepseek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"  # Или "deepseek-coder"
# Настройки RAG
CHUNK_SIZE = 1000  # Размер чанков для разделения документов
CHUNK_OVERLAP = 200  # Перекрытие между чанками
EMBEDDINGS_MODEL = "cointegrated/LaBSE-en-ru"  # Модель для эмбеддингов
VECTOR_DB_PATH = BASE_DIR / "vector_db"  # Путь к векторной БД

# Настройки контекста
MAX_CONTEXT_LENGTH = 10  # Максимальное количество сообщений в контексте
