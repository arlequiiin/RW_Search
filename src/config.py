import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
METADATA_DB = os.path.join(DATA_DIR, "metadata.db")  # sqlite or JSON

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"  # Лучшая для русского: 84% точность, быстрая
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 5  # Увеличено для лучшего покрытия

LLM_MODEL_NAME = "llama3:8b"  # имя в Ollama (пример)
LLM_MAX_TOKENS = 512

LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
