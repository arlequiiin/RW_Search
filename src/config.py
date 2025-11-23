import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
METADATA_DB = os.path.join(DATA_DIR, "metadata.db")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 5

LLM_MODEL_NAME = "qwen2.5:14b-instruct-q4_K_M"
LLM_MAX_TOKENS = 1024

LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
