import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
METADATA_DB = os.path.join(DATA_DIR, "metadata.db")

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 5 

LLM_MODEL_NAME = "llama3:8b"
LLM_MAX_TOKENS = 512

LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
