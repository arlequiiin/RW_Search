import chromadb
from chromadb.config import Settings
import os
from src.config import CHROMA_DIR

def get_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("documents")
    return client, collection
