import os
import uuid
from src.docs_parser import extract_text
from src.chunker import split_text
from src.embeddings import EmbeddingModel
from src.storage import get_chroma
from src.config import DOCS_DIR

def ingest_all():
    emb_model = EmbeddingModel()
    client, collection = get_chroma()

    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        text = extract_text(path)
        chunks = split_text(text)

        embeddings = emb_model.encode(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metas = [{"filename": fname, "chunk": i} for i in range(len(chunks))]

        collection.add(documents=chunks, metadatas=metas, ids=ids)
        print(f"✅ {fname} добавлен ({len(chunks)} чанков)")
    
    client.persist()
    print("✅ Индекс сохранён")

if __name__ == "__main__":
    ingest_all()
