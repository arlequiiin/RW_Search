#!/usr/bin/env python3
"""
Скрипт для загрузки документа в векторную базу
"""
import sys
import os
import uuid
from datetime import datetime

# Добавляем корневую директорию в PATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.docs_parser import extract_text
from src.chunker import split_text
from src.embeddings import EmbeddingModel
from src.storage import get_chroma
from src.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS


def ingest_document(file_path: str, active: bool = True, author: str = "unknown", tags: str = ""):
    """
    Загрузка документа в векторную базу

    Args:
        file_path: Путь к файлу
        active: Статус активности документа
        author: Автор документа
        tags: Теги через запятую
    """
    print(f"\n📄 Обработка файла: {file_path}")

    # 1. Проверка существования файла
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return False

    # 2. Извлечение текста
    try:
        text = extract_text(file_path)
        print(f"✅ Извлечено {len(text)} символов")
    except Exception as e:
        print(f"❌ Ошибка при извлечении текста: {e}")
        return False

    # 3. Разбиение на чанки
    chunks = split_text(text, max_length=CHUNK_SIZE_TOKENS * 4, overlap=CHUNK_OVERLAP_TOKENS * 4)
    print(f"✅ Создано {len(chunks)} чанков")

    if not chunks:
        print("⚠️  Документ пустой, пропускаем")
        return False

    # 4. Создание эмбеддингов
    print("🔄 Создание эмбеддингов...")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    embeddings = embedding_model.encode(chunks)
    print(f"✅ Создано {len(embeddings)} эмбеддингов")

    # 5. Подготовка метаданных
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    created_at = datetime.now().isoformat()

    # 6. Добавление в ChromaDB
    print("💾 Сохранение в векторную базу...")
    client, collection = get_chroma()

    chunk_ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        chunk_ids.append(chunk_id)

        metadata = {
            'doc_id': doc_id,
            'filename': filename,
            'file_path': file_path,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'active': active,
            'author': author,
            'tags': tags,
            'created_at': created_at
        }
        metadatas.append(metadata)

    # Добавление в коллекцию
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=chunk_ids
    )

    print(f"✅ Документ успешно добавлен!")
    print(f"   Doc ID: {doc_id}")
    print(f"   Чанков: {len(chunks)}")
    print(f"   Имя файла: {filename}")

    return True


def main():
    """Точка входа для скрипта"""
    if len(sys.argv) < 2:
        print("Использование: python ingest_file.py <путь_к_файлу> [author] [tags]")
        print("\nПример:")
        print("  python ingest_file.py data/docs/инструкция.docx")
        print("  python ingest_file.py data/docs/инструкция.docx 'Иван Иванов' 'инструкция,важно'")
        sys.exit(1)

    file_path = sys.argv[1]
    author = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    tags = sys.argv[3] if len(sys.argv) > 3 else ""

    success = ingest_document(file_path, active=True, author=author, tags=tags)

    if success:
        print("\n🎉 Готово!")
    else:
        print("\n❌ Произошла ошибка")
        sys.exit(1)


if __name__ == "__main__":
    main()
