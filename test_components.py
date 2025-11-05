#!/usr/bin/env python3
"""
Тестовый скрипт для проверки всех компонентов системы
"""
import sys
import os

print("=" * 60)
print("ПРОВЕРКА КОМПОНЕНТОВ RAG-СИСТЕМЫ")
print("=" * 60)

# 1. Проверка импорта docs_parser
print("\n[1/5] Проверка docs_parser...")
try:
    from src.docs_parser import extract_text
    test_file = "data/docs/Ошибка фильтра.docx"
    if os.path.exists(test_file):
        text = extract_text(test_file)
        print(f"✅ docs_parser работает! Извлечено {len(text)} символов")
        print(f"   Первые 100 символов: {text[:100]}...")
    else:
        print(f"⚠️  Тестовый файл {test_file} не найден")
except Exception as e:
    print(f"❌ Ошибка в docs_parser: {e}")

# 2. Проверка embeddings
print("\n[2/5] Проверка embeddings (SentenceTransformers)...")
try:
    from src.embeddings import EmbeddingModel
    model = EmbeddingModel("all-MiniLM-L6-v2")
    test_texts = ["Тестовый текст на русском", "Another test in English"]
    embeddings = model.encode(test_texts)
    print(f"✅ Embeddings работают! Размерность: {embeddings.shape}")
except Exception as e:
    print(f"❌ Ошибка в embeddings: {e}")

# 3. Проверка chunker
print("\n[3/5] Проверка chunker...")
try:
    from src.chunker import split_text
    test_text = "Это тестовый текст. " * 100
    chunks = split_text(test_text, max_length=200, overlap=50)
    print(f"✅ Chunker работает! Создано {len(chunks)} чанков")
except Exception as e:
    print(f"❌ Ошибка в chunker: {e}")

# 4. Проверка storage (ChromaDB)
print("\n[4/5] Проверка storage (ChromaDB)...")
try:
    from src.storage import get_chroma
    client, collection = get_chroma()
    count = collection.count()
    print(f"✅ ChromaDB подключена! Документов в коллекции: {count}")
except Exception as e:
    print(f"❌ Ошибка в storage: {e}")
    print(f"   Подсказка: возможно нужно исправить импорт chromadb")

# 5. Проверка Ollama
print("\n[5/5] Проверка Ollama (llama3:8b)...")
try:
    import ollama
    response = ollama.chat(model='llama3:8b', messages=[
        {'role': 'user', 'content': 'Ответь одним словом: работает?'}
    ])
    answer = response['message']['content']
    print(f"✅ Ollama работает! Ответ: {answer[:100]}")
except Exception as e:
    print(f"❌ Ошибка в Ollama: {e}")

print("\n" + "=" * 60)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 60)
