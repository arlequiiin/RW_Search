#!/usr/bin/env python3
"""Тест RAG pipeline"""

from src.rag_pipeline import create_rag_pipeline

print("=" * 60)
print("ТЕСТ RAG PIPELINE")
print("=" * 60)

# Создание pipeline
rag = create_rag_pipeline()

# Статистика
stats = rag.get_stats()
print(f"\n📊 Статистика базы:")
print(f"   Всего чанков: {stats['total_chunks']}")
print(f"   Коллекция: {stats['collection_name']}")

# Тестовый запрос
query = "Что делать при ошибке фильтра?"
print(f"\n❓ Запрос: {query}")

result = rag.query(query)

print(f"\n💬 Ответ:")
print(result['answer'])

print(f"\n📚 Источники:")
for source in result['sources']:
    print(f"   [{source['index']}] {source['filename']} (distance: {source['distance']:.4f})")

print("\n" + "=" * 60)
print("ТЕСТ ЗАВЕРШЕН")
print("=" * 60)
