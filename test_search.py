#!/usr/bin/env python3
"""Тест поиска на реальных данных"""

from src.rag_pipeline import create_rag_pipeline

print("=" * 70)
print("ТЕСТ ПОИСКА НА РЕАЛЬНОЙ БАЗЕ ДАННЫХ")
print("=" * 70)

# Создание pipeline
rag = create_rag_pipeline()

# Статистика
stats = rag.get_stats()
print(f"\n📊 Статистика базы: {stats['total_chunks']} чанков")

# Тестовые запросы
test_queries = [
    "Как бороться с вылетами 1С?",
    "Что делать если база Suspect?",
    "Как сделать возврат лотереи?",
    "Как восстановить базу данных?",
    "Что делать при ошибке 409 ККМ?"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*70}")
    print(f"[Запрос {i}/{len(test_queries)}]: {query}")
    print("="*70)
    
    result = rag.query(query, top_k=2)
    
    print(f"\n💬 ОТВЕТ:")
    print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
    
    print(f"\n📚 Источники:")
    for source in result['sources']:
        print(f"   • {source['filename']} (релевантность: {1-source['distance']:.1%})")

print(f"\n{'='*70}")
print("ТЕСТ ЗАВЕРШЕН")
print("="*70)
