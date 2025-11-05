#!/usr/bin/env python3
"""Тест качества эмбеддингов на русском"""

from sentence_transformers import SentenceTransformer, util
import numpy as np

print("=" * 70)
print("ТЕСТ КАЧЕСТВА ЭМБЕДДИНГОВ НА РУССКОМ ЯЗЫКЕ")
print("=" * 70)

# Текущая модель
model = SentenceTransformer("all-MiniLM-L6-v2")

# Тестовые тексты
query = "как написать расхождения?"
documents = [
    "Каждый день отдел закрытия пишут расхождения между 1 и 3 регистром",
    "Перед выполнением операции нужно открыть базу",
    "Для создания обращения укажите номер расхождения",
    "Настройка параметров конфигурации системы"
]

print(f"\n🔍 Запрос: '{query}'")
print(f"\n📄 Документы для сравнения:")
for i, doc in enumerate(documents, 1):
    print(f"   [{i}] {doc}")

# Создание эмбеддингов
query_emb = model.encode(query, convert_to_tensor=True)
doc_embs = model.encode(documents, convert_to_tensor=True)

# Вычисление косинусного сходства
similarities = util.cos_sim(query_emb, doc_embs)[0]

print(f"\n📊 Релевантность (cosine similarity):")
for i, sim in enumerate(similarities, 1):
    print(f"   [{i}] {sim.item():.4f} ({sim.item()*100:.2f}%)")

# Сортировка по релевантности
ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
print(f"\n🏆 Рейтинг по релевантности:")
for rank, (idx, sim) in enumerate(ranked, 1):
    print(f"   {rank}. Документ [{idx+1}] - {sim.item()*100:.2f}%")

print("\n" + "=" * 70)
